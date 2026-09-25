#!/usr/bin/env python

from __future__ import annotations

import copy
import json
import math
import os
import warnings
from bisect import bisect_right
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as functional
from torch import Tensor, nn
from torch.utils.data._utils.collate import default_collate

try:
    from pointops import knn_query as _pointops_knn_query
except Exception:  # pragma: no cover - optional CUDA extension.
    _pointops_knn_query = None

try:
    from pointops.sampling import farthest_point_sampling as _pointops_farthest_point_sampling
except Exception:  # pragma: no cover - optional CUDA extension.
    _pointops_farthest_point_sampling = None

ROLE_BACKGROUND = 0
ROLE_FOREGROUND = 1
ROLE_IGNORE = -100
ROLE_NAMES = ("background", "foreground")
ROLE_COLORS = np.array(
    [
        [128, 128, 128],
        [44, 202, 124],
    ],
    dtype=np.uint8,
)

DEFAULT_FUTURE_OFFSETS = (1, 2, 4, 8, 16, 31)
MOTION_PRIOR_DIM = 8
POINTSEG_CACHE_VERSION = 12
POINTSEG_CACHE_COMPATIBLE_VERSIONS = (11, 12)
POINTSEG_CACHE_FIELDS = (
    "point_cloud",
    "priors",
    "labels",
    "weights",
    "class_scores",
    "role_scores",
    "foreground_score",
    "episode_index",
    "frame_index",
    "dataset_index",
)
POINTSEG_CACHE_LABEL_FIELDS = (
    "point_indices",
    "labels",
    "weights",
    "class_scores",
    "role_scores",
    "foreground_score",
    "episode_index",
    "frame_index",
    "dataset_index",
)

_POINTOPS_KNN_FAILED = False
_POINTOPS_FPS_FALLBACK_WARNED = False
_TOOL_SWEEP_LOCAL_FALLBACK_WARNED = False
_TOOL_SWEEP_OFFSETS_WARNED = False


def infer_litept_output_channels(backbone: nn.Module) -> int:
    """Infer LitePT's final feature width without modifying the vendored backbone."""

    encoder_only = bool(getattr(backbone, "enc_mode", False))
    stages = getattr(backbone, "enc" if encoder_only else "dec", None)
    if stages is None:
        raise ValueError("LitePT backbone is missing its encoder/decoder stage container.")
    stage_modules = list(stages.children())
    if not stage_modules:
        raise ValueError("LitePT backbone has no output stage.")
    output_linears = [module for module in stage_modules[-1].modules() if isinstance(module, nn.Linear)]
    if not output_linears:
        raise ValueError("Could not infer LitePT output channels from its final stage.")
    return int(output_linears[-1].out_features)


def build_litept_grid_coord(coord: Tensor, batch: Tensor, grid_size: float) -> Tensor:
    """Build LitePT grid coordinates with an independent origin per sample.

    LitePT's fallback derives one origin from the minimum coordinate of the
    entire packed batch.  That makes a sample's sparse coordinates depend on
    which other samples happen to share the batch.  Compute the integer voxel
    coordinates first, then shift each sample by its own lower grid corner so
    the same point cloud receives the same ``grid_coord`` in every batch.
    """

    if coord.ndim != 2 or coord.shape[-1] != 3:
        raise ValueError(f"Expected coord shape (P, 3), got {tuple(coord.shape)}.")
    if batch.ndim != 1 or batch.shape[0] != coord.shape[0]:
        raise ValueError(f"Expected batch shape ({coord.shape[0]},), got {tuple(batch.shape)}.")
    if not math.isfinite(float(grid_size)) or float(grid_size) <= 0:
        raise ValueError(f"grid_size must be finite and positive, got {grid_size!r}.")
    if coord.shape[0] == 0:
        return torch.empty_like(coord, dtype=torch.int32)

    batch = batch.to(device=coord.device, dtype=torch.long)
    if bool((batch < 0).any().item()):
        raise ValueError("batch indices must be non-negative.")
    raw_grid = torch.floor(coord / float(grid_size)).to(dtype=torch.int32)
    sample_count = int(batch.max().item()) + 1
    sample_min = torch.full(
        (sample_count, 3),
        torch.iinfo(raw_grid.dtype).max,
        dtype=raw_grid.dtype,
        device=raw_grid.device,
    )
    sample_min.scatter_reduce_(
        0,
        batch[:, None].expand_as(raw_grid),
        raw_grid,
        reduce="amin",
        include_self=True,
    )
    return raw_grid - sample_min[batch]


def _env_flag(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


@dataclass(frozen=True)
class PseudoLabelConfig:
    held_sigma: float = 0.025
    static_sigma: float = 0.025
    motion_gap_eps: float = 0.005
    motion_rotation_radius: float = 0.08
    motion_baseline_threshold: float = 0.015
    motion_baseline_temperature: float = 0.005
    motion_evidence_topk: int = 3
    motion_relative_margin: float = 0.10
    motion_relative_tau: float = 0.10
    gripper_sigma: float = 0.045
    trajectory_sigma: float = 0.13
    contact_sigma: float = 0.10
    contact_radius: float = 0.12
    contact_temperature: float = 0.035
    approach_margin: float = 0.005
    approach_tau: float = 0.025
    background_trajectory_sigma: float = 0.20
    # Tool-conditioned interaction foreground.  The segmentation network and
    # cached tensor shapes stay unchanged; these parameters only affect the
    # temporal pseudo-label generator.
    tool_interaction_enable: bool = True
    tool_candidate_min_score: float = 0.35
    tool_candidate_max_points: int = 320
    tool_candidate_distance_boost: float = 0.35
    tool_candidate_distance_scale: float = 0.45
    # Restrict the conditioned tool cloud to the high-motion component that is
    # spatially connected to the gripper.  This prevents repeated table / wall
    # structure from entering the swept tool cloud merely because nearest-neighbour
    # motion residuals are ambiguous.
    tool_candidate_bridge_min_score: float = 0.18
    tool_candidate_seed_radius: float = 0.14
    tool_candidate_seed_min_score: float = 0.35
    tool_candidate_component_radius: float = 0.070
    tool_candidate_component_hops: int = 10
    tool_candidate_preselect_multiplier: int = 6
    tool_candidate_max_radius: float = 0.65
    tool_candidate_support_sigma: float = 0.035
    # Use the sparse episode trajectory only for the conditioned tool sweep.  The
    # legacy EEF contact / approach priors remain local to the temporal point-cloud
    # window.  A bounded frame window avoids sweeping the currently held object
    # through unrelated pre-grasp / post-release phases.
    tool_sweep_max_poses: int = 20
    tool_sweep_use_full_trajectory: bool = True
    tool_sweep_max_frame_offset: int = 96
    tool_sweep_max_translation: float = 0.65
    tool_sweep_contact_radius: float = 0.12
    tool_sweep_contact_temperature: float = 0.020
    # A scene point is a tool-conditioned target only when it is both absolutely
    # near the localized tool sweep and among the nearest fraction of scene points.
    # The rank gate is a safety valve against catastrophic all-scene foreground.
    tool_target_max_fraction: float = 0.18
    tool_target_static_min_score: float = 0.35
    tool_target_tool_exclusion_radius: float = 0.030
    tool_target_rigid_exclusion_score: float = 0.60
    tool_target_rank_temperature: float = 0.012
    tool_target_min_candidate_count: int = 4
    tool_target_candidate_count_temperature: float = 2.0
    tool_target_candidate_score_threshold: float = 0.40
    tool_target_candidate_score_temperature: float = 0.08
    tool_approach_margin: float = 0.030
    tool_approach_temperature: float = 0.015
    tool_sweep_dwell_sigma: float = 0.10
    interaction_motion_threshold: float = 0.020
    interaction_motion_temperature: float = 0.008
    background_tool_sweep_sigma: float = 0.16
    soft_background_weight: float = 0.20
    min_confidence: float = 0.20
    background_min_confidence: float = 0.55
    background_foreground_max: float = 0.12
    background_label_weight: float = 0.12
    min_foreground_fraction: float = 0.01
    forced_foreground_min_score: float = 0.05
    teacher_confidence: float = 0.86
    teacher_blend: float = 0.25
    teacher_geometry_gate: float = 0.18
    teacher_background_min_score: float = 0.65
    teacher_background_foreground_max: float = 0.08
    ignore_index: int = ROLE_IGNORE
    nn_chunk_size: int = 1024


@dataclass(frozen=True)
class SongPointSegLossConfig:
    soft_bce_weight: float = 1.0
    smoothness_weight: float = 0.04
    smooth_voxel_size: float = 0.025
    # The previous 4x foreground/background ratio amplified a few broad pseudo
    # masks into near-all-foreground predictions.  Keep a mild foreground boost,
    # then explicitly regularize frame-level foreground mass instead.
    class_weights: tuple[float, float] = (1.00, 1.25)
    foreground_mass_weight: float = 0.12
    foreground_mass_margin: float = 0.04
    foreground_mass_max: float = 0.45
    ignore_index: int = ROLE_IGNORE


def rot6d_to_matrix(d6: Tensor) -> Tensor:
    a1 = d6[..., 0:3]
    a2 = d6[..., 3:6]
    b1 = functional.normalize(a1, dim=-1, eps=1e-6)
    b2 = a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1
    b2 = functional.normalize(b2, dim=-1, eps=1e-6)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-1)


def matrix_to_rot6d(rot: Tensor) -> Tensor:
    return torch.cat([rot[..., :, 0], rot[..., :, 1]], dim=-1)


def pose9_to_matrix(pose9: Tensor) -> Tensor:
    pose9 = pose9.to(dtype=torch.float32)
    transform = torch.zeros(*pose9.shape[:-1], 4, 4, dtype=pose9.dtype, device=pose9.device)
    transform[..., 3, 3] = 1.0
    transform[..., :3, :3] = rot6d_to_matrix(pose9[..., 3:9])
    transform[..., :3, 3] = pose9[..., :3]
    return transform


def matrix_to_pose9(transform: Tensor) -> Tensor:
    return torch.cat([transform[..., :3, 3], matrix_to_rot6d(transform[..., :3, :3])], dim=-1)


def invert_transform(transform: Tensor) -> Tensor:
    rot = transform[..., :3, :3]
    trans = transform[..., :3, 3]
    rot_inv = rot.transpose(-1, -2)
    trans_inv = -(rot_inv @ trans.unsqueeze(-1)).squeeze(-1)
    out = torch.zeros_like(transform)
    out[..., :3, :3] = rot_inv
    out[..., :3, 3] = trans_inv
    out[..., 3, 3] = 1.0
    return out


def transform_points(points: Tensor, transform: Tensor) -> Tensor:
    rot = transform[..., :3, :3]
    trans = transform[..., :3, 3]
    return torch.matmul(points, rot.transpose(-1, -2)) + trans.unsqueeze(-2)


def relative_poses_to_first(pose9: Tensor) -> Tensor:
    """Convert a pose sequence in one common frame into poses relative to the first pose."""
    transforms = pose9_to_matrix(pose9)
    first_inv = invert_transform(transforms[..., :1, :, :])
    relative = first_inv @ transforms
    return matrix_to_pose9(relative)


def _identity_pose9(device: torch.device | None = None, dtype: torch.dtype = torch.float32) -> Tensor:
    pose = torch.zeros(9, device=device, dtype=dtype)
    pose[3] = 1.0
    pose[7] = 1.0
    return pose


def _sample_rows(array: np.ndarray, count: int, rng: np.random.Generator) -> np.ndarray:
    n_points = array.shape[0]
    if n_points == 0:
        channels = array.shape[-1] if array.ndim >= 2 else 6
        return np.zeros((0, channels), dtype=np.float32)
    if count <= 0 or n_points <= count:
        return np.ascontiguousarray(array, dtype=np.float32)
    indices = rng.choice(n_points, count, replace=False)
    return np.ascontiguousarray(array[indices], dtype=np.float32)


def _sample_rows_with_indices(
    array: np.ndarray, count: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    n_points = array.shape[0]
    if n_points == 0:
        channels = array.shape[-1] if array.ndim >= 2 else 6
        return np.zeros((0, channels), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    if count <= 0 or n_points <= count:
        indices = np.arange(n_points, dtype=np.int64)
        return np.ascontiguousarray(array, dtype=np.float32), indices
    indices = rng.choice(n_points, count, replace=False).astype(np.int64, copy=False)
    return np.ascontiguousarray(array[indices], dtype=np.float32), indices


def _require_zarr():
    try:
        import zarr  # type: ignore
        from numcodecs import Blosc  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on optional environment package
        raise ImportError(
            "Compressed point-cloud storage requires `zarr` and `numcodecs`. "
            "Install them in the active environment, e.g. `conda install -n reap -c conda-forge zarr numcodecs`."
        ) from exc
    return zarr, Blosc


class PackedPointCloudZarr:
    """Read packed xyz=float16, rgb=uint8 zarr point clouds as float32 XYZRGB."""

    def __init__(self, group: Any):
        self.group = group
        self.xyz = group["xyz"]
        self.rgb = group["rgb"]
        self.shape = (*self.xyz.shape[:-1], 6)
        self.dtype = np.dtype(np.float32)
        self.ndim = len(self.shape)

    def __len__(self) -> int:
        return int(self.shape[0])

    def __getitem__(self, item):
        xyz = np.asarray(self.xyz[item], dtype=np.float32)
        rgb = np.asarray(self.rgb[item], dtype=np.float32)
        return np.concatenate([xyz, rgb], axis=-1)

    def __array__(self, dtype=None):
        array = self[:]
        if dtype is not None:
            array = array.astype(dtype, copy=False)
        return array


def episode_point_cloud_npy_path(point_cloud_dir: str | Path, episode_index: int) -> Path:
    return Path(point_cloud_dir) / f"episode_{int(episode_index):06d}.npy"


def episode_point_cloud_zarr_path(point_cloud_dir: str | Path, episode_index: int) -> Path:
    return Path(point_cloud_dir) / f"episode_{int(episode_index):06d}.zarr"


def find_episode_point_cloud_path(point_cloud_dir: str | Path, episode_index: int) -> Path:
    zarr_path = episode_point_cloud_zarr_path(point_cloud_dir, episode_index)
    if zarr_path.exists():
        return zarr_path
    npy_path = episode_point_cloud_npy_path(point_cloud_dir, episode_index)
    if npy_path.exists():
        return npy_path
    raise FileNotFoundError(
        f"Point cloud episode file is missing: expected {zarr_path} or {npy_path}"
    )


def open_episode_point_clouds(
    point_cloud_dir: str | Path,
    episode_index: int,
    *,
    mmap_mode: str = "r",
) -> Any:
    path = find_episode_point_cloud_path(point_cloud_dir, episode_index)
    if path.suffix == ".zarr":
        zarr, _ = _require_zarr()
        zarr_obj = zarr.open(str(path), mode="r")
        if hasattr(zarr_obj, "array_keys") and "xyz" in zarr_obj and "rgb" in zarr_obj:
            return PackedPointCloudZarr(zarr_obj)
        return zarr_obj
    return np.load(path, mmap_mode=mmap_mode)


def _pack_point_cloud_rgb(rgb: np.ndarray) -> np.ndarray:
    rgb = np.asarray(rgb, dtype=np.float32)
    finite_rgb = rgb[np.isfinite(rgb)]
    if finite_rgb.size > 0 and float(finite_rgb.max(initial=0.0)) <= 1.0:
        rgb = rgb * 255.0
    return np.clip(np.rint(rgb), 0, 255).astype(np.uint8, copy=False)


def save_point_clouds_zarr(
    path: str | Path,
    point_clouds: np.ndarray,
    *,
    chunks: tuple[int, int, int] | None = None,
    compressor_name: str = "zstd",
    compression_level: int = 3,
    packed: bool = True,
) -> Path:
    zarr, Blosc = _require_zarr()
    point_clouds = np.ascontiguousarray(point_clouds, dtype=np.float32)
    if point_clouds.ndim != 3 or point_clouds.shape[-1] != 6:
        raise ValueError(f"Expected point clouds shape (T,N,6), got {point_clouds.shape}")
    path = Path(path)
    if path.exists():
        import shutil

        shutil.rmtree(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    compressor = Blosc(cname=compressor_name, clevel=int(compression_level), shuffle=Blosc.BITSHUFFLE)
    if packed:
        if chunks is None:
            chunks = (1, min(int(point_clouds.shape[1]), 16384), 3)
        group = zarr.open_group(str(path), mode="w")
        group.attrs.update(
            {
                "format": "song_point_cloud_packed_v1",
                "shape": list(point_clouds.shape),
                "xyz_dtype": "float16",
                "rgb_dtype": "uint8",
                "rgb_scale": "0_255",
            }
        )
        xyz = group.create_dataset(
            "xyz",
            shape=point_clouds[..., :3].shape,
            chunks=chunks,
            dtype=np.float16,
            compressor=compressor,
        )
        rgb = group.create_dataset(
            "rgb",
            shape=point_clouds[..., 3:6].shape,
            chunks=chunks,
            dtype=np.uint8,
            compressor=compressor,
        )
        xyz[:] = point_clouds[..., :3].astype(np.float16)
        rgb[:] = _pack_point_cloud_rgb(point_clouds[..., 3:6])
        return path

    if chunks is None:
        chunks = (1, min(int(point_clouds.shape[1]), 16384), 6)
    array = zarr.open(
        str(path),
        mode="w",
        shape=point_clouds.shape,
        chunks=chunks,
        dtype=point_clouds.dtype,
        compressor=compressor,
    )
    array[:] = point_clouds
    return path


def save_episode_point_clouds_zarr(
    point_cloud_dir: str | Path,
    episode_index: int,
    point_clouds: np.ndarray,
    *,
    chunks: tuple[int, int, int] | None = None,
    compressor_name: str = "zstd",
    compression_level: int = 3,
    packed: bool = True,
) -> Path:
    return save_point_clouds_zarr(
        episode_point_cloud_zarr_path(point_cloud_dir, episode_index),
        point_clouds,
        chunks=chunks,
        compressor_name=compressor_name,
        compression_level=compression_level,
        packed=packed,
    )


def _pad_point_tensors(tensors: list[Tensor], pad_value: float | int = 0) -> tuple[Tensor, Tensor]:
    if not tensors:
        raise ValueError("Cannot collate an empty point tensor list.")
    max_points = max(int(tensor.shape[0]) for tensor in tensors)
    trailing_shape = tuple(tensors[0].shape[1:])
    if max_points <= 0:
        raise ValueError("Cannot collate empty point clouds.")
    padded = tensors[0].new_full((len(tensors), max_points, *trailing_shape), pad_value)
    is_pad = torch.ones(len(tensors), max_points, dtype=torch.bool)
    for idx, tensor in enumerate(tensors):
        n_points = int(tensor.shape[0])
        if n_points <= 0:
            continue
        padded[idx, :n_points] = tensor
        is_pad[idx, :n_points] = False
    return padded, is_pad


def _pad_single_axis_tensors(tensors: list[Tensor], axis: int, pad_value: float | int = 0) -> tuple[Tensor, Tensor]:
    if axis < 0:
        axis += tensors[0].ndim
    max_points = max(int(tensor.shape[axis]) for tensor in tensors)
    if max_points <= 0:
        raise ValueError("Cannot collate tensors with an empty point axis.")
    out_shape = [len(tensors), *tensors[0].shape]
    out_shape[axis + 1] = max_points
    padded = tensors[0].new_full(tuple(out_shape), pad_value)
    mask_shape = [len(tensors), *tensors[0].shape[:axis], max_points]
    is_pad = torch.ones(tuple(mask_shape), dtype=torch.bool)
    for batch_idx, tensor in enumerate(tensors):
        n_points = int(tensor.shape[axis])
        slices = [batch_idx, *([slice(None)] * tensor.ndim)]
        slices[axis + 1] = slice(0, n_points)
        padded[tuple(slices)] = tensor
        mask_slices = [batch_idx, *([slice(None)] * axis), slice(0, n_points)]
        is_pad[tuple(mask_slices)] = False
    return padded, is_pad


def song_pointseg_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    if not batch:
        return {}
    out: dict[str, Any] = {}
    keys = set().union(*(item.keys() for item in batch))
    current_point_fields = {
        "observation.point_cloud": 0,
        "observation.point_cloud_world": 0,
        "pointseg.priors": 0,
        "pointseg.labels": ROLE_IGNORE,
        "pointseg.weights": 0,
        "pointseg.class_scores": 0,
        "pointseg.role_scores": 0,
        "pointseg.foreground_score": 0,
        "observation.point_cloud_indices": -1,
    }
    for key in sorted(keys):
        values = [item[key] for item in batch if key in item]
        if len(values) != len(batch):
            continue
        if key == "observation.point_cloud":
            first = values[0]
            if torch.is_tensor(first) and first.ndim == 2:
                out[key], out["observation.point_cloud_is_pad"] = _pad_point_tensors(values, 0)
                continue
            if torch.is_tensor(first) and first.ndim == 3 and first.shape[0] == 1:
                out[key], out["observation.point_cloud_is_pad"] = _pad_single_axis_tensors(values, 1, 0)
                continue
        if key == "observation.point_cloud_world":
            first = values[0]
            if torch.is_tensor(first) and first.ndim == 2:
                out[key], out["observation.point_cloud_world_is_pad"] = _pad_point_tensors(values, 0)
                continue
        if key in current_point_fields and torch.is_tensor(values[0]) and values[0].ndim >= 1:
            out[key], _ = _pad_point_tensors(values, current_point_fields[key])
            continue
        if key == "observation.point_cloud_future" and torch.is_tensor(values[0]):
            out[key], out["observation.point_cloud_future_is_pad"] = _pad_single_axis_tensors(values, 1, 0)
            continue
        if key == "observation.point_cloud_future_is_pad" and torch.is_tensor(values[0]):
            out[key], _ = _pad_single_axis_tensors(values, 1, True)
            continue
        out[key] = default_collate(values)
    return out


POINT_CLOUD_VIEW_DIRS = {
    "agentview": "point_clouds",
    # RLBench stores its external/front-camera cloud in the same canonical
    # directory that LIBERO calls ``agentview``.
    "front": "point_clouds",
    "robot0_eye_in_hand": "point_clouds_robot0_eye_in_hand",
}


def parse_camera_views(value: Any = None) -> tuple[str, ...]:
    """Parse the one training/cache view option used by the benchmark scripts."""

    if value is None:
        return ("agentview",)
    if isinstance(value, (list, tuple)):
        parts = [str(part).strip() for part in value]
    else:
        text = str(value).strip().strip("[]")
        parts = [part.strip().strip("\"'") for part in text.split(",")]
    views = tuple(part for part in parts if part) or ("agentview",)
    unknown = [view for view in views if view not in POINT_CLOUD_VIEW_DIRS]
    if unknown:
        raise ValueError(
            f"Unsupported camera view(s) {unknown}; supported views are {sorted(POINT_CLOUD_VIEW_DIRS)}."
        )
    if len(set(views)) != len(views):
        raise ValueError(f"camera views contain duplicates: {views}.")
    return views


def parse_camera_view_weights(
    value: Any = None,
    *,
    num_views: int | None = None,
) -> tuple[float, ...] | None:
    """Parse optional per-view scene-point sampling weights.

    ``None`` preserves the legacy equal-budget composition.  Weights are
    normalized by :func:`compose_point_cloud_views`; only their ratios matter.
    """

    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    if isinstance(value, (list, tuple)):
        parts = list(value)
    else:
        text = str(value).strip().strip("[]")
        parts = [part.strip().strip("\"'") for part in text.split(",")]
    weights = tuple(float(part) for part in parts if str(part).strip())
    if not weights:
        return None
    if num_views is not None and len(weights) != int(num_views):
        raise ValueError(
            f"camera view weights must contain one value per view: got {weights} "
            f"for {int(num_views)} views."
        )
    if any(not np.isfinite(weight) or weight <= 0.0 for weight in weights):
        raise ValueError(f"camera view weights must be finite and positive, got {weights}.")
    return weights


def point_cloud_dir_for_view(dataset_root: str | Path, view: str) -> Path:
    return Path(dataset_root) / POINT_CLOUD_VIEW_DIRS[str(view)]


def use_primary_view_for_training_index(dataset_index: int, seed: int = 20260812) -> bool:
    """Deterministic Bernoulli(0.5) assignment for input-level view dropout.

    SplitMix64 avoids correlations between sequential frame indices while
    remaining independent of DataLoader worker count and scheduling.  This is
    training input selection only; it never changes inference or model layers.
    """

    value = (int(dataset_index) + int(seed)) & 0xFFFFFFFFFFFFFFFF
    value = (value + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    value ^= value >> 31
    return bool(value & 1)


def paired_view_augmentation_index(
    augmented_index: int,
    num_samples: int,
    seed: int = 20260812,
) -> tuple[int, bool]:
    """Map a two-copy augmented index to its frame and complementary view mode."""

    augmented_index = int(augmented_index)
    num_samples = int(num_samples)
    if num_samples <= 0:
        raise ValueError(f"num_samples must be positive, got {num_samples}.")
    if augmented_index < 0 or augmented_index >= 2 * num_samples:
        raise IndexError(
            f"augmented_index must be in [0, {2 * num_samples}), got {augmented_index}."
        )
    base_index = augmented_index % num_samples
    second_copy = augmented_index >= num_samples
    use_primary = use_primary_view_for_training_index(base_index, seed)
    return base_index, bool(use_primary) ^ bool(second_copy)


def parse_camera_view_fusion(value: Any = None) -> str:
    """Parse the multi-view composition mode while preserving old checkpoints."""

    if value is None or not str(value).strip():
        return "legacy_budget"
    mode = str(value).strip().lower().replace("-", "_")
    aliases = {
        "legacy": "legacy_budget",
        "budget": "legacy_budget",
        "legacy_budget": "legacy_budget",
        # Older global/local geometry checkpoints used this name for the
        # fixed per-view scene budget implemented by legacy_budget.
        "global_local_geometry": "legacy_budget",
        "fps": "fps",
        "voxel_fps": "voxel_fps",
        "voxelized_fps": "voxel_fps",
        "voxel_cover_fps": "voxel_cover_fps",
        "voxel_coverage_fps": "voxel_cover_fps",
        "novelty_union": "novelty_union",
        "novelty": "novelty_union",
        "multiscale_novelty_union": "multiscale_novelty_union",
        "multiscale_novelty": "multiscale_novelty_union",
        "consensus_multiscale_novelty_union": "consensus_multiscale_novelty_union",
        "consensus_multiscale_novelty": "consensus_multiscale_novelty_union",
        "transport_novelty_union": "transport_novelty_union",
        "transport_novelty": "transport_novelty_union",
        "uniform_union": "uniform_union",
        "random_union": "uniform_union",
        "union_random": "uniform_union",
        "full_union": "full_union",
        "union_all": "full_union",
        "primary_residual": "primary_residual",
        "residual": "primary_residual",
    }
    if mode not in aliases:
        raise ValueError(
            "camera view fusion must be 'legacy_budget', 'fps', 'voxel_fps', "
            "'voxel_cover_fps', 'novelty_union', 'multiscale_novelty_union', "
            "'consensus_multiscale_novelty_union', "
            "'transport_novelty_union', "
            "'uniform_union', 'full_union', "
            "or 'primary_residual'; "
            f"got {value!r}."
        )
    return aliases[mode]


def _torch_farthest_point_indices(xyz: Tensor, count: int) -> Tensor:
    """Deterministic exact FPS fallback for CPU tests or missing pointops."""

    if xyz.ndim != 2 or xyz.shape[-1] != 3:
        raise ValueError(f"Expected xyz shape (N,3), got {tuple(xyz.shape)}.")
    count = int(count)
    if count <= 0 or count > xyz.shape[0]:
        raise ValueError(f"FPS count must be in [1, {xyz.shape[0]}], got {count}.")
    selected = torch.empty(count, dtype=torch.long, device=xyz.device)
    distances = torch.full((xyz.shape[0],), torch.inf, dtype=xyz.dtype, device=xyz.device)
    farthest = torch.zeros((), dtype=torch.long, device=xyz.device)
    for sample_index in range(count):
        selected[sample_index] = farthest
        centroid = xyz[farthest]
        distances = torch.minimum(distances, torch.sum((xyz - centroid) ** 2, dim=-1))
        farthest = torch.argmax(distances)
    return selected


def fps_sample_fused_point_cloud(
    point_cloud: Tensor,
    *,
    target_points: int = 10_000,
    gripper_points: int = 500,
    point_is_pad: Tensor | None = None,
) -> tuple[Tensor, Tensor | None, Tensor]:
    """FPS an equal multi-view scene union while preserving one gripper tail."""

    global _POINTOPS_FPS_FALLBACK_WARNED
    if point_cloud.ndim != 3 or point_cloud.shape[-1] != 6:
        raise ValueError(f"Expected point cloud shape (B,N,6), got {tuple(point_cloud.shape)}.")
    batch_size, source_points, _ = point_cloud.shape
    target_points = int(target_points)
    gripper_points = int(gripper_points)
    if target_points <= 0 or not 0 <= gripper_points < target_points:
        raise ValueError("FPS target/gripper counts are invalid.")
    if source_points < target_points:
        raise ValueError(f"Cannot FPS sample {target_points} points from {source_points} points.")
    identity = torch.arange(source_points, device=point_cloud.device, dtype=torch.long)
    identity = identity.unsqueeze(0).expand(batch_size, -1)
    if source_points == target_points:
        return point_cloud, point_is_pad, identity

    source_scene_points = source_points - gripper_points
    target_scene_points = target_points - gripper_points
    if source_scene_points < target_scene_points:
        raise ValueError("Source scene has fewer points than the requested FPS scene budget.")
    if torch.is_tensor(point_is_pad):
        if point_is_pad.shape != (batch_size, source_points):
            raise ValueError(
                f"point_is_pad must have shape {(batch_size, source_points)}, got {tuple(point_is_pad.shape)}."
            )
        if bool(point_is_pad[:, :source_scene_points].any().item()):
            raise ValueError("FPS fusion requires fixed-size, unpadded scene unions.")

    scene_xyz = point_cloud[:, :source_scene_points, :3].contiguous()
    if point_cloud.is_cuda and _pointops_farthest_point_sampling is not None:
        flat_xyz = scene_xyz.reshape(-1, 3).contiguous()
        offsets = torch.arange(1, batch_size + 1, device=point_cloud.device, dtype=torch.int32)
        offsets = offsets * source_scene_points
        new_offsets = torch.arange(1, batch_size + 1, device=point_cloud.device, dtype=torch.int32)
        new_offsets = new_offsets * target_scene_points
        flat_indices = _pointops_farthest_point_sampling(flat_xyz, offsets, new_offsets).long()
        batch_offsets = torch.arange(batch_size, device=point_cloud.device, dtype=torch.long)
        batch_offsets = batch_offsets.repeat_interleave(target_scene_points) * source_scene_points
        scene_indices = (flat_indices - batch_offsets).reshape(batch_size, target_scene_points)
    else:
        if not _POINTOPS_FPS_FALLBACK_WARNED:
            warnings.warn(
                "pointops CUDA FPS is unavailable; using deterministic PyTorch FPS. "
                "This fallback is intended for small tests only.",
                stacklevel=2,
            )
            _POINTOPS_FPS_FALLBACK_WARNED = True
        scene_indices = torch.stack(
            [_torch_farthest_point_indices(scene_xyz[index], target_scene_points) for index in range(batch_size)],
            dim=0,
        )

    gripper_indices = torch.arange(
        source_scene_points, source_points, device=point_cloud.device, dtype=torch.long
    ).unsqueeze(0).expand(batch_size, -1)
    indices = torch.cat([scene_indices, gripper_indices], dim=1)
    gather_indices = indices.unsqueeze(-1).expand(-1, -1, point_cloud.shape[-1])
    sampled = torch.gather(point_cloud, 1, gather_indices)
    sampled_pad = torch.gather(point_is_pad, 1, indices) if torch.is_tensor(point_is_pad) else None
    return sampled, sampled_pad, indices


def voxel_fps_sample_fused_point_cloud(
    point_cloud: Tensor,
    *,
    target_points: int = 10_000,
    gripper_points: int = 500,
    voxel_size: float = 0.005,
    point_is_pad: Tensor | None = None,
) -> tuple[Tensor, Tensor | None, Tensor]:
    """Deduplicate overlapping view samples by voxel, then apply shared-space FPS.

    The operation is view-count agnostic.  Each occupied voxel contributes its
    first point in the configured view order; when too few voxels remain for the
    requested budget, the function safely falls back to the original union FPS.
    The primary gripper tail is always preserved exactly.
    """

    if point_cloud.ndim != 3 or point_cloud.shape[-1] != 6:
        raise ValueError(f"Expected point cloud shape (B,N,6), got {tuple(point_cloud.shape)}.")
    target_points = int(target_points)
    gripper_points = int(gripper_points)
    voxel_size = float(voxel_size)
    if not np.isfinite(voxel_size) or voxel_size <= 0.0:
        raise ValueError(f"voxel_size must be finite and positive, got {voxel_size}.")
    batch_size, source_points, _ = point_cloud.shape
    if source_points <= target_points:
        return fps_sample_fused_point_cloud(
            point_cloud,
            target_points=target_points,
            gripper_points=gripper_points,
            point_is_pad=point_is_pad,
        )
    source_scene_points = source_points - gripper_points
    target_scene_points = target_points - gripper_points
    if torch.is_tensor(point_is_pad) and bool(point_is_pad[:, :source_scene_points].any().item()):
        raise ValueError("Voxel-FPS fusion requires fixed-size, unpadded scene unions.")

    scene_xyz = point_cloud[:, :source_scene_points, :3].contiguous()
    flat_scene_xyz = scene_xyz.reshape(-1, 3)
    batch_ids = torch.arange(batch_size, device=point_cloud.device, dtype=torch.int64)
    batch_ids = batch_ids.repeat_interleave(source_scene_points).unsqueeze(1)
    voxel_coords = torch.floor(flat_scene_xyz / voxel_size).to(torch.int64)
    unique_voxels, inverse = torch.unique(
        torch.cat([batch_ids, voxel_coords], dim=1), dim=0, return_inverse=True
    )
    flat_indices = torch.arange(
        batch_size * source_scene_points, device=point_cloud.device, dtype=torch.long
    )
    representatives = torch.full(
        (unique_voxels.shape[0],),
        batch_size * source_scene_points,
        device=point_cloud.device,
        dtype=torch.long,
    )
    representatives.scatter_reduce_(0, inverse, flat_indices, reduce="amin", include_self=True)
    voxel_counts = torch.bincount(unique_voxels[:, 0], minlength=batch_size).cpu().tolist()

    candidate_indices: list[Tensor] = []
    candidate_global_indices: list[Tensor] = []
    all_scene_indices = torch.arange(source_scene_points, device=point_cloud.device, dtype=torch.long)
    voxel_start = 0
    for batch_index in range(batch_size):
        voxel_count = int(voxel_counts[batch_index])
        if voxel_count >= target_scene_points:
            global_indices = representatives[voxel_start : voxel_start + voxel_count]
            candidate_global_indices.append(global_indices)
            candidate_indices.append(global_indices - batch_index * source_scene_points)
        else:
            candidate_global_indices.append(all_scene_indices + batch_index * source_scene_points)
            candidate_indices.append(all_scene_indices)
        voxel_start += voxel_count

    candidate_xyz = flat_scene_xyz.index_select(0, torch.cat(candidate_global_indices)).contiguous()
    candidate_source_indices = torch.cat(candidate_indices, dim=0)
    candidate_counts = torch.tensor(
        [indices.numel() for indices in candidate_indices], device=point_cloud.device, dtype=torch.int32
    )
    offsets = torch.cumsum(candidate_counts, dim=0)
    new_offsets = torch.arange(1, batch_size + 1, device=point_cloud.device, dtype=torch.int32)
    new_offsets = new_offsets * target_scene_points
    if point_cloud.is_cuda and _pointops_farthest_point_sampling is not None:
        selected_flat = _pointops_farthest_point_sampling(candidate_xyz, offsets, new_offsets).long()
        selected_source = candidate_source_indices.index_select(0, selected_flat)
        scene_indices = selected_source.reshape(batch_size, target_scene_points)
    else:
        starts = torch.cat([offsets.new_zeros(1), offsets[:-1]]).long()
        selected_rows = []
        for batch_index, indices in enumerate(candidate_indices):
            start = int(starts[batch_index].item())
            count = int(indices.numel())
            local = _torch_farthest_point_indices(candidate_xyz[start : start + count], target_scene_points)
            selected_rows.append(indices.index_select(0, local))
        scene_indices = torch.stack(selected_rows, dim=0)

    gripper_indices = torch.arange(
        source_scene_points, source_points, device=point_cloud.device, dtype=torch.long
    ).unsqueeze(0).expand(batch_size, -1)
    indices = torch.cat([scene_indices, gripper_indices], dim=1)
    sampled = torch.gather(point_cloud, 1, indices.unsqueeze(-1).expand(-1, -1, point_cloud.shape[-1]))
    sampled_pad = torch.gather(point_is_pad, 1, indices) if torch.is_tensor(point_is_pad) else None
    return sampled, sampled_pad, indices


def voxel_cover_fps_sample_fused_point_cloud(
    point_cloud: Tensor,
    *,
    target_points: int = 10_000,
    gripper_points: int = 500,
    voxel_size: float = 0.01,
    point_is_pad: Tensor | None = None,
) -> tuple[Tensor, Tensor | None, Tensor]:
    """Cover occupied shared-space voxels, then use union FPS for detail.

    Every occupied voxel is represented whenever the voxel count fits the
    scene budget.  Remaining slots are filled in deterministic union-FPS order,
    which retains geometric detail without assigning a fixed budget to any
    camera.  If occupied voxels exceed the budget, FPS is applied to the voxel
    representatives.  The operation is view-count agnostic and is an identity
    for an already target-sized single-view cloud.
    """

    if point_cloud.ndim != 3 or point_cloud.shape[-1] != 6:
        raise ValueError(f"Expected point cloud shape (B,N,6), got {tuple(point_cloud.shape)}.")
    target_points = int(target_points)
    gripper_points = int(gripper_points)
    voxel_size = float(voxel_size)
    if not np.isfinite(voxel_size) or voxel_size <= 0.0:
        raise ValueError(f"voxel_size must be finite and positive, got {voxel_size}.")
    batch_size, source_points, _ = point_cloud.shape
    if source_points == target_points:
        identity = torch.arange(source_points, device=point_cloud.device, dtype=torch.long)
        identity = identity.unsqueeze(0).expand(batch_size, -1)
        return point_cloud, point_is_pad, identity
    if source_points < target_points:
        raise ValueError(f"Cannot sample {target_points} points from {source_points} points.")
    source_scene_points = source_points - gripper_points
    target_scene_points = target_points - gripper_points
    if source_scene_points < target_scene_points:
        raise ValueError("Source scene has fewer points than the requested scene budget.")
    if torch.is_tensor(point_is_pad):
        if point_is_pad.shape != (batch_size, source_points):
            raise ValueError(
                f"point_is_pad must have shape {(batch_size, source_points)}, got {tuple(point_is_pad.shape)}."
            )
        if bool(point_is_pad[:, :source_scene_points].any().item()):
            raise ValueError("Voxel-cover FPS requires fixed-size, unpadded scene unions.")

    scene_xyz = point_cloud[:, :source_scene_points, :3].contiguous()
    flat_scene_xyz = scene_xyz.reshape(-1, 3)
    batch_ids = torch.arange(batch_size, device=point_cloud.device, dtype=torch.int64)
    batch_ids = batch_ids.repeat_interleave(source_scene_points).unsqueeze(1)
    voxel_coords = torch.floor(flat_scene_xyz / voxel_size).to(torch.int64)
    unique_voxels, inverse = torch.unique(
        torch.cat([batch_ids, voxel_coords], dim=1), dim=0, return_inverse=True
    )
    flat_indices = torch.arange(
        batch_size * source_scene_points, device=point_cloud.device, dtype=torch.long
    )
    representatives = torch.full(
        (unique_voxels.shape[0],),
        batch_size * source_scene_points,
        device=point_cloud.device,
        dtype=torch.long,
    )
    representatives.scatter_reduce_(0, inverse, flat_indices, reduce="amin", include_self=True)
    voxel_counts = torch.bincount(unique_voxels[:, 0], minlength=batch_size).cpu().tolist()

    # A complete union-FPS ordering supplies spatially distributed extra detail
    # for samples whose voxel cover uses fewer than the available scene slots.
    _, _, union_fps_indices = fps_sample_fused_point_cloud(
        point_cloud,
        target_points=target_points,
        gripper_points=gripper_points,
        point_is_pad=point_is_pad,
    )
    union_fps_scene = union_fps_indices[:, :target_scene_points]

    scene_rows: list[Tensor] = []
    voxel_start = 0
    all_scene_indices = torch.arange(source_scene_points, device=point_cloud.device, dtype=torch.long)
    for batch_index in range(batch_size):
        voxel_count = int(voxel_counts[batch_index])
        reps = representatives[voxel_start : voxel_start + voxel_count]
        reps = reps - batch_index * source_scene_points
        voxel_start += voxel_count
        if voxel_count >= target_scene_points:
            rep_xyz = scene_xyz[batch_index].index_select(0, reps).contiguous()
            if point_cloud.is_cuda and _pointops_farthest_point_sampling is not None:
                offsets = torch.tensor([voxel_count], device=point_cloud.device, dtype=torch.int32)
                new_offsets = torch.tensor(
                    [target_scene_points], device=point_cloud.device, dtype=torch.int32
                )
                local = _pointops_farthest_point_sampling(rep_xyz, offsets, new_offsets).long()
            else:
                local = _torch_farthest_point_indices(rep_xyz, target_scene_points)
            scene_rows.append(reps.index_select(0, local))
            continue

        selected_mask = torch.zeros(source_scene_points, dtype=torch.bool, device=point_cloud.device)
        selected_mask[reps] = True
        fps_extras = union_fps_scene[batch_index]
        fps_extras = fps_extras[~selected_mask[fps_extras]]
        need = target_scene_points - voxel_count
        if fps_extras.numel() < need:
            selected_mask[fps_extras] = True
            fallback_extras = all_scene_indices[~selected_mask]
            fps_extras = torch.cat([fps_extras, fallback_extras[: need - fps_extras.numel()]])
        scene_rows.append(torch.cat([reps, fps_extras[:need]]))
    scene_indices = torch.stack(scene_rows)

    gripper_indices = torch.arange(
        source_scene_points, source_points, device=point_cloud.device, dtype=torch.long
    ).unsqueeze(0).expand(batch_size, -1)
    indices = torch.cat([scene_indices, gripper_indices], dim=1)
    sampled = torch.gather(point_cloud, 1, indices.unsqueeze(-1).expand(-1, -1, point_cloud.shape[-1]))
    sampled_pad = torch.gather(point_is_pad, 1, indices) if torch.is_tensor(point_is_pad) else None
    return sampled, sampled_pad, indices


def novelty_union_sample_fused_point_cloud(
    point_cloud: Tensor,
    *,
    target_points: int = 10_000,
    gripper_points: int = 500,
    voxel_size: float = 0.01,
    point_is_pad: Tensor | None = None,
    local_transport: bool = False,
    coarse_novelty_scale: float | None = None,
) -> tuple[Tensor, Tensor | None, Tensor]:
    """Preserve the primary cloud and insert only secondary-view novel voxels.

    Secondary representatives replace primary samples only from voxels that
    already contain another retained primary point.  Thus every primary-view
    occupied voxel and the complete primary gripper tail survive, while the
    number of secondary points is determined by geometric coverage rather than
    a camera ratio or a learned gate.
    """

    if point_cloud.ndim != 3 or point_cloud.shape[-1] != 6:
        raise ValueError(f"Expected point cloud shape (B,N,6), got {tuple(point_cloud.shape)}.")
    target_points = int(target_points)
    gripper_points = int(gripper_points)
    voxel_size = float(voxel_size)
    if not np.isfinite(voxel_size) or voxel_size <= 0.0:
        raise ValueError(f"voxel_size must be finite and positive, got {voxel_size}.")
    if coarse_novelty_scale is not None and (
        not np.isfinite(float(coarse_novelty_scale)) or float(coarse_novelty_scale) <= 1.0
    ):
        raise ValueError("coarse_novelty_scale must be finite and greater than one.")
    batch_size, source_points, _ = point_cloud.shape
    if source_points == target_points:
        identity = torch.arange(source_points, device=point_cloud.device, dtype=torch.long)
        identity = identity.unsqueeze(0).expand(batch_size, -1)
        return point_cloud, point_is_pad, identity
    source_scene_points = source_points - gripper_points
    target_scene_points = target_points - gripper_points
    if source_scene_points < target_scene_points or source_scene_points % target_scene_points != 0:
        raise ValueError(
            "Novelty-union expects an ordered union of equal per-view scene budgets; "
            f"source_scene_points={source_scene_points}, target_scene_points={target_scene_points}."
        )
    if torch.is_tensor(point_is_pad) and bool(point_is_pad[:, :source_scene_points].any().item()):
        raise ValueError("Novelty-union fusion requires fixed-size, unpadded scene unions.")

    scene_xyz = point_cloud[:, :source_scene_points, :3].contiguous()
    flat_xyz = scene_xyz.reshape(-1, 3)
    local_indices = torch.arange(source_scene_points, device=point_cloud.device, dtype=torch.long)
    flat_local_indices = local_indices.repeat(batch_size)
    batch_ids = torch.arange(batch_size, device=point_cloud.device, dtype=torch.int64)
    batch_ids = batch_ids.repeat_interleave(source_scene_points)
    quantized = torch.floor(flat_xyz / voxel_size).to(torch.int64)
    unique_voxels, inverse = torch.unique(
        torch.cat([batch_ids.unsqueeze(1), quantized], dim=1), dim=0, return_inverse=True
    )
    total_scene_points = batch_size * source_scene_points
    flat_global_indices = torch.arange(total_scene_points, device=point_cloud.device, dtype=torch.long)
    sentinel = total_scene_points
    first_primary = torch.full(
        (unique_voxels.shape[0],), sentinel, device=point_cloud.device, dtype=torch.long
    )
    first_secondary = first_primary.clone()
    primary_mask = flat_local_indices < target_scene_points
    first_primary.scatter_reduce_(
        0,
        inverse[primary_mask],
        flat_global_indices[primary_mask],
        reduce="amin",
        include_self=True,
    )
    first_secondary.scatter_reduce_(
        0,
        inverse[~primary_mask],
        flat_global_indices[~primary_mask],
        reduce="amin",
        include_self=True,
    )
    if coarse_novelty_scale is None:
        novel_global = first_secondary[(first_primary == sentinel) & (first_secondary != sentinel)]
    else:
        # A secondary point is admitted only when its coarser occupied cell is
        # absent from the complete primary cloud.  One representative per
        # coarse novel cell suppresses fine-grid boundary/sampling noise while
        # the independent fine grid below still protects every primary
        # occupied cell.  The insertion count is determined by geometry; no
        # camera ratio, point quota, task rule, or primary/secondary pairing is
        # used.
        coarse_quantized = torch.floor(
            flat_xyz / (voxel_size * float(coarse_novelty_scale))
        ).to(torch.int64)
        coarse_voxels, coarse_inverse = torch.unique(
            torch.cat([batch_ids.unsqueeze(1), coarse_quantized], dim=1),
            dim=0,
            return_inverse=True,
        )
        coarse_first_primary = torch.full(
            (coarse_voxels.shape[0],), sentinel, device=point_cloud.device, dtype=torch.long
        )
        coarse_first_secondary = coarse_first_primary.clone()
        coarse_first_primary.scatter_reduce_(
            0,
            coarse_inverse[primary_mask],
            flat_global_indices[primary_mask],
            reduce="amin",
            include_self=True,
        )
        coarse_first_secondary.scatter_reduce_(
            0,
            coarse_inverse[~primary_mask],
            flat_global_indices[~primary_mask],
            reduce="amin",
            include_self=True,
        )
        novel_global = coarse_first_secondary[
            (coarse_first_primary == sentinel) & (coarse_first_secondary != sentinel)
        ]
    redundant_primary_global = flat_global_indices[
        primary_mask & (flat_global_indices != first_primary[inverse])
    ]
    novel_batches = torch.div(novel_global, source_scene_points, rounding_mode="floor")
    redundant_batches = torch.div(
        redundant_primary_global, source_scene_points, rounding_mode="floor"
    )
    novel_counts = torch.bincount(novel_batches, minlength=batch_size).cpu().tolist()
    redundant_counts = torch.bincount(redundant_batches, minlength=batch_size).cpu().tolist()

    scene_indices = torch.arange(
        target_scene_points, device=point_cloud.device, dtype=torch.long
    ).unsqueeze(0).repeat(batch_size, 1)
    novel_start = 0
    redundant_start = 0
    for batch_index in range(batch_size):
        novel_count = int(novel_counts[batch_index])
        redundant_count = int(redundant_counts[batch_index])
        insert_count = min(novel_count, redundant_count)
        if insert_count:
            removable = redundant_primary_global[
                redundant_start : redundant_start + redundant_count
            ] % source_scene_points
            additions = novel_global[novel_start : novel_start + novel_count] % source_scene_points
            if local_transport:
                removable_local, additions_local = _local_transport_replacement_pairs(
                    scene_xyz[batch_index].index_select(0, additions),
                    scene_xyz[batch_index].index_select(0, removable),
                )
                removable = removable.index_select(0, removable_local)
                additions = additions.index_select(0, additions_local)
            else:
                removable = removable[:insert_count]
                additions = additions[:insert_count]
            scene_indices[batch_index, removable] = additions
        novel_start += novel_count
        redundant_start += redundant_count

    gripper_indices = torch.arange(
        source_scene_points, source_points, device=point_cloud.device, dtype=torch.long
    ).unsqueeze(0).expand(batch_size, -1)
    indices = torch.cat([scene_indices, gripper_indices], dim=1)
    sampled = torch.gather(point_cloud, 1, indices.unsqueeze(-1).expand(-1, -1, point_cloud.shape[-1]))
    sampled_pad = torch.gather(point_is_pad, 1, indices) if torch.is_tensor(point_is_pad) else None
    return sampled, sampled_pad, indices


def multiscale_novelty_union_sample_fused_point_cloud(
    point_cloud: Tensor,
    *,
    target_points: int = 10_000,
    gripper_points: int = 500,
    voxel_size: float = 0.01,
    coarse_novelty_scale: float = 3.0,
    point_is_pad: Tensor | None = None,
) -> tuple[Tensor, Tensor | None, Tensor]:
    """Insert only coarse-persistent secondary coverage into a fine-protected primary cloud.

    The configured voxel size protects every occupied primary cell.  Secondary
    coverage must remain novel on the configured coarser grid, and only one
    representative of each coarse novel cell is inserted.  This is a
    view-count-agnostic input coreset: it changes no model module and assigns no
    fixed point budget to any camera.
    """

    return novelty_union_sample_fused_point_cloud(
        point_cloud,
        target_points=target_points,
        gripper_points=gripper_points,
        voxel_size=voxel_size,
        point_is_pad=point_is_pad,
        coarse_novelty_scale=coarse_novelty_scale,
    )


def consensus_multiscale_novelty_union_sample_fused_point_cloud(
    point_cloud: Tensor,
    *,
    target_points: int = 10_000,
    gripper_points: int = 500,
    voxel_size: float = 0.01,
    coarse_novelty_scale: float = 4.0,
    point_is_pad: Tensor | None = None,
) -> tuple[Tensor, Tensor | None, Tensor]:
    """Fuse overlapping views by voxel medoids and retain coarse novel coverage.

    Every fine voxel occupied by the primary view keeps exactly one protected
    representative.  When points from other views occupy that same voxel, the
    protected representative is the *real input point* nearest the union
    centroid, rather than always the first (therefore primary) array entry.
    Secondary-only coverage is admitted only when it remains novel on the
    configured coarser grid, with one secondary medoid per coarse cell.

    The output is an exact-index coreset: no synthetic point, camera quota,
    learned gate, semantic rule, or local point-to-point transport is used.
    The primary fine-cell cover and primary gripper tail are preserved, and an
    already target-sized single-view cloud remains byte-identical.
    """

    if point_cloud.ndim != 3 or point_cloud.shape[-1] != 6:
        raise ValueError(f"Expected point cloud shape (B,N,6), got {tuple(point_cloud.shape)}.")
    target_points = int(target_points)
    gripper_points = int(gripper_points)
    voxel_size = float(voxel_size)
    coarse_novelty_scale = float(coarse_novelty_scale)
    if not np.isfinite(voxel_size) or voxel_size <= 0.0:
        raise ValueError(f"voxel_size must be finite and positive, got {voxel_size}.")
    if not np.isfinite(coarse_novelty_scale) or coarse_novelty_scale <= 1.0:
        raise ValueError("coarse_novelty_scale must be finite and greater than one.")

    batch_size, source_points, _ = point_cloud.shape
    if source_points == target_points:
        identity = torch.arange(source_points, device=point_cloud.device, dtype=torch.long)
        identity = identity.unsqueeze(0).expand(batch_size, -1)
        return point_cloud, point_is_pad, identity

    source_scene_points = source_points - gripper_points
    target_scene_points = target_points - gripper_points
    if source_scene_points < target_scene_points or source_scene_points % target_scene_points != 0:
        raise ValueError(
            "Consensus multiscale fusion expects an ordered union of equal per-view scene budgets; "
            f"source_scene_points={source_scene_points}, target_scene_points={target_scene_points}."
        )
    if torch.is_tensor(point_is_pad):
        if point_is_pad.shape != (batch_size, source_points):
            raise ValueError(
                f"point_is_pad must have shape {(batch_size, source_points)}, got {tuple(point_is_pad.shape)}."
            )
        if bool(point_is_pad[:, :source_scene_points].any().item()):
            raise ValueError("Consensus multiscale fusion requires fixed-size, unpadded scene unions.")

    scene_xyz = point_cloud[:, :source_scene_points, :3].contiguous()
    flat_xyz = scene_xyz.reshape(-1, 3)
    flat_local_indices = torch.arange(
        source_scene_points, device=point_cloud.device, dtype=torch.long
    ).repeat(batch_size)
    batch_ids = torch.arange(batch_size, device=point_cloud.device, dtype=torch.int64)
    batch_ids = batch_ids.repeat_interleave(source_scene_points)
    primary_mask = flat_local_indices < target_scene_points
    flat_global_indices = torch.arange(
        batch_size * source_scene_points, device=point_cloud.device, dtype=torch.long
    )
    sentinel = batch_size * source_scene_points

    fine_quantized = torch.floor(flat_xyz / voxel_size).to(torch.int64)
    fine_voxels, fine_inverse = torch.unique(
        torch.cat([batch_ids.unsqueeze(1), fine_quantized], dim=1),
        dim=0,
        return_inverse=True,
    )
    fine_count = torch.bincount(fine_inverse, minlength=fine_voxels.shape[0])
    fine_sum = torch.zeros(
        (fine_voxels.shape[0], 3), device=point_cloud.device, dtype=flat_xyz.dtype
    )
    fine_sum.index_add_(0, fine_inverse, flat_xyz)
    fine_centroid = fine_sum / fine_count.to(flat_xyz.dtype).unsqueeze(1)
    fine_distance = torch.sum((flat_xyz - fine_centroid.index_select(0, fine_inverse)) ** 2, dim=1)
    fine_min_distance = torch.full(
        (fine_voxels.shape[0],), torch.inf, device=point_cloud.device, dtype=flat_xyz.dtype
    )
    fine_min_distance.scatter_reduce_(
        0, fine_inverse, fine_distance, reduce="amin", include_self=True
    )
    fine_medoid = torch.full(
        (fine_voxels.shape[0],), sentinel, device=point_cloud.device, dtype=torch.long
    )
    fine_is_medoid = fine_distance == fine_min_distance.index_select(0, fine_inverse)
    fine_medoid.scatter_reduce_(
        0,
        fine_inverse[fine_is_medoid],
        flat_global_indices[fine_is_medoid],
        reduce="amin",
        include_self=True,
    )
    first_primary = torch.full_like(fine_medoid, sentinel)
    first_primary.scatter_reduce_(
        0,
        fine_inverse[primary_mask],
        flat_global_indices[primary_mask],
        reduce="amin",
        include_self=True,
    )
    primary_fine_voxel = first_primary != sentinel

    coarse_quantized = torch.floor(
        flat_xyz / (voxel_size * coarse_novelty_scale)
    ).to(torch.int64)
    coarse_voxels, coarse_inverse = torch.unique(
        torch.cat([batch_ids.unsqueeze(1), coarse_quantized], dim=1),
        dim=0,
        return_inverse=True,
    )
    coarse_first_primary = torch.full(
        (coarse_voxels.shape[0],), sentinel, device=point_cloud.device, dtype=torch.long
    )
    coarse_first_primary.scatter_reduce_(
        0,
        coarse_inverse[primary_mask],
        flat_global_indices[primary_mask],
        reduce="amin",
        include_self=True,
    )
    secondary_mask = ~primary_mask
    coarse_secondary_count = torch.bincount(
        coarse_inverse[secondary_mask], minlength=coarse_voxels.shape[0]
    )
    coarse_secondary_sum = torch.zeros(
        (coarse_voxels.shape[0], 3), device=point_cloud.device, dtype=flat_xyz.dtype
    )
    coarse_secondary_sum.index_add_(
        0, coarse_inverse[secondary_mask], flat_xyz[secondary_mask]
    )
    coarse_secondary_centroid = coarse_secondary_sum / coarse_secondary_count.clamp_min(1).to(
        flat_xyz.dtype
    ).unsqueeze(1)
    secondary_coarse_inverse = coarse_inverse[secondary_mask]
    secondary_distance = torch.sum(
        (
            flat_xyz[secondary_mask]
            - coarse_secondary_centroid.index_select(0, secondary_coarse_inverse)
        )
        ** 2,
        dim=1,
    )
    coarse_min_secondary_distance = torch.full(
        (coarse_voxels.shape[0],), torch.inf, device=point_cloud.device, dtype=flat_xyz.dtype
    )
    coarse_min_secondary_distance.scatter_reduce_(
        0,
        secondary_coarse_inverse,
        secondary_distance,
        reduce="amin",
        include_self=True,
    )
    coarse_secondary_medoid = torch.full(
        (coarse_voxels.shape[0],), sentinel, device=point_cloud.device, dtype=torch.long
    )
    secondary_global_indices = flat_global_indices[secondary_mask]
    secondary_is_medoid = secondary_distance == coarse_min_secondary_distance.index_select(
        0, secondary_coarse_inverse
    )
    coarse_secondary_medoid.scatter_reduce_(
        0,
        secondary_coarse_inverse[secondary_is_medoid],
        secondary_global_indices[secondary_is_medoid],
        reduce="amin",
        include_self=True,
    )
    coarse_novel = (coarse_first_primary == sentinel) & (coarse_secondary_count > 0)
    novel_global = coarse_secondary_medoid[coarse_novel]

    scene_indices = torch.arange(
        target_scene_points, device=point_cloud.device, dtype=torch.long
    ).unsqueeze(0).repeat(batch_size, 1)
    protected_positions = torch.zeros(
        (batch_size, target_scene_points), device=point_cloud.device, dtype=torch.bool
    )

    primary_voxel_ids = torch.nonzero(primary_fine_voxel, as_tuple=False).flatten()
    primary_representatives = fine_medoid.index_select(0, primary_voxel_ids)
    primary_slots = first_primary.index_select(0, primary_voxel_ids)
    representative_batches = fine_voxels.index_select(0, primary_voxel_ids)[:, 0].long()
    representative_local = primary_representatives % source_scene_points
    slot_local = primary_slots % source_scene_points
    secondary_representative = representative_local >= target_scene_points
    if bool(secondary_representative.any().item()):
        scene_indices[
            representative_batches[secondary_representative],
            slot_local[secondary_representative],
        ] = representative_local[secondary_representative]
    protected_local = torch.where(secondary_representative, slot_local, representative_local)
    protected_positions[representative_batches, protected_local] = True

    novel_batches = torch.div(novel_global, source_scene_points, rounding_mode="floor")
    novel_counts = torch.bincount(novel_batches, minlength=batch_size).cpu().tolist()
    novel_start = 0
    for batch_index in range(batch_size):
        additions = novel_global[novel_start : novel_start + int(novel_counts[batch_index])]
        additions = additions % source_scene_points
        removable = torch.nonzero(~protected_positions[batch_index], as_tuple=False).flatten()
        insert_count = min(int(additions.numel()), int(removable.numel()))
        if insert_count:
            scene_indices[batch_index, removable[:insert_count]] = additions[:insert_count]
        novel_start += int(novel_counts[batch_index])

    gripper_indices = torch.arange(
        source_scene_points, source_points, device=point_cloud.device, dtype=torch.long
    ).unsqueeze(0).expand(batch_size, -1)
    indices = torch.cat([scene_indices, gripper_indices], dim=1)
    sampled = torch.gather(point_cloud, 1, indices.unsqueeze(-1).expand(-1, -1, point_cloud.shape[-1]))
    sampled_pad = torch.gather(point_is_pad, 1, indices) if torch.is_tensor(point_is_pad) else None
    return sampled, sampled_pad, indices


def _local_transport_replacement_pairs(
    novel_xyz: Tensor,
    redundant_primary_xyz: Tensor,
) -> tuple[Tensor, Tensor]:
    """Greedily match every possible novel/redundant pair by nearest XYZ.

    In each round every unmatched secondary point proposes to its nearest still
    available redundant primary sample. Each primary sample accepts the closest
    proposal (stable secondary order resolves exact ties), then both endpoints
    leave the next round. This yields exactly ``min(N, M)`` one-to-one pairs,
    without a camera quota, distance threshold, or point-array-order pairing.
    """

    if novel_xyz.ndim != 2 or novel_xyz.shape[-1] != 3:
        raise ValueError(f"Expected novel_xyz shape (N,3), got {tuple(novel_xyz.shape)}.")
    if redundant_primary_xyz.ndim != 2 or redundant_primary_xyz.shape[-1] != 3:
        raise ValueError(
            "Expected redundant_primary_xyz shape (M,3), got "
            f"{tuple(redundant_primary_xyz.shape)}."
        )
    novel_count = int(novel_xyz.shape[0])
    redundant_count = int(redundant_primary_xyz.shape[0])
    if novel_count == 0 or redundant_count == 0:
        empty = torch.empty(0, dtype=torch.long, device=novel_xyz.device)
        return empty, empty

    novel_ids = torch.arange(novel_count, device=novel_xyz.device, dtype=torch.long)
    matched = torch.zeros(novel_count, device=novel_xyz.device, dtype=torch.bool)
    used_redundant = torch.zeros(
        redundant_count, device=novel_xyz.device, dtype=torch.bool
    )
    selected_redundant = torch.full(
        (novel_count,), -1, device=novel_xyz.device, dtype=torch.long
    )
    sentinel = novel_count
    target_matches = min(novel_count, redundant_count)

    while int(matched.sum().item()) < target_matches:
        remaining_novel = novel_ids[~matched]
        remaining_redundant = torch.arange(
            redundant_count, device=novel_xyz.device, dtype=torch.long
        )[~used_redundant]
        query = novel_xyz.index_select(0, remaining_novel).contiguous()
        target = redundant_primary_xyz.index_select(0, remaining_redundant).contiguous()
        if novel_xyz.is_cuda and _pointops_knn_query is not None:
            query_offset = torch.tensor(
                [query.shape[0]], device=novel_xyz.device, dtype=torch.int32
            )
            target_offset = torch.tensor(
                [target.shape[0]], device=novel_xyz.device, dtype=torch.int32
            )
            try:
                with torch.cuda.device(novel_xyz.device):
                    proposed_local, proposal_distances = _pointops_knn_query(
                        1, target, target_offset, query, query_offset
                    )
                proposed_local = proposed_local.reshape(-1).long()
                proposal_distances = proposal_distances.reshape(-1)
            except Exception as exc:
                if _env_flag("SONG_POINTSEG_REQUIRE_POINTOPS", True):
                    raise RuntimeError(
                        "pointops KNN failed while matching local multi-view transport."
                    ) from exc
                distances = torch.cdist(query, target)
                proposal_distances, proposed_local = distances.min(dim=1)
        else:
            if novel_xyz.is_cuda and _env_flag("SONG_POINTSEG_REQUIRE_POINTOPS", True):
                raise _pointops_required_error()
            distances = torch.cdist(query, target)
            proposal_distances, proposed_local = distances.min(dim=1)

        proposed = remaining_redundant.index_select(0, proposed_local)
        best_distance = torch.full(
            (redundant_count,),
            torch.inf,
            device=novel_xyz.device,
            dtype=proposal_distances.dtype,
        )
        best_distance.scatter_reduce_(
            0, proposed, proposal_distances, reduce="amin", include_self=True
        )
        winners = proposal_distances == best_distance[proposed]
        # Resolve exact-distance ties by the stable secondary point order.
        best_novel = torch.full(
            (redundant_count,), sentinel, device=novel_xyz.device, dtype=torch.long
        )
        best_novel.scatter_reduce_(
            0,
            proposed[winners],
            remaining_novel[winners],
            reduce="amin",
            include_self=True,
        )
        winners &= remaining_novel == best_novel[proposed]
        accepted_novel = remaining_novel[winners]
        accepted_redundant = proposed[winners]
        if accepted_novel.numel() == 0:
            raise RuntimeError("Local transport matching made no progress.")
        selected_redundant[accepted_novel] = accepted_redundant
        matched[accepted_novel] = True
        used_redundant[accepted_redundant] = True

    matched_novel = novel_ids[matched]
    return selected_redundant[matched], matched_novel


def transport_novelty_union_sample_fused_point_cloud(
    point_cloud: Tensor,
    *,
    target_points: int = 10_000,
    gripper_points: int = 500,
    voxel_size: float = 0.01,
    point_is_pad: Tensor | None = None,
) -> tuple[Tensor, Tensor | None, Tensor]:
    """Novelty union with local one-to-one primary replacement transport."""

    return novelty_union_sample_fused_point_cloud(
        point_cloud,
        target_points=target_points,
        gripper_points=gripper_points,
        voxel_size=voxel_size,
        point_is_pad=point_is_pad,
        local_transport=True,
    )


def compose_point_cloud_views(
    view_clouds: list[np.ndarray] | tuple[np.ndarray, ...],
    *,
    gripper_points: int = 500,
    seed: int = 0,
    view_weights: Any = None,
    fusion: Any = "legacy_budget",
) -> np.ndarray:
    """Compose one or more stored camera clouds for the selected fusion policy.

    Every stored camera cloud keeps the original ``num_points`` layout and the
    original gripper tail. Single-view mode returns that cloud unchanged.
    ``legacy_budget`` divides a fixed scene budget across views and appends the
    primary-view gripper tail once. ``fps`` returns the equal scene union plus
    that same gripper tail; the shared GPU sampler reduces it to the model point
    budget later so cache generation, online training, and inference agree.
    ``uniform_union`` draws the checkpoint-native scene budget uniformly
    without replacement from the complete scene union, so camera contribution
    is sampled instead of assigned by a fixed quota. ``full_union`` returns
    that ordered union directly, preserving every scene
    point from every view and one primary-view gripper tail. The downstream
    point stack is length agnostic and the collator supplies padding masks when
    single- and multi-view examples share a batch. ``primary_residual`` returns
    the same ordered union without downsampling:
    the model reconstructs a byte-identical primary cloud and a separate wrist
    cloud, then injects the wrist representation through a zero-initialized
    matrix residual. This preserves the primary checkpoint function without a
    learned gate or a hand-chosen view ratio.
    """

    clouds = [np.asarray(cloud, dtype=np.float32) for cloud in view_clouds]
    if not clouds:
        raise ValueError("At least one camera cloud is required.")
    for cloud in clouds:
        if cloud.ndim != 2 or cloud.shape[-1] != 6:
            raise ValueError(f"Expected point cloud shape (N,6), got {cloud.shape}.")
    if len(clouds) == 1:
        return np.ascontiguousarray(clouds[0], dtype=np.float32)
    fusion = parse_camera_view_fusion(fusion)
    total_points = int(clouds[0].shape[0])
    if any(int(cloud.shape[0]) != total_points for cloud in clouds[1:]):
        raise ValueError(
            f"All camera clouds must contain the same number of points, got {[c.shape[0] for c in clouds]}."
        )
    gripper_points = int(gripper_points)
    if gripper_points < 0 or gripper_points >= total_points:
        raise ValueError(
            f"gripper_points must be in [0, {total_points - 1}], got {gripper_points}."
        )
    if fusion == "uniform_union":
        if parse_camera_view_weights(view_weights, num_views=len(clouds)) is not None:
            raise ValueError("camera view weights cannot be combined with uniform_union fusion.")
        scene_budget = total_points - gripper_points
        scene_union = np.concatenate(
            [cloud[:-gripper_points] if gripper_points else cloud for cloud in clouds],
            axis=0,
        )
        if scene_union.shape[0] < scene_budget:
            raise ValueError(
                f"Scene union has {scene_union.shape[0]} points but requires {scene_budget}."
            )
        rng = np.random.default_rng(int(seed))
        indices = rng.choice(scene_union.shape[0], scene_budget, replace=False)
        parts = [np.ascontiguousarray(scene_union[indices], dtype=np.float32)]
        if gripper_points:
            parts.append(np.ascontiguousarray(clouds[0][-gripper_points:], dtype=np.float32))
        return np.ascontiguousarray(np.concatenate(parts, axis=0), dtype=np.float32)
    if fusion in {
        "fps",
        "voxel_fps",
        "voxel_cover_fps",
        "novelty_union",
        "multiscale_novelty_union",
        "consensus_multiscale_novelty_union",
        "transport_novelty_union",
        "full_union",
        "primary_residual",
    }:
        if parse_camera_view_weights(view_weights, num_views=len(clouds)) is not None:
            raise ValueError(f"camera view weights cannot be combined with {fusion} fusion.")
        parts = [
            np.ascontiguousarray(cloud[:-gripper_points] if gripper_points else cloud, dtype=np.float32)
            for cloud in clouds
        ]
        if gripper_points:
            parts.append(np.ascontiguousarray(clouds[0][-gripper_points:], dtype=np.float32))
        return np.ascontiguousarray(np.concatenate(parts, axis=0), dtype=np.float32)

    scene_budget = total_points - gripper_points
    parsed_weights = parse_camera_view_weights(view_weights, num_views=len(clouds))
    weights = np.ones(len(clouds), dtype=np.float64)
    if parsed_weights is not None:
        weights = np.asarray(parsed_weights, dtype=np.float64)
    expected = scene_budget * weights / weights.sum()
    allocations_array = np.floor(expected).astype(np.int64)
    remainder = int(scene_budget - allocations_array.sum())
    if remainder:
        fractional_order = np.argsort(-(expected - allocations_array), kind="stable")
        allocations_array[fractional_order[:remainder]] += 1
    allocations = allocations_array.tolist()
    rng = np.random.default_rng(int(seed))
    parts: list[np.ndarray] = []
    for cloud, count in zip(clouds, allocations, strict=True):
        scene = cloud[:-gripper_points] if gripper_points else cloud
        if len(scene) == 0:
            raise ValueError("A camera cloud contains no non-gripper scene points.")
        if len(scene) == count:
            sample = scene
        else:
            indices = rng.choice(len(scene), count, replace=len(scene) < count)
            sample = scene[indices]
        parts.append(np.ascontiguousarray(sample, dtype=np.float32))
    if gripper_points:
        parts.append(np.ascontiguousarray(clouds[0][-gripper_points:], dtype=np.float32))
    composed = np.ascontiguousarray(np.concatenate(parts, axis=0), dtype=np.float32)
    if composed.shape != clouds[0].shape:
        raise RuntimeError(
            f"Composed cloud shape {composed.shape} does not match stored shape {clouds[0].shape}."
        )
    return composed


def infer_single_view_point_count(
    fused_point_count: int,
    *,
    num_views: int,
    gripper_points: int,
) -> int:
    """Recover the native per-view point count from an equal-view union.

    Reduction-based multi-view fusion concatenates every view's scene points
    and appends exactly one primary-view gripper tail.  Its input adapter must
    therefore reduce the union back to the point count of one stored view,
    whatever that count happens to be (for example 10k in simulation or 50k
    on a real robot).  This helper deliberately encodes that relationship
    instead of a dataset-specific numeric constant.
    """

    fused_point_count = int(fused_point_count)
    num_views = int(num_views)
    gripper_points = int(gripper_points)
    if fused_point_count <= 0:
        raise ValueError(f"fused_point_count must be positive, got {fused_point_count}.")
    if num_views <= 0:
        raise ValueError(f"num_views must be positive, got {num_views}.")
    if gripper_points < 0 or gripper_points >= fused_point_count:
        raise ValueError(
            f"gripper_points must be in [0, {fused_point_count - 1}], got {gripper_points}."
        )
    if num_views == 1:
        return fused_point_count
    fused_scene_points = fused_point_count - gripper_points
    if fused_scene_points % num_views != 0:
        raise ValueError(
            "The fused cloud does not match the equal-view union layout: "
            f"({fused_point_count} total - {gripper_points} gripper) scene points "
            f"are not divisible by {num_views} views."
        )
    return fused_scene_points // num_views + gripper_points


class SongTemporalPointCloudDataset(torch.utils.data.Dataset):
    """Adds a temporal point-cloud window used only to build PointSeg supervision.

    The current frame is always stored at temporal index 0.  With ``bidirectional=True``
    (the default), the remaining entries contain both past and future frames.  This
    wrapper leaves the policy action chunk untouched; temporal poses are read from the
    episode's achieved ``observation.state`` trajectory.  Legacy datasets where state
    and action were duplicates can still fall back to the action column.
    """

    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        point_cloud_dir: str | Path,
        *,
        camera_views: str | tuple[str, ...] | list[str] | None = None,
        camera_view_weights: Any = None,
        camera_view_fusion: Any = "legacy_budget",
        gripper_points: int | None = None,
        future_offsets: tuple[int, ...] | list[int] = DEFAULT_FUTURE_OFFSETS,
        bidirectional: bool = True,
        trajectory_samples: int = 32,
        current_points: int | None = 8192,
        future_points: int | None = 16384,
        seed: int = 1000,
        return_full_point_cloud: bool = False,
        include_base_item: bool = True,
        mmap_mode: str = "r",
    ):
        self.dataset = dataset
        self.point_cloud_dir = Path(point_cloud_dir)
        if camera_views is None:
            camera_views = os.environ.get("SONG_CAMERA_VIEWS", "agentview")
        if camera_view_weights is None:
            camera_view_weights = os.environ.get("SONG_CAMERA_VIEW_WEIGHTS")
        if gripper_points is None:
            gripper_points = int(os.environ.get("SONG_POINTCLOUD_GRIPPER_POINTS", "500"))
        self.camera_views = parse_camera_views(camera_views)
        self.camera_view_weights = parse_camera_view_weights(
            camera_view_weights,
            num_views=len(self.camera_views),
        )
        self.camera_view_fusion = parse_camera_view_fusion(camera_view_fusion)
        self.dataset_root = self.point_cloud_dir.parent
        self.point_cloud_dirs = {
            view: (
                self.point_cloud_dir
                if view == "agentview"
                else point_cloud_dir_for_view(self.dataset_root, view)
            )
            for view in self.camera_views
        }
        self.gripper_points = int(gripper_points)
        self.future_offsets = tuple(int(offset) for offset in future_offsets)
        if any(offset <= 0 for offset in self.future_offsets):
            raise ValueError("future_offsets should contain positive frame offsets; current frame is added automatically.")
        self.bidirectional = bool(bidirectional)
        past_offsets = tuple(-offset for offset in reversed(self.future_offsets)) if self.bidirectional else ()
        # Keep the current frame at index 0 because the motion-prior implementation
        # treats every following entry as context relative to this anchor.
        self.temporal_offsets = (0, *past_offsets, *self.future_offsets)
        self.trajectory_samples = int(trajectory_samples)
        if self.trajectory_samples < 0:
            raise ValueError("trajectory_samples must be non-negative.")
        self.current_points = None if current_points is None else int(current_points)
        self.future_points = None if future_points is None else int(future_points)
        self.seed = int(seed)
        self.return_full_point_cloud = return_full_point_cloud
        self.include_base_item = bool(include_base_item)
        self.mmap_mode = mmap_mode
        self._point_cloud_cache: dict[tuple[str, int], np.ndarray] = {}
        self._episode_motion_state_cache: dict[int, Tensor] = {}
        self._index_dataset = None

    def __getattr__(self, name: str):
        return getattr(self.dataset, name)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_point_cloud_cache"] = {}
        state["_episode_motion_state_cache"] = {}
        state["_index_dataset"] = None
        return state

    def __len__(self) -> int:
        return len(self.dataset)

    @staticmethod
    def _to_int(value: Any) -> int:
        if torch.is_tensor(value):
            return int(value.reshape(-1)[0].item())
        if isinstance(value, np.ndarray):
            return int(value.reshape(-1)[0].item())
        return int(value)

    def _episode_point_clouds(self, view: str, episode_index: int) -> np.ndarray:
        cache_key = (str(view), int(episode_index))
        point_clouds = self._point_cloud_cache.get(cache_key)
        if point_clouds is None:
            point_clouds = open_episode_point_clouds(
                self.point_cloud_dirs[str(view)],
                episode_index,
                mmap_mode=self.mmap_mode,
            )
            self._point_cloud_cache[cache_key] = point_clouds
        return point_clouds

    def _point_cloud_frame(self, episode_index: int, frame_index: int) -> np.ndarray:
        clouds = [
            np.asarray(self._episode_point_clouds(view, episode_index)[frame_index], dtype=np.float32)
            for view in self.camera_views
        ]
        # Keep view composition independent of the cache sampling seed so an
        # indices cache can be reconstructed exactly during training.
        seed = 1000 + int(episode_index) * 1_000_003 + int(frame_index) * 97
        return compose_point_cloud_views(
            clouds,
            gripper_points=self.gripper_points,
            seed=seed,
            view_weights=self.camera_view_weights,
            fusion=self.camera_view_fusion,
        )

    def _episode_motion_states(self, episode_index: int) -> Tensor:
        """Load achieved EEF poses used by the geometric motion prior."""

        cached = self._episode_motion_state_cache.get(episode_index)
        if cached is not None:
            return cached

        # Lightweight test/custom datasets can expose per-episode state arrays.
        observation_states = getattr(self.dataset, "observation_states", None)
        if observation_states is not None:
            episode_states = torch.as_tensor(
                observation_states[episode_index], dtype=torch.float32
            )
            self._episode_motion_state_cache[episode_index] = episode_states
            return episode_states

        # Backward-compatible fallback for old custom datasets where
        # observation.state and action were intentionally identical.
        actions = getattr(self.dataset, "actions", None)
        if actions is not None:
            episode_states = torch.as_tensor(actions[episode_index], dtype=torch.float32)
            self._episode_motion_state_cache[episode_index] = episode_states
            return episode_states

        ensure_loaded = getattr(self.dataset, "_ensure_hf_dataset_loaded", None)
        if callable(ensure_loaded):
            ensure_loaded()
        hf_dataset = getattr(self.dataset, "hf_dataset", None)
        meta = getattr(self.dataset, "meta", None)
        episodes = getattr(meta, "episodes", None)
        if hf_dataset is None or episodes is None:
            raise RuntimeError(
                "Bidirectional PointSeg supervision needs access to the raw per-frame "
                "observation.state trajectory. Expected a LeRobotDataset or a dataset "
                "exposing per-episode `observation_states`."
            )

        episode_record = None
        for position, candidate in enumerate(episodes):
            candidate_index = int(candidate.get("episode_index", position))
            if candidate_index == episode_index:
                episode_record = candidate
                break
        if episode_record is None:
            raise KeyError(f"Could not find episode metadata for episode_index={episode_index}.")

        start = int(episode_record["dataset_from_index"])
        end = int(episode_record["dataset_to_index"])
        absolute_indices = list(range(start, end))
        absolute_to_relative = getattr(self.dataset, "_absolute_to_relative_idx", None)
        if absolute_to_relative is None:
            relative_indices = absolute_indices
        else:
            relative_indices = [absolute_to_relative[index] for index in absolute_indices]
        state_column = "observation.state"
        column_names = set(getattr(hf_dataset, "column_names", ()) or ())
        if state_column not in column_names:
            # Old generated datasets duplicated achieved poses into action.
            # This fallback is intentionally schema-based; new datasets must
            # never use commanded targets as observed motion.
            state_column = "action"
        try:
            values = hf_dataset[state_column][relative_indices]
        except (KeyError, TypeError, IndexError):
            values = hf_dataset[relative_indices][state_column]
        episode_states = torch.stack([torch.as_tensor(value) for value in values]).to(
            dtype=torch.float32
        )
        self._episode_motion_state_cache[episode_index] = episode_states
        return episode_states

    def _relative_temporal_poses(self, episode_index: int, frame_indices: list[int]) -> Tensor:
        motion_states = self._episode_motion_states(episode_index)
        poses = motion_states[frame_indices, :9].to(dtype=torch.float32)
        rel = relative_poses_to_first(poses.unsqueeze(0)).squeeze(0)
        rel[0] = _identity_pose9(device=rel.device, dtype=rel.dtype)
        return rel

    def _relative_episode_trajectory(self, episode_index: int, frame_index: int) -> tuple[Tensor, Tensor]:
        """Return a sparse full-episode EE trajectory anchored at the current frame.

        Point-cloud KNN remains local.  This pose-only trajectory supplies long-range
        approach/contact evidence without loading point clouds from the whole episode.
        Index 0 is always the current pose so downstream code can use it as the anchor.
        """
        motion_states = self._episode_motion_states(episode_index)
        episode_len = int(motion_states.shape[0])
        if self.trajectory_samples == 0:
            sample_indices = torch.tensor([frame_index], dtype=torch.long)
        else:
            sample_indices = torch.linspace(
                0,
                max(episode_len - 1, 0),
                steps=self.trajectory_samples,
                dtype=torch.float32,
            ).round().to(dtype=torch.long)
            sample_indices = torch.cat([torch.tensor([frame_index], dtype=torch.long), sample_indices])
        poses = motion_states[sample_indices, :9].to(dtype=torch.float32)
        relative = relative_poses_to_first(poses.unsqueeze(0)).squeeze(0)
        relative[0] = _identity_pose9(device=relative.device, dtype=relative.dtype)
        return relative, sample_indices - int(frame_index)

    def _index_only_item(self, idx: int) -> dict[str, Any]:
        """Read frame identity without decoding image/video columns for offline cache generation."""
        ensure_loaded = getattr(self.dataset, "_ensure_hf_dataset_loaded", None)
        if callable(ensure_loaded):
            ensure_loaded()
        hf_dataset = getattr(self.dataset, "hf_dataset", None)
        if hf_dataset is None:
            full_item = self.dataset[idx]
            return {key: full_item[key] for key in ("episode_index", "frame_index")}
        if self._index_dataset is None:
            self._index_dataset = hf_dataset.select_columns(["episode_index", "frame_index"])
        return dict(self._index_dataset[idx])

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = dict(self.dataset[idx]) if self.include_base_item else self._index_only_item(idx)
        episode_index = self._to_int(item["episode_index"])
        frame_index = self._to_int(item["frame_index"])
        first_view_clouds = self._episode_point_clouds(self.camera_views[0], episode_index)
        episode_len = int(first_view_clouds.shape[0])
        rng = np.random.default_rng(self.seed + idx)

        current_full = self._point_cloud_frame(episode_index, frame_index)
        if self.camera_view_fusion in {
            "fps",
            "voxel_fps",
            "voxel_cover_fps",
            "novelty_union",
            "multiscale_novelty_union",
            "consensus_multiscale_novelty_union",
            "transport_novelty_union",
            "uniform_union",
            "full_union",
            "primary_residual",
        }:
            current_sample = current_full
            current_indices = np.arange(current_full.shape[0], dtype=np.int64)
        elif self.current_points is None:
            current_sample = current_full
            current_indices = np.arange(current_full.shape[0], dtype=np.int64)
        else:
            current_sample, current_indices = _sample_rows_with_indices(
                current_full, self.current_points, rng
            )
        item["observation.point_cloud"] = torch.from_numpy(current_sample)
        item["observation.point_cloud_indices"] = torch.from_numpy(current_indices)
        # Explicit source metadata lets diagnostics distinguish a raw high-resolution
        # frame from an already-reduced cloud without inferring it from fixed sizes.
        item["pointseg_source_num_points"] = torch.tensor(int(current_full.shape[0]), dtype=torch.long)
        item["pointseg_sample_num_points"] = torch.tensor(int(current_sample.shape[0]), dtype=torch.long)

        future_samples = []
        future_is_pad = []
        temporal_frame_indices = []
        for offset in self.temporal_offsets:
            raw_index = frame_index + offset
            clamped_index = min(max(raw_index, 0), episode_len - 1)
            future_is_pad.append(raw_index < 0 or raw_index >= episode_len)
            temporal_frame_indices.append(clamped_index)
            future_cloud = self._point_cloud_frame(episode_index, clamped_index)
            future_samples.append(
                future_cloud
                if self.camera_view_fusion
                in {
                    "fps",
                    "voxel_fps",
                    "voxel_cover_fps",
                    "novelty_union",
                    "multiscale_novelty_union",
                    "consensus_multiscale_novelty_union",
                    "transport_novelty_union",
                    "uniform_union",
                    "full_union",
                    "primary_residual",
                }
                else (
                    future_cloud
                    if self.future_points is None
                    else _sample_rows(future_cloud, self.future_points, rng)
                )
            )

        max_future_points = max(sample.shape[0] for sample in future_samples)
        if max_future_points <= 0:
            raise ValueError(f"Future point clouds are empty for dataset index {idx}.")
        future_array = np.zeros((len(future_samples), max_future_points, future_samples[0].shape[-1]), dtype=np.float32)
        future_mask = np.ones((len(future_samples), max_future_points), dtype=bool)
        for future_idx, sample in enumerate(future_samples):
            n_points = sample.shape[0]
            future_array[future_idx, :n_points] = sample
            future_mask[future_idx, :n_points] = False
        item["observation.point_cloud_future"] = torch.from_numpy(future_array)
        item["observation.point_cloud_future_is_pad"] = torch.from_numpy(future_mask)
        item["future_is_pad"] = torch.tensor(future_is_pad, dtype=torch.bool)
        item["future_offsets"] = torch.tensor(self.temporal_offsets, dtype=torch.long)
        item["future_ee_poses"] = self._relative_temporal_poses(episode_index, temporal_frame_indices)
        trajectory_poses, trajectory_offsets = self._relative_episode_trajectory(episode_index, frame_index)
        item["pointseg_trajectory_ee_poses"] = trajectory_poses
        item["pointseg_trajectory_offsets"] = trajectory_offsets

        if self.return_full_point_cloud:
            item["observation.point_cloud_full"] = torch.from_numpy(np.ascontiguousarray(current_full))

        return item


class SongPointSegCachedDataset(torch.utils.data.Dataset):
    """Reads offline Song pointseg training samples from sharded mmap arrays."""

    def __init__(self, cache_dir: str | Path, *, mmap_mode: str = "r"):
        self.cache_dir = Path(cache_dir)
        self.mmap_mode = mmap_mode
        manifest_path = self.cache_dir / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Song pointseg cache manifest is missing: {manifest_path}")

        with open(manifest_path) as f:
            self.manifest = json.load(f)

        version = int(self.manifest.get("version", -1))
        if version not in POINTSEG_CACHE_COMPATIBLE_VERSIONS:
            raise ValueError(
                f"Unsupported Song pointseg cache version {version}; expected one of "
                f"{POINTSEG_CACHE_COMPATIBLE_VERSIONS}. "
                "Rebuild the cache because motion-prior semantics changed."
            )

        fields = tuple(self.manifest.get("fields", ()))
        self.cache_mode = str(
            self.manifest.get(
                "cache_mode",
                "embedded_point_cloud" if "point_cloud" in fields else "indices",
            )
        )
        expected_fields = POINTSEG_CACHE_FIELDS if self.cache_mode == "embedded_point_cloud" else POINTSEG_CACHE_LABEL_FIELDS
        missing_fields = [field for field in expected_fields if field not in fields]
        if missing_fields:
            raise ValueError(f"Song pointseg cache is missing fields: {missing_fields}")
        self.fields = fields
        self.variable_num_points = bool(self.manifest.get("variable_num_points", False))

        self.shards = list(self.manifest.get("shards", []))
        if not self.shards:
            raise ValueError(f"Song pointseg cache has no shards: {manifest_path}")
        self.shard_lengths = [int(shard["length"]) for shard in self.shards]
        self._cumulative_lengths = np.cumsum(self.shard_lengths).tolist()
        self._array_cache: dict[int, dict[str, np.ndarray]] = {}

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_array_cache"] = {}
        return state

    def __len__(self) -> int:
        return int(self._cumulative_lengths[-1])

    def _locate_index(self, idx: int) -> tuple[int, int]:
        if idx < 0:
            idx += len(self)
        if idx < 0 or idx >= len(self):
            raise IndexError(idx)
        shard_index = bisect_right(self._cumulative_lengths, idx)
        previous = self._cumulative_lengths[shard_index - 1] if shard_index > 0 else 0
        return shard_index, idx - previous

    def _open_shard(self, shard_index: int) -> dict[str, np.ndarray]:
        arrays = self._array_cache.get(shard_index)
        if arrays is not None:
            return arrays

        shard_dir = self.cache_dir / self.shards[shard_index]["path"]
        arrays = {}
        for field in self.fields:
            path = shard_dir / f"{field}.npy"
            if not path.exists():
                raise FileNotFoundError(f"Song pointseg cache array is missing: {path}")
            arrays[field] = np.load(path, mmap_mode=self.mmap_mode)
        offsets_path = shard_dir / "sample_offsets.npy"
        if offsets_path.exists():
            arrays["sample_offsets"] = np.load(offsets_path, mmap_mode=self.mmap_mode)
        self._array_cache[shard_index] = arrays
        return arrays

    @staticmethod
    def _float_tensor(array: np.ndarray) -> Tensor:
        return torch.from_numpy(np.asarray(array, dtype=np.float32).copy())

    @staticmethod
    def _long_tensor(array: np.ndarray) -> Tensor:
        return torch.from_numpy(np.asarray(array, dtype=np.int64).copy())

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        shard_index, local_index = self._locate_index(idx)
        arrays = self._open_shard(shard_index)
        if "sample_offsets" in arrays:
            offsets = arrays["sample_offsets"]
            point_slice = slice(int(offsets[local_index]), int(offsets[local_index + 1]))
        else:
            point_slice = local_index
        item = {
            "pointseg.labels": self._long_tensor(arrays["labels"][point_slice]),
            "pointseg.weights": self._float_tensor(arrays["weights"][point_slice]),
            "pointseg.class_scores": self._float_tensor(arrays["class_scores"][point_slice]),
            "pointseg.foreground_score": self._float_tensor(arrays["foreground_score"][point_slice]),
            "episode_index": torch.tensor(int(arrays["episode_index"][local_index]), dtype=torch.long),
            "frame_index": torch.tensor(int(arrays["frame_index"][local_index]), dtype=torch.long),
            "dataset_index": torch.tensor(int(arrays["dataset_index"][local_index]), dtype=torch.long),
        }
        if "priors" in arrays:
            item["pointseg.priors"] = self._float_tensor(arrays["priors"][point_slice])
        if "role_scores" in arrays:
            item["pointseg.role_scores"] = self._float_tensor(arrays["role_scores"][point_slice])
        if self.cache_mode == "embedded_point_cloud":
            item["observation.point_cloud"] = self._float_tensor(arrays["point_cloud"][point_slice])
        else:
            item["observation.point_cloud_indices"] = self._long_tensor(arrays["point_indices"][point_slice])
        return item


def _use_pointops_knn(device: torch.device) -> bool:
    return (
        _pointops_knn_query is not None
        and not _POINTOPS_KNN_FAILED
        and device.type == "cuda"
    )


def _require_pointops_knn(device: torch.device) -> bool:
    return device.type == "cuda" and _env_flag("SONG_POINTSEG_REQUIRE_POINTOPS", True)


def _pointops_required_error() -> RuntimeError:
    if _pointops_knn_query is None:
        reason = "pointops.knn_query could not be imported"
    elif _POINTOPS_KNN_FAILED:
        reason = "pointops KNN was disabled by an earlier runtime failure"
    else:
        reason = "pointops KNN is unavailable"
    return RuntimeError(
        f"CUDA Song pointseg KNN requires pointops, but {reason}. "
        "Set SONG_POINTSEG_REQUIRE_POINTOPS=0 to allow the slower torch GEMM fallback."
    )


def _nearest_distances_pointops(query: Tensor, target: Tensor, target_is_pad: Tensor | None = None) -> Tensor:
    """Return exact nearest L2 distance using the pointops CUDA KNN kernel."""
    group_count, query_count = query.shape[:2]
    if target_is_pad is not None:
        target_is_pad = target_is_pad.to(device=query.device, dtype=torch.bool)
        if target_is_pad.shape != target.shape[:2]:
            raise ValueError(f"Expected target_is_pad shape {target.shape[:2]}, got {target_is_pad.shape}.")
        if not bool(target_is_pad.any().item()):
            target_is_pad = None

    if target_is_pad is None:
        target_count = target.shape[1]
        query_flat = query.reshape(group_count * query_count, 3).contiguous()
        target_flat = target.reshape(group_count * target_count, 3).contiguous()
        query_offset = (
            torch.arange(1, group_count + 1, device=query.device, dtype=torch.int32) * query_count
        ).contiguous()
        target_offset = (
            torch.arange(1, group_count + 1, device=query.device, dtype=torch.int32) * target_count
        ).contiguous()
        dist_out = None
    else:
        target_counts = (~target_is_pad).sum(dim=1)
        valid_group = target_counts > 0
        dist_out = query.new_full((group_count, query_count), float("inf"))
        if not bool(valid_group.any().item()):
            return dist_out

        query = query[valid_group].contiguous()
        target = target[valid_group].contiguous()
        target_is_pad = target_is_pad[valid_group].contiguous()
        target_counts = target_counts[valid_group]
        valid_group_count = int(target_counts.numel())
        query_flat = query.reshape(valid_group_count * query_count, 3).contiguous()
        target_flat = target[~target_is_pad].reshape(-1, 3).contiguous()
        query_offset = (
            torch.arange(1, valid_group_count + 1, device=query.device, dtype=torch.int32) * query_count
        ).contiguous()
        target_offset = torch.cumsum(target_counts.to(dtype=torch.int32), dim=0).contiguous()

    device_ctx = torch.cuda.device(query.device) if query.device.type == "cuda" else nullcontext()
    with device_ctx:
        _, dist = _pointops_knn_query(1, target_flat, target_offset, query_flat, query_offset)
    dist = dist.reshape(query.shape[0], query_count)
    if dist_out is None:
        return dist
    dist_out[valid_group] = dist
    return dist_out


def _nearest_distances_from_grouped_queries(
    query: Tensor, target: Tensor, target_is_pad: Tensor | None = None
) -> Tensor:
    """Return nearest L2 distance for each grouped query point.

    query: (G, N, 3), target: (G, M, 3)
    """
    global _POINTOPS_KNN_FAILED
    if _use_pointops_knn(query.device):
        try:
            return _nearest_distances_pointops(query, target, target_is_pad)
        except Exception as exc:
            if _require_pointops_knn(query.device):
                raise RuntimeError(
                    "pointops KNN failed while computing Song pointseg motion priors. "
                    "Keeping the fast CUDA path strict prevents silent mid-training slowdowns; "
                    "set SONG_POINTSEG_REQUIRE_POINTOPS=0 to allow the slower torch GEMM fallback."
                ) from exc
            _POINTOPS_KNN_FAILED = True
            warnings.warn(
                f"pointops KNN failed; falling back to torch GEMM nearest-neighbor. Error: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
    elif _require_pointops_knn(query.device):
        raise _pointops_required_error()

    target_t = target.transpose(1, 2).contiguous()
    dist_sq = torch.bmm(query, target_t)
    dist_sq.mul_(-2.0)
    dist_sq.add_(query.square().sum(dim=-1, keepdim=True))
    dist_sq.add_(target.square().sum(dim=-1).unsqueeze(-2))
    if target_is_pad is not None:
        dist_sq = dist_sq.masked_fill(target_is_pad[:, None, :].to(device=dist_sq.device), float("inf"))
    min_sq = dist_sq.clamp_min_(0.0).amin(dim=-1)
    return min_sq.sqrt_()


def _nearest_distances(query: Tensor, target: Tensor, chunk_size: int) -> Tensor:
    chunks = []
    target = target.contiguous()
    target_t = target.transpose(0, 1).contiguous()
    target_norm = target.square().sum(dim=-1).unsqueeze(0)
    for start in range(0, query.shape[0], chunk_size):
        query_chunk = query[start : start + chunk_size]
        dist_sq = query_chunk @ target_t
        dist_sq.mul_(-2.0)
        dist_sq.add_(query_chunk.square().sum(dim=-1, keepdim=True))
        dist_sq.add_(target_norm)
        chunks.append(dist_sq.clamp_min_(0.0).amin(dim=-1).sqrt_())
    return torch.cat(chunks, dim=0)


def _future_group_size(chunk_len: int, target_points: int, valid_count: int, device: torch.device) -> int:
    if valid_count <= 1:
        return 1
    if _use_pointops_knn(device):
        return valid_count
    elements_per_future = max(1, 2 * chunk_len * target_points)
    max_elements = 128_000_000
    if device.type == "cuda":
        try:
            free_bytes, total_bytes = torch.cuda.mem_get_info(device)
            max_bytes = min(int(total_bytes * 0.25), int(free_bytes * 0.60), 1_750_000_000)
            max_elements = max(32_000_000, max_bytes // 4)
        except RuntimeError:
            pass
    return max(1, min(valid_count, max_elements // elements_per_future))


def _future_motion_weights(
    transforms: Tensor,
    valid_future: Tensor,
    *,
    rotation_radius: float,
    baseline_threshold: float,
    baseline_temperature: float,
) -> tuple[Tensor, Tensor]:
    """Return continuous confidence weights for how informative each future pose is."""
    future_transforms = transforms[:, 1:]
    translation = torch.linalg.norm(future_transforms[..., :3, 3], dim=-1)
    trace = future_transforms[..., :3, :3].diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    rotation = torch.acos(((trace - 1.0) * 0.5).clamp(-1.0, 1.0))
    baseline = translation + float(rotation_radius) * rotation
    temperature = max(float(baseline_temperature), 1e-6)
    weights = torch.sigmoid((baseline - float(baseline_threshold)) / temperature)
    weights = weights.masked_fill(~valid_future, 0.0)
    return weights, baseline


def _aggregate_motion_hypotheses(
    held_by_future: Tensor,
    static_by_future: Tensor,
    motion_weights: Tensor,
    valid_future: Tensor,
    *,
    relative_gap_eps: float,
    topk: int,
) -> tuple[Tensor, Tensor, Tensor]:
    """Compare held/static hypotheses within each future frame, then aggregate robustly."""
    if held_by_future.shape != static_by_future.shape or held_by_future.ndim != 3:
        raise ValueError(
            "Expected held/static residuals with matching shape (B,T,N), got "
            f"{held_by_future.shape} and {static_by_future.shape}."
        )
    if motion_weights.shape != held_by_future.shape[:2] or valid_future.shape != held_by_future.shape[:2]:
        raise ValueError(
            f"Expected motion weights and valid mask shape {held_by_future.shape[:2]}, got "
            f"{motion_weights.shape} and {valid_future.shape}."
        )

    denominator = held_by_future + static_by_future + float(relative_gap_eps)
    relative_gap = (static_by_future - held_by_future) / denominator
    weighted_gap = relative_gap * motion_weights[..., None]
    weighted_gap = weighted_gap.masked_fill(~valid_future[..., None], -torch.inf)

    k = max(1, min(int(topk), held_by_future.shape[1]))
    top_scores, top_indices = torch.topk(weighted_gap, k=k, dim=1)
    selected_valid = torch.isfinite(top_scores)
    selected_weights = torch.gather(
        motion_weights[..., None].expand_as(held_by_future),
        1,
        top_indices,
    )
    selected_weights = selected_weights.masked_fill(~selected_valid, 0.0)

    selected_held = torch.gather(held_by_future, 1, top_indices)
    selected_static = torch.gather(static_by_future, 1, top_indices)
    selected_held = selected_held.masked_fill(~selected_valid, 0.0)
    selected_static = selected_static.masked_fill(~selected_valid, 0.0)

    weight_sum = selected_weights.sum(dim=1)
    held_residual = (selected_held * selected_weights).sum(dim=1) / weight_sum.clamp_min(1e-6)
    static_residual = (selected_static * selected_weights).sum(dim=1) / weight_sum.clamp_min(1e-6)
    residual_gap = top_scores.masked_fill(~selected_valid, 0.0).sum(dim=1)
    residual_gap = residual_gap / selected_valid.sum(dim=1).clamp_min(1)

    no_valid = ~valid_future.any(dim=1)
    if bool(no_valid.any().item()):
        held_residual[no_valid] = 1.0
        static_residual[no_valid] = 1.0
        residual_gap[no_valid] = 0.0
    return held_residual, static_residual, residual_gap


def _motion_residuals_batched(
    current_xyz: Tensor,
    future_xyz: Tensor,
    future_poses: Tensor,
    future_is_pad: Tensor,
    future_point_is_pad: Tensor | None = None,
    *,
    chunk_size: int,
    motion_gap_eps: float,
    motion_rotation_radius: float,
    motion_baseline_threshold: float,
    motion_baseline_temperature: float,
    motion_evidence_topk: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    transforms = pose9_to_matrix(future_poses)
    inverse_transforms = invert_transform(transforms)
    valid_future = ~future_is_pad[:, 1:]
    if future_point_is_pad is not None:
        valid_future = valid_future & (~future_point_is_pad[:, 1:].all(dim=-1))
    pair_batch, pair_future_offset = torch.nonzero(valid_future, as_tuple=True)
    pair_future = pair_future_offset + 1

    residual_shape = (current_xyz.shape[0], future_xyz.shape[1] - 1, current_xyz.shape[1])
    held_by_future = torch.full(
        residual_shape, float("inf"), dtype=current_xyz.dtype, device=current_xyz.device
    )
    static_by_future = torch.full_like(held_by_future, float("inf"))
    motion_weights, motion_baseline = _future_motion_weights(
        transforms,
        valid_future,
        rotation_radius=motion_rotation_radius,
        baseline_threshold=motion_baseline_threshold,
        baseline_temperature=motion_baseline_temperature,
    )

    if pair_batch.numel() == 0:
        empty = current_xyz.new_ones(current_xyz.shape[:2])
        return empty, empty.clone(), current_xyz.new_zeros(current_xyz.shape[:2]), motion_weights, motion_baseline

    for start in range(0, current_xyz.shape[1], chunk_size):
        current_chunk_len = min(chunk_size, current_xyz.shape[1] - start)
        group_size = _future_group_size(
            current_chunk_len, future_xyz.shape[2], int(pair_batch.numel()), current_xyz.device
        )
        for group_start in range(0, int(pair_batch.numel()), group_size):
            group_end = min(group_start + group_size, int(pair_batch.numel()))
            batch_group = pair_batch[group_start:group_end]
            future_group = pair_future[group_start:group_end]
            current_group = current_xyz[batch_group, start : start + current_chunk_len].contiguous()
            targets = future_xyz[batch_group, future_group].contiguous()
            target_is_pad = (
                future_point_is_pad[batch_group, future_group].contiguous()
                if future_point_is_pad is not None
                else None
            )
            inverse = inverse_transforms[batch_group, future_group]
            static_query = transform_points(current_group, inverse)
            grouped_query = torch.cat([current_group, static_query], dim=1).contiguous()
            nearest = _nearest_distances_from_grouped_queries(grouped_query, targets, target_is_pad)
            held_group = nearest[:, :current_chunk_len]
            static_group = nearest[:, current_chunk_len:]
            future_offset_group = pair_future_offset[group_start:group_end]
            held_by_future[batch_group, future_offset_group, start : start + current_chunk_len] = held_group
            static_by_future[batch_group, future_offset_group, start : start + current_chunk_len] = static_group

    held_residual, static_residual, residual_gap = _aggregate_motion_hypotheses(
        held_by_future,
        static_by_future,
        motion_weights,
        valid_future,
        relative_gap_eps=motion_gap_eps,
        topk=motion_evidence_topk,
    )
    return held_residual, static_residual, residual_gap, motion_weights, motion_baseline



def _subsample_pose_sequence(
    poses: Tensor,
    pose_is_pad: Tensor,
    max_poses: int,
) -> tuple[Tensor, Tensor]:
    """Select an evenly spaced valid pose subset independently for each batch item."""
    if poses.ndim != 3 or pose_is_pad.shape != poses.shape[:2]:
        raise ValueError(
            f"Expected poses (B,L,D) and pose_is_pad {poses.shape[:2]}, got "
            f"{poses.shape} and {pose_is_pad.shape}."
        )
    max_poses = max(1, int(max_poses))
    bsize, sequence_len, pose_dim = poses.shape
    if sequence_len <= max_poses and not bool(pose_is_pad.any().item()):
        return poses, pose_is_pad

    selected = poses.new_zeros((bsize, max_poses, pose_dim))
    selected_is_pad = torch.ones((bsize, max_poses), dtype=torch.bool, device=poses.device)
    for batch_index in range(bsize):
        valid_indices = torch.nonzero(~pose_is_pad[batch_index], as_tuple=False).flatten()
        if valid_indices.numel() == 0:
            continue
        if valid_indices.numel() > max_poses:
            positions = torch.linspace(
                0,
                valid_indices.numel() - 1,
                steps=max_poses,
                device=poses.device,
            ).round().to(dtype=torch.long)
            valid_indices = valid_indices[positions]
        count = int(valid_indices.numel())
        selected[batch_index, :count] = poses[batch_index, valid_indices]
        selected_is_pad[batch_index, :count] = False
    return selected, selected_is_pad



def _filter_tool_sweep_poses(
    poses: Tensor,
    pose_is_pad: Tensor,
    pose_offsets: Tensor | None,
    *,
    max_frame_offset: int,
    max_translation: float,
) -> tuple[Tensor, Tensor]:
    """Mask unrelated pre-grasp / post-release poses from the conditioned sweep."""
    filtered_is_pad = pose_is_pad.clone()
    if pose_offsets is not None:
        pose_offsets = pose_offsets.to(device=poses.device)
        if pose_offsets.shape != poses.shape[:2]:
            raise ValueError(
                f"Expected pose_offsets shape {poses.shape[:2]}, got {tuple(pose_offsets.shape)}."
            )
        filtered_is_pad |= pose_offsets.abs() > max(0, int(max_frame_offset))
    translation = torch.linalg.norm(poses[..., :3], dim=-1)
    filtered_is_pad |= translation > max(float(max_translation), 0.0)
    # Index zero is the current pose in SongTemporalPointCloudDataset.  Keep it
    # whenever it is structurally valid so current-distance diagnostics remain
    # well-defined even when all other poses are filtered.
    filtered_is_pad[:, 0] = pose_is_pad[:, 0]
    return poses, filtered_is_pad


def _fractional_distance_gate(
    distance: Tensor,
    point_is_pad: Tensor,
    fraction: float,
    temperature: float,
    eligible: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Keep the nearest eligible targets under a full-frame point budget.

    The budget is still ``fraction * all_valid_scene_points`` so this branch cannot
    activate the whole scene.  Ranking, however, is performed only over eligible
    static non-tool points.  Robot and held-tool points therefore no longer consume
    the complete target budget before a nearby flower pot can enter it.
    """
    if distance.shape != point_is_pad.shape:
        raise ValueError(f"Expected matching distance/mask shapes, got {distance.shape} and {point_is_pad.shape}.")
    if eligible is not None and eligible.shape != distance.shape:
        raise ValueError(f"Expected eligible shape {distance.shape}, got {eligible.shape}.")
    fraction = float(min(max(fraction, 1e-4), 1.0))
    temperature = max(float(temperature), 1e-6)
    gate = distance.new_zeros(distance.shape)
    radii = distance.new_zeros((distance.shape[0],))
    for batch_index in range(distance.shape[0]):
        all_valid = (~point_is_pad[batch_index]) & torch.isfinite(distance[batch_index])
        candidate = all_valid
        if eligible is not None:
            candidate = candidate & eligible[batch_index].to(device=distance.device, dtype=torch.bool)
        candidate_indices = torch.nonzero(candidate, as_tuple=False).flatten()
        if candidate_indices.numel() == 0:
            continue
        # Preserve the original global safety cap, but do not spend it on the
        # gripper / robot / held object itself.
        full_budget = max(1, int(np.ceil(float(all_valid.sum().item()) * fraction)))
        k = min(int(candidate_indices.numel()), full_budget)
        values = distance[batch_index, candidate_indices]
        selected_values, selected_local = torch.topk(values, k=k, largest=False)
        selected_indices = candidate_indices[selected_local]
        radius = selected_values.max()
        radii[batch_index] = radius
        selected_gate = 0.5 + 0.5 * torch.sigmoid((radius - selected_values) / temperature)
        gate[batch_index, selected_indices] = selected_gate
    return gate, radii

def _select_tool_sweep_candidates(
    current_xyz: Tensor,
    rigid_tool_score: Tensor,
    current_is_pad: Tensor,
    *,
    max_points: int,
    min_score: float,
    bridge_min_score: float,
    distance_boost: float,
    distance_scale: float,
    seed_radius: float,
    seed_min_score: float,
    component_radius: float,
    component_hops: int,
    preselect_multiplier: int,
    max_radius: float,
) -> tuple[Tensor, Tensor, Tensor]:
    """Select the gripper-connected rigid component as the conditioned tool cloud.

    Motion residuals on repeated planes can assign moderate ``rigid_tool_score`` to
    distant background.  A global top-k therefore pollutes the swept cloud and can
    make almost the entire scene appear close to the tool at some pose.  We first
    preselect high-score points, seed the component near the EEF origin, and then
    grow only through short spatial links.  The resulting component can include a
    long grasped object, but cannot jump directly to a far table, wall, or cabinet.
    """
    if current_xyz.ndim != 3 or current_xyz.shape[-1] != 3:
        raise ValueError(f"Expected current_xyz (B,N,3), got {current_xyz.shape}.")
    if rigid_tool_score.shape != current_xyz.shape[:2]:
        raise ValueError(
            f"Expected rigid_tool_score shape {current_xyz.shape[:2]}, got {rigid_tool_score.shape}."
        )
    if current_is_pad.shape != current_xyz.shape[:2]:
        raise ValueError(f"Expected current_is_pad shape {current_xyz.shape[:2]}, got {current_is_pad.shape}.")

    bsize, n_points = current_xyz.shape[:2]
    count = max(1, min(int(max_points), n_points))
    pre_count = max(count, min(n_points, count * max(1, int(preselect_multiplier))))
    scale = max(float(distance_scale), 1e-6)
    radial = torch.linalg.norm(current_xyz, dim=-1)
    radial_factor = 1.0 + float(distance_boost) * (radial / scale).clamp(0.0, 1.0)
    eligible = (
        (~current_is_pad)
        & (rigid_tool_score >= float(bridge_min_score))
        & (radial <= float(max_radius))
    )
    ranking_score = (rigid_tool_score * radial_factor).masked_fill(~eligible, -torch.inf)
    pre_rank, pre_indices = torch.topk(ranking_score, k=pre_count, dim=1, largest=True)
    pre_xyz = current_xyz.gather(1, pre_indices[..., None].expand(bsize, pre_count, 3))
    pre_raw_scores = rigid_tool_score.gather(1, pre_indices)
    pre_radial = radial.gather(1, pre_indices)
    pre_valid = torch.isfinite(pre_rank)

    selected_xyz = current_xyz.new_zeros((bsize, count, 3))
    selected_scores = rigid_tool_score.new_zeros((bsize, count))
    selected_is_pad = torch.ones((bsize, count), dtype=torch.bool, device=current_xyz.device)
    link_radius = max(float(component_radius), 1e-6)
    hops = max(1, int(component_hops))

    for batch_index in range(bsize):
        valid = pre_valid[batch_index]
        if not bool(valid.any().item()):
            continue
        xyz = pre_xyz[batch_index]
        scores = pre_raw_scores[batch_index]
        seed = (
            valid
            & (pre_radial[batch_index] <= float(seed_radius))
            & (scores >= max(float(seed_min_score), float(min_score)))
        )
        if not bool(seed.any().item()):
            # Do not invent a tool component from a far high-score patch.  A weak
            # near-EEF fallback is allowed only inside a slightly enlarged seed ball.
            fallback = valid & (pre_radial[batch_index] <= 1.5 * float(seed_radius))
            if not bool(fallback.any().item()):
                continue
            fallback_scores = scores.masked_fill(~fallback, -torch.inf)
            seed = torch.zeros_like(valid)
            seed[int(torch.argmax(fallback_scores).item())] = True

        distance = torch.cdist(xyz.unsqueeze(0), xyz.unsqueeze(0)).squeeze(0)
        adjacency = (distance <= link_radius) & valid[:, None] & valid[None, :]
        reachable = seed.clone()
        for _ in range(hops):
            expanded = adjacency[reachable].any(dim=0) if bool(reachable.any().item()) else reachable
            expanded = expanded & valid
            if torch.equal(expanded, reachable):
                break
            reachable = expanded
        if not bool(reachable.any().item()):
            continue

        connected_rank = pre_rank[batch_index].masked_fill(~reachable, -torch.inf)
        connected_count = min(count, int(reachable.sum().item()))
        _, local_indices = torch.topk(connected_rank, k=connected_count, largest=True)
        selected_xyz[batch_index, :connected_count] = xyz[local_indices]
        selected_scores[batch_index, :connected_count] = scores[local_indices]
        selected_is_pad[batch_index, :connected_count] = False

    return selected_xyz, selected_is_pad, selected_scores

def _nearest_distances_to_swept_points(
    current_xyz: Tensor,
    swept_xyz: Tensor,
    swept_is_pad: Tensor,
    current_is_pad: Tensor,
    *,
    chunk_size: int,
) -> Tensor:
    """Compute current-point distance to a per-sample swept tool cloud.

    The per-sample loop deliberately bounds the GEMM fallback memory when the
    optional pointops KNN extension is unavailable.  With pointops installed the
    same helper still uses its fast CUDA nearest-neighbour kernel.
    """
    if swept_xyz.ndim != 3 or swept_xyz.shape[-1] != 3:
        raise ValueError(f"Expected swept_xyz (B,M,3), got {swept_xyz.shape}.")
    if swept_is_pad.shape != swept_xyz.shape[:2]:
        raise ValueError(f"Expected swept_is_pad shape {swept_xyz.shape[:2]}, got {swept_is_pad.shape}.")
    if current_is_pad.shape != current_xyz.shape[:2]:
        raise ValueError(f"Expected current_is_pad shape {current_xyz.shape[:2]}, got {current_is_pad.shape}.")

    bsize, n_points = current_xyz.shape[:2]
    out = current_xyz.new_full((bsize, n_points), float("inf"))
    query_chunk = max(1, int(chunk_size))
    for batch_index in range(bsize):
        valid_target = ~swept_is_pad[batch_index]
        if not bool(valid_target.any().item()):
            continue
        target = swept_xyz[batch_index, valid_target].unsqueeze(0).contiguous()
        for start in range(0, n_points, query_chunk):
            end = min(start + query_chunk, n_points)
            query = current_xyz[batch_index : batch_index + 1, start:end].contiguous()
            out[batch_index, start:end] = _nearest_distances_from_grouped_queries(query, target)[0]
    return out.masked_fill(current_is_pad, 0.0)


def _nearest_distances_to_tool_pose_clouds(
    current_xyz: Tensor,
    tool_xyz_by_pose: Tensor,
    tool_is_pad_by_pose: Tensor,
    current_is_pad: Tensor,
    *,
    chunk_size: int,
) -> Tensor:
    """Return point-to-tool distance for every selected tool pose.

    Shapes are ``current_xyz=(B,N,3)``, ``tool_xyz_by_pose=(B,L,C,3)`` and the
    result is ``(B,L,N)``.  Keeping the pose axis lets the caller distinguish a
    target that the conditioned tool *approaches* from a point that is merely near
    one flattened sweep sample.
    """
    if tool_xyz_by_pose.ndim != 4 or tool_xyz_by_pose.shape[-1] != 3:
        raise ValueError(f"Expected tool_xyz_by_pose (B,L,C,3), got {tool_xyz_by_pose.shape}.")
    if tool_is_pad_by_pose.shape != tool_xyz_by_pose.shape[:3]:
        raise ValueError(
            f"Expected tool_is_pad_by_pose shape {tool_xyz_by_pose.shape[:3]}, "
            f"got {tool_is_pad_by_pose.shape}."
        )
    bsize, pose_count = tool_xyz_by_pose.shape[:2]
    n_points = current_xyz.shape[1]
    out = current_xyz.new_full((bsize, pose_count, n_points), float("inf"))
    query_chunk = max(1, int(chunk_size))
    for batch_index in range(bsize):
        valid_pose = ~tool_is_pad_by_pose[batch_index].all(dim=-1)
        valid_indices = torch.nonzero(valid_pose, as_tuple=False).flatten()
        if valid_indices.numel() == 0:
            continue
        targets = tool_xyz_by_pose[batch_index, valid_indices].contiguous()
        target_is_pad = tool_is_pad_by_pose[batch_index, valid_indices].contiguous()
        for start in range(0, n_points, query_chunk):
            end = min(start + query_chunk, n_points)
            query = current_xyz[batch_index, start:end]
            query = query.unsqueeze(0).expand(valid_indices.numel(), -1, -1).contiguous()
            distances = _nearest_distances_from_grouped_queries(query, targets, target_is_pad)
            out[batch_index, valid_indices, start:end] = distances
    return out.masked_fill(current_is_pad[:, None, :], 0.0)


def _compute_tool_sweep_distance(
    current_xyz: Tensor,
    rigid_tool_score: Tensor,
    current_is_pad: Tensor,
    sweep_poses: Tensor,
    sweep_is_pad: Tensor,
    *,
    sweep_offsets: Tensor | None,
    candidate_max_points: int,
    candidate_min_score: float,
    candidate_bridge_min_score: float,
    candidate_distance_boost: float,
    candidate_distance_scale: float,
    candidate_seed_radius: float,
    candidate_seed_min_score: float,
    candidate_component_radius: float,
    candidate_component_hops: int,
    candidate_preselect_multiplier: int,
    candidate_max_radius: float,
    max_poses: int,
    max_frame_offset: int,
    max_translation: float,
    dwell_sigma: float,
    nn_chunk_size: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Build the rigid gripper+condition sweep and its temporal approach cues.

    The selected points are the cloud that co-moves with the end effector: gripper
    points plus any grasped / conditioned object points.  For every scene point we
    retain the distance at each sampled pose, then derive:

    * minimum swept distance;
    * distance to the conditioned cloud at the current pose;
    * distance span across the trajectory (far-to-near evidence);
    * trajectory dwell near that point.

    This is the key distinction needed by watering-style tasks: the flower pot may
    remain perfectly static, but the kettle-conditioned cloud clearly approaches it.
    """
    candidates, candidate_is_pad, candidate_scores = _select_tool_sweep_candidates(
        current_xyz,
        rigid_tool_score,
        current_is_pad,
        max_points=candidate_max_points,
        min_score=candidate_min_score,
        bridge_min_score=candidate_bridge_min_score,
        distance_boost=candidate_distance_boost,
        distance_scale=candidate_distance_scale,
        seed_radius=candidate_seed_radius,
        seed_min_score=candidate_seed_min_score,
        component_radius=candidate_component_radius,
        component_hops=candidate_component_hops,
        preselect_multiplier=candidate_preselect_multiplier,
        max_radius=candidate_max_radius,
    )
    sweep_poses, sweep_is_pad = _filter_tool_sweep_poses(
        sweep_poses,
        sweep_is_pad,
        sweep_offsets,
        max_frame_offset=max_frame_offset,
        max_translation=max_translation,
    )
    selected_poses, selected_pose_is_pad = _subsample_pose_sequence(
        sweep_poses,
        sweep_is_pad,
        max_poses=max_poses,
    )
    transforms = pose9_to_matrix(selected_poses[..., :9])
    rotation = transforms[..., :3, :3]
    translation = transforms[..., :3, 3]
    swept_by_pose = torch.einsum("blij,bkj->blki", rotation, candidates)
    swept_by_pose = swept_by_pose + translation[:, :, None, :]
    swept_is_pad_by_pose = selected_pose_is_pad[:, :, None] | candidate_is_pad[:, None, :]

    distance_by_pose = _nearest_distances_to_tool_pose_clouds(
        current_xyz,
        swept_by_pose,
        swept_is_pad_by_pose,
        current_is_pad,
        chunk_size=nn_chunk_size,
    )
    valid_pose = ~selected_pose_is_pad
    finite_pose = valid_pose[:, :, None] & torch.isfinite(distance_by_pose)
    min_distance = distance_by_pose.masked_fill(~finite_pose, float("inf")).amin(dim=1)
    max_distance = distance_by_pose.masked_fill(~finite_pose, -torch.inf).amax(dim=1)
    any_valid = finite_pose.any(dim=1)
    distance_span = torch.where(
        any_valid,
        (max_distance - min_distance).clamp_min(0.0),
        torch.zeros_like(min_distance),
    )

    current_distance = _nearest_distances_to_tool_pose_clouds(
        current_xyz,
        candidates[:, None, :, :],
        candidate_is_pad[:, None, :],
        current_is_pad,
        chunk_size=nn_chunk_size,
    )[:, 0]
    current_to_min_delta = torch.where(
        torch.isfinite(current_distance) & torch.isfinite(min_distance),
        (current_distance - min_distance).clamp_min(0.0),
        torch.zeros_like(min_distance),
    )

    sigma = max(float(dwell_sigma), 1e-6)
    pose_proximity = torch.exp(-distance_by_pose / sigma)
    pose_proximity = torch.where(finite_pose, pose_proximity, torch.zeros_like(pose_proximity))
    valid_count = finite_pose.sum(dim=1).clamp_min(1).to(dtype=pose_proximity.dtype)
    dwell = pose_proximity.sum(dim=1) / valid_count

    min_distance = torch.where(any_valid, min_distance, torch.full_like(min_distance, float("inf")))
    min_distance = min_distance.masked_fill(current_is_pad, 0.0)
    current_distance = current_distance.masked_fill(current_is_pad, 0.0)
    distance_span = distance_span.masked_fill(current_is_pad, 0.0)
    current_to_min_delta = current_to_min_delta.masked_fill(current_is_pad, 0.0)
    dwell = dwell.masked_fill(current_is_pad, 0.0)
    valid_candidate_count = (~candidate_is_pad).sum(dim=1)
    return (
        min_distance,
        current_distance,
        current_to_min_delta,
        distance_span,
        dwell,
        valid_candidate_count,
        candidate_scores,
    )


def compute_motion_priors(
    current_pc: Tensor,
    future_pc: Tensor,
    future_poses: Tensor,
    future_is_pad: Tensor | None = None,
    *,
    current_is_pad: Tensor | None = None,
    future_point_is_pad: Tensor | None = None,
    trajectory_poses: Tensor | None = None,
    trajectory_is_pad: Tensor | None = None,
    trajectory_offsets: Tensor | None = None,
    nn_chunk_size: int = 1024,
    motion_gap_eps: float = 0.005,
    motion_rotation_radius: float = 0.08,
    motion_baseline_threshold: float = 0.015,
    motion_baseline_temperature: float = 0.005,
    motion_evidence_topk: int = 3,
    contact_sigma: float = 0.10,
    tool_candidate_min_score: float = 0.35,
    tool_candidate_bridge_min_score: float = 0.18,
    tool_candidate_max_points: int = 320,
    tool_candidate_distance_boost: float = 0.35,
    tool_candidate_distance_scale: float = 0.45,
    tool_candidate_seed_radius: float = 0.14,
    tool_candidate_seed_min_score: float = 0.35,
    tool_candidate_component_radius: float = 0.070,
    tool_candidate_component_hops: int = 10,
    tool_candidate_preselect_multiplier: int = 6,
    tool_candidate_max_radius: float = 0.65,
    tool_sweep_max_poses: int = 20,
    tool_sweep_use_full_trajectory: bool = True,
    tool_sweep_max_frame_offset: int = 96,
    tool_sweep_max_translation: float = 0.65,
    tool_sweep_dwell_sigma: float = 0.10,
    tool_interaction_enable: bool = True,
    held_sigma: float = 0.025,
    motion_relative_margin: float = 0.10,
    motion_relative_tau: float = 0.10,
) -> dict[str, Tensor]:
    """Compute geometry/motion priors used by pseudo labels and the segmentation network."""
    if current_pc.ndim != 3 or future_pc.ndim != 4 or future_poses.ndim != 3:
        raise ValueError("Expected current_pc (B,N,6), future_pc (B,K,M,6), future_poses (B,K,9).")
    bsize, n_points = current_pc.shape[:2]
    if future_is_pad is None:
        future_is_pad = torch.zeros(
            bsize, future_pc.shape[1], dtype=torch.bool, device=current_pc.device
        )
    future_is_pad = future_is_pad.to(device=current_pc.device, dtype=torch.bool)
    if current_is_pad is None:
        current_is_pad = torch.zeros(bsize, n_points, dtype=torch.bool, device=current_pc.device)
    else:
        current_is_pad = current_is_pad.to(device=current_pc.device, dtype=torch.bool)
        if current_is_pad.shape != current_pc.shape[:2]:
            raise ValueError(f"Expected current_is_pad shape {current_pc.shape[:2]}, got {current_is_pad.shape}.")
    if future_point_is_pad is not None:
        future_point_is_pad = future_point_is_pad.to(device=current_pc.device, dtype=torch.bool)
        if future_point_is_pad.shape != future_pc.shape[:3]:
            raise ValueError(
                f"Expected future_point_is_pad shape {future_pc.shape[:3]}, got {future_point_is_pad.shape}."
            )
        if not bool(future_point_is_pad.any().item()):
            future_point_is_pad = None
    current_xyz = current_pc[..., :3].to(dtype=torch.float32)
    future_xyz = future_pc[..., :3].to(device=current_pc.device, dtype=torch.float32)
    future_poses = future_poses.to(device=current_pc.device, dtype=torch.float32)

    held_residual, static_residual, residual_gap, motion_weights, motion_baseline = _motion_residuals_batched(
        current_xyz,
        future_xyz,
        future_poses,
        future_is_pad,
        future_point_is_pad,
        chunk_size=nn_chunk_size,
        motion_gap_eps=motion_gap_eps,
        motion_rotation_radius=motion_rotation_radius,
        motion_baseline_threshold=motion_baseline_threshold,
        motion_baseline_temperature=motion_baseline_temperature,
        motion_evidence_topk=motion_evidence_topk,
    )
    held_residual = torch.where(current_is_pad, torch.ones_like(held_residual), held_residual)
    static_residual = torch.where(current_is_pad, torch.ones_like(static_residual), static_residual)

    if trajectory_poses is None:
        global _TOOL_SWEEP_LOCAL_FALLBACK_WARNED
        if bool(tool_sweep_use_full_trajectory) and not _TOOL_SWEEP_LOCAL_FALLBACK_WARNED:
            warnings.warn(
                "tool_sweep_use_full_trajectory=True but trajectory_poses was not provided; "
                "falling back to the local temporal window. Pass "
                "batch['pointseg_trajectory_ee_poses'] so delayed tool-target interactions "
                "such as kettle-to-flower watering are visible to the pseudo-label generator.",
                RuntimeWarning,
                stacklevel=2,
            )
            _TOOL_SWEEP_LOCAL_FALLBACK_WARNED = True
        trajectory_poses = future_poses
        trajectory_is_pad = future_is_pad
        trajectory_offsets = None
    else:
        if trajectory_poses.ndim != 3 or trajectory_poses.shape[0] != bsize or trajectory_poses.shape[-1] < 3:
            raise ValueError(
                "Expected trajectory_poses with shape (B,L,>=3), got "
                f"{tuple(trajectory_poses.shape)}."
            )
        trajectory_poses = trajectory_poses.to(device=current_pc.device, dtype=torch.float32)
        if trajectory_is_pad is None:
            trajectory_is_pad = torch.zeros(
                trajectory_poses.shape[:2], dtype=torch.bool, device=current_pc.device
            )
        else:
            trajectory_is_pad = trajectory_is_pad.to(device=current_pc.device, dtype=torch.bool)
            if trajectory_is_pad.shape != trajectory_poses.shape[:2]:
                raise ValueError(
                    f"Expected trajectory_is_pad shape {trajectory_poses.shape[:2]}, "
                    f"got {tuple(trajectory_is_pad.shape)}."
                )
        if trajectory_offsets is not None:
            trajectory_offsets = trajectory_offsets.to(device=current_pc.device)
            if trajectory_offsets.shape != trajectory_poses.shape[:2]:
                raise ValueError(
                    f"Expected trajectory_offsets shape {trajectory_poses.shape[:2]}, "
                    f"got {tuple(trajectory_offsets.shape)}."
                )
        elif bool(tool_sweep_use_full_trajectory):
            global _TOOL_SWEEP_OFFSETS_WARNED
            if not _TOOL_SWEEP_OFFSETS_WARNED:
                warnings.warn(
                    "Full-trajectory conditioned tool sweep received no trajectory_offsets; "
                    "using translation-only filtering. Pass batch['pointseg_trajectory_offsets'] "
                    "to prevent unrelated pre-grasp / post-release poses from entering the sweep.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                _TOOL_SWEEP_OFFSETS_WARNED = True

    # Crucial separation: EEF-local contact / approach evidence must use the
    # local temporal point-cloud window.  The sparse episode trajectory is only
    # for the conditioned-tool sweep; using it here marks the entire robot path
    # as foreground in every frame.
    local_trajectory_poses = future_poses
    local_trajectory_is_pad = future_is_pad
    trajectory_xyz = local_trajectory_poses[..., :3]
    valid_traj = ~local_trajectory_is_pad
    point_to_traj = torch.linalg.norm(current_xyz[:, :, None, :] - trajectory_xyz[:, None, :, :], dim=-1)
    point_to_traj = point_to_traj.masked_fill(~valid_traj[:, None, :], float("inf"))
    point_to_traj = point_to_traj.masked_fill(current_is_pad[:, :, None], float("inf"))
    min_traj_dist = point_to_traj.amin(dim=-1)
    min_traj_dist = torch.where(torch.isfinite(min_traj_dist), min_traj_dist, torch.zeros_like(min_traj_dist))

    contact_scale = max(float(contact_sigma), 1e-6)
    trajectory_proximity = torch.exp(-point_to_traj / contact_scale)
    trajectory_proximity = torch.where(
        torch.isfinite(trajectory_proximity), trajectory_proximity, torch.zeros_like(trajectory_proximity)
    )
    valid_traj_count = valid_traj.sum(dim=-1, keepdim=True).clamp_min(1).to(dtype=trajectory_proximity.dtype)
    trajectory_dwell = trajectory_proximity.sum(dim=-1) / valid_traj_count
    context_valid = (~future_is_pad)[:, 1:]
    if context_valid.shape[1] > 0:
        context_observability = context_valid.to(dtype=current_xyz.dtype).mean(dim=-1, keepdim=True)
    else:
        context_observability = current_xyz.new_zeros((bsize, 1))
    context_observability = context_observability.expand(-1, n_points)

    ee_dist = torch.linalg.norm(current_xyz, dim=-1)
    ee_dist = torch.where(current_is_pad, torch.zeros_like(ee_dist), ee_dist)
    # The trajectory is anchored at the current EE pose, so current point-to-EE
    # distance is the stable reference even if sparse trajectory samples repeat.
    approach_delta = ee_dist - min_traj_dist
    min_traj_dist = torch.where(current_is_pad, torch.zeros_like(min_traj_dist), min_traj_dist)
    approach_delta = torch.where(current_is_pad, torch.zeros_like(approach_delta), approach_delta)
    residual_gap = torch.where(current_is_pad, torch.zeros_like(residual_gap), residual_gap)

    held_score_for_sweep = torch.exp(-held_residual / max(float(held_sigma), 1e-6))
    motion_score_for_sweep = torch.sigmoid(
        (residual_gap - float(motion_relative_margin)) / max(float(motion_relative_tau), 1e-6)
    )
    rigid_tool_score_for_sweep = (held_score_for_sweep * motion_score_for_sweep).clamp(0.0, 1.0)
    if bool(tool_sweep_use_full_trajectory):
        sweep_poses = trajectory_poses
        sweep_is_pad = trajectory_is_pad
        sweep_offsets = trajectory_offsets
    else:
        sweep_poses = future_poses
        sweep_is_pad = future_is_pad
        sweep_offsets = None
    if bool(tool_interaction_enable):
        (
            tool_sweep_dist,
            tool_current_dist,
            tool_approach_delta,
            tool_distance_span,
            tool_sweep_dwell,
            tool_candidate_count,
            tool_candidate_scores,
        ) = _compute_tool_sweep_distance(
            current_xyz,
            rigid_tool_score_for_sweep,
            current_is_pad,
            sweep_poses,
            sweep_is_pad,
            sweep_offsets=sweep_offsets,
            candidate_max_points=tool_candidate_max_points,
            candidate_min_score=tool_candidate_min_score,
            candidate_bridge_min_score=tool_candidate_bridge_min_score,
            candidate_distance_boost=tool_candidate_distance_boost,
            candidate_distance_scale=tool_candidate_distance_scale,
            candidate_seed_radius=tool_candidate_seed_radius,
            candidate_seed_min_score=tool_candidate_seed_min_score,
            candidate_component_radius=tool_candidate_component_radius,
            candidate_component_hops=tool_candidate_component_hops,
            candidate_preselect_multiplier=tool_candidate_preselect_multiplier,
            candidate_max_radius=tool_candidate_max_radius,
            max_poses=tool_sweep_max_poses,
            max_frame_offset=tool_sweep_max_frame_offset,
            max_translation=tool_sweep_max_translation,
            dwell_sigma=tool_sweep_dwell_sigma,
            nn_chunk_size=nn_chunk_size,
        )
    else:
        tool_sweep_dist = current_xyz.new_full(current_xyz.shape[:2], float("inf"))
        tool_sweep_dist = tool_sweep_dist.masked_fill(current_is_pad, 0.0)
        tool_current_dist = tool_sweep_dist.clone()
        tool_approach_delta = current_xyz.new_zeros(current_xyz.shape[:2])
        tool_distance_span = current_xyz.new_zeros(current_xyz.shape[:2])
        tool_sweep_dwell = current_xyz.new_zeros(current_xyz.shape[:2])
        tool_candidate_count = torch.zeros(bsize, dtype=torch.long, device=current_xyz.device)
        tool_candidate_scores = current_xyz.new_zeros((bsize, 1))

    priors = torch.stack(
        [
            ee_dist,
            min_traj_dist,
            approach_delta,
            held_residual,
            static_residual,
            residual_gap,
            trajectory_dwell,
            context_observability,
        ],
        dim=-1,
    )

    return {
        "priors": priors,
        "point_is_pad": current_is_pad,
        "ee_dist": ee_dist,
        "min_traj_dist": min_traj_dist,
        "approach_delta": approach_delta,
        "held_residual": held_residual,
        "static_residual": static_residual,
        "residual_gap": residual_gap,
        "trajectory_dwell": trajectory_dwell,
        "context_observability": context_observability,
        "motion_weights": motion_weights,
        "motion_baseline": motion_baseline,
        # Extra dictionary-only priors.  They are intentionally not appended to
        # the fixed 8-D `priors` tensor, preserving cache/model tensor shapes.
        "tool_sweep_dist": tool_sweep_dist,
        "tool_current_dist": tool_current_dist,
        "tool_approach_delta": tool_approach_delta,
        "tool_distance_span": tool_distance_span,
        "tool_sweep_dwell": tool_sweep_dwell,
        "tool_candidate_count": tool_candidate_count,
        "tool_candidate_scores": tool_candidate_scores,
    }


def generate_pseudo_labels_from_priors(
    priors_or_dict: Tensor | dict[str, Tensor],
    *,
    config: PseudoLabelConfig | None = None,
) -> dict[str, Tensor]:
    config = config or PseudoLabelConfig()
    if torch.is_tensor(priors_or_dict):
        priors = priors_or_dict
        if priors.shape[-1] != MOTION_PRIOR_DIM:
            raise ValueError(f"Expected priors last dim={MOTION_PRIOR_DIM}, got {priors.shape[-1]}.")
        prior_dict = {
            "priors": priors,
            "ee_dist": priors[..., 0],
            "min_traj_dist": priors[..., 1],
            "approach_delta": priors[..., 2],
            "held_residual": priors[..., 3],
            "static_residual": priors[..., 4],
            "residual_gap": priors[..., 5],
            "trajectory_dwell": priors[..., 6],
            "context_observability": priors[..., 7],
        }
    else:
        prior_dict = dict(priors_or_dict)

    ee_dist = prior_dict["ee_dist"]
    min_traj_dist = prior_dict["min_traj_dist"]
    approach_delta = prior_dict["approach_delta"]
    held_residual = prior_dict["held_residual"]
    static_residual = prior_dict["static_residual"]
    residual_gap = prior_dict["residual_gap"]
    trajectory_dwell = prior_dict.get("trajectory_dwell", torch.zeros_like(ee_dist))
    context_observability = prior_dict.get("context_observability", torch.ones_like(ee_dist))

    held_score = torch.exp(-held_residual / config.held_sigma)
    static_score = torch.exp(-static_residual / config.static_sigma)
    motion_score = torch.sigmoid(
        (residual_gap - config.motion_relative_margin) / max(config.motion_relative_tau, 1e-6)
    )
    trajectory_near = torch.exp(-min_traj_dist / config.trajectory_sigma)
    approach_score = torch.sigmoid((approach_delta - config.approach_margin) / config.approach_tau)

    # Keep the legacy EEF-local evidence for cache/API compatibility, but add a
    # trajectory-independent rigid carrier score so the far end of a long tool
    # is not suppressed merely because it is distant from the robot hand.
    legacy_tool_score = held_score * motion_score * trajectory_near
    rigid_tool_score = (held_score * motion_score).clamp(0.0, 1.0)
    approached_score = approach_score * trajectory_near
    contact_score = torch.sigmoid(
        (float(config.contact_radius) - min_traj_dist)
        / max(float(config.contact_temperature), 1e-6)
    )

    has_conditioned_tool_sweep = "tool_sweep_dist" in prior_dict
    tool_sweep_dist = prior_dict.get("tool_sweep_dist", min_traj_dist)
    tool_sweep_dist = tool_sweep_dist.to(device=min_traj_dist.device, dtype=min_traj_dist.dtype)
    tool_current_dist = prior_dict.get("tool_current_dist", ee_dist)
    tool_current_dist = tool_current_dist.to(device=min_traj_dist.device, dtype=min_traj_dist.dtype)
    tool_approach_delta = prior_dict.get(
        "tool_approach_delta", (tool_current_dist - tool_sweep_dist).clamp_min(0.0)
    ).to(device=min_traj_dist.device, dtype=min_traj_dist.dtype)
    tool_distance_span = prior_dict.get("tool_distance_span", tool_approach_delta)
    tool_distance_span = tool_distance_span.to(device=min_traj_dist.device, dtype=min_traj_dist.dtype)
    tool_sweep_dwell = prior_dict.get("tool_sweep_dwell", torch.zeros_like(tool_sweep_dist))
    tool_sweep_dwell = tool_sweep_dwell.to(device=min_traj_dist.device, dtype=min_traj_dist.dtype)

    point_is_pad_for_gate = prior_dict.get("point_is_pad")
    if point_is_pad_for_gate is None:
        point_is_pad_for_gate = torch.zeros_like(ee_dist, dtype=torch.bool)
    else:
        point_is_pad_for_gate = point_is_pad_for_gate.to(device=ee_dist.device, dtype=torch.bool)

    # Absolute proximity alone is insufficient when the candidate cloud is
    # contaminated: a flattened sweep can pass near most of the scene.  Keep only
    # the nearest configurable fraction, with a soft rank boundary.
    target_rank_eligible = (
        (static_score >= float(config.tool_target_static_min_score))
        & (tool_current_dist >= float(config.tool_target_tool_exclusion_radius))
        & (rigid_tool_score <= float(config.tool_target_rigid_exclusion_score))
        & torch.isfinite(tool_sweep_dist)
    )
    tool_rank_gate, tool_rank_radius = _fractional_distance_gate(
        tool_sweep_dist,
        point_is_pad_for_gate,
        config.tool_target_max_fraction,
        config.tool_target_rank_temperature,
        eligible=target_rank_eligible,
    )
    tool_sweep_contact_score = torch.sigmoid(
        (float(config.tool_sweep_contact_radius) - tool_sweep_dist)
        / max(float(config.tool_sweep_contact_temperature), 1e-6)
    )
    tool_spatial_gate = (tool_sweep_contact_score * tool_rank_gate).clamp(0.0, 1.0)

    # Use both current-to-closest change and the bounded sweep span.  The span
    # keeps a flower pot active at the actual pouring frame, while the spatial
    # gate prevents that temporal contrast from activating distant background.
    tool_temporal_delta = torch.maximum(tool_approach_delta, tool_distance_span)
    tool_temporal_approach_score = torch.sigmoid(
        (tool_temporal_delta - float(config.tool_approach_margin))
        / max(float(config.tool_approach_temperature), 1e-6)
    )
    tool_temporal_score = (
        1.0 - (1.0 - tool_temporal_approach_score) * (1.0 - tool_sweep_dwell.clamp(0.0, 1.0))
    ).clamp(0.0, 1.0)

    nonstatic_score = torch.sigmoid(
        (static_residual - float(config.interaction_motion_threshold))
        / max(float(config.interaction_motion_temperature), 1e-6)
    )
    context_score = context_observability.clamp(0.0, 1.0)
    tool_candidate_count = prior_dict.get("tool_candidate_count")
    tool_candidate_scores = prior_dict.get("tool_candidate_scores")
    if tool_candidate_count is None or tool_candidate_scores is None:
        candidate_quality_scalar = torch.zeros(ee_dist.shape[0], device=ee_dist.device, dtype=ee_dist.dtype)
    else:
        candidate_count = tool_candidate_count.to(device=ee_dist.device, dtype=ee_dist.dtype)
        candidate_scores = tool_candidate_scores.to(device=ee_dist.device, dtype=ee_dist.dtype)
        count_quality = torch.sigmoid(
            (candidate_count - float(config.tool_target_min_candidate_count))
            / max(float(config.tool_target_candidate_count_temperature), 1e-6)
        )
        peak_score = candidate_scores.amax(dim=-1)
        score_quality = torch.sigmoid(
            (peak_score - float(config.tool_target_candidate_score_threshold))
            / max(float(config.tool_target_candidate_score_temperature), 1e-6)
        )
        candidate_quality_scalar = (count_quality * score_quality).clamp(0.0, 1.0)
    candidate_quality = candidate_quality_scalar
    while candidate_quality.ndim < ee_dist.ndim:
        candidate_quality = candidate_quality.unsqueeze(-1)
    candidate_quality = candidate_quality.expand_as(ee_dist)

    connected_tool_support = torch.exp(
        -tool_current_dist / max(float(config.tool_candidate_support_sigma), 1e-6)
    )
    connected_rigid_tool_score = (
        rigid_tool_score * connected_tool_support * candidate_quality
    ).clamp(0.0, 1.0)

    # Static targets are allowed, but only inside the localized, rank-limited
    # conditioned sweep.  This retains the flower pot while rejecting far tables,
    # walls, drawers and repeated planar background.
    static_tool_target_score = (
        static_score
        * tool_spatial_gate
        * tool_temporal_score
        * candidate_quality
    ).clamp(0.0, 1.0)
    moved_interaction_score = (
        nonstatic_score
        * tool_spatial_gate
        * context_score
        * candidate_quality
    ).clamp(0.0, 1.0)
    if not has_conditioned_tool_sweep:
        static_tool_target_score = torch.zeros_like(static_tool_target_score)
        moved_interaction_score = torch.zeros_like(moved_interaction_score)
        connected_rigid_tool_score = legacy_tool_score
    interaction_score = (
        1.0 - (1.0 - static_tool_target_score) * (1.0 - moved_interaction_score)
    ).clamp(0.0, 1.0)
    conditioned_approach_score = (
        1.0 - (1.0 - approached_score) * (1.0 - static_tool_target_score)
    ).clamp(0.0, 1.0)

    if bool(config.tool_interaction_enable):
        foreground_evidence = torch.stack(
            [connected_rigid_tool_score, conditioned_approach_score, contact_score, moved_interaction_score],
            dim=-1,
        ).clamp(0.0, 1.0)
    else:
        foreground_evidence = torch.stack(
            [legacy_tool_score, approached_score, contact_score], dim=-1
        ).clamp(0.0, 1.0)
    foreground_score = (1.0 - (1.0 - foreground_evidence).prod(dim=-1)).clamp(0.0, 1.0)
    class_scores = torch.stack([1.0 - foreground_score, foreground_score], dim=-1)

    # Preserve the established three-channel tensor for checkpoint compatibility.
    evidence_scores = torch.stack(
        [legacy_tool_score, conditioned_approach_score, contact_score], dim=-1
    ).clamp(0.0, 1.0)

    far_from_trajectory = 1.0 - torch.exp(
        -min_traj_dist / max(config.background_trajectory_sigma, 1e-6)
    )
    far_from_tool_sweep = 1.0 - torch.exp(
        -tool_sweep_dist / max(config.background_tool_sweep_sigma, 1e-6)
    )
    background_confidence = (
        (1.0 - foreground_score)
        * far_from_trajectory
        * far_from_tool_sweep
        * (0.25 + 0.75 * context_observability.clamp(0.0, 1.0))
    )
    weights = (
        foreground_score
        + float(config.soft_background_weight) * background_confidence
    ).detach().clamp(0.0, 1.0)
    # Kept only for cache/API compatibility.  PointSeg supervision uses
    # class_scores and weights exclusively; there are no hard pseudo labels.
    labels = torch.full_like(foreground_score, config.ignore_index, dtype=torch.long)
    point_is_pad = prior_dict.get("point_is_pad")
    if point_is_pad is not None:
        point_is_pad = point_is_pad.to(device=labels.device, dtype=torch.bool)
        labels = torch.where(point_is_pad, torch.full_like(labels, config.ignore_index), labels)
        weights = torch.where(point_is_pad, torch.zeros_like(weights), weights)
        class_scores = torch.where(point_is_pad[..., None], torch.zeros_like(class_scores), class_scores)
        evidence_scores = torch.where(point_is_pad[..., None], torch.zeros_like(evidence_scores), evidence_scores)
        foreground_score = torch.where(point_is_pad, torch.zeros_like(foreground_score), foreground_score)
        rigid_tool_score = torch.where(point_is_pad, torch.zeros_like(rigid_tool_score), rigid_tool_score)
        connected_rigid_tool_score = torch.where(
            point_is_pad, torch.zeros_like(connected_rigid_tool_score), connected_rigid_tool_score
        )
        target_rank_eligible = torch.where(
            point_is_pad, torch.zeros_like(target_rank_eligible), target_rank_eligible
        )
        tool_rank_gate = torch.where(point_is_pad, torch.zeros_like(tool_rank_gate), tool_rank_gate)
        tool_spatial_gate = torch.where(point_is_pad, torch.zeros_like(tool_spatial_gate), tool_spatial_gate)
        candidate_quality = torch.where(point_is_pad, torch.zeros_like(candidate_quality), candidate_quality)
        tool_sweep_contact_score = torch.where(
            point_is_pad, torch.zeros_like(tool_sweep_contact_score), tool_sweep_contact_score
        )
        tool_temporal_approach_score = torch.where(
            point_is_pad, torch.zeros_like(tool_temporal_approach_score), tool_temporal_approach_score
        )
        tool_temporal_score = torch.where(point_is_pad, torch.zeros_like(tool_temporal_score), tool_temporal_score)
        static_tool_target_score = torch.where(
            point_is_pad, torch.zeros_like(static_tool_target_score), static_tool_target_score
        )
        moved_interaction_score = torch.where(
            point_is_pad, torch.zeros_like(moved_interaction_score), moved_interaction_score
        )
        conditioned_approach_score = torch.where(
            point_is_pad, torch.zeros_like(conditioned_approach_score), conditioned_approach_score
        )
        nonstatic_score = torch.where(point_is_pad, torch.zeros_like(nonstatic_score), nonstatic_score)
        interaction_score = torch.where(point_is_pad, torch.zeros_like(interaction_score), interaction_score)

    return {
        **prior_dict,
        "labels": labels,
        "weights": weights,
        "class_scores": class_scores.detach(),
        # Legacy field name retained for cache compatibility.  Its channels are
        # now [tool_comotion, EEF-or-conditioned-tool approach, near_contact].
        "role_scores": evidence_scores.detach(),
        "foreground_score": foreground_score.detach(),
        "background_confidence": background_confidence.detach(),
        "held_score": held_score.detach(),
        "static_score": static_score.detach(),
        "approach_score": approach_score.detach(),
        "conditioned_approach_score": conditioned_approach_score.detach(),
        "contact_score": contact_score.detach(),
        "rigid_tool_score": rigid_tool_score.detach(),
        "connected_rigid_tool_score": connected_rigid_tool_score.detach(),
        "tool_sweep_dist": tool_sweep_dist.detach(),
        "tool_current_dist": tool_current_dist.detach(),
        "tool_approach_delta": tool_approach_delta.detach(),
        "tool_distance_span": tool_distance_span.detach(),
        "tool_sweep_dwell": tool_sweep_dwell.detach(),
        "tool_candidate_quality": candidate_quality.detach(),
        "tool_rank_radius": tool_rank_radius.detach(),
        "tool_target_rank_eligible": target_rank_eligible.detach(),
        "tool_rank_gate": tool_rank_gate.detach(),
        "tool_spatial_gate": tool_spatial_gate.detach(),
        "tool_sweep_contact_score": tool_sweep_contact_score.detach(),
        "tool_temporal_approach_score": tool_temporal_approach_score.detach(),
        "tool_temporal_score": tool_temporal_score.detach(),
        "static_tool_target_score": static_tool_target_score.detach(),
        "moved_interaction_score": moved_interaction_score.detach(),
        "nonstatic_score": nonstatic_score.detach(),
        "interaction_score": interaction_score.detach(),
    }


def generate_pseudo_labels(
    current_pc: Tensor,
    future_pc: Tensor,
    future_poses: Tensor,
    future_is_pad: Tensor | None = None,
    *,
    current_is_pad: Tensor | None = None,
    future_point_is_pad: Tensor | None = None,
    trajectory_poses: Tensor | None = None,
    trajectory_is_pad: Tensor | None = None,
    trajectory_offsets: Tensor | None = None,
    config: PseudoLabelConfig | None = None,
) -> dict[str, Tensor]:
    config = config or PseudoLabelConfig()
    prior_dict = compute_motion_priors(
        current_pc,
        future_pc,
        future_poses,
        future_is_pad,
        current_is_pad=current_is_pad,
        future_point_is_pad=future_point_is_pad,
        trajectory_poses=trajectory_poses,
        trajectory_is_pad=trajectory_is_pad,
        trajectory_offsets=trajectory_offsets,
        nn_chunk_size=config.nn_chunk_size,
        motion_gap_eps=config.motion_gap_eps,
        motion_rotation_radius=config.motion_rotation_radius,
        motion_baseline_threshold=config.motion_baseline_threshold,
        motion_baseline_temperature=config.motion_baseline_temperature,
        motion_evidence_topk=config.motion_evidence_topk,
        contact_sigma=config.contact_sigma,
        tool_candidate_min_score=config.tool_candidate_min_score,
        tool_candidate_bridge_min_score=config.tool_candidate_bridge_min_score,
        tool_candidate_max_points=config.tool_candidate_max_points,
        tool_candidate_distance_boost=config.tool_candidate_distance_boost,
        tool_candidate_distance_scale=config.tool_candidate_distance_scale,
        tool_candidate_seed_radius=config.tool_candidate_seed_radius,
        tool_candidate_seed_min_score=config.tool_candidate_seed_min_score,
        tool_candidate_component_radius=config.tool_candidate_component_radius,
        tool_candidate_component_hops=config.tool_candidate_component_hops,
        tool_candidate_preselect_multiplier=config.tool_candidate_preselect_multiplier,
        tool_candidate_max_radius=config.tool_candidate_max_radius,
        tool_sweep_max_poses=config.tool_sweep_max_poses,
        tool_sweep_use_full_trajectory=config.tool_sweep_use_full_trajectory,
        tool_sweep_max_frame_offset=config.tool_sweep_max_frame_offset,
        tool_sweep_max_translation=config.tool_sweep_max_translation,
        tool_sweep_dwell_sigma=config.tool_sweep_dwell_sigma,
        tool_interaction_enable=config.tool_interaction_enable,
        held_sigma=config.held_sigma,
        motion_relative_margin=config.motion_relative_margin,
        motion_relative_tau=config.motion_relative_tau,
    )
    return generate_pseudo_labels_from_priors(prior_dict, config=config)


def force_small_current_clouds_foreground(
    pseudo: dict[str, Tensor],
    current_pc: Tensor,
    configured_current_points: int,
    point_is_pad: Tensor | None = None,
) -> dict[str, Tensor]:
    """Deprecated compatibility shim; soft labels must never depend on point count."""
    del current_pc, configured_current_points
    if point_is_pad is None:
        return pseudo
    out = dict(pseudo)
    out["point_is_pad"] = point_is_pad
    return out


def refine_pseudo_labels_with_teacher(
    pseudo: dict[str, Tensor],
    teacher_logits: Tensor,
    *,
    config: PseudoLabelConfig | None = None,
) -> dict[str, Tensor]:
    config = config or PseudoLabelConfig()
    refined = dict(pseudo)
    probs = teacher_logits.softmax(dim=-1).detach()
    teacher_conf, teacher_labels = probs.max(dim=-1)
    class_scores = pseudo["class_scores"]
    geometry_gate = class_scores.gather(-1, teacher_labels.unsqueeze(-1)).squeeze(-1)
    foreground_accept = (
        (teacher_labels == ROLE_FOREGROUND)
        & (teacher_conf >= config.teacher_confidence)
        & (geometry_gate >= config.teacher_geometry_gate)
    )
    background_accept = (
        (teacher_labels == ROLE_BACKGROUND)
        & (teacher_conf >= config.teacher_confidence)
        & (class_scores[..., ROLE_BACKGROUND] >= config.teacher_background_min_score)
        & (pseudo["foreground_score"] <= config.teacher_background_foreground_max)
    )
    accept = foreground_accept | background_accept

    geometric_foreground = pseudo["class_scores"][..., ROLE_FOREGROUND]
    blend = (float(config.teacher_blend) * teacher_conf).clamp(0.0, 1.0)
    blend = torch.where(accept, blend, torch.zeros_like(blend))
    refined_foreground = (
        (1.0 - blend) * geometric_foreground + blend * probs[..., ROLE_FOREGROUND]
    ).clamp(0.0, 1.0)
    refined["foreground_score"] = refined_foreground
    refined["class_scores"] = torch.stack([1.0 - refined_foreground, refined_foreground], dim=-1)
    refined["weights"] = torch.where(
        accept,
        torch.maximum(pseudo["weights"], blend),
        pseudo["weights"],
    )
    # Teacher refinement remains soft; the legacy label tensor is never changed.
    refined["labels"] = pseudo["labels"]
    refined["teacher_accept_mask"] = accept
    return refined


class SongPointSegNet(nn.Module):
    """LitePT-based role segmentation for Song manipulation point clouds."""

    def __init__(
        self,
        *,
        backbone_type: str = "litept",
        hidden_dim: int = 128,
        grid_size: float = 0.01,
        in_channels: int = 6,
    ):
        super().__init__()
        self.backbone_type = backbone_type
        self.grid_size = float(grid_size)
        self.in_channels = int(in_channels)
        if self.in_channels != 6:
            raise ValueError("SongPointSegNet now uses only raw XYZRGB point features, so in_channels must be 6.")

        if backbone_type == "litept":
            try:
                from lerobot.policies.smolvla.litept.model import LitePT
            except Exception as exc:  # pragma: no cover - exercised only when optional deps are missing
                raise ImportError(
                    "SongPointSegNet(backbone_type='litept') requires LitePT optional dependencies "
                    "(flash_attn, spconv, torch_scatter, pointrope). Use backbone_type='mlp' for CPU smoke tests."
                ) from exc
            self.backbone = LitePT(
                in_channels=self.in_channels,
                enc_mode=False,
                dec_depths=(1, 1, 1, 1),
                dec_conv=(True, True, False, False),
                dec_attn=(False, False, False, False),
            )
            self.head = nn.Sequential(
                nn.Linear(infer_litept_output_channels(self.backbone), hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, len(ROLE_NAMES)),
            )
        elif backbone_type == "mlp":
            self.backbone = nn.Sequential(
                nn.Linear(self.in_channels, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
            )
            self.head = nn.Linear(hidden_dim, len(ROLE_NAMES))
        else:
            raise ValueError(f"Unknown backbone_type={backbone_type!r}. Expected 'litept' or 'mlp'.")

    def train(self, mode: bool = True):
        """Use serialization-order shuffling as training augmentation only.

        LitePT defaults to randomly permuting its serialization orders on every
        forward, including evaluation.  Keeping that behavior during training is
        useful augmentation, while disabling it in eval makes PointSeg predictions
        and thresholded foreground sets deterministic.
        """
        super().train(mode)
        if self.backbone_type == "litept" and hasattr(self.backbone, "shuffle_orders"):
            self.backbone.shuffle_orders = bool(mode)
        return self

    def _make_features(self, current_pc: Tensor) -> Tensor:
        xyz = current_pc[..., :3].to(dtype=torch.float32)
        rgb = current_pc[..., 3:6].to(dtype=torch.float32) / 255.0
        return torch.cat([xyz, rgb], dim=-1)

    def _forward_litept(self, current_pc: Tensor, features: Tensor, point_is_pad: Tensor | None = None) -> Tensor:
        bsize, n_points = current_pc.shape[:2]
        if point_is_pad is None:
            point_is_pad = torch.zeros(bsize, n_points, dtype=torch.bool, device=current_pc.device)
        valid = ~point_is_pad
        counts = valid.sum(dim=1).to(dtype=torch.long)
        valid_batch = counts > 0
        if not bool(valid_batch.any().item()):
            return current_pc.new_zeros(bsize, n_points, len(ROLE_NAMES))

        valid_local = torch.nonzero(valid[valid_batch], as_tuple=False)
        flat_valid = valid[valid_batch].reshape(-1)
        coord = current_pc[valid_batch, :, :3].reshape(-1, 3)[flat_valid].contiguous().to(dtype=torch.float32)
        feat = features[valid_batch].reshape(-1, features.shape[-1])[flat_valid].contiguous().to(dtype=torch.float32)
        counts = counts[valid_batch]
        offset = torch.cumsum(counts, dim=0)
        grid_coord = build_litept_grid_coord(coord, valid_local[:, 0], self.grid_size)
        point = self.backbone(
            {
                "coord": coord,
                "feat": feat,
                "offset": offset,
                "grid_coord": grid_coord,
                "grid_size": self.grid_size,
            }
        )
        valid_logits = self.head(point.feat)
        logits = valid_logits.new_zeros(bsize, n_points, valid_logits.shape[-1])
        valid_batch_indices = torch.nonzero(valid_batch, as_tuple=False).flatten()
        logits[valid_batch_indices[valid_local[:, 0]], valid_local[:, 1]] = valid_logits
        return logits

    def forward(
        self,
        current_pc: Tensor,
        future_pc: Tensor | None = None,
        future_poses: Tensor | None = None,
        future_is_pad: Tensor | None = None,
        *,
        priors: Tensor | None = None,
        point_is_pad: Tensor | None = None,
        future_point_is_pad: Tensor | None = None,
    ) -> dict[str, Tensor]:
        if priors is None:
            if future_pc is not None and future_poses is not None:
                prior_dict = compute_motion_priors(
                    current_pc,
                    future_pc,
                    future_poses,
                    future_is_pad,
                    current_is_pad=point_is_pad,
                    future_point_is_pad=future_point_is_pad,
                )
            else:
                prior_dict = {}
                if point_is_pad is not None:
                    prior_dict["point_is_pad"] = point_is_pad
        else:
            prior_dict = {"priors": priors}
            if point_is_pad is not None:
                prior_dict["point_is_pad"] = point_is_pad

        features = self._make_features(current_pc)
        if self.backbone_type == "litept":
            role_logits = self._forward_litept(current_pc, features, point_is_pad)
        else:
            role_logits = self.head(self.backbone(features))
            if point_is_pad is not None:
                role_logits = role_logits.masked_fill(point_is_pad[..., None].to(device=role_logits.device), 0.0)
        role_probs = role_logits.softmax(dim=-1)
        return {
            **prior_dict,
            "role_logits": role_logits,
            "role_probs": role_probs,
            "operation_prob": role_probs[..., ROLE_FOREGROUND],
        }


def _voxel_smoothness_loss(
    logits: Tensor, xyz: Tensor, voxel_size: float, point_is_pad: Tensor | None = None
) -> Tensor:
    probs = logits.softmax(dim=-1)
    total = logits.sum() * 0.0
    used = 0
    for bidx in range(logits.shape[0]):
        valid = ~point_is_pad[bidx] if point_is_pad is not None else torch.ones(
            xyz.shape[1], dtype=torch.bool, device=xyz.device
        )
        if int(valid.sum().item()) <= 1:
            continue
        xyz_b = xyz[bidx, valid]
        probs_b = probs[bidx, valid]
        voxel = torch.floor(xyz_b / voxel_size).to(dtype=torch.long)
        _, inverse = torch.unique(voxel, dim=0, return_inverse=True)
        groups = int(inverse.max().item()) + 1 if inverse.numel() > 0 else 0
        if groups <= 1:
            continue
        sums = torch.zeros(groups, probs.shape[-1], device=probs.device, dtype=probs.dtype)
        counts = torch.zeros(groups, 1, device=probs.device, dtype=probs.dtype)
        sums.scatter_add_(0, inverse[:, None].expand(-1, probs.shape[-1]), probs_b)
        counts.scatter_add_(0, inverse[:, None], torch.ones_like(counts[inverse]))
        means = sums / counts.clamp_min(1.0)
        total = total + (probs_b - means[inverse]).square().mean()
        used += 1
    if used == 0:
        return total
    return total / used


def _soft_foreground_bce(
    logits: Tensor,
    pseudo: dict[str, Tensor],
    config: SongPointSegLossConfig,
    point_is_pad: Tensor | None = None,
) -> Tensor:
    class_scores = pseudo["class_scores"].to(device=logits.device, dtype=logits.dtype)
    target = class_scores[..., ROLE_FOREGROUND].clamp(0.0, 1.0)
    weights = pseudo["weights"].to(device=logits.device, dtype=logits.dtype).clamp_min(0.0)
    if point_is_pad is not None:
        weights = weights.masked_fill(point_is_pad.to(device=logits.device, dtype=torch.bool), 0.0)

    operation_logits = logits[..., ROLE_FOREGROUND] - logits[..., ROLE_BACKGROUND]
    losses = functional.binary_cross_entropy_with_logits(operation_logits, target, reduction="none")
    background_weight, foreground_weight = config.class_weights
    soft_class_weight = (1.0 - target) * float(background_weight) + target * float(foreground_weight)
    effective_weight = weights * soft_class_weight
    return (losses * effective_weight).sum() / effective_weight.sum().clamp_min(1e-6)


class SongPointSegLoss(nn.Module):
    def __init__(self, config: SongPointSegLossConfig | None = None):
        super().__init__()
        self.config = config or SongPointSegLossConfig()

    def forward(self, outputs: dict[str, Tensor], pseudo: dict[str, Tensor], current_pc: Tensor) -> tuple[Tensor, dict[str, Tensor]]:
        logits = outputs["role_logits"]
        weights = pseudo["weights"].to(device=logits.device, dtype=logits.dtype)
        soft_target = pseudo["class_scores"][..., ROLE_FOREGROUND].to(device=logits.device, dtype=logits.dtype)
        point_is_pad = pseudo.get("point_is_pad", outputs.get("point_is_pad"))
        if point_is_pad is not None:
            point_is_pad = point_is_pad.to(device=logits.device, dtype=torch.bool)
            weights = torch.where(point_is_pad, torch.zeros_like(weights), weights)
        valid = weights > 0
        valid_f = valid.to(dtype=torch.float32)

        soft_bce = _soft_foreground_bce(logits, pseudo, self.config, point_is_pad)
        smoothness = _voxel_smoothness_loss(
            logits, current_pc[..., :3], self.config.smooth_voxel_size, point_is_pad
        )
        if point_is_pad is None:
            geometric_valid = torch.ones_like(soft_target, dtype=torch.bool)
        else:
            geometric_valid = ~point_is_pad
        geometric_valid_f = geometric_valid.to(dtype=logits.dtype)
        denom = geometric_valid_f.sum(dim=-1).clamp_min(1.0)
        pred_mean_by_frame = (
            outputs["operation_prob"] * geometric_valid_f
        ).sum(dim=-1) / denom
        target_mean_by_frame = (
            soft_target * geometric_valid_f
        ).sum(dim=-1) / denom
        allowed_mass = torch.minimum(
            target_mean_by_frame + float(self.config.foreground_mass_margin),
            torch.full_like(target_mean_by_frame, float(self.config.foreground_mass_max)),
        )
        foreground_mass = torch.relu(pred_mean_by_frame - allowed_mass).square().mean()
        loss = (
            self.config.soft_bce_weight * soft_bce
            + self.config.smoothness_weight * smoothness
            + self.config.foreground_mass_weight * foreground_mass
        )
        zero = logits.sum().detach() * 0.0
        metrics = {
            "loss": loss.detach(),
            "loss_soft_bce": soft_bce.detach(),
            # Legacy names are kept so existing log dashboards do not fail.
            "loss_ce": zero,
            "loss_foreground_bce": soft_bce.detach(),
            "loss_smoothness": smoothness.detach(),
            "loss_foreground_mass": foreground_mass.detach(),
            "pred_foreground_mean_by_frame_max": pred_mean_by_frame.max().detach(),
            "target_foreground_mean_by_frame_max": target_mean_by_frame.max().detach(),
            "loss_motion": zero,
            "pseudo_valid_ratio": valid_f.mean().detach(),
            "pseudo_valid_foreground_ratio": (
                ((soft_target >= 0.5) & valid).to(dtype=torch.float32).sum()
                / valid_f.sum().clamp_min(1.0)
            ).detach(),
            "pseudo_background_ratio": (
                ((soft_target < 0.5) & valid).to(dtype=torch.float32).sum() / valid_f.sum().clamp_min(1.0)
            ).detach(),
            "pseudo_foreground_ratio": (((soft_target >= 0.5) & valid).to(dtype=torch.float32).sum()
                                           / valid_f.sum().clamp_min(1.0)).detach(),
            "pseudo_soft_foreground_mean": (
                (soft_target * valid.to(dtype=soft_target.dtype)).sum() / valid_f.sum().clamp_min(1.0)
            ).detach(),
            "pred_foreground_ratio": (
                ((outputs["operation_prob"] >= 0.5) & valid)
                .to(dtype=torch.float32)
                .sum()
                / valid_f.sum().clamp_min(1.0)
            ).detach(),
            "pred_operation_prob": (
                (outputs["operation_prob"] * valid_f.to(dtype=outputs["operation_prob"].dtype)).sum()
                / valid_f.to(dtype=outputs["operation_prob"].dtype).sum().clamp_min(1.0)
            ).detach(),
        }
        if "teacher_accept_mask" in pseudo:
            metrics["teacher_accept_ratio"] = pseudo["teacher_accept_mask"].to(dtype=torch.float32).mean().detach()
        return loss, metrics


class EMATeacher:
    def __init__(self, model: nn.Module, decay: float = 0.995):
        self.decay = float(decay)
        self.model = copy.deepcopy(model).eval()
        for param in self.model.parameters():
            param.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        teacher_state = self.model.state_dict()
        student_state = model.state_dict()
        for key, teacher_value in teacher_state.items():
            student_value = student_state[key].detach()
            if torch.is_floating_point(teacher_value):
                teacher_value.mul_(self.decay).add_(student_value, alpha=1.0 - self.decay)
            else:
                teacher_value.copy_(student_value)


def write_role_ply(path: str | Path, points_xyzrgb: np.ndarray, labels: np.ndarray, operation_prob: np.ndarray | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    labels = np.asarray(labels, dtype=np.int64)
    colors = ROLE_COLORS[np.clip(labels, 0, len(ROLE_COLORS) - 1)]
    if operation_prob is not None:
        operation_prob = np.asarray(operation_prob, dtype=np.float32)
        colors = np.clip(colors.astype(np.float32) * (0.35 + 0.65 * operation_prob[:, None]), 0, 255).astype(
            np.uint8
        )
    xyz = np.asarray(points_xyzrgb[:, :3], dtype=np.float32)
    with open(path, "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {xyz.shape[0]}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        for point, color in zip(xyz, colors, strict=False):
            f.write(f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f} {int(color[0])} {int(color[1])} {int(color[2])}\n")


def save_pointseg_npz(
    path: str | Path,
    current_pc: Tensor,
    outputs: dict[str, Tensor],
    pseudo: dict[str, Tensor] | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "point_cloud": current_pc.detach().cpu().numpy(),
        "role_logits": outputs["role_logits"].detach().cpu().numpy(),
        "operation_prob": outputs["operation_prob"].detach().cpu().numpy(),
    }
    if pseudo is not None:
        data["pseudo_labels"] = pseudo["labels"].detach().cpu().numpy()
        data["pseudo_weights"] = pseudo["weights"].detach().cpu().numpy()
        if "role_scores" in pseudo:
            data["pseudo_role_scores_gripper_condition_target"] = pseudo["role_scores"].detach().cpu().numpy()
        if "foreground_score" in pseudo:
            data["pseudo_foreground_score"] = pseudo["foreground_score"].detach().cpu().numpy()
        for diagnostic_key in (
            "rigid_tool_score",
            "connected_rigid_tool_score",
            "conditioned_approach_score",
            "static_tool_target_score",
            "moved_interaction_score",
            "interaction_score",
            "tool_sweep_dist",
            "tool_current_dist",
            "tool_approach_delta",
            "tool_distance_span",
            "tool_sweep_dwell",
            "tool_candidate_quality",
            "tool_target_rank_eligible",
            "tool_rank_radius",
            "tool_rank_gate",
            "tool_spatial_gate",
            "tool_sweep_contact_score",
            "tool_temporal_approach_score",
            "tool_temporal_score",
        ):
            if diagnostic_key in pseudo:
                data[f"pseudo_{diagnostic_key}"] = pseudo[diagnostic_key].detach().cpu().numpy()
    if metadata is not None:
        for key, value in metadata.items():
            data[f"meta_{key}"] = np.asarray(value)
    np.savez_compressed(path, **data)


def save_pointseg_config(path: str | Path, args: Any, pseudo_cfg: PseudoLabelConfig, loss_cfg: SongPointSegLossConfig) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    def jsonable(value: Any) -> Any:
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, tuple | list):
            return [jsonable(item) for item in value]
        if isinstance(value, dict):
            return {str(key): jsonable(item) for key, item in value.items()}
        return value

    payload = {
        "args": jsonable(vars(args) if hasattr(args, "__dict__") else args),
        "pseudo_label_config": asdict(pseudo_cfg),
        "loss_config": asdict(loss_cfg),
        "role_names": ROLE_NAMES,
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def move_batch_to_device(batch: dict[str, Any], device: torch.device | str) -> dict[str, Any]:
    moved = {}
    for key, value in batch.items():
        moved[key] = value.to(device, non_blocking=True) if torch.is_tensor(value) else value
    return moved


def parse_future_offsets(value: str) -> tuple[int, ...]:
    offsets = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not offsets:
        raise ValueError("At least one future offset is required.")
    if any(offset <= 0 for offset in offsets):
        raise ValueError("Future offsets must be positive integers.")
    return offsets


def pretty_metrics(metrics: dict[str, Tensor | float], step: int) -> str:
    values = []
    for key, value in metrics.items():
        scalar = float(value.item()) if torch.is_tensor(value) else float(value)
        values.append(f"{key}={scalar:.4f}")
    return f"step={step} " + " ".join(values)
