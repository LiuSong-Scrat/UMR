#!/usr/bin/env python

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("HF_DATASETS_CACHE", "/tmp/lerobot_hf_datasets_cache")
os.environ.setdefault("SONG_POINTSEG_REQUIRE_POINTOPS", "1")

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]
RL_BENCHMARK_ROOT = REPO_ROOT / "benchmarks" / "RLBench"
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.policies.smolvla.song_pointseg import (
    DEFAULT_FUTURE_OFFSETS,
    POINTSEG_CACHE_LABEL_FIELDS,
    POINTSEG_CACHE_VERSION,
    ROLE_NAMES,
    PseudoLabelConfig,
    SongTemporalPointCloudDataset,
    SongPointSegCachedDataset,
    generate_pseudo_labels,
    move_batch_to_device,
    open_episode_point_clouds,
    parse_future_offsets,
    song_pointseg_collate,
)
from lerobot.utils.random_utils import set_seed

DEFAULT_DATASET_ROOT = Path(
    os.environ.get(
        "RLBENCH_POINTSEG_DATASET",
        str(RL_BENCHMARK_ROOT / "datasets" / "rlbench_water_lerobot"),
    )
)
DEFAULT_CACHE_DIR = Path(
    os.environ.get(
        "RLBENCH_POINTSEG_SAMPLE_CACHE",
        str(RL_BENCHMARK_ROOT / "datasets" / "rlbench_water_lerobot_pointseg_cache"),
    )
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cache Song pointseg sampled points and pseudo labels offline.")
    parser.add_argument("--dataset.repo_id", dest="dataset_repo_id", type=str, default=str(DEFAULT_DATASET_ROOT))
    parser.add_argument("--dataset.root", dest="dataset_root", type=str, default=None)
    parser.add_argument("--point-cloud-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--future-offsets", type=parse_future_offsets, default=DEFAULT_FUTURE_OFFSETS)
    parser.add_argument("--current-points", type=int, default=10000)
    parser.add_argument("--future-points", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--shard-size", type=int, default=256)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--storage-dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument("--nn-chunk-size", type=int, default=1024)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--vis-count", type=int, default=8)
    parser.add_argument(
        "--preview-color-mode",
        choices=["original", "heatmap", "both"],
        default="original",
        help="PLY colors: original RGB, foreground heatmap, or both files.",
    )
    parser.add_argument(
        "--rebuild-visualizations",
        action="store_true",
        help="Rebuild only visualizations from an existing cache; keep all cache shards unchanged.",
    )
    # Pouring-oriented soft pseudo-label parameters. These are saved in the cache manifest.
    parser.add_argument("--motion-rotation-radius", type=float, default=0.18)
    parser.add_argument("--motion-baseline-threshold", type=float, default=0.010)
    parser.add_argument("--motion-baseline-temperature", type=float, default=0.006)
    parser.add_argument("--motion-relative-margin", type=float, default=0.05)
    parser.add_argument("--motion-relative-tau", type=float, default=0.08)
    parser.add_argument("--trajectory-sigma", type=float, default=0.22)
    parser.add_argument("--contact-radius", type=float, default=0.22)
    parser.add_argument("--contact-temperature", type=float, default=0.055)
    parser.add_argument("--approach-margin", type=float, default=0.0)
    parser.add_argument("--approach-tau", type=float, default=0.04)
    parser.add_argument("--background-trajectory-sigma", type=float, default=0.32)
    parser.add_argument(
        "--one-episode-per-task",
        action="store_true",
        help="Cache the first episode of each task only and visualize every selected episode.",
    )
    parser.add_argument(
        "--vis-one-episode-per-task",
        action="store_true",
        help="Keep the full cache, but save previews from one episode per task.",
    )
    parser.add_argument(
        "--episode-indices",
        type=str,
        default=None,
        help="Comma-separated episode indices to cache, for example 0,100,200.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--rank-wait-timeout-sec", type=int, default=0, help="Timeout while rank0 waits for rank done marker files. 0 means wait forever.")
    return parser.parse_args()


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple | list):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    return value


def _make_lerobot_dataset(args: argparse.Namespace) -> LeRobotDataset:
    repo_id = args.dataset_repo_id
    root = Path(args.dataset_root) if args.dataset_root else None
    max_offset = max(args.future_offsets)
    metadata = LeRobotDatasetMetadata(repo_id, root=root)
    fps = int(metadata.fps)
    return LeRobotDataset(
        repo_id,
        root=root,
        delta_timestamps={
            "action": [i / fps for i in range(max_offset + 1)],
            "observation.state": [0.0],
        },
    )


def make_dataset(args: argparse.Namespace) -> SongTemporalPointCloudDataset:
    dataset = _make_lerobot_dataset(args)
    dataset_root = Path(getattr(dataset, "root", args.dataset_repo_id))
    point_cloud_dir = args.point_cloud_dir or dataset_root / "point_clouds"
    return SongTemporalPointCloudDataset(
        dataset,
        point_cloud_dir=point_cloud_dir,
        future_offsets=args.future_offsets,
        current_points=args.current_points,
        future_points=args.future_points,
        seed=args.seed,
        include_base_item=False,
    )


def _prepare_output_dir(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"Cache output dir is not empty: {output_dir}. Pass --overwrite to rebuild it."
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def _make_shard_manifest(total_samples: int, shard_size: int) -> list[dict[str, Any]]:
    if shard_size <= 0:
        raise ValueError("--shard-size must be positive.")
    shards = []
    for start in range(0, total_samples, shard_size):
        length = min(shard_size, total_samples - start)
        shard_index = len(shards)
        shards.append(
            {
                "path": f"shard_{shard_index:06d}",
                "start": start,
                "length": length,
            }
        )
    return shards


def _slice_batch_to_size(batch: dict[str, Any], batch_size: int) -> dict[str, Any]:
    sliced = {}
    for key, value in batch.items():
        if torch.is_tensor(value) and value.ndim > 0 and value.shape[0] >= batch_size:
            sliced[key] = value[:batch_size]
        else:
            sliced[key] = value
    return sliced


def _save_variable_shard(
    output_dir: Path,
    shard: dict[str, Any],
    samples: list[dict[str, np.ndarray | int]],
    *,
    storage_dtype: np.dtype,
) -> dict[str, Any]:
    shard_dir = output_dir / shard["path"]
    shard_dir.mkdir(parents=True, exist_ok=True)
    lengths = [int(sample["point_indices"].shape[0]) for sample in samples]
    offsets = np.zeros(len(samples) + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(lengths, dtype=np.int64)

    np.save(shard_dir / "sample_offsets.npy", offsets)
    np.save(shard_dir / "point_indices.npy", np.concatenate([sample["point_indices"] for sample in samples], axis=0).astype(np.int64, copy=False))
    np.save(shard_dir / "labels.npy", np.concatenate([sample["labels"] for sample in samples], axis=0).astype(np.int16, copy=False))
    np.save(shard_dir / "weights.npy", np.concatenate([sample["weights"] for sample in samples], axis=0).astype(storage_dtype, copy=False))
    np.save(shard_dir / "class_scores.npy", np.concatenate([sample["class_scores"] for sample in samples], axis=0).astype(storage_dtype, copy=False))
    np.save(shard_dir / "role_scores.npy", np.concatenate([sample["role_scores"] for sample in samples], axis=0).astype(storage_dtype, copy=False))
    np.save(shard_dir / "foreground_score.npy", np.concatenate([sample["foreground_score"] for sample in samples], axis=0).astype(storage_dtype, copy=False))
    np.save(shard_dir / "episode_index.npy", np.asarray([sample["episode_index"] for sample in samples], dtype=np.int64))
    np.save(shard_dir / "frame_index.npy", np.asarray([sample["frame_index"] for sample in samples], dtype=np.int64))
    np.save(shard_dir / "dataset_index.npy", np.asarray([sample["dataset_index"] for sample in samples], dtype=np.int64))
    shard["num_points"] = int(offsets[-1])
    return shard


def _sample_from_batch(
    current_pc: torch.Tensor,
    pseudo: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    batch_index: int,
    dataset_index: int,
) -> dict[str, np.ndarray | int]:
    is_pad = batch.get("observation.point_cloud_is_pad")
    valid = ~is_pad[batch_index].bool().detach().cpu() if is_pad is not None else torch.ones(
        current_pc.shape[1], dtype=torch.bool
    )
    point_indices = batch.get("observation.point_cloud_indices")
    if point_indices is None:
        point_indices_np = torch.arange(current_pc.shape[1], dtype=torch.long)[valid].numpy()
    else:
        point_indices_np = point_indices[batch_index].detach().cpu()[valid].to(dtype=torch.long).numpy()
    return {
        "point_indices": point_indices_np,
        "labels": pseudo["labels"][batch_index].detach().cpu()[valid].numpy(),
        "weights": pseudo["weights"][batch_index].detach().cpu()[valid].numpy(),
        "class_scores": pseudo["class_scores"][batch_index].detach().cpu()[valid].numpy(),
        "role_scores": pseudo["role_scores"][batch_index].detach().cpu()[valid].numpy(),
        "foreground_score": pseudo["foreground_score"][batch_index].detach().cpu()[valid].numpy(),
        "episode_index": int(batch["episode_index"][batch_index].detach().cpu().reshape(-1)[0].item()),
        "frame_index": int(batch["frame_index"][batch_index].detach().cpu().reshape(-1)[0].item()),
        "dataset_index": int(dataset_index),
    }


def _episode_preview_targets(
    dataset: SongTemporalPointCloudDataset,
    total_samples: int,
    vis_count: int,
    selected_dataset_indices: list[int] | None = None,
    vis_one_episode_per_task: bool = False,
) -> dict[int, list[tuple[str, int, str, int]]]:
    episodes = dataset.meta.episodes
    if episodes is None:
        raise ValueError("Episode metadata is required to save first/middle/last pseudo-label previews.")

    targets: dict[int, list[tuple[str, int, str, int]]] = {}
    if vis_count <= 0 or len(episodes) == 0:
        return targets
    selected_index_to_local = None
    def task_name_for_position(position: int) -> str:
        task_names = episodes[position].get("tasks", [])
        return str(task_names[0]) if task_names else f"task_{position:03d}"

    if selected_dataset_indices is None:
        selected_positions = np.linspace(
            0,
            len(episodes) - 1,
            num=min(int(vis_count), len(episodes)),
            dtype=np.int64,
        )
        episode_positions = np.unique(selected_positions).tolist()
        if vis_one_episode_per_task:
            episode_positions = []
            seen_tasks: set[str] = set()
            for episode_position in range(len(episodes)):
                task_name = task_name_for_position(episode_position)
                if task_name in seen_tasks:
                    continue
                seen_tasks.add(task_name)
                episode_positions.append(episode_position)
    else:
        selected_index_to_local = {
            int(dataset_index): local_index
            for local_index, dataset_index in enumerate(selected_dataset_indices)
        }
        selected_episode_ids: set[int] = set()
        for episode_position in range(len(episodes)):
            episode = episodes[episode_position]
            start_index = int(episode["dataset_from_index"])
            end_index = int(episode["dataset_to_index"])
            if any(start_index <= index < end_index for index in selected_index_to_local):
                selected_episode_ids.add(int(episode.get("episode_index", episode_position)))
        episode_positions = [
            episode_position
            for episode_position in range(len(episodes))
            if int(episodes[episode_position].get("episode_index", episode_position)) in selected_episode_ids
        ]
        if vis_one_episode_per_task:
            one_per_task: list[int] = []
            seen_tasks: set[str] = set()
            for episode_position in episode_positions:
                task_name = task_name_for_position(episode_position)
                if task_name in seen_tasks:
                    continue
                seen_tasks.add(task_name)
                one_per_task.append(episode_position)
            episode_positions = one_per_task

    for episode_position in episode_positions:
        episode = episodes[episode_position]
        episode_index = int(episode.get("episode_index", episode_position))
        task_name = task_name_for_position(episode_position)
        start_index = int(episode["dataset_from_index"])
        end_index = int(episode["dataset_to_index"])
        episode_length = end_index - start_index
        if episode_length <= 0:
            continue

        max_frame_index = episode_length - 1
        frame_targets = [
            ("p25", min(max_frame_index, int(episode_length * 0.25))),
            ("p50", min(max_frame_index, int(episode_length * 0.50))),
            ("p75", min(max_frame_index, int(episode_length * 0.75))),
        ]
        # Dense terminal previews make it possible to verify that a static target
        # (for example the rack) does not disappear when future frames run out.
        for distance_to_end in (8, 4, 2, 1, 0):
            terminal_frame = max(0, max_frame_index - distance_to_end)
            frame_targets.append((f"terminal_m{distance_to_end:02d}", terminal_frame))
        frame_targets = list(dict.fromkeys(frame_targets))
        for position, frame_index in frame_targets:
            dataset_index = start_index + frame_index
            if selected_index_to_local is not None:
                local_index = selected_index_to_local.get(dataset_index)
                if local_index is not None and 0 <= local_index < total_samples:
                    targets.setdefault(local_index, []).append(
                        (task_name, episode_index, position, frame_index)
                    )
            elif 0 <= dataset_index < total_samples:
                targets.setdefault(dataset_index, []).append(
                    (task_name, episode_index, position, frame_index)
                )
    return targets


def _parse_episode_indices(value: str | None) -> tuple[int, ...] | None:
    if value is None:
        return None
    values = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not values:
        raise ValueError("--episode-indices must contain at least one integer.")
    if any(value < 0 for value in values):
        raise ValueError("--episode-indices must be non-negative.")
    return values


def _select_episode_indices(
    dataset: SongTemporalPointCloudDataset,
    *,
    one_episode_per_task: bool,
    episode_indices: tuple[int, ...] | None,
) -> tuple[int, ...] | None:
    if one_episode_per_task and episode_indices is not None:
        raise ValueError("Use only one of --one-episode-per-task and --episode-indices.")
    episodes = dataset.meta.episodes
    if episode_indices is not None:
        available = {
            int(episodes[index].get("episode_index", index))
            for index in range(len(episodes))
        }
        missing = sorted(set(episode_indices) - available)
        if missing:
            raise ValueError(f"Requested episode indices do not exist: {missing}")
        return tuple(dict.fromkeys(episode_indices))
    if not one_episode_per_task:
        return None

    selected: list[int] = []
    seen_tasks: set[str] = set()
    for episode_position in range(len(episodes)):
        episode = episodes[episode_position]
        tasks = episode.get("tasks", [])
        task_name = str(tasks[0]) if tasks else f"episode_{episode_position}"
        if task_name in seen_tasks:
            continue
        seen_tasks.add(task_name)
        selected.append(int(episode.get("episode_index", episode_position)))
    return tuple(selected)


def _preview_time_gradient(num_steps: int) -> np.ndarray:
    if num_steps <= 1:
        return np.array([[1.0, 0.0, 1.0]], dtype=np.float64)
    t = np.linspace(0.0, 1.0, num_steps, dtype=np.float64)[:, None]
    start = np.array([1.0, 0.0, 1.0], dtype=np.float64)
    end = np.array([0.0, 1.0, 1.0], dtype=np.float64)
    return ((1.0 - t) * start + t * end).clip(0.0, 1.0)


def _preview_rot6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    d6 = d6.to(dtype=torch.float32)
    a1 = d6[..., 0:3]
    a2 = d6[..., 3:6]
    b1 = torch.nn.functional.normalize(a1, dim=-1, eps=1e-6)
    b2 = a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1
    b2 = torch.nn.functional.normalize(b2, dim=-1, eps=1e-6)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-1)


def _preview_relative_ee_positions(pose9: torch.Tensor) -> np.ndarray:
    pose_np = pose9.detach().cpu().to(dtype=torch.float32).numpy()
    if pose_np.ndim != 2 or pose_np.shape[-1] < 9:
        return np.zeros((0, 3), dtype=np.float32)
    finite = np.isfinite(pose_np[:, :9]).all(axis=1)
    pose_np = pose_np[finite]
    if pose_np.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.float32)

    rotmats = _preview_rot6d_to_matrix(torch.from_numpy(pose_np[:, 3:9])).cpu().numpy()
    transforms = np.tile(np.eye(4, dtype=np.float32), (pose_np.shape[0], 1, 1))
    transforms[:, :3, :3] = rotmats
    transforms[:, :3, 3] = pose_np[:, :3]
    try:
        first_inv = np.linalg.inv(transforms[0])
    except np.linalg.LinAlgError:
        return transforms[:, :3, 3]
    return (first_inv[None] @ transforms)[:, :3, 3]


def _preview_future_ee_positions(batch: dict[str, torch.Tensor], batch_index: int, horizon: int = 32) -> np.ndarray:
    trajectory_poses = batch.get("pointseg_trajectory_ee_poses")
    if torch.is_tensor(trajectory_poses) and trajectory_poses.ndim >= 3 and trajectory_poses.shape[-1] >= 9:
        pose9 = trajectory_poses[batch_index].detach().cpu()
        offsets = batch.get("pointseg_trajectory_offsets")
        if torch.is_tensor(offsets):
            order = torch.argsort(offsets[batch_index].detach().cpu())
            pose9 = pose9[order]
        pose9 = pose9[: int(horizon), :9]
        if pose9.shape[0] > 0:
            return pose9.numpy()[:, :3]

    future_poses = batch.get("future_ee_poses")
    if torch.is_tensor(future_poses) and future_poses.ndim >= 3 and future_poses.shape[-1] >= 9:
        pose9 = future_poses[batch_index].detach().cpu()
        valid = torch.ones(pose9.shape[0], dtype=torch.bool)
        future_is_pad = batch.get("future_is_pad")
        if torch.is_tensor(future_is_pad):
            candidate = ~future_is_pad[batch_index].detach().cpu().bool()
            if candidate.numel() == pose9.shape[0]:
                valid &= candidate
        offsets = batch.get("future_offsets")
        if torch.is_tensor(offsets):
            offset_values = offsets[batch_index].detach().cpu()
            order = torch.argsort(offset_values)
            pose9 = pose9[order]
            valid = valid[order]
        pose9 = pose9[valid][: int(horizon), :9].to(dtype=torch.float32)
        if pose9.shape[0] > 0:
            return pose9.numpy()[:, :3]

    action = batch.get("action")
    if torch.is_tensor(action) and action.ndim >= 3 and action.shape[-1] >= 9:
        pose9 = action[batch_index, : min(int(horizon), action.shape[1]), :9]
        return _preview_relative_ee_positions(pose9)
    return np.zeros((0, 3), dtype=np.float32)


def _preview_heatmap_colors(foreground_score: np.ndarray) -> np.ndarray:
    score = np.asarray(foreground_score, dtype=np.float32).reshape(-1).clip(0.0, 1.0)
    heat = np.stack(
        [
            np.clip(2.0 * score, 0.0, 1.0),
            np.clip(1.0 - np.abs(2.0 * score - 1.0), 0.0, 1.0),
            np.clip(1.0 - 2.0 * score, 0.0, 1.0),
        ],
        axis=-1,
    )
    return np.rint(255.0 * heat).astype(np.uint8)


def _write_preview_ply_with_trajectory(
    path: Path,
    points_xyzrgb: np.ndarray,
    foreground_score: np.ndarray,
    ee_positions: np.ndarray,
    *,
    color_mode: str = "original",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    points_xyzrgb = np.asarray(points_xyzrgb, dtype=np.float32)
    xyz = points_xyzrgb[:, :3]
    foreground_score = np.asarray(foreground_score, dtype=np.float32).reshape(-1).clip(0.0, 1.0)

    if points_xyzrgb.shape[-1] >= 6:
        point_colors = np.asarray(points_xyzrgb[:, 3:6], dtype=np.float32)
        finite_colors = point_colors[np.isfinite(point_colors)]
        if finite_colors.size > 0 and float(finite_colors.max(initial=0.0)) <= 1.0:
            point_colors = point_colors * 255.0
        point_colors = np.clip(np.rint(point_colors), 0, 255).astype(np.uint8)
    else:
        point_colors = np.full((xyz.shape[0], 3), 128, dtype=np.uint8)

    if color_mode == "heatmap":
        # Continuous blue->yellow->red heatmap. The score is never thresholded,
        # so small structures remain inspectable in the diagnostic view.
        point_colors = _preview_heatmap_colors(foreground_score)
    elif color_mode != "original":
        raise ValueError(f"Unknown preview color mode: {color_mode}")

    ee_positions = np.asarray(ee_positions, dtype=np.float32)
    if ee_positions.ndim != 2 or ee_positions.shape[-1] != 3:
        ee_positions = np.zeros((0, 3), dtype=np.float32)
    ee_positions = ee_positions[np.isfinite(ee_positions).all(axis=1)]
    trajectory_colors = np.rint(_preview_time_gradient(ee_positions.shape[0]) * 255.0).astype(np.uint8)

    vertices = np.concatenate([xyz, ee_positions], axis=0)
    colors = np.concatenate([point_colors, trajectory_colors], axis=0)
    vertex_foreground_scores = np.concatenate(
        [foreground_score, np.full(ee_positions.shape[0], np.nan, dtype=np.float32)],
        axis=0,
    )
    first_trajectory_vertex = xyz.shape[0]
    edges = [
        (
            first_trajectory_vertex + idx,
            first_trajectory_vertex + idx + 1,
            trajectory_colors[idx],
            trajectory_colors[idx + 1],
        )
        for idx in range(ee_positions.shape[0] - 1)
    ]

    with open(path, "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {vertices.shape[0]}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("property float foreground_score\n")
        f.write(f"element edge {len(edges)}\n")
        f.write("property int vertex1\n")
        f.write("property int vertex2\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        for point, color, score in zip(vertices, colors, vertex_foreground_scores, strict=True):
            f.write(
                f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f} "
                f"{int(color[0])} {int(color[1])} {int(color[2])} {score:.8f}\n"
            )
        for start, end, start_color, end_color in edges:
            color = np.rint((start_color.astype(np.float32) + end_color.astype(np.float32)) * 0.5).astype(
                np.uint8
            )
            f.write(f"{start} {end} {int(color[0])} {int(color[1])} {int(color[2])}\n")


def _safe_task_folder(task_name: str) -> str:
    folder = "_".join("".join(char if char.isalnum() else "_" for char in task_name).split())
    return folder.strip("_").lower() or "unknown_task"


def _save_episode_preview(
    output_dir: Path,
    task_name: str,
    episode_index: int,
    position: str,
    frame_index: int,
    current_pc: torch.Tensor,
    foreground_score: torch.Tensor,
    ee_positions: np.ndarray,
    *,
    color_mode: str = "original",
) -> None:
    episode_dir = (
        output_dir
        / "visualizations"
        / _safe_task_folder(task_name)
        / f"episode_{episode_index:06d}"
    )
    points_xyzrgb = current_pc.detach().cpu().numpy()
    score_np = foreground_score.detach().cpu().numpy()
    base_path = episode_dir / f"{position}_frame_{frame_index:06d}_soft.ply"
    _write_preview_ply_with_trajectory(
        base_path,
        points_xyzrgb,
        score_np,
        ee_positions,
        color_mode="heatmap" if color_mode == "heatmap" else "original",
    )
    if color_mode == "both":
        _write_preview_ply_with_trajectory(
            episode_dir / f"{position}_frame_{frame_index:06d}_soft_heatmap.ply",
            points_xyzrgb,
            score_np,
            ee_positions,
            color_mode="heatmap",
        )
    score = foreground_score.detach().float().cpu()
    xyz = current_pc.detach().float().cpu()[..., :3]
    ee_distance = torch.linalg.norm(xyz, dim=-1)
    stats: dict[str, Any] = {
        "task": task_name,
        "episode_index": int(episode_index),
        "position": position,
        "frame_index": int(frame_index),
        "num_points": int(score.numel()),
        "coordinate_frame": "current_eef",
        "preview_color_mode": color_mode,
        "xyz_min": xyz.min(dim=0).values.tolist(),
        "xyz_max": xyz.max(dim=0).values.tolist(),
        "rgb_min": points_xyzrgb[:, 3:6].min(axis=0).tolist()
        if points_xyzrgb.shape[-1] >= 6
        else None,
        "rgb_max": points_xyzrgb[:, 3:6].max(axis=0).tolist()
        if points_xyzrgb.shape[-1] >= 6
        else None,
        "rgb_nonzero_points": int(np.count_nonzero(points_xyzrgb[:, 3:6].sum(axis=1) > 0))
        if points_xyzrgb.shape[-1] >= 6
        else None,
        "soft_mean": float(score.mean().item()),
        "soft_max": float(score.max().item()),
        "soft_quantiles": {
            str(q): float(torch.quantile(score, q).item()) for q in (0.5, 0.75, 0.9, 0.95, 0.99)
        },
        "fraction_above": {
            str(threshold): float((score >= threshold).float().mean().item())
            for threshold in (0.25, 0.5, 0.75)
        },
    }
    for radius in (0.10, 0.15, 0.25):
        near = ee_distance <= radius
        stats[f"near_{radius:.2f}m_count"] = int(near.sum().item())
        stats[f"near_{radius:.2f}m_soft_mean"] = (
            float(score[near].mean().item()) if bool(near.any()) else None
        )
    _atomic_write_json(episode_dir / f"{position}_frame_{frame_index:06d}_soft_stats.json", stats)


def _rebuild_visualizations_from_cache(args: argparse.Namespace) -> None:
    """Recreate previews from cached indices/scores without changing cache shards."""
    if not args.output_dir.is_dir() or not (args.output_dir / "manifest.json").is_file():
        raise FileNotFoundError(
            "--rebuild-visualizations requires an existing cache with manifest.json: "
            + str(args.output_dir)
        )

    dataset = make_dataset(args)
    cached = SongPointSegCachedDataset(args.output_dir)
    if len(cached) != len(dataset):
        raise ValueError(
            f"Cache samples ({len(cached)}) do not match dataset samples ({len(dataset)})."
        )

    visualization_dir = args.output_dir / "visualizations"
    if visualization_dir.exists():
        shutil.rmtree(visualization_dir)

    preview_targets = _episode_preview_targets(
        dataset,
        len(dataset),
        args.vis_count,
        vis_one_episode_per_task=args.vis_one_episode_per_task,
    )
    point_cloud_dir = args.point_cloud_dir or Path(getattr(dataset.dataset, "root", args.dataset_repo_id)) / "point_clouds"
    saved = 0
    for dataset_index in sorted(preview_targets):
        targets = preview_targets[dataset_index]
        cached_item = cached[dataset_index]
        episode_index = int(cached_item["episode_index"].item())
        frame_index = int(cached_item["frame_index"].item())
        point_indices = cached_item["observation.point_cloud_indices"].numpy()
        score = cached_item["pointseg.foreground_score"]
        point_clouds = open_episode_point_clouds(point_cloud_dir, episode_index)
        current_pc = torch.from_numpy(np.asarray(point_clouds[frame_index], dtype=np.float32)[point_indices])

        # The trajectory stored by SongTemporalPointCloudDataset is already
        # relative to the current EEF. Only sort by offsets for display.
        source_item = dataset[dataset_index]
        pose9 = source_item["pointseg_trajectory_ee_poses"]
        offsets = source_item["pointseg_trajectory_offsets"]
        order = torch.argsort(offsets)
        ee_positions = pose9[order][:32, :3].detach().cpu().numpy()
        for task_name, target_episode, position, target_frame in targets:
            _save_episode_preview(
                args.output_dir,
                task_name,
                target_episode,
                position,
                target_frame,
                current_pc,
                score,
                ee_positions,
                color_mode=args.preview_color_mode,
            )
            saved += 1

    _write_terminal_continuity_summary(args.output_dir)
    print(
        f"Rebuilt {saved} previews from existing cache; shards were not modified: {args.output_dir}",
        flush=True,
    )



def _is_torchrun_env() -> bool:
    return int(os.environ.get("WORLD_SIZE", "1")) > 1


def _init_multiprocess(args: argparse.Namespace) -> tuple[int, int, int, torch.device]:
    """Read torchrun rank env vars and choose one GPU per rank.

    This cache job does not need gradient collectives, so it intentionally avoids
    torch.distributed/NCCL. Synchronization is done with marker files to prevent
    NCCL watchdog timeouts when some ranks finish much earlier than others.
    """
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    wants_cuda = str(args.device).startswith("cuda") and torch.cuda.is_available()
    if wants_cuda:
        visible_count = torch.cuda.device_count()
        if visible_count <= 0:
            raise RuntimeError("args.device requests CUDA but torch.cuda.device_count() is 0")
        device_index = local_rank % visible_count
        torch.cuda.set_device(device_index)
        device = torch.device(f"cuda:{device_index}")
    else:
        device = torch.device(args.device)

    return rank, local_rank, world_size, device


def _sync_dir(output_dir: Path) -> Path:
    return output_dir / "_dist_sync"


def _write_marker(path: Path, text: str = "ok") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(text)
    os.replace(tmp, path)


def _wait_for_marker(path: Path, *, timeout_sec: int = 0, poll_sec: float = 2.0) -> None:
    start = time.time()
    while not path.exists():
        if timeout_sec > 0 and (time.time() - start) > timeout_sec:
            raise TimeoutError(f"Timed out waiting for marker: {path}")
        time.sleep(poll_sec)


def _wait_for_all_rank_done(output_dir: Path, world_size: int, *, timeout_sec: int = 0) -> None:
    sync = _sync_dir(output_dir)
    start = time.time()
    missing_report_t = 0.0
    while True:
        missing = [r for r in range(world_size) if not (sync / f"rank_{r:03d}.done").exists()]
        failed = sorted(sync.glob("rank_*.failed"))
        if failed:
            details = []
            for path in failed:
                try:
                    details.append(f"{path.name}: {path.read_text()[:2000]}")
                except Exception:
                    details.append(str(path))
            raise RuntimeError("One or more ranks failed:\n" + "\n".join(details))
        if not missing:
            return
        now = time.time()
        if now - missing_report_t > 60:
            print(f"[rank 0] waiting for done markers from ranks: {missing}", flush=True)
            missing_report_t = now
        if timeout_sec > 0 and (now - start) > timeout_sec:
            raise TimeoutError(f"Timed out waiting for ranks {missing} to finish")
        time.sleep(2.0)


def _rank_bounds(total: int, world_size: int, rank: int) -> tuple[int, int]:
    """Contiguous split, preserving global dataset/cache order."""
    base = total // world_size
    rem = total % world_size
    start = rank * base + min(rank, rem)
    length = base + (1 if rank < rem else 0)
    return start, start + length


def _make_rank_shards(total_samples: int, shard_size: int, world_size: int, rank: int) -> list[dict[str, Any]]:
    start_index, end_index = _rank_bounds(total_samples, world_size, rank)
    local_samples = end_index - start_index
    local_shards = _make_shard_manifest(local_samples, shard_size)
    for shard in local_shards:
        shard["path"] = f"rank_{rank:03d}/{shard['path']}"
        shard["start"] = start_index + int(shard["start"])
    return local_shards


def _build_all_shards_from_disk(output_dir: Path, total_samples: int, shard_size: int, world_size: int) -> list[dict[str, Any]]:
    """Rank0 rebuilds manifest shard list after all ranks have written shard arrays."""
    all_shards: list[dict[str, Any]] = []
    for rank in range(world_size):
        for shard in _make_rank_shards(total_samples, shard_size, world_size, rank):
            offsets_path = output_dir / shard["path"] / "sample_offsets.npy"
            if not offsets_path.exists():
                raise FileNotFoundError(f"Missing shard offsets written by rank {rank}: {offsets_path}")
            offsets = np.load(offsets_path, mmap_mode="r")
            shard["num_points"] = int(offsets[-1])
            all_shards.append(shard)
    return all_shards


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, path)


def _write_terminal_continuity_summary(output_dir: Path) -> None:
    records_by_episode: dict[int, list[dict[str, Any]]] = {}
    for path in sorted((output_dir / "visualizations").rglob("terminal_*_soft_stats.json")):
        with open(path) as f:
            record = json.load(f)
        records_by_episode.setdefault(int(record["episode_index"]), []).append(record)

    episode_summaries = []
    for episode_index, records in sorted(records_by_episode.items()):
        records.sort(key=lambda value: int(value["frame_index"]))
        soft_means = np.asarray([record["soft_mean"] for record in records], dtype=np.float64)
        near_means = np.asarray(
            [
                np.nan if record.get("near_0.15m_soft_mean") is None else record["near_0.15m_soft_mean"]
                for record in records
            ],
            dtype=np.float64,
        )
        episode_summaries.append(
            {
                "episode_index": episode_index,
                "frames": [int(record["frame_index"]) for record in records],
                "soft_mean_min": float(soft_means.min()),
                "soft_mean_max": float(soft_means.max()),
                "soft_mean_max_adjacent_change": float(np.abs(np.diff(soft_means)).max(initial=0.0)),
                "near_0.15m_soft_mean_min": (
                    float(np.nanmin(near_means)) if np.isfinite(near_means).any() else None
                ),
                "near_0.15m_soft_mean_max_adjacent_change": (
                    float(np.nanmax(np.abs(np.diff(near_means))))
                    if near_means.size > 1 and np.isfinite(np.diff(near_means)).any()
                    else 0.0
                ),
            }
        )
    _atomic_write_json(
        output_dir / "terminal_soft_continuity_summary.json",
        {
            "description": (
                "Distribution-level terminal diagnostics. Inspect the matching *_soft.ply files to verify "
                "individual mug, handle, and rack surfaces; no semantic masks are used."
            ),
            "num_episodes": len(episode_summaries),
            "episodes": episode_summaries,
        },
    )


def cache_samples(args: argparse.Namespace) -> None:
    if args.rebuild_visualizations:
        _rebuild_visualizations_from_cache(args)
        return

    if args.smoke_test:
        args.current_points = min(args.current_points, 256)
        args.future_points = min(args.future_points, 512)
        args.batch_size = min(args.batch_size, 2)
        args.shard_size = min(args.shard_size, 4)
        args.max_samples = 4 if args.max_samples is None else min(args.max_samples, 4)
        args.vis_count = min(args.vis_count, 2)

    rank, local_rank, world_size, device = _init_multiprocess(args)
    is_main = rank == 0

    # Make each rank deterministic but distinct for DataLoader workers.
    set_seed(args.seed + rank)
    storage_dtype = np.dtype(args.storage_dtype)

    sync = _sync_dir(args.output_dir)
    if is_main:
        _prepare_output_dir(args.output_dir, args.overwrite)
        sync.mkdir(parents=True, exist_ok=True)
        for marker in sync.glob("rank_*.done"):
            marker.unlink(missing_ok=True)
        for marker in sync.glob("rank_*.failed"):
            marker.unlink(missing_ok=True)
        _write_marker(sync / "ready", f"ready pid={os.getpid()} time={time.time()}\n")
    else:
        _wait_for_marker(sync / "ready", timeout_sec=args.rank_wait_timeout_sec)

    pseudo_cfg = replace(
        PseudoLabelConfig(),
        nn_chunk_size=args.nn_chunk_size,
        motion_rotation_radius=args.motion_rotation_radius,
        motion_baseline_threshold=args.motion_baseline_threshold,
        motion_baseline_temperature=args.motion_baseline_temperature,
        motion_relative_margin=args.motion_relative_margin,
        motion_relative_tau=args.motion_relative_tau,
        trajectory_sigma=args.trajectory_sigma,
        contact_radius=args.contact_radius,
        contact_temperature=args.contact_temperature,
        approach_margin=args.approach_margin,
        approach_tau=args.approach_tau,
        background_trajectory_sigma=args.background_trajectory_sigma,
    )
    if is_main:
        _atomic_write_json(args.output_dir / "pseudo_label_config.json", asdict(pseudo_cfg))
    full_dataset = make_dataset(args)
    requested_episode_indices = _parse_episode_indices(args.episode_indices)
    selected_episode_indices = _select_episode_indices(
        full_dataset,
        one_episode_per_task=args.one_episode_per_task,
        episode_indices=requested_episode_indices,
    )
    selected_dataset_indices = None
    if selected_episode_indices is not None:
        episodes_by_id = {
            int(full_dataset.meta.episodes[index].get("episode_index", index)): full_dataset.meta.episodes[index]
            for index in range(len(full_dataset.meta.episodes))
        }
        selected_dataset_indices = []
        for episode_index in selected_episode_indices:
            episode = episodes_by_id[episode_index]
            selected_dataset_indices.extend(
                range(int(episode["dataset_from_index"]), int(episode["dataset_to_index"]))
            )
        if args.max_samples is not None:
            selected_dataset_indices = selected_dataset_indices[: args.max_samples]
        total_samples = len(selected_dataset_indices)
    else:
        total_samples = len(full_dataset) if args.max_samples is None else min(len(full_dataset), args.max_samples)
    if total_samples <= 0:
        raise ValueError("Song pointseg cache needs at least one sample.")
    preview_targets = _episode_preview_targets(
        full_dataset,
        total_samples,
        args.vis_count,
        selected_dataset_indices=selected_dataset_indices,
        vis_one_episode_per_task=args.vis_one_episode_per_task,
    )

    start_index, end_index = _rank_bounds(total_samples, world_size, rank)
    local_samples = end_index - start_index
    local_indices = range(start_index, end_index)
    if selected_dataset_indices is not None:
        local_indices = selected_dataset_indices[start_index:end_index]
    dataset = Subset(full_dataset, local_indices)
    shards = _make_rank_shards(total_samples, args.shard_size, world_size, rank)

    if is_main:
        manifest = {
            "version": POINTSEG_CACHE_VERSION,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "role_names": list(ROLE_NAMES),
            "fields": list(POINTSEG_CACHE_LABEL_FIELDS),
            "cache_mode": "indices",
            "num_samples": total_samples,
            "selected_episode_indices": list(selected_episode_indices) if selected_episode_indices is not None else None,
            "future_offsets": list(args.future_offsets),
            "temporal_offsets": list(full_dataset.temporal_offsets),
            "temporal_mode": "bidirectional" if full_dataset.bidirectional else "future_only",
            "trajectory_mode": "sparse_full_episode",
            "trajectory_pose_source": "observation.state (achieved EEF pose)",
            "trajectory_samples": full_dataset.trajectory_samples,
            "current_points": args.current_points,
            "future_points": args.future_points,
            "variable_num_points": True,
            "point_count_policy": "cap_without_repeat",
            "pseudo_label_policy": "soft_binary_trajectory_v1",
            "evidence_channels": ["tool_comotion", "trajectory_approach", "near_contact"],
            "storage_dtype": args.storage_dtype,
            "pseudo_label_config": asdict(pseudo_cfg),
            "args": _jsonable(vars(args)),
            "distributed": {
                "world_size": world_size,
                "launcher": "torchrun",
                "split": "contiguous_by_dataset_index",
            },
            "shards": [],
        }
    else:
        manifest = None

    if is_main:
        print(
            f"[cache] total_samples={total_samples} world_size={world_size} "
            f"batch_size_per_gpu={args.batch_size} device={device}",
            flush=True,
        )
    print(
        f"[rank {rank}/{world_size}] local_rank={local_rank} device={device} "
        f"range=[{start_index}, {end_index}) local_samples={local_samples} shards={len(shards)}",
        flush=True,
    )

    if local_samples == 0:
        _write_marker(sync / f"rank_{rank:03d}.done", "empty\n")
        if is_main and manifest is not None:
            _wait_for_all_rank_done(args.output_dir, world_size, timeout_sec=args.rank_wait_timeout_sec)
            manifest["shards"] = _build_all_shards_from_disk(
                args.output_dir, total_samples, args.shard_size, world_size
            )
            _atomic_write_json(args.output_dir / "manifest.json", manifest)
        return

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
        prefetch_factor=2 if args.num_workers > 0 else None,
        persistent_workers=args.num_workers > 0,
        collate_fn=song_pointseg_collate,
    )

    current_shard_index = 0
    shard_samples: list[dict[str, np.ndarray | int]] = []
    written = 0
    previews_saved = 0
    t0 = time.time()
    progress = tqdm(
        total=local_samples,
        desc=f"Rank {rank} cache",
        unit="sample",
        position=rank,
        disable=not is_main,
    )

    try:
        with torch.inference_mode():
            for batch in dataloader:
                if written >= local_samples:
                    break

                batch_size = min(int(batch["observation.point_cloud"].shape[0]), local_samples - written)
                batch = _slice_batch_to_size(batch, batch_size)
                batch = move_batch_to_device(batch, device)
                current_pc = batch["observation.point_cloud"]

                geometric_pseudo = generate_pseudo_labels(
                    current_pc,
                    batch["observation.point_cloud_future"],
                    batch["future_ee_poses"],
                    batch["future_is_pad"],
                    current_is_pad=batch.get("observation.point_cloud_is_pad"),
                    future_point_is_pad=batch.get("observation.point_cloud_future_is_pad"),
                    trajectory_poses=batch.get("pointseg_trajectory_ee_poses"),
                    trajectory_offsets=batch.get("pointseg_trajectory_offsets"),
                    config=pseudo_cfg,
                )
                pseudo = geometric_pseudo

                current_is_pad = batch.get("observation.point_cloud_is_pad")
                for batch_index in range(batch_size):
                    dataset_index = start_index + written + batch_index
                    targets = preview_targets.get(dataset_index, ())
                    if not targets:
                        continue
                    valid = (
                        ~current_is_pad[batch_index].bool()
                        if current_is_pad is not None
                        else torch.ones(current_pc.shape[1], dtype=torch.bool, device=current_pc.device)
                    )
                    preview_pc = current_pc[batch_index][valid]
                    preview_score = pseudo["foreground_score"][batch_index][valid]
                    preview_ee_positions = _preview_future_ee_positions(batch, batch_index)
                    for task_name, episode_index, position, frame_index in targets:
                        _save_episode_preview(
                            args.output_dir,
                            task_name,
                            episode_index,
                            position,
                            frame_index,
                            preview_pc,
                            preview_score,
                            preview_ee_positions,
                            color_mode=args.preview_color_mode,
                        )
                        previews_saved += 1

                for batch_index in range(batch_size):
                    global_dataset_index = start_index + written
                    shard_samples.append(
                        _sample_from_batch(current_pc, pseudo, batch, batch_index, global_dataset_index)
                    )
                    written += 1
                    progress.update(1)

                    if len(shard_samples) == int(shards[current_shard_index]["length"]):
                        _save_variable_shard(
                            args.output_dir,
                            shards[current_shard_index],
                            shard_samples,
                            storage_dtype=storage_dtype,
                        )
                        shard_samples = []
                        current_shard_index += 1
    finally:
        progress.close()

    if shard_samples:
        _save_variable_shard(
            args.output_dir,
            shards[current_shard_index],
            shard_samples,
            storage_dtype=storage_dtype,
        )

    elapsed = time.time() - t0
    speed = written / max(elapsed, 1e-6)
    print(
        f"[rank {rank}/{world_size}] wrote {written} samples in {elapsed:.1f}s "
        f"({speed:.2f} samples/s); episode previews saved={previews_saved}",
        flush=True,
    )

    _write_marker(
        sync / f"rank_{rank:03d}.done",
        f"written={written} elapsed={elapsed:.3f} speed={speed:.3f} pid={os.getpid()}\n",
    )

    if is_main:
        assert manifest is not None
        _wait_for_all_rank_done(args.output_dir, world_size, timeout_sec=args.rank_wait_timeout_sec)
        manifest["shards"] = _build_all_shards_from_disk(
            args.output_dir, total_samples, args.shard_size, world_size
        )
        _atomic_write_json(args.output_dir / "manifest.json", manifest)
        _write_terminal_continuity_summary(args.output_dir)
        print(f"Cached {total_samples} Song pointseg samples to {args.output_dir}", flush=True)


def main() -> None:
    args = parse_args()
    rank = int(os.environ.get("RANK", "0"))
    try:
        cache_samples(args)
    except Exception as exc:
        try:
            _write_marker(_sync_dir(args.output_dir) / f"rank_{rank:03d}.failed", repr(exc))
        except Exception:
            pass
        raise


if __name__ == "__main__":
    main()
