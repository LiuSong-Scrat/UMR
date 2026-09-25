#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import dataclasses
import logging
import os
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from pprint import pformat
from typing import Any

import numpy as np
import torch
from accelerate import Accelerator
from termcolor import colored
from torch.optim import Optimizer
from tqdm import tqdm

from lerobot.configs import parser
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.factory import make_dataset
from lerobot.datasets.sampler import EpisodeAwareSampler
from lerobot.datasets.utils import cycle
from lerobot.envs.factory import make_env, make_env_pre_post_processors
from lerobot.envs.utils import close_envs
from lerobot.optim.factory import make_optimizer_and_scheduler
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.smolvla.processor_smolvla import validate_smolvla_worldflow_preprocessor
from lerobot.policies.smolvla.song_pointseg import (
    DEFAULT_FUTURE_OFFSETS,
    PseudoLabelConfig,
    ROLE_FOREGROUND,
    SongPointSegCachedDataset,
    SongTemporalPointCloudDataset,
    generate_pseudo_labels,
    open_episode_point_clouds,
    song_pointseg_collate,
    write_role_ply,
)
from lerobot.rl.wandb_utils import WandBLogger
from lerobot.scripts.lerobot_eval import eval_policy_all
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.logging_utils import AverageMeter, MetricsTracker
from lerobot.utils.random_utils import set_seed
from lerobot.utils.train_utils import (
    get_step_checkpoint_dir,
    get_step_identifier,
    load_training_state,
    save_checkpoint,
    update_last_checkpoint,
)
from lerobot.utils.utils import (
    format_big_number,
    has_method,
    init_logging,
    inside_slurm,
)


class PointCloudMemmapDataset(torch.utils.data.Dataset):
    """Inject point clouds from per-episode zarr/npy arrays into a LeRobotDataset item."""

    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        point_cloud_dir: str | Path,
        key: str = "observation.point_cloud",
        mmap_mode: str = "r",
    ):
        self.dataset = dataset
        self.point_cloud_dir = Path(point_cloud_dir)
        self.key = key
        self.mmap_mode = mmap_mode
        self._point_cloud_cache: dict[int, np.ndarray] = {}

    def __getattr__(self, name):
        return getattr(self.dataset, name)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_point_cloud_cache"] = {}
        return state

    def __len__(self):
        return len(self.dataset)

    @staticmethod
    def _to_int(value) -> int:
        if torch.is_tensor(value):
            return int(value.reshape(-1)[0].item())
        if isinstance(value, np.ndarray):
            return int(value.reshape(-1)[0].item())
        return int(value)

    def _episode_point_clouds(self, episode_index: int) -> np.ndarray:
        point_clouds = self._point_cloud_cache.get(episode_index)
        if point_clouds is None:
            point_clouds = open_episode_point_clouds(
                self.point_cloud_dir,
                episode_index,
                mmap_mode=self.mmap_mode,
            )
            self._point_cloud_cache[episode_index] = point_clouds
        return point_clouds

    def __getitem__(self, idx):
        item = self.dataset[idx]
        episode_index = self._to_int(item["episode_index"])
        frame_index = self._to_int(item["frame_index"])
        point_cloud = np.asarray(self._episode_point_clouds(episode_index)[frame_index], dtype=np.float32).copy()
        item[self.key] = torch.from_numpy(point_cloud).unsqueeze(0)
        return item


class WorldFlowMemmapDataset(torch.utils.data.Dataset):
    """Inject fixed-reference EEF poses for WorldFlow supervision.

    The current pose is the achieved observation pose. Future World targets use
    ``action_target_ee_poses`` when present so they describe the same controller
    targets as the Ego action chunk. Legacy datasets without that sidecar fall
    back to achieved future poses.
    """

    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        root: str | Path,
        *,
        chunk_size: int,
        action_start_offset: int = 0,
        require_action_target_sidecar: bool = False,
        mmap_mode: str = "r",
    ):
        self.dataset = dataset
        self.root = Path(root)
        self.pose_dir = self.root / "world_ee_poses"
        command_target_dir = self.root / "action_target_ee_poses"
        if require_action_target_sidecar and not command_target_dir.is_dir():
            raise FileNotFoundError(
                "WorldFlow requires commanded action targets but the sidecar directory is missing: "
                f"{command_target_dir}. Regenerate the dataset with action_target_ee_poses or set "
                "worldflow_require_action_target_sidecar=False only for an explicitly achieved-trajectory dataset."
            )
        self.target_pose_dir = command_target_dir if command_target_dir.is_dir() else self.pose_dir
        self.chunk_size = int(chunk_size)
        self.action_start_offset = int(action_start_offset)
        if self.action_start_offset < 0:
            raise ValueError("WorldFlow action_start_offset must be non-negative.")
        self.mmap_mode = mmap_mode
        self._pose_cache: dict[int, np.ndarray] = {}
        self._target_pose_cache: dict[int, np.ndarray] = {}

        if not self.pose_dir.is_dir():
            raise FileNotFoundError(
                f"WorldFlow is enabled but reference-frame ee pose directory is missing: {self.pose_dir}"
            )
        if self.target_pose_dir == self.pose_dir:
            logging.warning(
                "WorldFlow command-target sidecar is absent at %s; falling back to achieved future poses. "
                "The World--Ego bridge is exactly label-consistent only when action_target_ee_poses is present.",
                command_target_dir,
            )

    def __getattr__(self, name):
        return getattr(self.dataset, name)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_pose_cache"] = {}
        state["_target_pose_cache"] = {}
        return state

    def __len__(self):
        return len(self.dataset)

    @staticmethod
    def _to_int(value) -> int:
        if torch.is_tensor(value):
            return int(value.reshape(-1)[0].item())
        if isinstance(value, np.ndarray):
            return int(value.reshape(-1)[0].item())
        return int(value)

    def _load_episode_poses(
        self,
        episode_index: int,
        *,
        directory: Path,
        cache: dict[int, np.ndarray],
        description: str,
    ) -> np.ndarray:
        poses = cache.get(episode_index)
        if poses is None:
            path = directory / f"episode_{episode_index:06d}.npy"
            if not path.exists():
                raise FileNotFoundError(f"WorldFlow {description} pose memmap file is missing: {path}")
            poses = np.load(path, mmap_mode=self.mmap_mode)
            if poses.ndim != 2 or poses.shape[-1] != 9:
                raise ValueError(f"Expected WorldFlow {description} poses shape (T,9), got {poses.shape}.")
            cache[episode_index] = poses
        return poses

    def _episode_poses(self, episode_index: int) -> np.ndarray:
        return self._load_episode_poses(
            episode_index,
            directory=self.pose_dir,
            cache=self._pose_cache,
            description="achieved current",
        )

    def _episode_target_poses(self, episode_index: int) -> np.ndarray:
        return self._load_episode_poses(
            episode_index,
            directory=self.target_pose_dir,
            cache=self._target_pose_cache,
            description="command target",
        )

    def __getitem__(self, idx):
        item = dict(self.dataset[idx])
        episode_index = self._to_int(item["episode_index"])
        frame_index = self._to_int(item["frame_index"])
        poses = self._episode_poses(episode_index)
        target_poses = self._episode_target_poses(episode_index)
        episode_len = int(len(poses))
        if episode_len <= 0:
            raise ValueError(f"Worldflow episode {episode_index} is empty.")
        if len(target_poses) != episode_len:
            raise ValueError(
                f"WorldFlow episode {episode_index} achieved/target lengths differ: "
                f"{episode_len} != {len(target_poses)}."
            )

        current_index = min(max(frame_index, 0), episode_len - 1)
        current_pose = np.array(poses[current_index], dtype=np.float32, copy=True)
        item["worldflow.current_ee_pose"] = torch.from_numpy(
            current_pose
        )

        action = item.get("action")
        if (torch.is_tensor(action) or isinstance(action, np.ndarray)) and action.ndim >= 2:
            chunk_size = int(action.shape[0])
        else:
            chunk_size = self.chunk_size
        frame_indices = (
            frame_index
            + self.action_start_offset
            + np.arange(chunk_size, dtype=np.int64)
        )
        clamped_indices = np.clip(frame_indices, 0, episode_len - 1)
        item["worldflow.ee_poses"] = torch.from_numpy(
            np.array(target_poses[clamped_indices], dtype=np.float32, copy=True)
        )
        item["worldflow.step_is_pad"] = torch.from_numpy(frame_indices >= episode_len)
        return item


class PointSegCacheInjectedDataset(torch.utils.data.Dataset):
    """Inject offline temporal pointseg samples into the action-training dataset."""

    pointseg_keys = (
        "observation.point_cloud",
        "pointseg.priors",
        "pointseg.labels",
        "pointseg.weights",
        "pointseg.class_scores",
        "pointseg.role_scores",
        "pointseg.foreground_score",
    )

    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        cache_dir: str | Path,
        *,
        point_cloud_dir: str | Path | None = None,
        strict: bool = True,
        mmap_mode: str = "r",
    ):
        self.dataset = dataset
        self.cache = SongPointSegCachedDataset(cache_dir)
        root_value = getattr(dataset, "root", None)
        if root_value is None:
            root_value = dataset.meta.root
        root = Path(root_value)
        self.point_cloud_dir = Path(point_cloud_dir) if point_cloud_dir is not None else root / "point_clouds"
        self.strict = strict
        self.mmap_mode = mmap_mode
        self._point_cloud_cache: dict[int, np.ndarray] = {}
        if self.strict and len(self.cache) < len(self.dataset):
            raise ValueError(
                f"Song pointseg cache has {len(self.cache)} samples but action dataset has {len(self.dataset)}. "
                "Rebuild the cache without --max-samples, or set SONG_POINTSEG_CACHE_STRICT=0 for debugging."
            )

    def __getattr__(self, name):
        return getattr(self.dataset, name)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_point_cloud_cache"] = {}
        return state

    def __len__(self):
        return len(self.dataset)

    @staticmethod
    def _to_int(value) -> int:
        if torch.is_tensor(value):
            return int(value.reshape(-1)[0].item())
        if isinstance(value, np.ndarray):
            return int(value.reshape(-1)[0].item())
        return int(value)

    def _episode_point_clouds(self, episode_index: int) -> np.ndarray:
        point_clouds = self._point_cloud_cache.get(episode_index)
        if point_clouds is None:
            point_clouds = open_episode_point_clouds(
                self.point_cloud_dir,
                episode_index,
                mmap_mode=self.mmap_mode,
            )
            self._point_cloud_cache[episode_index] = point_clouds
        return point_clouds

    def _check_alignment(self, item: dict[str, Any], cache_item: dict[str, torch.Tensor], idx: int) -> None:
        if "episode_index" in item and "episode_index" in cache_item:
            item_episode = self._to_int(item["episode_index"])
            cache_episode = self._to_int(cache_item["episode_index"])
            if item_episode != cache_episode:
                raise ValueError(
                    f"Song pointseg cache is not aligned at dataset index {idx}: "
                    f"episode_index {cache_episode} != {item_episode}."
                )
        if "frame_index" in item and "frame_index" in cache_item:
            item_frame = self._to_int(item["frame_index"])
            cache_frame = self._to_int(cache_item["frame_index"])
            if item_frame != cache_frame:
                raise ValueError(
                    f"Song pointseg cache is not aligned at dataset index {idx}: "
                    f"frame_index {cache_frame} != {item_frame}."
                )

    def __getitem__(self, idx):
        item = self.dataset[idx]
        if idx >= len(self.cache):
            if self.strict:
                raise IndexError(f"Song pointseg cache is missing dataset index {idx}.")
            return item

        cache_item = self.cache[idx]
        self._check_alignment(item, cache_item, idx)
        if "observation.point_cloud" not in cache_item and "observation.point_cloud_indices" in cache_item:
            episode_index = self._to_int(cache_item["episode_index"])
            frame_index = self._to_int(cache_item["frame_index"])
            indices = cache_item["observation.point_cloud_indices"].detach().cpu().numpy().astype(np.int64)
            point_clouds = self._episode_point_clouds(episode_index)
            point_cloud = np.asarray(point_clouds[frame_index][indices], dtype=np.float32).copy()
            cache_item["observation.point_cloud"] = torch.from_numpy(point_cloud)
        for key in self.pointseg_keys:
            if key in cache_item:
                item[key] = cache_item[key]
        if "observation.point_cloud_indices" in cache_item:
            item["observation.point_cloud_indices"] = cache_item["observation.point_cloud_indices"]
        return item


class OnlinePointSegPseudoDataset(torch.utils.data.Dataset):
    """Adds temporal point-cloud fields for batch-level online pseudo-label generation."""

    transient_keys = (
        "observation.point_cloud_future",
        "observation.point_cloud_future_is_pad",
        "future_is_pad",
        "future_offsets",
        "future_ee_poses",
        "pointseg_trajectory_ee_poses",
        "pointseg_trajectory_offsets",
    )

    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        *,
        point_cloud_dir: str | Path,
        policy_cfg: Any,
        mmap_mode: str = "r",
    ):
        self.dataset = SongTemporalPointCloudDataset(
            dataset,
            point_cloud_dir=point_cloud_dir,
            future_offsets=self._future_offsets(policy_cfg),
            current_points=self._env_int("SONG_POINTSEG_ONLINE_CURRENT_POINTS", 10000),
            future_points=self._env_int("SONG_POINTSEG_ONLINE_FUTURE_POINTS", 10000),
            seed=self._env_int("SONG_POINTSEG_ONLINE_SEED", 1000),
            mmap_mode=mmap_mode,
        )
        self.current_points = int(self.dataset.current_points)
        default_device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(os.environ.get("SONG_POINTSEG_ONLINE_DEVICE", default_device))
        self.pseudo_config = PseudoLabelConfig(
            nn_chunk_size=self._env_int("SONG_POINTSEG_ONLINE_NN_CHUNK_SIZE", 512)
        )

    def __getattr__(self, name):
        return getattr(self.dataset, name)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["device"] = torch.device("cpu")
        return state

    def __len__(self):
        return len(self.dataset)

    @staticmethod
    def _env_int(name: str, default: int) -> int:
        value = os.environ.get(name)
        if value is None or str(value).strip() == "":
            return int(default)
        return int(value)

    @staticmethod
    def _future_offsets(policy_cfg: Any) -> tuple[int, ...]:
        env_value = os.environ.get("SONG_POINTSEG_ONLINE_FUTURE_OFFSETS")
        if env_value:
            offsets = tuple(int(part) for part in env_value.replace(";", ",").split(",") if part.strip())
        else:
            offsets = DEFAULT_FUTURE_OFFSETS
        chunk_size = int(getattr(policy_cfg, "chunk_size", max(offsets) + 1))
        offsets = tuple(offset for offset in offsets if 0 < int(offset) < chunk_size)
        if not offsets:
            offsets = (1,)
        return offsets

    def __getitem__(self, idx):
        return dict(self.dataset[idx])

    def make_collate_fn(self):
        return OnlinePointSegBatchCollator(
            current_points=self.current_points,
            device=self.device,
            pseudo_config=self.pseudo_config,
        )


class OnlinePointSegBatchCollator:
    """Collate samples, compute Song pseudo labels once for the whole batch, and drop future fields."""

    transient_keys = OnlinePointSegPseudoDataset.transient_keys

    def __init__(self, *, current_points: int, device: torch.device, pseudo_config: PseudoLabelConfig):
        self.current_points = int(current_points)
        self.device = torch.device(device)
        self.pseudo_config = pseudo_config
        self.profile_freq = int(os.environ.get("SONG_POINTSEG_PROFILE_FREQ", "0") or 0)
        self._calls = 0

    def __call__(self, samples: list[dict[str, Any]]) -> dict[str, Any]:
        self._calls += 1
        t0 = time.perf_counter()
        batch = song_pointseg_collate(samples)
        t1 = time.perf_counter()
        current_pc = batch["observation.point_cloud"].to(device=self.device, dtype=torch.float32)
        future_pc = batch["observation.point_cloud_future"].to(device=self.device, dtype=torch.float32)
        future_poses = batch["future_ee_poses"].to(device=self.device, dtype=torch.float32)
        future_is_pad = batch["future_is_pad"].to(device=self.device, dtype=torch.bool)
        current_is_pad = batch.get("observation.point_cloud_is_pad")
        if torch.is_tensor(current_is_pad):
            current_is_pad = current_is_pad.to(device=self.device, dtype=torch.bool)
        future_point_is_pad = batch.get("observation.point_cloud_future_is_pad")
        if torch.is_tensor(future_point_is_pad):
            # Fixed-size zarr point clouds produce an all-False mask. Dropping it avoids a
            # per-KNN CUDA synchronization in the fast pointops path.
            future_point_is_pad = (
                future_point_is_pad.to(device=self.device, dtype=torch.bool)
                if bool(future_point_is_pad.any().item())
                else None
            )
        t2 = time.perf_counter()
        with torch.inference_mode():
            pseudo = generate_pseudo_labels(
                current_pc,
                future_pc,
                future_poses,
                future_is_pad,
                current_is_pad=current_is_pad,
                future_point_is_pad=future_point_is_pad,
                trajectory_poses=batch["pointseg_trajectory_ee_poses"].to(
                    device=self.device, dtype=torch.float32
                ),
                config=self.pseudo_config,
            )
        t3 = time.perf_counter()
        for source_key, dest_key in (
            ("labels", "pointseg.labels"),
            ("weights", "pointseg.weights"),
            ("class_scores", "pointseg.class_scores"),
            ("role_scores", "pointseg.role_scores"),
            ("foreground_score", "pointseg.foreground_score"),
        ):
            if source_key in pseudo:
                batch[dest_key] = pseudo[source_key].detach().cpu()
        for key in self.transient_keys:
            batch.pop(key, None)
        t4 = time.perf_counter()
        if self.profile_freq > 0 and self._calls % self.profile_freq == 0:
            logging.info(
                "Song pointseg online profile call=%s device=%s collate_s=%.3f to_device_s=%.3f "
                "pseudo_s=%.3f cpu_copy_s=%.3f future_mask=%s",
                self._calls,
                self.device,
                t1 - t0,
                t2 - t1,
                t3 - t2,
                t4 - t3,
                future_point_is_pad is not None,
            )
        return batch


def maybe_wrap_pointseg_cache_dataset(dataset, cache_dir_value: str | Path | None = None, policy_cfg=None):
    def maybe_online_fallback(reason: str):
        if not bool(getattr(policy_cfg, "pointseg_enable", False)):
            logging.info(f"{reason}; pointseg is disabled, so no online pseudo labels are needed.")
            return dataset
        if os.environ.get("SONG_POINTSEG_ONLINE", "1").lower() in {"0", "false", "no"}:
            logging.info(f"{reason}; online pointseg pseudo labels are disabled by SONG_POINTSEG_ONLINE=0.")
            return dataset
        root = Path(getattr(dataset, "root", dataset.meta.root))
        point_cloud_dir = root / "point_clouds"
        if not point_cloud_dir.is_dir():
            logging.info(f"{reason}; point cloud dir not found at {point_cloud_dir}, using fallback point cloud loader.")
            return dataset
        mmap_mode = os.environ.get("SONG_POINTCLOUD_MMAP_MODE", "r")
        logging.info(
            f"{reason}; computing Song pointseg pseudo labels online from {point_cloud_dir}. "
            "This matches the offline cache supervision but is much slower."
        )
        return OnlinePointSegPseudoDataset(
            dataset,
            point_cloud_dir=point_cloud_dir,
            policy_cfg=policy_cfg,
            mmap_mode=mmap_mode,
        )

    if cache_dir_value is None:
        cache_dir_value = ""
    cache_dir_value = str(cache_dir_value).strip()
    if not cache_dir_value or cache_dir_value.lower() in {"0", "false", "none"}:
        return maybe_online_fallback("Song pointseg cache is disabled")

    cache_dir = Path(cache_dir_value)
    manifest = cache_dir / "manifest.json"
    if not manifest.exists():
        return maybe_online_fallback(f"Song pointseg cache not found at {cache_dir}")

    strict = os.environ.get("SONG_POINTSEG_CACHE_STRICT", "1") != "0"
    root = Path(getattr(dataset, "root", dataset.meta.root))
    point_cloud_dir = root / "point_clouds"
    mmap_mode = os.environ.get("SONG_POINTCLOUD_MMAP_MODE", "r")
    logging.info(f"Injecting Song pointseg temporal cache from {cache_dir}")
    return PointSegCacheInjectedDataset(
        dataset,
        cache_dir=cache_dir,
        point_cloud_dir=point_cloud_dir,
        strict=strict,
        mmap_mode=mmap_mode,
    )


def maybe_wrap_point_cloud_memmap_dataset(dataset):
    if isinstance(dataset, (PointSegCacheInjectedDataset, OnlinePointSegPseudoDataset)):
        return dataset
    root = Path(getattr(dataset, "root", dataset.meta.root))
    point_cloud_dir = root / "point_clouds"
    if not point_cloud_dir.is_dir():
        return dataset
    logging.info(f"Loading point clouds from per-episode memmap files in {point_cloud_dir}")
    mmap_mode = os.environ.get("SONG_POINTCLOUD_MMAP_MODE", "r")
    return PointCloudMemmapDataset(dataset, point_cloud_dir=point_cloud_dir, mmap_mode=mmap_mode)


def _find_wrapped_dataset(dataset, cls):
    current = dataset
    while current is not None:
        if isinstance(current, cls):
            return current
        next_dataset = getattr(current, "dataset", None)
        if next_dataset is None or next_dataset is current:
            return None
        current = next_dataset
    return None


def make_song_train_collate_fn(dataset):
    online = _find_wrapped_dataset(dataset, OnlinePointSegPseudoDataset)
    if online is not None:
        return online.make_collate_fn()
    return song_pointseg_collate


def maybe_wrap_worldflow_dataset(dataset, policy_cfg):
    if not bool(getattr(policy_cfg, "worldflow_enable", False)):
        return dataset
    root = Path(getattr(dataset, "root", dataset.meta.root))
    mmap_mode = os.environ.get("SONG_WORLDFLOW_MMAP_MODE", os.environ.get("SONG_POINTCLOUD_MMAP_MODE", "r"))
    logging.info(f"Injecting worldflow supervision from {root}")
    return WorldFlowMemmapDataset(
        dataset,
        root=root,
        chunk_size=int(getattr(policy_cfg, "chunk_size", 32)),
        action_start_offset=int(getattr(policy_cfg, "action_chunk_start_offset", 0)),
        require_action_target_sidecar=bool(
            getattr(policy_cfg, "worldflow_require_action_target_sidecar", False)
        ),
        mmap_mode=mmap_mode,
    )


def visualize_res(batch, result, batch_idx=0, ood_test_sno=0, step=0, output_dir: str | Path | None = None):
    import numpy as np
    import open3d as o3d

    # ===== rot6d → rotation matrix =====
    def rot6d_to_matrix(rot6d):
        a1 = rot6d[..., 0:3]
        a2 = rot6d[..., 3:6]

        b1 = a1 / np.linalg.norm(a1, axis=-1, keepdims=True)
        b2 = a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1
        b2 = b2 / np.linalg.norm(b2, axis=-1, keepdims=True)
        b3 = np.cross(b1, b2)

        return np.stack([b1, b2, b3], axis=-1)

    def create_frame(position, rot_matrix, scale=0.03):
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=scale,
            origin=[0, 0, 0]
        )
        frame.rotate(rot_matrix, center=np.zeros(3))
        frame.translate(position)
        return frame

    geometries = []

    # ================= GT =================
    gt_action = batch['action'][batch_idx].cpu().numpy()
    gt_xyz = gt_action[:, :3]
    gt_rot6d = gt_action[:, 3:9]
    gt_rotmat = rot6d_to_matrix(gt_rot6d)

    for i in range(len(gt_xyz)):
        frame = create_frame(gt_xyz[i], gt_rotmat[i], scale=0.05)
        geometries.append(frame)

    # ================= Pred =================
    pred_action = result[batch_idx].cpu().numpy()
    pred_xyz = pred_action[:, :3]
    pred_rot6d = pred_action[:, 3:9]
    pred_rotmat = rot6d_to_matrix(pred_rot6d)

    for i in range(len(pred_xyz)):
        frame = create_frame(pred_xyz[i], pred_rotmat[i], scale=0.03)
        geometries.append(frame)

    # ================= Scene Point Cloud =================
    point_cloud_value = batch["observation.point_cloud"]
    if point_cloud_value.ndim == 4:
        cloud = point_cloud_value[batch_idx][0].cpu().numpy()
    else:
        cloud = point_cloud_value[batch_idx].cpu().numpy()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(cloud[:, :3])
    pcd.colors = o3d.utility.Vector3dVector(cloud[:, 3:] / 255)
    geometries.append(pcd)

    # ================= 转换并合并 =================
    all_points = []
    all_colors = []

    # 1. 添加场景点云的点和颜色
    scene_points = np.asarray(pcd.points)
    scene_colors = np.asarray(pcd.colors)
    all_points.append(scene_points)
    all_colors.append(scene_colors)

    # 2. 将每个坐标轴网格采样为点云
    for frame in geometries:
        if isinstance(frame, o3d.geometry.TriangleMesh):
            # 从网格中采样点
            frame_pcd = frame.sample_points_poisson_disk(number_of_points=100) # 每个坐标轴采样100个点
            all_points.append(np.asarray(frame_pcd.points))
            all_colors.append(np.asarray(frame_pcd.colors))

    # 3. 合并所有点和颜色
    final_points = np.vstack(all_points)
    final_colors = np.vstack(all_colors)

    # 4. 创建最终的点云对象
    final_pcd = o3d.geometry.PointCloud()
    final_pcd.points = o3d.utility.Vector3dVector(final_points)
    final_pcd.colors = o3d.utility.Vector3dVector(final_colors)


    # 5. 保存
    vis_dir = Path(output_dir or "/home/liusong/ProgramFiles/Huggingface/lerobot/outputs/train/my_smolvla_song")
    vis_dir = vis_dir / "visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)
    ply_save_path = vis_dir / f"step{step}_{ood_test_sno}.ply"
    if o3d.io.write_point_cloud(str(ply_save_path), final_pcd):
        print(f"合并后的点云已保存为 {ply_save_path}")
    else:
        logging.warning(f"合并后的点云保存失败: {ply_save_path}")

    # o3d.visualization.draw_geometries(geometries)


def _unwrap_policy_module(policy: PreTrainedPolicy) -> PreTrainedPolicy:
    while hasattr(policy, "module"):
        policy = policy.module
    return policy


@torch.no_grad()
def save_joint_pointseg_visualization(
    policy: PreTrainedPolicy,
    batch: dict[str, torch.Tensor],
    *,
    step: int,
    output_dir: str | Path | None = None,
    tag: str = "train",
    threshold: float = 0.5,
    max_items: int = 2,
) -> None:
    """Save foreground/background masks produced by the joint pointseg branch."""
    raw_policy = _unwrap_policy_module(policy)
    model = getattr(raw_policy, "model", None)
    conditioner = getattr(model, "pointseg_conditioner", None)
    if conditioner is None:
        return

    point_cloud_payloads, _ = raw_policy.prepare_point_clouds(batch)
    payload = point_cloud_payloads[0]
    if not isinstance(payload, dict):
        return

    conditioned = conditioner(payload)
    point_cloud = payload["point_cloud"].detach().float().cpu().numpy()
    operation_prob = conditioned["operation_prob"].detach().float().cpu().numpy()
    selection_scores = conditioned["pointseg_selection_scores"].detach().float().cpu().numpy()
    point_is_pad = payload.get("point_is_pad")
    if torch.is_tensor(point_is_pad):
        point_is_pad = point_is_pad.detach().bool().cpu().numpy()
    vis_dir = Path(output_dir or "/home/liusong/ProgramFiles/Huggingface/lerobot/outputs/train/my_smolvla_song")
    vis_dir = vis_dir / "visualizations" / "pointseg"
    vis_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    skipped_without_split = 0
    for batch_idx in range(point_cloud.shape[0]):
        valid = ~point_is_pad[batch_idx] if point_is_pad is not None else np.ones(
            point_cloud.shape[1], dtype=bool
        )
        if not np.any(valid):
            skipped_without_split += 1
            continue

        current_point_cloud = point_cloud[batch_idx][valid]
        probs = operation_prob[batch_idx][valid]
        scores = selection_scores[batch_idx][valid]
        n_points = probs.shape[0]
        labels_threshold = (probs >= threshold).astype(np.int64)
        threshold_foreground_count = int(np.count_nonzero(labels_threshold == ROLE_FOREGROUND))
        if threshold_foreground_count == 0 or threshold_foreground_count == n_points:
            skipped_without_split += 1
            continue

        foreground_count = min(
            n_points,
            conditioner._target_count(n_points, conditioner.foreground_ratio, conditioner.min_foreground_points),
        )
        if foreground_count >= n_points:
            skipped_without_split += 1
            continue

        labels_topk = np.zeros(n_points, dtype=np.int64)
        topk_idx = np.argpartition(-scores, foreground_count - 1)[:foreground_count]
        labels_topk[topk_idx] = ROLE_FOREGROUND

        stem = f"{tag}_step{step}_b{batch_idx}"
        write_role_ply(vis_dir / f"{stem}_thr{threshold:.2f}.ply", current_point_cloud, labels_threshold, probs)
        write_role_ply(vis_dir / f"{stem}_topk.ply", current_point_cloud, labels_topk, probs)
        np.savez_compressed(
            vis_dir / f"{stem}.npz",
            point_cloud=current_point_cloud,
            operation_prob=probs,
            selection_scores=scores,
            labels_threshold=labels_threshold,
            labels_topk=labels_topk,
            foreground_count=np.asarray(foreground_count, dtype=np.int64),
            threshold=np.asarray(threshold, dtype=np.float32),
        )
        saved += 1
        if saved >= max_items:
            break

    if saved:
        logging.info(f"Joint pointseg visualization saved to {vis_dir} ({tag}, step {step}, {saved} item(s))")
    elif skipped_without_split:
        logging.info(
            "Skipped joint pointseg visualization (%s, step %s): no prediction contained both foreground and background.",
            tag,
            step,
        )


def ood_case_inference(
    policy,
    preprocessor,
    postprocessor,
    batch,
    step,
    output_dir: str | Path | None = None,
    ood_num_points: int = 10000,
    ood_tasks: dict[int, str] | list[str] | tuple[str, ...] | None = None,
):
    ######OOD task may differ from the training batch, so rebuild language tokens with processor.
    import open3d as o3d
    ood_num_points = int(os.environ.get("SONG_OOD_NUM_POINTS", str(ood_num_points)))
    if ood_num_points <= 0:
        raise ValueError(f"ood_num_points should be positive, got {ood_num_points}.")

    def clone_first_batch_item(src: dict[str, Any]) -> dict[str, Any]:
        cloned = {}
        pc = src.get("observation.point_cloud")
        batch_size = int(pc.shape[0]) if torch.is_tensor(pc) and pc.ndim >= 3 else 1
        for key, value in src.items():
            if torch.is_tensor(value):
                if value.ndim > 0 and int(value.shape[0]) == batch_size:
                    cloned[key] = value[:1].clone()
                else:
                    cloned[key] = value.clone()
            elif isinstance(value, list):
                cloned[key] = [value[0]] if len(value) == batch_size and batch_size > 0 else list(value)
            elif isinstance(value, tuple):
                cloned[key] = (value[0],) if len(value) == batch_size and batch_size > 0 else tuple(value)
            elif isinstance(value, dict):
                cloned[key] = dict(value)
            else:
                cloned[key] = value
        return cloned

    def random_repeat_sample_points(xyzrgb: np.ndarray, M: int, rng: np.random.Generator):
        N = xyzrgb.shape[0]
        if N == 0:
            return xyzrgb
        if N >= M:
            idx = rng.choice(N, M, replace=False)
            return xyzrgb[idx]
        extra = rng.choice(N, M - N, replace=True)
        return np.concatenate([xyzrgb, xyzrgb[extra]], axis=0)

    def load_ply_xyzrgb(path: Path) -> np.ndarray | None:
        if not path.exists():
            logging.warning(f"OOD ply file is missing: {path}")
            return None
        scene_pcd = o3d.io.read_point_cloud(str(path))
        points = np.asarray(scene_pcd.points, dtype=np.float32)
        if points.size == 0:
            logging.warning(f"OOD ply file has no points: {path}")
            return None
        colors = np.asarray(scene_pcd.colors, dtype=np.float32)
        if colors.shape != points.shape:
            colors = np.zeros_like(points, dtype=np.float32)
        elif colors.max(initial=0.0) <= 1.0:
            colors = colors * 255.0
        colors = np.clip(colors, 0.0, 255.0)
        return np.concatenate((points, colors), axis=1).astype(np.float32, copy=False)

    def load_ood_task(sno: int) -> str:
        if isinstance(ood_tasks, dict) and sno in ood_tasks:
            task = ood_tasks[sno]
        elif isinstance(ood_tasks, (list, tuple)) and 0 <= sno - 1 < len(ood_tasks):
            task = ood_tasks[sno - 1]
        else:
            task_path = Path(f"/home/liusong/temp/ood_test_new{sno}.txt")
            if task_path.exists():
                task = task_path.read_text(encoding="utf-8").strip()
            else:
                task = os.environ.get("SONG_OOD_TASK", 'Place the Red Cube on the Blue Cube\n')
        task = str(task).strip()
        return task if task.endswith("\n") else f"{task}\n"

    def set_ood_point_cloud(dst_batch: dict[str, Any], scene_tensor: torch.Tensor) -> None:
        point_cloud_value = dst_batch["observation.point_cloud"]
        if point_cloud_value.ndim == 4:
            dst_batch["observation.point_cloud"] = scene_tensor.unsqueeze(0).unsqueeze(0)
            dst_batch["observation.point_cloud_is_pad"] = torch.zeros(
                1, 1, scene_tensor.shape[0], dtype=torch.bool, device=scene_tensor.device
            )
        elif point_cloud_value.ndim == 3:
            dst_batch["observation.point_cloud"] = scene_tensor.unsqueeze(0)
            dst_batch["observation.point_cloud_is_pad"] = torch.zeros(
                1, scene_tensor.shape[0], dtype=torch.bool, device=scene_tensor.device
            )
        elif point_cloud_value.ndim == 2:
            dst_batch["observation.point_cloud"] = scene_tensor
            dst_batch["observation.point_cloud_is_pad"] = torch.zeros(
                scene_tensor.shape[0], dtype=torch.bool, device=scene_tensor.device
            )
        else:
            raise ValueError(f"Expected observation.point_cloud ndim 2/3/4, got {point_cloud_value.shape}")

    def remove_stale_pointseg_fields(dst_batch: dict[str, Any]) -> None:
        for key in list(dst_batch):
            if key.startswith("pointseg."):
                del dst_batch[key]

    def remove_language_token_fields(dst_batch: dict[str, Any]) -> None:
        dst_batch.pop("observation.language.tokens", None)
        dst_batch.pop("observation.language.attention_mask", None)

    def make_identity_pose_action_like(action: torch.Tensor) -> torch.Tensor:
        identity = torch.zeros_like(action)
        if identity.shape[-1] < 9:
            raise ValueError(f"Expected pose9 action with last dim >= 9, got {tuple(identity.shape)}")
        identity[..., 3] = 1.0
        identity[..., 7] = 1.0
        if action.shape[-1] > 9:
            identity[..., 9:] = action[..., 9:]
        return identity

    result_dir = Path(output_dir or "/home/liusong/ProgramFiles/Huggingface/lerobot/outputs/train/my_smolvla_song")
    result_dir = result_dir / "visualizations" / "ood"
    result_dir.mkdir(parents=True, exist_ok=True)

    results = []
    ood_test_sno = list(range(1,7))
    for sno in ood_test_sno:
        ply_path = Path(f"/home/liusong/temp/ood_test_new{sno}.ply")
        scene_point_cloud = load_ply_xyzrgb(ply_path)
        if scene_point_cloud is None:
            continue

        ood_batch = clone_first_batch_item(batch)
        if torch.is_tensor(ood_batch.get("action")):
            ood_batch["action"] = make_identity_pose_action_like(ood_batch["action"])
        ood_batch["task"] = [load_ood_task(sno)]

        point_cloud_value = ood_batch["observation.point_cloud"]
        rng = np.random.default_rng(1000 + int(step) * 31 + sno)
        scene_point_cloud = random_repeat_sample_points(scene_point_cloud, int(ood_num_points), rng)
        scene_tensor = torch.tensor(scene_point_cloud, device=point_cloud_value.device, dtype=point_cloud_value.dtype)
        set_ood_point_cloud(ood_batch, scene_tensor)
        remove_stale_pointseg_fields(ood_batch)
        remove_language_token_fields(ood_batch)

        model_batch = preprocessor(ood_batch)
        remove_stale_pointseg_fields(model_batch)

        action_chunk = policy.predict_action_chunk(model_batch)
        action_chunk = postprocessor(action_chunk)
        visualize_res(ood_batch, action_chunk, ood_test_sno=sno, step=step, output_dir=output_dir)
        save_joint_pointseg_visualization(
            policy,
            model_batch,
            step=step,
            output_dir=output_dir,
            tag=f"ood{sno}",
            max_items=1,
        )
        npz_path = result_dir / f"step{step}_ood{sno}.npz"
        np.savez_compressed(
            npz_path,
            source_ply=np.asarray(str(ply_path)),
            task=np.asarray(ood_batch["task"][0]),
            point_cloud=scene_point_cloud,
            action=action_chunk[0].detach().cpu().numpy(),
        )
        results.append(
            {
                "sno": sno,
                "source_ply": str(ply_path),
                "result_npz": str(npz_path),
                "merged_ply": str(Path(output_dir or "/home/liusong/ProgramFiles/Huggingface/lerobot/outputs/train/my_smolvla_song") / "visualizations" / f"step{step}_{sno}.ply"),
            }
        )
    return results



def random_repeat_sample_points(xyzrgb: np.ndarray, M: int):
    N = xyzrgb.shape[0]
    if N == 0:
        return xyzrgb
    if N >= M:
        idx = np.random.choice(N, M, replace=False)
        return xyzrgb[idx]
    else:
        extra = np.random.choice(N, M - N, replace=True)
        return np.concatenate([xyzrgb, xyzrgb[extra]], axis=0)  
def count_parameters(module: torch.nn.Module, only_trainable: bool = False) -> int:
    skipped = 0
    total = 0
    for p in module.parameters():
        if only_trainable and not p.requires_grad:
            continue
        if getattr(p, "is_uninitialized", False):
            skipped += 1
            continue
        try:
            total += p.numel()
        except (ValueError, RuntimeError):
            skipped += 1
    if skipped > 0:
        logging.warning(
            f"Skipped {skipped} uninitialized parameters while counting {module.__class__.__name__}. "
            "Lazy modules will initialize on first forward."
        )
    return total


def ensure_ddp_parameters_initialized(module: torch.nn.Module, accelerator: Accelerator) -> None:
    """Fail before Accelerator/DDP wrapping and report the exact lazy parameters."""

    uninitialized = []
    for name, parameter in module.named_parameters():
        if isinstance(parameter, torch.nn.parameter.UninitializedParameter) or getattr(
            parameter, "is_uninitialized", False
        ):
            uninitialized.append(name)
    if not uninitialized:
        return

    details = ", ".join(uninitialized)
    message = (
        "Policy still contains uninitialized parameters before distributed wrapping: "
        f"{details}. Replace Lazy modules with explicit input dimensions or initialize them before "
        "creating the optimizer and calling accelerator.prepare()."
    )
    if accelerator.num_processes > 1:
        raise RuntimeError(message)
    logging.warning(message)


def update_policy(
    train_metrics: MetricsTracker,
    policy: PreTrainedPolicy,
    batch: Any,
    optimizer: Optimizer,
    grad_clip_norm: float,
    accelerator: Accelerator,
    lr_scheduler=None,
    lock=None,
    rabc_weights_provider=None,
) -> tuple[MetricsTracker, dict]:
    """
    Performs a single training step to update the policy's weights.

    This function executes the forward and backward passes, clips gradients, and steps the optimizer and
    learning rate scheduler. Accelerator handles mixed-precision training automatically.

    Args:
        train_metrics: A MetricsTracker instance to record training statistics.
        policy: The policy model to be trained.
        batch: A batch of training data.
        optimizer: The optimizer used to update the policy's parameters.
        grad_clip_norm: The maximum norm for gradient clipping.
        accelerator: The Accelerator instance for distributed training and mixed precision.
        lr_scheduler: An optional learning rate scheduler.
        lock: An optional lock for thread-safe optimizer updates.
        rabc_weights_provider: Optional RABCWeights instance for sample weighting.

    Returns:
        A tuple containing:
        - The updated MetricsTracker with new statistics for this step.
        - A dictionary of outputs from the policy's forward pass, for logging purposes.
    """
    start_time = time.perf_counter()
    policy.train()

    # Get RA-BC weights if enabled
    rabc_batch_weights = None
    rabc_batch_stats = None
    if rabc_weights_provider is not None:
        rabc_batch_weights, rabc_batch_stats = rabc_weights_provider.compute_batch_weights(batch)

    # Let accelerator handle mixed precision
    with accelerator.autocast():
        # Use per-sample loss when RA-BC is enabled for proper weighting
        if rabc_batch_weights is not None:
            # Get per-sample losses
            per_sample_loss, output_dict = policy.forward(batch, reduction="none")

            # Apply RA-BC weights: L_RA-BC = Σ(w_i * l_i) / (Σw_i + ε)
            # rabc_batch_weights is already normalized to sum to batch_size
            epsilon = 1e-6
            loss = (per_sample_loss * rabc_batch_weights).sum() / (rabc_batch_weights.sum() + epsilon)
            # Log raw mean weight (before normalization) - this is the meaningful metric
            output_dict["rabc_mean_weight"] = rabc_batch_stats["raw_mean_weight"]
            output_dict["rabc_num_zero_weight"] = rabc_batch_stats["num_zero_weight"]
            output_dict["rabc_num_full_weight"] = rabc_batch_stats["num_full_weight"]
        else:
            loss, output_dict = policy.forward(batch)

        # TODO(rcadene): policy.unnormalize_outputs(out_dict)

    if not torch.isfinite(loss):
        optimizer.zero_grad(set_to_none=True)
        output_dict["skipped_nonfinite_loss"] = 1
        train_metrics.loss = loss.item()
        train_metrics.grad_norm = float("nan")
        train_metrics.lr = optimizer.param_groups[0]["lr"]
        train_metrics.update_s = time.perf_counter() - start_time
        return train_metrics, output_dict

    # Use accelerator's backward method
    accelerator.backward(loss)

    # Clip gradients if specified
    if grad_clip_norm > 0:
        grad_norm = accelerator.clip_grad_norm_(policy.parameters(), grad_clip_norm)
    else:
        grad_norm = torch.nn.utils.clip_grad_norm_(
            policy.parameters(), float("inf"), error_if_nonfinite=False
        )

    if not torch.isfinite(torch.as_tensor(grad_norm)):
        optimizer.zero_grad(set_to_none=True)
        output_dict["skipped_nonfinite_grad"] = 1
        train_metrics.loss = loss.item()
        train_metrics.grad_norm = grad_norm.item()
        train_metrics.lr = optimizer.param_groups[0]["lr"]
        train_metrics.update_s = time.perf_counter() - start_time
        return train_metrics, output_dict

    # Optimizer step
    with lock if lock is not None else nullcontext():
        optimizer.step()

    optimizer.zero_grad()

    # Step through pytorch scheduler at every batch instead of epoch
    if lr_scheduler is not None:
        lr_scheduler.step()

    # Update internal buffers if policy has update method
    if has_method(accelerator.unwrap_model(policy, keep_fp32_wrapper=True), "update"):
        accelerator.unwrap_model(policy, keep_fp32_wrapper=True).update()

    train_metrics.loss = loss.item()
    train_metrics.grad_norm = grad_norm.item()
    train_metrics.lr = optimizer.param_groups[0]["lr"]
    train_metrics.update_s = time.perf_counter() - start_time
    return train_metrics, output_dict




@parser.wrap()
def train(cfg: TrainPipelineConfig, accelerator: Accelerator | None = None):
    """
    Main function to train a policy.

    This function orchestrates the entire training pipeline, including:
    - Setting up logging, seeding, and device configuration.
    - Creating the dataset, evaluation environment (if applicable), policy, and optimizer.
    - Handling resumption from a checkpoint.
    - Running the main training loop, which involves fetching data batches and calling `update_policy`.
    - Periodically logging metrics, saving model checkpoints, and evaluating the policy.
    - Pushing the final trained model to the Hugging Face Hub if configured.

    Args:
        cfg: A `TrainPipelineConfig` object containing all training configurations.
        accelerator: Optional Accelerator instance. If None, one will be created automatically.
    """
    cfg.validate()

    # Create Accelerator if not provided
    # It will automatically detect if running in distributed mode or single-process mode
    # We set step_scheduler_with_optimizer=False to prevent accelerate from adjusting the lr_scheduler steps based on the num_processes
    # We set find_unused_parameters=True to handle models with conditional computation
    if accelerator is None:
        from accelerate.utils import DistributedDataParallelKwargs

        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        # Accelerate auto-detects the device based on the available hardware and ignores the policy.device setting.
        # Force the device to be CPU when policy.device is set to CPU.
        force_cpu = cfg.policy.device == "cpu"
        accelerator = Accelerator(
            step_scheduler_with_optimizer=False,
            kwargs_handlers=[ddp_kwargs],
            cpu=force_cpu,
        )

    init_logging(accelerator=accelerator)

    # Determine if this is the main process (for logging and checkpointing)
    # When using accelerate, only the main process should log to avoid duplicate outputs
    is_main_process = accelerator.is_main_process

    # Only log on main process
    if is_main_process:
        logging.info(pformat(cfg.to_dict()))

    # Initialize wandb only on main process
    if cfg.wandb.enable and cfg.wandb.project and is_main_process:
        wandb_logger = WandBLogger(cfg)
    else:
        wandb_logger = None
        if is_main_process:
            logging.info(colored("Logs will be saved locally.", "yellow", attrs=["bold"]))

    if cfg.seed is not None:
        set_seed(cfg.seed, accelerator=accelerator)

    # Use accelerator's device
    device = accelerator.device
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    # Dataset loading synchronization: main process downloads first to avoid race conditions
    if is_main_process:
        logging.info("Creating dataset")
        dataset = make_dataset(cfg)
        dataset = maybe_wrap_pointseg_cache_dataset(dataset, cfg.pointseg_sample_cache_dir, cfg.policy)
        dataset = maybe_wrap_point_cloud_memmap_dataset(dataset)
        dataset = maybe_wrap_worldflow_dataset(dataset, cfg.policy)

    accelerator.wait_for_everyone()

    # Now all other processes can safely load the dataset
    if not is_main_process:
        dataset = make_dataset(cfg)
        dataset = maybe_wrap_pointseg_cache_dataset(dataset, cfg.pointseg_sample_cache_dir, cfg.policy)
        dataset = maybe_wrap_point_cloud_memmap_dataset(dataset)
        dataset = maybe_wrap_worldflow_dataset(dataset, cfg.policy)

    # Create environment used for evaluating checkpoints during training on simulation data.
    # On real-world data, no need to create an environment as evaluations are done outside train.py,
    # using the eval.py instead, with gym_dora environment and dora-rs.
    eval_env = None
    if cfg.eval_freq > 0 and cfg.env is not None and is_main_process:
        logging.info("Creating env")
        eval_env = make_env(cfg.env, n_envs=cfg.eval.batch_size, use_async_envs=cfg.eval.use_async_envs)

    if is_main_process:
        logging.info("Creating policy")
    policy = make_policy(
        cfg=cfg.policy,
        ds_meta=dataset.meta,
        rename_map=cfg.rename_map,
    )
    if is_main_process and hasattr(policy.config, "flow_contract_summary"):
        logging.info("Resolved flow contract: %s", policy.config.flow_contract_summary())

    if cfg.peft is not None:
        logging.info("Using PEFT! Wrapping model.")
        # Convert CLI peft config to dict for overrides
        peft_cli_overrides = dataclasses.asdict(cfg.peft)
        policy = policy.wrap_with_peft(peft_cli_overrides=peft_cli_overrides)

    ensure_ddp_parameters_initialized(policy, accelerator)

    # Wait for all processes to finish policy creation before continuing
    accelerator.wait_for_everyone()

    # Create processors - only provide dataset_stats if not resuming from saved processors
    processor_kwargs = {}
    postprocessor_kwargs = {}


    if (cfg.policy.pretrained_path and not cfg.resume) or not cfg.policy.pretrained_path:
        # Only provide dataset_stats when not resuming from saved processor state
        processor_kwargs["dataset_stats"] = dataset.meta.stats
    # For SARM, always provide dataset_meta for progress normalization
    if cfg.policy.type == "sarm":
        processor_kwargs["dataset_meta"] = dataset.meta
    if cfg.policy.pretrained_path is not None:
        processor_kwargs["preprocessor_overrides"] = {
            "device_processor": {"device": device.type},
            "normalizer_processor": {
                "stats": dataset.meta.stats,
                "features": {**policy.config.input_features, **policy.config.output_features},
                "norm_map": policy.config.normalization_mapping,
            },
        }
        processor_kwargs["preprocessor_overrides"]["rename_observations_processor"] = {
            "rename_map": cfg.rename_map
        }
        postprocessor_kwargs["postprocessor_overrides"] = {
            "unnormalizer_processor": {
                "stats": dataset.meta.stats,
                "features": policy.config.output_features,
                "norm_map": policy.config.normalization_mapping,
            },
        }

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg.policy,
        pretrained_path=cfg.policy.pretrained_path,
        **processor_kwargs,
        **postprocessor_kwargs,
    )
    if bool(getattr(policy.config, "worldflow_enable", False)):
        validate_smolvla_worldflow_preprocessor(preprocessor)

    if is_main_process:
        logging.info("Creating optimizer and scheduler")
    optimizer, lr_scheduler = make_optimizer_and_scheduler(cfg, policy)

    # Load precomputed SARM progress for RA-BC if enabled
    # Generate progress using: src/lerobot/policies/sarm/compute_rabc_weights.py
    rabc_weights = None
    if cfg.use_rabc:
        from lerobot.utils.rabc import RABCWeights

        # Get chunk_size from policy config
        chunk_size = getattr(policy.config, "chunk_size", None)
        if chunk_size is None:
            raise ValueError("Chunk size is not found in policy config")

        head_mode = getattr(cfg, "rabc_head_mode", "sparse")
        logging.info(f"Loading SARM progress for RA-BC from {cfg.rabc_progress_path}")
        logging.info(f"Using chunk_size={chunk_size} from policy config, head_mode={head_mode}")
        rabc_weights = RABCWeights(
            progress_path=cfg.rabc_progress_path,
            chunk_size=chunk_size,
            head_mode=head_mode,
            kappa=getattr(cfg, "rabc_kappa", 0.01),
            epsilon=getattr(cfg, "rabc_epsilon", 1e-6),
            device=device,
        )

    step = 0  # number of policy updates (forward + backward + optim)

    if cfg.resume:
        step, optimizer, lr_scheduler = load_training_state(cfg.checkpoint_path, optimizer, lr_scheduler)

    num_learnable_params = count_parameters(policy, only_trainable=True)
    num_total_params = count_parameters(policy, only_trainable=False)


    if is_main_process:
        logging.info(colored("Output dir:", "yellow", attrs=["bold"]) + f" {cfg.output_dir}")
        if cfg.env is not None:
            logging.info(f"{cfg.env.task=}")
            logging.info("Creating environment processors")
            env_preprocessor, env_postprocessor = make_env_pre_post_processors(
                env_cfg=cfg.env, policy_cfg=cfg.policy
            )
        logging.info(f"{cfg.steps=} ({format_big_number(cfg.steps)})")
        logging.info(f"{dataset.num_frames=} ({format_big_number(dataset.num_frames)})")
        logging.info(f"{dataset.num_episodes=}")
        num_processes = accelerator.num_processes
        effective_bs = cfg.batch_size * num_processes
        logging.info(f"Effective batch size: {cfg.batch_size} x {num_processes} = {effective_bs}")
        logging.info(f"{num_learnable_params=} ({format_big_number(num_learnable_params)})")
        logging.info(f"{num_total_params=} ({format_big_number(num_total_params)})")

    # create dataloader for offline training
    if hasattr(cfg.policy, "drop_n_last_frames"):
        shuffle = False
        sampler = EpisodeAwareSampler(
            dataset.meta.episodes["dataset_from_index"],
            dataset.meta.episodes["dataset_to_index"],
            episode_indices_to_use=dataset.episodes,
            drop_n_last_frames=cfg.policy.drop_n_last_frames,
            shuffle=True,
        )
    else:
        shuffle = True
        sampler = None

    collate_fn = make_song_train_collate_fn(dataset)
    dataloader_num_workers = int(cfg.num_workers)
    if is_main_process and isinstance(collate_fn, OnlinePointSegBatchCollator):
        logging.info("Song pointseg online pseudo labels will be computed once per DataLoader batch.")
    if isinstance(collate_fn, OnlinePointSegBatchCollator) and collate_fn.device.type == "cuda" and dataloader_num_workers > 0:
        if is_main_process:
            logging.warning(
                "Song pointseg online pseudo labels use CUDA; setting DataLoader num_workers=0 "
                "to avoid CUDA initialization in forked worker processes."
            )
        dataloader_num_workers = 0

    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=dataloader_num_workers,
        batch_size=cfg.batch_size,
        shuffle=shuffle and not cfg.dataset.streaming,
        sampler=sampler,
        pin_memory=device.type == "cuda",
        drop_last=False,
        prefetch_factor=2 if dataloader_num_workers > 0 else None,
        persistent_workers=dataloader_num_workers > 0,
        collate_fn=collate_fn,
    )

    # Prepare everything with accelerator
    accelerator.wait_for_everyone()
    policy, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        policy, optimizer, dataloader, lr_scheduler
    )
    dl_iter = cycle(dataloader)

    policy.train()

    train_metrics = {
        "loss": AverageMeter("loss", ":.3f"),
        "grad_norm": AverageMeter("grdn", ":.3f"),
        "lr": AverageMeter("lr", ":0.1e"),
        "update_s": AverageMeter("updt_s", ":.3f"),
        "dataloading_s": AverageMeter("data_s", ":.3f"),
    }

    # Keep global batch size for logging; MetricsTracker handles world size internally.
    effective_batch_size = cfg.batch_size * accelerator.num_processes
    train_tracker = MetricsTracker(
        cfg.batch_size,
        dataset.num_frames,
        dataset.num_episodes,
        train_metrics,
        initial_step=step,
        accelerator=accelerator,
    )

    if is_main_process:
        progbar = tqdm(
            total=cfg.steps - step,
            desc="Training",
            unit="step",
            disable=inside_slurm(),
            position=0,
            leave=True,
        )
        logging.info(
            f"Start offline training on a fixed dataset, with effective batch size: {effective_batch_size}"
        )

    for _ in range(step, cfg.steps):
        start_time = time.perf_counter()
        batch = next(dl_iter)
        batch = preprocessor(batch)
        train_tracker.dataloading_s = time.perf_counter() - start_time
        train_tracker, output_dict = update_policy(
            train_tracker,
            policy,
            batch,
            optimizer,
            cfg.optimizer.grad_clip_norm,
            accelerator=accelerator,
            lr_scheduler=lr_scheduler,
            rabc_weights_provider=rabc_weights,
        )

        # Note: eval and checkpoint happens *after* the `step`th training update has completed, so we
        # increment `step` here.
        step += 1
        if is_main_process:
            progbar.update(1)
        train_tracker.step()
        is_log_step = cfg.log_freq > 0 and step % cfg.log_freq == 0 and is_main_process
        is_saving_step = step % cfg.save_freq == 0 or step == cfg.steps
        is_eval_step = cfg.eval_freq > 0 and step % cfg.eval_freq == 0

        if is_log_step:
            logging.info(train_tracker)
            if output_dict:
                debug_keys = (
                    "loss_action",
                    "loss_action_translation",
                    "loss_action_rotation6d",
                    "loss_action_gripper",
                    "action_endpoint_trans_err",
                    "action_endpoint_rot_err_deg",
                    "action_endpoint_gripper_err",
                    "loss_pointseg_aux",
                    "loss_se3_twist",
                    "loss_se3_endpoint",
                    "loss_se3_gripper",
                    "loss_se3_equivariance",
                    "se3_action_trans_err",
                    "se3_action_rot_err_deg",
                    "loss_worldflow_flow",
                    "loss_worldflow_geo",
                    "loss_worldflow_bridge",
                    "loss_worldflow_equiv",
                    "worldflow_trans_err",
                    "worldflow_rot_err_deg",
                    "worldflow_valid_ratio",
                    "worldflow_foreground_points",
                    "worldflow_noise_conjugacy_error",
                    "worldflow_path_conjugacy_error",
                    "pointseg_foreground_ratio",
                    "pointseg_operation_prob_mean",
                    "pointseg_selection_score_mean",
                    "pred_operation_prob",
                    "pseudo_valid_ratio",
                    "pseudo_foreground_ratio",
                    "pred_foreground_ratio",
                )
                debug_items = []
                for key in debug_keys:
                    value = output_dict.get(key)
                    if value is not None:
                        debug_items.append(f"{key}:{float(value):.4g}")
                if debug_items:
                    logging.info(" ".join(debug_items))
            if wandb_logger:
                wandb_log_dict = train_tracker.to_dict()
                if output_dict:
                    wandb_log_dict.update(output_dict)
                # Log RA-BC statistics if enabled
                if rabc_weights is not None:
                    rabc_stats = rabc_weights.get_stats()
                    wandb_log_dict.update(
                        {
                            "rabc_delta_mean": rabc_stats["delta_mean"],
                            "rabc_delta_std": rabc_stats["delta_std"],
                            "rabc_num_frames": rabc_stats["num_frames"],
                        }
                    )
                wandb_logger.log_dict(wandb_log_dict, step)
            train_tracker.reset_averages()

        if cfg.save_checkpoint and is_saving_step:
            with torch.no_grad(), accelerator.autocast():
                save_joint_pointseg_visualization(
                    policy,
                    batch,
                    step=step,
                    output_dir=cfg.output_dir,
                    tag="train",
                    max_items=2,
                )
                try:
                    ood_case_inference(policy, preprocessor, postprocessor, batch, step, output_dir=cfg.output_dir,ood_num_points=50000)
                except Exception:
                    logging.exception("OOD case inference failed at step %s; continuing training/checkpoint save.", step)

            if is_main_process:
                logging.info(f"Checkpoint policy after step {step}")
                checkpoint_dir = get_step_checkpoint_dir(cfg.output_dir, cfg.steps, step)
                save_checkpoint(
                    checkpoint_dir=checkpoint_dir,
                    step=step,
                    cfg=cfg,
                    policy=accelerator.unwrap_model(policy),
                    optimizer=optimizer,
                    scheduler=lr_scheduler,
                    preprocessor=preprocessor,
                    postprocessor=postprocessor,
                )
                update_last_checkpoint(checkpoint_dir)
                if wandb_logger:
                    wandb_logger.log_policy(checkpoint_dir)

            accelerator.wait_for_everyone()

        if cfg.env and is_eval_step:
            if is_main_process:
                step_id = get_step_identifier(step, cfg.steps)
                logging.info(f"Eval policy at step {step}")
                with torch.no_grad(), accelerator.autocast():
                    eval_info = eval_policy_all(
                        envs=eval_env,  # dict[suite][task_id] -> vec_env
                        policy=accelerator.unwrap_model(policy),
                        env_preprocessor=env_preprocessor,
                        env_postprocessor=env_postprocessor,
                        preprocessor=preprocessor,
                        postprocessor=postprocessor,
                        n_episodes=cfg.eval.n_episodes,
                        videos_dir=cfg.output_dir / "eval" / f"videos_step_{step_id}",
                        max_episodes_rendered=4,
                        start_seed=cfg.seed,
                        max_parallel_tasks=cfg.env.max_parallel_tasks,
                    )
                # overall metrics (suite-agnostic)
                aggregated = eval_info["overall"]

                # optional: per-suite logging
                for suite, suite_info in eval_info.items():
                    logging.info("Suite %s aggregated: %s", suite, suite_info)

                # meters/tracker
                eval_metrics = {
                    "avg_sum_reward": AverageMeter("∑rwrd", ":.3f"),
                    "pc_success": AverageMeter("success", ":.1f"),
                    "eval_s": AverageMeter("eval_s", ":.3f"),
                }
                eval_tracker = MetricsTracker(
                    cfg.batch_size,
                    dataset.num_frames,
                    dataset.num_episodes,
                    eval_metrics,
                    initial_step=step,
                    accelerator=accelerator,
                )
                eval_tracker.eval_s = aggregated.pop("eval_s")
                eval_tracker.avg_sum_reward = aggregated.pop("avg_sum_reward")
                eval_tracker.pc_success = aggregated.pop("pc_success")
                if wandb_logger:
                    wandb_log_dict = {**eval_tracker.to_dict(), **eval_info}
                    wandb_logger.log_dict(wandb_log_dict, step, mode="eval")
                    wandb_logger.log_video(eval_info["overall"]["video_paths"][0], step, mode="eval")

            accelerator.wait_for_everyone()

    if is_main_process:
        progbar.close()

    if eval_env:
        close_envs(eval_env)

    if is_main_process:
        logging.info("End of training")

        if cfg.policy.push_to_hub:
            unwrapped_policy = accelerator.unwrap_model(policy)
            if cfg.policy.use_peft:
                unwrapped_policy.push_model_to_hub(cfg, peft_model=unwrapped_policy)
            else:
                unwrapped_policy.push_model_to_hub(cfg)
            preprocessor.push_to_hub(cfg.policy.repo_id)
            postprocessor.push_to_hub(cfg.policy.repo_id)

    # Properly clean up the distributed process group
    accelerator.wait_for_everyone()
    accelerator.end_training()


def main():
    register_third_party_plugins()
    train()


def _apply_song_debug_defaults() -> None:
    if len(sys.argv) > 1 and os.environ.get("SONG_TRAIN_DEBUG_DEFAULTS") != "1":
        return
    sys.argv = [
        "train_song.py",
        "--policy.path=/home/liusong/ProgramFiles/Huggingface/lerobot/benchmarks/song_real_libero/outputs/libero_setting/train_libero_fresh_post/checkpoints/last/pretrained_model",
        # "--policy.type=smolvla",
        # "--policy.repo_id=/home/liusong/scp_receive/smolvla",
        "--policy.push_to_hub=false",
        "--dataset.repo_id=/home/liusong/ProgramFiles/Huggingface/lerobot/benchmarks/song_real_libero/data/libero_setting/libero_4suite_lerobot_dataset",
        #"--pointseg_sample_cache_dir=/home/liusong/ProgramFiles/Huggingface/lerobot/benchmarks/song_real_libero/data/real_setting/real_priorseg_cache",
        "--policy.vlm_model_name=/home/liusong/SmolVLM2-500M-Video-Instruct",
        "--policy.load_vlm_weights=false",
        "--batch_size=6",
        "--steps=500000",
        "--log_freq=1",
        "--output_dir=/home/liusong/ProgramFiles/Huggingface/lerobot/benchmarks/song_real_libero/outputs/real_setting/train/temp",
        "--job_name=temp",
        "--policy.device=cuda",
        "--wandb.enable=true",
        "--wandb.disable_artifact=true",
        "--save_freq=30000",
        "--eval_freq=30000",
        "--num_workers=8",
        "--policy.pointseg_enable=true",
        "--policy.pointseg_backbone_type=litept",
        "--policy.pointseg_grid_size=0.01",
        "--policy.pointseg_feature_dim=64",
        "--policy.pointseg_aux_loss_weight=0.002",
        "--policy.pointseg_foreground_ratio=0.08",
        "--policy.pointseg_background_ratio=0.08",
        "--policy.pointseg_min_foreground_points=4000",
        "--policy.pointseg_min_background_points=0",
        "--policy.pointseg_use_temporal_priors_as_input=false",
        "--policy.pointseg_use_pseudo_selection=false",
        "--policy.worldflow_enable=false",
        "--policy.worldflow_feature_dim=64",
        "--policy.worldflow_grid_size=0.01",
        "--policy.worldflow_loss_weight=0.05",
        "--policy.worldflow_geo_loss_weight=0.05",
        "--policy.worldflow_trans_weight=1.0",
        "--policy.worldflow_rot_weight=1.0",
        "--policy.worldflow_se3_head_enable=false",
        "--policy.worldflow_equiv_loss_weight=0.02",
        "--policy.se3_enable=false",
        "--policy.se3_noise_trans_scale=0.15",
        "--policy.se3_noise_rot_scale=0.75",    
        "--policy.se3_pose_loss_weight=1.0",
        "--policy.se3_gripper_loss_weight=1.0",
        "--policy.se3_endpoint_loss_weight=0.25",
        "--policy.se3_final_correction_enable=false",
        "--policy.se3_final_correction_loss_weight=0.20",
        "--policy.se3_equivariance_loss_weight=0.02",
    ]

if __name__ == "__main__":
    _apply_song_debug_defaults()
    main()


# def random_repeat_sample_points(xyzrgb: np.ndarray, M: int):
#     N = xyzrgb.shape[0]
#     if N == 0:
#         return xyzrgb
#     if N >= M:
#         idx = np.random.choice(N, M, replace=False)
#         return xyzrgb[idx]
#     else:
#         extra = np.random.choice(N, M - N, replace=True)
#         return np.concatenate([xyzrgb, xyzrgb[extra]], axis=0)   
# batch['task'][0] = "Place the Red Cube on the Blue Cube"
# scene_pcd = o3d.io.read_point_cloud(f"/home/liusong/temp/ood_test_new4.ply",)
# scene_point_cloud = np.concatenate((np.asarray(scene_pcd.points[:]),np.asarray(scene_pcd.colors[:])*255), axis=1)
# scene_point_cloud = random_repeat_sample_points(scene_point_cloud, 10000)
# batch['observation.point_cloud'][0] = torch.tensor(scene_point_cloud).to("cuda")
# action_pred = self.predict_action_chunk(batch)
# vis_umi_data(action_pred.cpu().numpy()[0],batch['observation.point_cloud'].cpu().numpy()[0])
# print(action_pred[0][:,-1])
