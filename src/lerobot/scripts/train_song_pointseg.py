#!/usr/bin/env python

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

os.environ.setdefault("HF_DATASETS_CACHE", "/tmp/lerobot_hf_datasets_cache")

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.policies.smolvla.song_pointseg import (
    DEFAULT_FUTURE_OFFSETS,
    EMATeacher,
    PseudoLabelConfig,
    SongPointSegCachedDataset,
    SongPointSegLoss,
    SongPointSegLossConfig,
    SongPointSegNet,
    SongTemporalPointCloudDataset,
    generate_pseudo_labels,
    move_batch_to_device,
    open_episode_point_clouds,
    parse_future_offsets,
    pretty_metrics,
    refine_pseudo_labels_with_teacher,
    save_pointseg_config,
    save_pointseg_npz,
    song_pointseg_collate,
    write_role_ply,
)
from lerobot.utils.random_utils import set_seed

DEFAULT_DATASET_ROOT = Path(
    os.environ.get(
        "SONG_POINTSEG_DATASET",
        "/home/liusong/ProgramFiles/Huggingface/lerobot/benchmarks/song_real_libero/data/real_setting/real_lerobot_dataset",
    )
)
DEFAULT_OUTPUT_DIR = Path(
    os.environ.get(
        "SONG_POINTSEG_OUTPUT_DIR",
        "/home/liusong/ProgramFiles/Huggingface/lerobot/benchmarks/song_real_libero/outputs/real_setting/train/pointseg/song_pointseg",
    )
)
import math
from collections import deque
from contextlib import contextmanager
from pathlib import Path
from typing import TypedDict

import numpy as np
import open3d as o3d
import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn
from typing_extensions import Unpack
def create_frame(position, rot_matrix, scale=0.03):
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=scale,
        origin=[0, 0, 0]
    )
    frame.rotate(rot_matrix, center=np.zeros(3))
    frame.translate(position)
    return frame
def rot6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    a1 = d6[..., 0:3]
    a2 = d6[..., 3:6]
    b1 = F.normalize(a1, dim=-1, eps=1e-6)
    b2 = a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1, eps=1e-6)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-1)  # columns


def _skew(vec: Tensor) -> Tensor:
    x, y, z = vec.unbind(dim=-1)
    zeros = torch.zeros_like(x)
    return torch.stack(
        [
            torch.stack([zeros, -z, y], dim=-1),
            torch.stack([z, zeros, -x], dim=-1),
            torch.stack([-y, x, zeros], dim=-1),
        ],
        dim=-2,
    )


def _eye4_like(shape: torch.Size | tuple[int, ...], *, device: torch.device, dtype: torch.dtype) -> Tensor:
    eye = torch.eye(4, device=device, dtype=dtype)
    return eye.expand(*shape, 4, 4).clone()


def so3_exp(omega: Tensor, eps: float = 1e-6) -> Tensor:
    omega = omega.to(dtype=torch.float32)
    theta = torch.linalg.norm(omega, dim=-1, keepdim=True)
    theta2 = theta * theta
    k = _skew(omega)
    k2 = k @ k
    small = theta < eps
    a = torch.where(
        small,
        1.0 - theta2 / 6.0 + theta2 * theta2 / 120.0,
        torch.sin(theta) / theta.clamp_min(eps),
    )
    b = torch.where(
        small,
        0.5 - theta2 / 24.0 + theta2 * theta2 / 720.0,
        (1.0 - torch.cos(theta)) / theta2.clamp_min(eps),
    )
    eye = torch.eye(3, device=omega.device, dtype=omega.dtype).expand(*omega.shape[:-1], 3, 3)
    return eye + a.unsqueeze(-1) * k + b.unsqueeze(-1) * k2


def so3_log(rot: Tensor, eps: float = 1e-6) -> Tensor:
    rot = rot.to(dtype=torch.float32)
    trace = rot.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    cosine = ((trace - 1.0) * 0.5).clamp(-1.0, 1.0)
    vee = torch.stack(
        [
            rot[..., 2, 1] - rot[..., 1, 2],
            rot[..., 0, 2] - rot[..., 2, 0],
            rot[..., 1, 0] - rot[..., 0, 1],
        ],
        dim=-1,
    )
    sine = 0.5 * torch.linalg.norm(vee, dim=-1)
    theta = torch.atan2(sine, cosine)
    theta2 = theta * theta
    factor = torch.where(
        sine > eps,
        theta / (2.0 * sine.clamp_min(eps)),
        0.5 + theta2 / 12.0 + theta2 * theta2 / 720.0,
    )
    return factor.unsqueeze(-1) * vee


def se3_exp(xi: Tensor, eps: float = 1e-6) -> Tensor:
    xi = xi.to(dtype=torch.float32)
    v = xi[..., :3]
    omega = xi[..., 3:6]
    rot = so3_exp(omega, eps=eps)
    theta = torch.linalg.norm(omega, dim=-1, keepdim=True)
    theta2 = theta * theta
    k = _skew(omega)
    k2 = k @ k
    small = theta < eps
    a = torch.where(
        small,
        0.5 - theta2 / 24.0 + theta2 * theta2 / 720.0,
        (1.0 - torch.cos(theta)) / theta2.clamp_min(eps),
    )
    b = torch.where(
        small,
        1.0 / 6.0 - theta2 / 120.0 + theta2 * theta2 / 5040.0,
        (theta - torch.sin(theta)) / (theta2 * theta).clamp_min(eps),
    )
    eye3 = torch.eye(3, device=xi.device, dtype=xi.dtype).expand(*xi.shape[:-1], 3, 3)
    v_matrix = eye3 + a.unsqueeze(-1) * k + b.unsqueeze(-1) * k2
    trans = (v_matrix @ v.unsqueeze(-1)).squeeze(-1)
    out = _eye4_like(xi.shape[:-1], device=xi.device, dtype=xi.dtype)
    out[..., :3, :3] = rot
    out[..., :3, 3] = trans
    return out


def se3_log(transform: Tensor, eps: float = 1e-6) -> Tensor:
    transform = transform.to(dtype=torch.float32)
    rot = transform[..., :3, :3]
    trans = transform[..., :3, 3]
    omega = so3_log(rot, eps=eps)
    theta = torch.linalg.norm(omega, dim=-1, keepdim=True)
    theta2 = theta * theta
    k = _skew(omega)
    k2 = k @ k
    small = theta < eps
    half_theta = 0.5 * theta
    c = torch.where(
        small,
        1.0 / 12.0 + theta2 / 720.0 + theta2 * theta2 / 30240.0,
        (1.0 / theta2.clamp_min(eps))
        - (1.0 + torch.cos(theta)) / (2.0 * theta * torch.sin(theta).clamp_min(eps)),
    )
    eye3 = torch.eye(3, device=transform.device, dtype=transform.dtype).expand(*transform.shape[:-2], 3, 3)
    v_inv = eye3 - 0.5 * k + c.unsqueeze(-1) * k2
    v = (v_inv @ trans.unsqueeze(-1)).squeeze(-1)
    # `half_theta` is kept to make the small-angle branch explicit and silence over-eager simplifiers.
    _ = half_theta
    return torch.cat([v, omega], dim=-1)


def se3_left_apply(delta_xi: Tensor, transform: Tensor) -> Tensor:
    return se3_exp(delta_xi) @ transform


def se3_geodesic_loss(pred: Tensor, target: Tensor, trans_weight: float = 1.0, rot_weight: float = 1.0) -> Tensor:
    trans = F.smooth_l1_loss(pred[..., :3, 3], target[..., :3, 3], reduction="none").sum(dim=-1)
    rot = _rotation_geodesic(pred[..., :3, :3], target[..., :3, :3])
    return trans_weight * trans + rot_weight * rot


def _transform_point_cloud_xyzrgb(point_cloud: Tensor, transform: Tensor) -> Tensor:
    xyz = point_cloud[..., :3].to(dtype=torch.float32)
    rot = transform[..., :3, :3]
    trans = transform[..., :3, 3]
    xyz_out = torch.matmul(xyz, rot.transpose(-1, -2)) + trans.unsqueeze(-2)
    return torch.cat([xyz_out, point_cloud[..., 3:6].to(dtype=torch.float32)], dim=-1)


def _sample_random_se3(
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    *,
    trans_scale: float = 0.20,
    rot_scale: float = 0.75,
) -> Tensor:
    xi = torch.randn(batch_size, 6, device=device, dtype=dtype)
    xi[..., :3] = xi[..., :3] * float(trans_scale)
    xi[..., 3:6] = xi[..., 3:6] * float(rot_scale)
    return se3_exp(xi)


def _to_numpy_array(value) -> np.ndarray:
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _time_gradient(num_steps: int) -> np.ndarray:
    if num_steps <= 1:
        return np.array([[0.1, 0.75, 0.25]], dtype=np.float64)
    t = np.linspace(0.0, 1.0, num_steps, dtype=np.float64)[:, None]
    start = np.array([0.05, 0.55, 1.0], dtype=np.float64)
    middle = np.array([0.10, 0.85, 0.25], dtype=np.float64)
    end = np.array([1.0, 0.18, 0.05], dtype=np.float64)
    first_half = (1.0 - 2.0 * t) * start + (2.0 * t) * middle
    second_half = (2.0 - 2.0 * t) * middle + (2.0 * t - 1.0) * end
    return np.where(t <= 0.5, first_half, second_half).clip(0.0, 1.0)


def _make_sphere(center: np.ndarray, radius: float, color: np.ndarray) -> o3d.geometry.TriangleMesh:
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius, resolution=12)
    sphere.translate(center)
    sphere.paint_uniform_color(color.tolist())
    return sphere


def _make_trajectory_lines(positions: np.ndarray, colors: np.ndarray) -> o3d.geometry.LineSet | None:
    if positions.shape[0] < 2:
        return None
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(positions)
    line_set.lines = o3d.utility.Vector2iVector([[idx, idx + 1] for idx in range(positions.shape[0] - 1)])
    line_colors = 0.5 * (colors[:-1] + colors[1:])
    line_set.colors = o3d.utility.Vector3dVector(line_colors)
    return line_set


def vis_umi_data(
    action,
    pointcloud,
    *,
    frame_stride: int | None = None,
    max_frames: int = 12,
    frame_scale: float = 0.035,
    point_radius: float = 0.008,
):
    """Visualize a UMI pose9 trajectory with explicit temporal order.

    The trajectory is colored from blue/green at the beginning to red at the end.
    Coordinate frames are drawn sparsely so dense chunks remain readable.
    """
    actions = _to_numpy_array(action).astype(np.float32, copy=False)
    while actions.ndim > 2 and actions.shape[0] == 1:
        actions = actions[0]
    if actions.ndim != 2 or actions.shape[-1] < 9:
        raise ValueError(f"Expected action shape (T, >=9), got {actions.shape}.")

    cloud = _to_numpy_array(pointcloud).astype(np.float32, copy=False)
    while cloud.ndim > 2 and cloud.shape[0] == 1:
        cloud = cloud[0]
    if cloud.ndim != 2 or cloud.shape[-1] < 3:
        raise ValueError(f"Expected pointcloud shape (N, >=3), got {cloud.shape}.")

    positions = actions[:, :3]
    colors = _time_gradient(positions.shape[0])
    if frame_stride is None:
        frame_stride = max(1, int(math.ceil(positions.shape[0] / max(1, max_frames))))
    frame_indices = list(range(0, positions.shape[0], max(1, int(frame_stride))))
    if positions.shape[0] - 1 not in frame_indices:
        frame_indices.append(positions.shape[0] - 1)

    geometries = [create_frame(np.array([0.0, 0.0, 0.0]), np.eye(3), scale=frame_scale * 1.2)]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(cloud[:, :3])
    if cloud.shape[-1] >= 6:
        rgb = np.clip(cloud[:, 3:6] / 255.0, 0.0, 1.0)
    else:
        rgb = np.full((cloud.shape[0], 3), 0.55, dtype=np.float32)
    pcd.colors = o3d.utility.Vector3dVector(rgb)
    geometries.append(pcd)

    trajectory_lines = _make_trajectory_lines(positions, colors)
    if trajectory_lines is not None:
        geometries.append(trajectory_lines)

    # Small colored beads make the time direction visible even when frames overlap.
    for idx, (position, color) in enumerate(zip(positions, colors, strict=True)):
        radius = point_radius * (1.6 if idx in (0, positions.shape[0] - 1) else 1.0)
        geometries.append(_make_sphere(position, radius, color))

    for idx in frame_indices:
        rot6d = torch.as_tensor(actions[idx, 3:9], dtype=torch.float32)
        rotmat = rot6d_to_matrix(rot6d).cpu().numpy()
        scale = frame_scale * (1.35 if idx in (0, positions.shape[0] - 1) else 1.0)
        geometries.append(create_frame(positions[idx], rotmat, scale=scale))

    print(
        f"Visualizing {positions.shape[0]} poses: blue/green=start, red=end, "
        f"frames={frame_indices}, start={positions[0].round(4)}, end={positions[-1].round(4)}"
    )
    o3d.visualization.draw_geometries(
        geometries,
        window_name="UMI trajectory: blue/green=start, red=end",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train unsupervised Song manipulation point-cloud segmentation.")
    parser.add_argument("--dataset.repo_id", dest="dataset_repo_id", type=str, default=str(DEFAULT_DATASET_ROOT))
    parser.add_argument("--dataset.root", dest="dataset_root", type=str, default=None)
    parser.add_argument("--point-cloud-dir", type=Path, default=None)
    parser.add_argument("--sample-cache-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--future-offsets", type=parse_future_offsets, default=DEFAULT_FUTURE_OFFSETS)
    parser.add_argument("--current-points", type=int, default=50000)
    parser.add_argument("--future-points", type=int, default=16384)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip-norm", type=float, default=5.0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--backbone-type", choices=["litept", "mlp"], default="litept")
    parser.add_argument("--grid-size", type=float, default=0.01)
    parser.add_argument("--ema-decay", type=float, default=0.995)
    parser.add_argument("--ema-start-step", type=int, default=200)
    parser.add_argument("--no-ema", action="store_true")
    parser.add_argument("--vis-freq", type=int, default=500)
    parser.add_argument("--save-freq", type=int, default=1000)
    parser.add_argument("--log-freq", type=int, default=10)
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


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


def make_dataset(args: argparse.Namespace) -> torch.utils.data.Dataset:
    if args.sample_cache_dir is not None:
        return SongPointSegCachedDataset(args.sample_cache_dir)

    return make_temporal_dataset(args)


def make_temporal_dataset(args: argparse.Namespace) -> SongTemporalPointCloudDataset:
    """Build the uncached dataset used for temporal priors and full-resolution previews."""

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
    )


def find_max_resolution_visualization_indices(
    dataset: SongTemporalPointCloudDataset,
) -> tuple[list[int], int, int]:
    """Find frames from the dataset's highest source resolution without fixed thresholds.

    The mixed real dataset currently contains already-reduced clouds and raw camera
    clouds.  Their exact sizes are data properties, not model assumptions, so this
    function discovers the maximum source point count from episode array metadata.
    """
    episodes = getattr(dataset.meta, "episodes", None)
    if not episodes:
        raise ValueError("Episode metadata is required for full-resolution PointSeg visualization.")

    episode_records: list[tuple[int, list[int], int]] = []
    max_source_points = 0
    absolute_to_relative = getattr(dataset.dataset, "_absolute_to_relative_idx", None)
    for position, episode in enumerate(episodes):
        episode_index = int(episode.get("episode_index", position))
        start = int(episode["dataset_from_index"])
        end = int(episode["dataset_to_index"])
        if absolute_to_relative is None:
            relative_indices = list(range(start, end))
        else:
            relative_indices = [
                int(absolute_to_relative[index])
                for index in range(start, end)
                if index in absolute_to_relative
            ]
        if not relative_indices:
            continue
        point_clouds = open_episode_point_clouds(
            dataset.point_cloud_dir,
            episode_index,
            mmap_mode=dataset.mmap_mode,
        )
        if len(point_clouds.shape) != 3 or int(point_clouds.shape[-1]) < 3:
            raise ValueError(
                f"Expected episode {episode_index} point clouds with shape (T,N,C), got {point_clouds.shape}."
            )
        source_points = int(point_clouds.shape[1])
        max_source_points = max(max_source_points, source_points)
        episode_records.append((source_points, relative_indices, episode_index))

    if max_source_points <= 0:
        raise ValueError("Could not discover a positive source point count for visualization.")

    candidate_indices: list[int] = []
    candidate_episodes = 0
    for source_points, relative_indices, _episode_index in episode_records:
        if source_points != max_source_points:
            continue
        candidate_episodes += 1
        candidate_indices.extend(relative_indices)
    if not candidate_indices:
        raise ValueError("No samples were found at the dataset's maximum source point-cloud resolution.")
    return candidate_indices, max_source_points, candidate_episodes


def pseudo_from_cached_batch(batch: dict) -> dict[str, torch.Tensor]:
    return {
        "priors": batch["pointseg.priors"],
        "labels": batch["pointseg.labels"],
        "weights": batch["pointseg.weights"],
        "class_scores": batch["pointseg.class_scores"],
        "role_scores": batch["pointseg.role_scores"],
        "foreground_score": batch["pointseg.foreground_score"],
    }


def save_visualization(
    output_dir: Path,
    step: int,
    batch: dict,
    outputs: dict,
    pseudo: dict,
    batch_index: int = 0,
    tag: str | None = None,
) -> None:
    vis_dir = output_dir / "visualizations"
    stem = f"step_{step:06d}" if not tag else f"step_{step:06d}_{tag}"
    current_pc = batch["observation.point_cloud"][batch_index].detach().cpu()
    point_is_pad = batch.get("observation.point_cloud_is_pad")
    if torch.is_tensor(point_is_pad):
        valid_points = ~point_is_pad[batch_index].detach().cpu().to(dtype=torch.bool)
    else:
        valid_points = torch.ones(current_pc.shape[0], dtype=torch.bool)
    current_pc = current_pc[valid_points]
    probs = outputs["role_probs"][batch_index].detach().cpu()[valid_points]
    pred_labels = probs.argmax(dim=-1).numpy()
    operation_prob = outputs["operation_prob"][batch_index].detach().cpu()[valid_points].numpy()
    if "foreground_score" in pseudo:
        pseudo_foreground_score = (
            pseudo["foreground_score"][batch_index].detach().cpu()[valid_points].numpy()
        )
    else:
        pseudo_foreground_score = (
            pseudo["class_scores"][batch_index, :, 1].detach().cpu()[valid_points].numpy()
        )
    # Pseudo supervision remains fully soft. This threshold is only used to
    # render it with the same foreground/background convention as pred.ply.
    pseudo_labels = (pseudo_foreground_score > 0.5).astype(np.int64)

    def _relative_pose_matrices(pose9: torch.Tensor) -> np.ndarray:
        pose_np = pose9.detach().cpu().to(dtype=torch.float32).numpy()
        finite = np.isfinite(pose_np[:, :9]).all(axis=1)
        pose_np = pose_np[finite]
        if pose_np.shape[0] == 0:
            return np.zeros((0, 4, 4), dtype=np.float32)

        rotmats = rot6d_to_matrix(torch.from_numpy(pose_np[:, 3:9])).cpu().numpy()
        transforms = np.tile(np.eye(4, dtype=np.float32), (pose_np.shape[0], 1, 1))
        transforms[:, :3, :3] = rotmats
        transforms[:, :3, 3] = pose_np[:, :3]
        try:
            first_inv = np.linalg.inv(transforms[0])
        except np.linalg.LinAlgError:
            return transforms
        return first_inv[None] @ transforms

    def _future_ee_trajectory() -> tuple[np.ndarray, np.ndarray] | None:
        horizon = 32
        trajectory_poses = batch.get("pointseg_trajectory_ee_poses")
        if torch.is_tensor(trajectory_poses) and trajectory_poses.ndim >= 3 and trajectory_poses.shape[-1] >= 9:
            pose9 = trajectory_poses[batch_index].detach().cpu()
            offsets = batch.get("pointseg_trajectory_offsets")
            if torch.is_tensor(offsets):
                pose9 = pose9[torch.argsort(offsets[batch_index].detach().cpu())]
            pose9 = pose9[:horizon, :9].to(dtype=torch.float32)
            if pose9.shape[0] > 0:
                return pose9.numpy()[:, :3], rot6d_to_matrix(pose9[:, 3:9]).cpu().numpy()

        action = batch.get("action")
        if torch.is_tensor(action) and action.ndim >= 3 and action.shape[-1] >= 9:
            pose9 = action[batch_index, : min(horizon, action.shape[1]), :9]
            transforms = _relative_pose_matrices(pose9)
            if transforms.shape[0] > 0:
                return transforms[:, :3, 3], transforms[:, :3, :3]

        future_poses = batch.get("future_ee_poses")
        if not torch.is_tensor(future_poses) or future_poses.ndim < 3 or future_poses.shape[-1] < 9:
            return None
        pose9 = future_poses[batch_index].detach().cpu()
        future_is_pad = batch.get("future_is_pad")
        if torch.is_tensor(future_is_pad):
            valid = ~future_is_pad[batch_index].detach().cpu().bool()
            pose9 = pose9[valid] if valid.numel() == pose9.shape[0] else pose9
        pose9 = pose9[:horizon, :9]
        if pose9.shape[0] == 0:
            return None
        pose9 = pose9.to(dtype=torch.float32)
        positions = pose9.numpy()[:, :3]
        rotmats = rot6d_to_matrix(pose9[:, 3:9]).cpu().numpy()
        return positions, rotmats

    def _write_role_ply_with_trajectory(
        path: Path,
        points_xyzrgb: np.ndarray,
        labels: np.ndarray,
        *,
        operation_prob: np.ndarray | None = None,
        trajectory: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> None:
        from lerobot.policies.smolvla.song_pointseg import ROLE_COLORS, ROLE_FOREGROUND

        path.parent.mkdir(parents=True, exist_ok=True)
        xyz = np.asarray(points_xyzrgb[:, :3], dtype=np.float32)
        role_labels = np.asarray(labels, dtype=np.int64)
        if points_xyzrgb.shape[-1] >= 6:
            point_colors = np.asarray(points_xyzrgb[:, 3:6], dtype=np.float32)
            finite_colors = point_colors[np.isfinite(point_colors)]
            if finite_colors.size > 0 and float(finite_colors.max(initial=0.0)) <= 1.0:
                point_colors = point_colors * 255.0
            point_colors = np.clip(np.rint(point_colors), 0, 255).astype(np.uint8)
        else:
            point_colors = np.full((xyz.shape[0], 3), 128, dtype=np.uint8)

        foreground_mask = role_labels == ROLE_FOREGROUND
        point_colors[foreground_mask] = ROLE_COLORS[ROLE_FOREGROUND]
        if operation_prob is not None:
            prob = np.nan_to_num(
                np.asarray(operation_prob, dtype=np.float32).reshape(-1),
                nan=0.0,
                posinf=1.0,
                neginf=0.0,
            ).clip(0.0, 1.0)
            if prob.shape[0] != xyz.shape[0]:
                raise ValueError(
                    f"operation_prob has {prob.shape[0]} points, but point cloud has {xyz.shape[0]}"
                )
            point_colors[foreground_mask] = np.clip(
                point_colors[foreground_mask].astype(np.float32)
                * (0.35 + 0.65 * prob[foreground_mask, None]),
                0,
                255,
            ).astype(np.uint8)

        if trajectory is None:
            traj_positions = np.zeros((0, 3), dtype=np.float32)
        else:
            traj_positions, _ = trajectory
            traj_positions = np.asarray(traj_positions, dtype=np.float32)
            finite = np.isfinite(traj_positions).all(axis=1)
            traj_positions = traj_positions[finite]

        traj_colors = np.rint(_time_gradient(traj_positions.shape[0]) * 255.0).astype(np.uint8)
        vertices = np.concatenate([xyz, traj_positions], axis=0)
        colors = np.concatenate([point_colors, traj_colors], axis=0)
        first_traj_vertex = xyz.shape[0]
        edges = [
            (first_traj_vertex + idx, first_traj_vertex + idx + 1, traj_colors[idx], traj_colors[idx + 1])
            for idx in range(traj_positions.shape[0] - 1)
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
            f.write(f"element edge {len(edges)}\n")
            f.write("property int vertex1\n")
            f.write("property int vertex2\n")
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
            f.write("end_header\n")
            for point, color in zip(vertices, colors, strict=False):
                f.write(
                    f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f} "
                    f"{int(color[0])} {int(color[1])} {int(color[2])}\n"
                )
            for start, end, start_color, end_color in edges:
                color = np.rint((start_color.astype(np.float32) + end_color.astype(np.float32)) * 0.5).astype(np.uint8)
                f.write(f"{start} {end} {int(color[0])} {int(color[1])} {int(color[2])}\n")

    trajectory = _future_ee_trajectory()
    current_pc_np = current_pc.numpy()
    _write_role_ply_with_trajectory(
        vis_dir / f"{stem}_pred.ply",
        current_pc_np,
        pred_labels,
        operation_prob=operation_prob,
        trajectory=trajectory,
    )
    _write_role_ply_with_trajectory(
        vis_dir / f"{stem}_pseudo.ply",
        current_pc_np,
        pseudo_labels,
        operation_prob=pseudo_foreground_score,
        trajectory=trajectory,
    )
    def _select_valid(values: dict) -> dict[str, torch.Tensor]:
        selected = {}
        for key, value in values.items():
            if not torch.is_tensor(value):
                continue
            item = value[batch_index]
            if item.ndim >= 1 and item.shape[0] == valid_points.shape[0]:
                item = item[valid_points.to(device=item.device)]
            selected[key] = item
        return selected

    def _batch_scalar(key: str, default: int = -1) -> int:
        value = batch.get(key)
        if not torch.is_tensor(value):
            return int(default)
        return int(value[batch_index].detach().cpu().reshape(-1)[0].item())

    source_num_points = _batch_scalar("pointseg_source_num_points", current_pc.shape[0])
    save_pointseg_npz(
        vis_dir / f"{stem}.npz",
        current_pc,
        _select_valid(outputs),
        _select_valid(pseudo),
        metadata={
            "tag": tag or "train_batch",
            "source_num_points": source_num_points,
            "sample_num_points": int(current_pc.shape[0]),
            "episode_index": _batch_scalar("episode_index"),
            "frame_index": _batch_scalar("frame_index"),
            "dataset_index": _batch_scalar("dataset_index"),
        },
    )


def save_checkpoint(
    output_dir: Path,
    step: int,
    model: SongPointSegNet,
    optimizer: torch.optim.Optimizer,
    teacher: EMATeacher | None,
    args: argparse.Namespace,
) -> None:
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "step": step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "args": vars(args),
    }
    if teacher is not None:
        payload["teacher"] = teacher.model.state_dict()
    torch.save(payload, checkpoint_dir / f"step_{step:06d}.pt")
    torch.save(payload, checkpoint_dir / "last.pt")


def train(args: argparse.Namespace) -> None:
    if args.smoke_test:
        args.steps = min(args.steps, 2)
        args.current_points = min(args.current_points, 512)
        args.future_points = min(args.future_points, 1024)
        args.batch_size = min(args.batch_size, 2)
        args.vis_freq = 1
        args.save_freq = 2

    set_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    pseudo_cfg = PseudoLabelConfig()
    loss_cfg = SongPointSegLossConfig()
    save_pointseg_config(args.output_dir / "pointseg_config.json", args, pseudo_cfg, loss_cfg)

    dataset = make_dataset(args)
    visualization_dataset: SongTemporalPointCloudDataset | None = None
    fullres_indices: list[int] = []
    fullres_source_points = 0
    fullres_cursor = 0
    if args.vis_freq > 0:
        visualization_dataset = (
            dataset if isinstance(dataset, SongTemporalPointCloudDataset) else make_temporal_dataset(args)
        )
        fullres_indices, fullres_source_points, fullres_episode_count = (
            find_max_resolution_visualization_indices(visualization_dataset)
        )
        fullres_rng = np.random.default_rng(args.seed + 104729)
        fullres_rng.shuffle(fullres_indices)
        print(
            "PointSeg full-resolution visualization pool: "
            f"source_points={fullres_source_points}, episodes={fullres_episode_count}, "
            f"frames={len(fullres_indices)}"
        )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
        prefetch_factor=2 if args.num_workers > 0 else None,
        collate_fn=song_pointseg_collate,
    )
    iterator = iter(dataloader)

    model = SongPointSegNet(backbone_type=args.backbone_type, grid_size=args.grid_size).to(device)
    criterion = SongPointSegLoss(loss_cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    teacher: EMATeacher | None = None

    progress = tqdm(range(1, args.steps + 1), desc="Song pointseg", unit="step")
    for step in progress:
        start_time = time.perf_counter()
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(dataloader)
            batch = next(iterator)

        batch = move_batch_to_device(batch, device)
        current_pc = batch["observation.point_cloud"]
        current_is_pad = batch.get("observation.point_cloud_is_pad")
        uses_cached_pseudo = "pointseg.priors" in batch

        with torch.no_grad():
            if uses_cached_pseudo:
                pseudo = pseudo_from_cached_batch(batch)
            else:
                pseudo = generate_pseudo_labels(
                    current_pc,
                    batch["observation.point_cloud_future"],
                    batch["future_ee_poses"],
                    batch["future_is_pad"],
                    current_is_pad=current_is_pad,
                    future_point_is_pad=batch.get("observation.point_cloud_future_is_pad"),
                    trajectory_poses=batch.get("pointseg_trajectory_ee_poses"),
                    config=pseudo_cfg,
                )
            if current_is_pad is not None:
                pseudo["point_is_pad"] = current_is_pad
            if teacher is not None:
                teacher_outputs = teacher.model(current_pc, priors=pseudo["priors"], point_is_pad=current_is_pad)
                pseudo = refine_pseudo_labels_with_teacher(pseudo, teacher_outputs["role_logits"], config=pseudo_cfg)

        model.train()
        outputs = model(current_pc, priors=pseudo["priors"], point_is_pad=current_is_pad)
        loss, metrics = criterion(outputs, pseudo, current_pc)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if args.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
        optimizer.step()

        if teacher is None and not args.no_ema and step >= args.ema_start_step:
            teacher = EMATeacher(model, decay=args.ema_decay)
        elif teacher is not None:
            teacher.update(model)

        metrics_for_log = {**metrics, "step_s": time.perf_counter() - start_time}
        progress.set_postfix(
            {
                key: f"{float(value.item()) if torch.is_tensor(value) else value:.3f}"
                for key, value in metrics_for_log.items()
                if key in {"loss", "pseudo_foreground_ratio", "pred_foreground_ratio", "step_s"}
            }
        )

        if args.log_freq > 0 and step % args.log_freq == 0:
            print(pretty_metrics(metrics_for_log, step))

        if args.vis_freq > 0 and step % args.vis_freq == 0:
            with torch.no_grad():
                # Re-run after the optimizer update in deterministic eval mode so
                # pred.ply represents checkpoint inference rather than train-mode
                # BatchNorm/random serialization output from before the update.
                model.eval()
                vis_outputs = model(current_pc, priors=pseudo["priors"], point_is_pad=current_is_pad)
                save_visualization(args.output_dir, step, batch, vis_outputs, pseudo)

                # Always save a second result from the dataset's dynamically
                # discovered maximum source resolution.  This is independent of
                # whichever resolutions happened to occur in the shuffled batch.
                if visualization_dataset is None or not fullres_indices:
                    raise RuntimeError("Full-resolution visualization pool was not initialized.")
                fullres_index = fullres_indices[fullres_cursor % len(fullres_indices)]
                fullres_cursor += 1
                fullres_item = dict(visualization_dataset[fullres_index])
                source_num_points = int(
                    torch.as_tensor(fullres_item["pointseg_source_num_points"]).reshape(-1)[0].item()
                )
                if source_num_points != fullres_source_points:
                    raise RuntimeError(
                        "Full-resolution visualization pool became inconsistent: "
                        f"dataset index {fullres_index} has {source_num_points} source points, "
                        f"expected {fullres_source_points}."
                    )
                fullres_item["dataset_index"] = torch.tensor(fullres_index, dtype=torch.long)
                fullres_batch = move_batch_to_device(song_pointseg_collate([fullres_item]), device)
                fullres_pc = fullres_batch["observation.point_cloud"]
                fullres_is_pad = fullres_batch.get("observation.point_cloud_is_pad")
                fullres_pseudo = generate_pseudo_labels(
                    fullres_pc,
                    fullres_batch["observation.point_cloud_future"],
                    fullres_batch["future_ee_poses"],
                    fullres_batch["future_is_pad"],
                    current_is_pad=fullres_is_pad,
                    future_point_is_pad=fullres_batch.get("observation.point_cloud_future_is_pad"),
                    trajectory_poses=fullres_batch.get("pointseg_trajectory_ee_poses"),
                    config=pseudo_cfg,
                )
                if fullres_is_pad is not None:
                    fullres_pseudo["point_is_pad"] = fullres_is_pad
                if teacher is not None:
                    fullres_teacher_outputs = teacher.model(
                        fullres_pc,
                        priors=fullres_pseudo["priors"],
                        point_is_pad=fullres_is_pad,
                    )
                    fullres_pseudo = refine_pseudo_labels_with_teacher(
                        fullres_pseudo,
                        fullres_teacher_outputs["role_logits"],
                        config=pseudo_cfg,
                    )
                fullres_outputs = model(
                    fullres_pc,
                    priors=fullres_pseudo["priors"],
                    point_is_pad=fullres_is_pad,
                )
                save_visualization(
                    args.output_dir,
                    step,
                    fullres_batch,
                    fullres_outputs,
                    fullres_pseudo,
                    tag="fullres",
                )
                model.train()

        if args.save_freq > 0 and step % args.save_freq == 0:
            save_checkpoint(args.output_dir, step, model, optimizer, teacher, args)

    save_checkpoint(args.output_dir, args.steps, model, optimizer, teacher, args)


def main() -> None:
    train(parse_args())


if __name__ == "__main__":
    main()



# idx = 0
# valid_mask = (batch["observation.point_cloud_is_pad"]).sum(1)==0
# for idx,label in enumerate(valid_mask):
#     if label==True:
#         batch_idx = idx

#         point_valid = ~batch["observation.point_cloud_is_pad"][batch_idx]
#         step_valid = ~batch["future_is_pad"][batch_idx]

#         point_cloud = batch["observation.point_cloud"][batch_idx][point_valid]
#         rgb = point_cloud[...,3:]
#         rgb[pseudo['labels'][batch_idx].cpu().numpy()!=1] = 0
#         point_cloud[...,3:] = rgb


#         trajectory = batch["future_ee_poses"][batch_idx][step_valid]

#         print("episode:", batch["episode_index"][batch_idx].item())
#         print("frame:", batch["frame_index"][batch_idx].item())
#         print("raw action start:", batch["action"][batch_idx, 0, :9])
#         print("relative start:", trajectory[0])

#         vis_umi_data(
#             trajectory.cpu().numpy(),
#             point_cloud.cpu().numpy(),
#         )
