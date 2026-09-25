#!/usr/bin/env python
from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation as R

if __package__ and __package__.startswith("benchmarks."):
    from .._paths import LIBERO_DATA_ROOT
else:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from _paths import LIBERO_DATA_ROOT


def ensure_libero_config(config_path: str | Path | None = None, dataset_root: str | Path | None = None) -> Path:
    """Create LIBERO's config.yaml non-interactively before importing libero.libero."""
    config_dir = Path(
        config_path
        or os.environ.get("LIBERO_CONFIG_PATH")
        or LIBERO_DATA_ROOT / "libero_config"
    ).expanduser().resolve()
    os.environ["LIBERO_CONFIG_PATH"] = str(config_dir)
    config_file = config_dir / "config.yaml"
    requested_dataset_root = Path(dataset_root).expanduser().resolve() if dataset_root else None
    if config_file.exists() and requested_dataset_root is None:
        return config_file

    spec = importlib.util.find_spec("libero")
    if spec is None or not spec.submodule_search_locations:
        raise ModuleNotFoundError("LIBERO is not installed. Install it before running LIBERO benchmark scripts.")
    package_root = Path(list(spec.submodule_search_locations)[0]).resolve()
    benchmark_root = package_root / "libero"
    if not benchmark_root.is_dir():
        raise FileNotFoundError(f"Could not locate LIBERO benchmark root under {package_root}")

    datasets = requested_dataset_root if requested_dataset_root is not None else benchmark_root.parent / "datasets"
    if config_file.exists():
        existing = config_file.read_text(encoding="utf-8")
        if f"datasets: {datasets}" in existing:
            return config_file
    config_dir.mkdir(parents=True, exist_ok=True)
    config_file.write_text(
        "\n".join(
            [
                f"benchmark_root: {benchmark_root}",
                f"bddl_files: {benchmark_root / 'bddl_files'}",
                f"init_states: {benchmark_root / 'init_files'}",
                f"datasets: {datasets}",
                f"assets: {benchmark_root / 'assets'}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return config_file


def normalize_camera_name(camera_name: str) -> str:
    camera_name = str(camera_name)
    return camera_name[: -len("_image")] if camera_name.endswith("_image") else camera_name


def pointcloud_camera_names_from_config(cfg: dict[str, Any]) -> list[str]:
    camera_names = cfg.get("pointcloud_camera_names") or cfg.get("camera_names") or []
    camera_names = [normalize_camera_name(camera_name) for camera_name in camera_names]
    if not camera_names:
        raise ValueError("At least one LIBERO point-cloud camera must be configured.")
    reference_camera = normalize_camera_name(cfg.get("pointcloud_reference_camera", camera_names[0]))
    if reference_camera not in camera_names:
        raise ValueError(
            f"pointcloud_reference_camera={reference_camera!r} must be included in "
            f"pointcloud_camera_names={camera_names!r}."
        )
    camera_names = [reference_camera, *(name for name in camera_names if name != reference_camera)]
    return camera_names


def render_camera_names_from_config(cfg: dict[str, Any]) -> list[str]:
    merged: list[str] = []
    for camera_name in list(cfg.get("camera_names") or []) + pointcloud_camera_names_from_config(cfg):
        camera_name = normalize_camera_name(camera_name)
        if camera_name not in merged:
            merged.append(camera_name)
    if not merged:
        raise ValueError("At least one LIBERO render camera must be configured.")
    return merged


def normalize_render_camera_name(camera_name: str | None, fallback: str | None = None) -> str | None:
    if camera_name is None:
        return normalize_camera_name(fallback) if fallback else None
    camera_name = str(camera_name).strip()
    if not camera_name or camera_name.lower() in {"none", "free"}:
        return None
    return normalize_camera_name(camera_name)


def sample_or_repeat_points(xyzrgb: np.ndarray, num_points: int, seed: int = 0) -> np.ndarray:
    xyzrgb = np.asarray(xyzrgb, dtype=np.float32)
    if xyzrgb.ndim != 2 or xyzrgb.shape[1] != 6:
        raise ValueError(f"Expected point cloud shape (N, 6), got {xyzrgb.shape}")
    if num_points <= 0 or xyzrgb.shape[0] == num_points:
        return xyzrgb
    if xyzrgb.shape[0] == 0:
        raise ValueError("Cannot sample from an empty point cloud.")
    rng = np.random.default_rng(seed)
    replace = xyzrgb.shape[0] < num_points
    indices = rng.choice(xyzrgb.shape[0], num_points, replace=replace)
    return xyzrgb[indices].astype(np.float32, copy=False)


def normalize_gripper_widths(
    widths: np.ndarray,
    already_normalized: bool = False,
    max_physical_width: float | None = None,
) -> np.ndarray:
    widths = np.asarray(widths, dtype=np.float32).reshape(-1).copy()
    if widths.size == 0:
        return widths
    if already_normalized:
        return np.clip(widths, 0.0, 1.0)
    if max_physical_width is not None and float(max_physical_width) > 0:
        return np.clip(widths / float(max_physical_width), 0.0, 1.0)

    width_range = widths.max() - widths.min()
    if width_range != 0:
        widths = (widths - widths.min()) / width_range
    else:
        widths[widths >= 0] = 1.0
    return np.clip(widths, 0.0, 1.0)


def gripper_width_percent_from_scalar(width: float, max_physical_width: float = 0.08) -> float:
    width = float(width)
    if not np.isfinite(width):
        return 0.0
    physical_limit = max(float(max_physical_width), 1e-6)
    if 0.0 <= width <= physical_limit * 1.25:
        return float(np.clip(width / max(max_physical_width, 1e-6), 0.0, 1.0))
    return float(np.clip(width, 0.0, 1.0))


def allocate_counts(total: int, weights: list[float] | np.ndarray) -> np.ndarray:
    weights = np.asarray(weights, dtype=np.float64)
    if total <= 0:
        return np.zeros(len(weights), dtype=np.int64)
    if weights.sum() <= 0:
        counts = np.zeros(len(weights), dtype=np.int64)
        counts[:total] = 1
        return counts

    expected = total * weights / weights.sum()
    counts = np.floor(expected).astype(np.int64)
    remainder = int(total - counts.sum())
    if remainder > 0:
        order = np.argsort(expected - counts)[::-1]
        counts[order[:remainder]] += 1
    return counts


def box_faces(min_corner: np.ndarray, size: np.ndarray) -> list[tuple[float, np.ndarray, np.ndarray, np.ndarray]]:
    sx, sy, sz = size
    x0, y0, z0 = min_corner
    x1, y1, z1 = min_corner + size
    return [
        (sy * sz, np.array([x0, y0, z0]), np.array([0.0, sy, 0.0]), np.array([0.0, 0.0, sz])),
        (sy * sz, np.array([x1, y0, z0]), np.array([0.0, sy, 0.0]), np.array([0.0, 0.0, sz])),
        (sx * sz, np.array([x0, y0, z0]), np.array([sx, 0.0, 0.0]), np.array([0.0, 0.0, sz])),
        (sx * sz, np.array([x0, y1, z0]), np.array([sx, 0.0, 0.0]), np.array([0.0, 0.0, sz])),
        (sx * sy, np.array([x0, y0, z0]), np.array([sx, 0.0, 0.0]), np.array([0.0, sy, 0.0])),
        (sx * sy, np.array([x0, y0, z1]), np.array([sx, 0.0, 0.0]), np.array([0.0, sy, 0.0])),
    ]


def sample_box_surface(min_corner: np.ndarray, size: np.ndarray, count: int, rng: np.random.Generator) -> np.ndarray:
    faces = box_faces(np.asarray(min_corner, dtype=np.float64), np.asarray(size, dtype=np.float64))
    counts = allocate_counts(count, [face[0] for face in faces])
    samples = []
    for face_count, (_, origin, axis_a, axis_b) in zip(counts, faces):
        if face_count == 0:
            continue
        uv = rng.random((face_count, 2), dtype=np.float64)
        samples.append(origin + uv[:, :1] * axis_a + uv[:, 1:] * axis_b)
    if not samples:
        return np.empty((0, 3), dtype=np.float64)
    return np.vstack(samples)


def create_gripper_points(
    width_percent: float,
    pose: np.ndarray,
    count: int,
    rng: np.random.Generator,
    gripper_len: float = 0.06,
    max_width: float = 0.06,
    finger_length: float = 0.08,
    finger_thickness: float = 0.01,
    base_thickness: float = 0.01,
    handle_length: float = 0.05,
    opening_max_width: float | None = None,
) -> np.ndarray:
    # Keep the fixed REAP body width (`max_width`) independent from the finger
    # opening scale when a simulator exposes a different physical aperture.
    opening_scale = max_width if opening_max_width is None else float(opening_max_width)
    width = float(np.clip(width_percent, 0.0, 1.0)) * opening_scale
    boxes = [
        (
            np.array([width / 2.0, base_thickness, 0.0]),
            np.array([finger_thickness, finger_length, finger_thickness]),
        ),
        (
            np.array([-width / 2.0 - finger_thickness, base_thickness, 0.0]),
            np.array([finger_thickness, finger_length, finger_thickness]),
        ),
        (
            np.array([-max_width / 2.0 - finger_thickness / 2.0, 0.0, 0.0]),
            np.array([max_width + finger_thickness, base_thickness, finger_thickness]),
        ),
        (
            np.array([-base_thickness / 2.0, -handle_length, 0.0]),
            np.array([base_thickness, handle_length, base_thickness]),
        ),
    ]

    box_areas = [2.0 * (size[0] * size[1] + size[0] * size[2] + size[1] * size[2]) for _, size in boxes]
    box_counts = allocate_counts(count, box_areas)
    points = []
    for box_count, (min_corner, size) in zip(box_counts, boxes):
        points.append(sample_box_surface(min_corner, size, box_count, rng))
    points = np.vstack(points) if points else np.empty((0, 3), dtype=np.float64)

    static_rot = R.from_euler("zyx", [np.pi / 2.0, np.pi / 2.0, 0.0]).as_matrix()
    pose_rot = R.from_euler("zyx", pose[3:]).as_matrix()
    # Keep the REAP gripper template aligned with the real-data HDF5 path:
    # `gripper_len` is a positive length, while the template offset direction
    # is fixed by the end-effector frame convention.
    points = points @ static_rot.T + np.array([0.0, 0.0, -gripper_len])
    points = points @ pose_rot.T + pose[:3]
    return points


def create_panda_gripper_points(
    width_percent: float,
    pose: np.ndarray,
    count: int,
    rng: np.random.Generator,
    max_width: float = 0.08,
    finger_length: float = 0.065,
    finger_thickness: float = 0.012,
    finger_depth: float = 0.018,
    palm_depth: float = 0.05,
) -> np.ndarray:
    """Approximate robosuite PandaGripper geometry in the grip-site frame."""
    width = float(np.clip(width_percent, 0.0, 1.0)) * max_width
    half_width = width / 2.0
    boxes = [
        (
            np.array([half_width, -finger_depth / 2.0, -0.045]),
            np.array([finger_thickness, finger_depth, finger_length]),
        ),
        (
            np.array([-half_width - finger_thickness, -finger_depth / 2.0, -0.045]),
            np.array([finger_thickness, finger_depth, finger_length]),
        ),
        (
            np.array([-max_width / 2.0 - finger_thickness, -0.035, -0.095]),
            np.array([max_width + 2.0 * finger_thickness, 0.07, palm_depth]),
        ),
    ]

    box_areas = [2.0 * (size[0] * size[1] + size[0] * size[2] + size[1] * size[2]) for _, size in boxes]
    box_counts = allocate_counts(count, box_areas)
    points = []
    for box_count, (min_corner, size) in zip(box_counts, boxes):
        points.append(sample_box_surface(min_corner, size, box_count, rng))
    points = np.vstack(points) if points else np.empty((0, 3), dtype=np.float64)

    pose_rot = R.from_euler("zyx", pose[3:]).as_matrix()
    points = points @ pose_rot.T + pose[:3]
    return points


RLBENCH_PANDA_GRIPPER_TEMPLATE = "rlbench_panda"
RLBENCH_PANDA_GRIPPER_TEMPLATE_VERSION = "rlbench_panda_tip_ttm_v1"
RLBENCH_MINIMAL_TWO_FINGER_TEMPLATE = "rlbench_minimal_two_finger"
RLBENCH_MINIMAL_TWO_FINGER_TEMPLATE_VERSION = "rlbench_panda_tip_minimal_two_finger_v1"
RLBENCH_PANDA_MAX_WIDTH = 0.08


def rlbench_panda_gripper_local_boxes(width_percent: float) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Approximate the RLBench Panda tip geometry in its local frame."""
    width = float(np.clip(width_percent, 0.0, 1.0)) * RLBENCH_PANDA_MAX_WIDTH
    half_gap = width / 2.0
    finger_half_x = 0.0106
    finger_depth = 0.0265
    finger_min_z = -0.0536
    finger_size = np.array([2.0 * finger_half_x, finger_depth, 0.0539], dtype=np.float64)
    palm_half_x = 0.0316
    palm_half_y = 0.1043
    return {
        "left_finger": (
            np.array([-finger_half_x, -half_gap - finger_depth, finger_min_z]),
            finger_size.copy(),
        ),
        "right_finger": (
            np.array([-finger_half_x, half_gap, finger_min_z]),
            finger_size.copy(),
        ),
        "palm": (
            np.array([-palm_half_x, -palm_half_y, -0.1381]),
            np.array([2.0 * palm_half_x, 2.0 * palm_half_y, 0.0921]),
        ),
    }


def create_rlbench_panda_gripper_points(
    width_percent: float,
    pose: np.ndarray,
    count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    boxes = list(rlbench_panda_gripper_local_boxes(width_percent).values())
    areas = [2.0 * (size[0] * size[1] + size[0] * size[2] + size[1] * size[2]) for _, size in boxes]
    counts = allocate_counts(int(count), areas)
    points = [sample_box_surface(min_corner, size, n, rng) for n, (min_corner, size) in zip(counts, boxes)]
    local = np.vstack(points) if points else np.empty((0, 3), dtype=np.float64)
    pose = np.asarray(pose, dtype=np.float64)
    return local @ R.from_euler("zyx", pose[3:]).as_matrix().T + pose[:3]


def create_rlbench_minimal_two_finger_points(
    width_percent: float,
    pose: np.ndarray,
    count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    boxes = rlbench_panda_gripper_local_boxes(width_percent)
    finger_boxes = [boxes["left_finger"], boxes["right_finger"]]
    areas = [2.0 * (size[0] * size[1] + size[0] * size[2] + size[1] * size[2]) for _, size in finger_boxes]
    counts = allocate_counts(int(count), areas)
    points = [sample_box_surface(min_corner, size, n, rng) for n, (min_corner, size) in zip(counts, finger_boxes)]
    local = np.vstack(points) if points else np.empty((0, 3), dtype=np.float64)
    pose = np.asarray(pose, dtype=np.float64)
    return local @ R.from_euler("zyx", pose[3:]).as_matrix().T + pose[:3]


def create_gripper_cloud_rgb(
    width_percent: float,
    pose: np.ndarray,
    count: int,
    rng: np.random.Generator,
    gripper_len: float,
    gripper_template: str = "reap",
    reap_max_width: float = 0.06,
    reap_opening_max_width: float | None = None,
) -> np.ndarray:
    if gripper_template == RLBENCH_MINIMAL_TWO_FINGER_TEMPLATE:
        points = create_rlbench_minimal_two_finger_points(width_percent, pose, count, rng)
    elif gripper_template == RLBENCH_PANDA_GRIPPER_TEMPLATE:
        points = create_rlbench_panda_gripper_points(width_percent, pose, count, rng)
    elif gripper_template == "panda":
        points = create_panda_gripper_points(width_percent, pose, count, rng)
    else:
        points = create_gripper_points(
            width_percent,
            pose,
            count,
            rng,
            gripper_len=gripper_len,
            max_width=float(reap_max_width),
            opening_max_width=reap_opening_max_width,
        )
    colors = np.tile(np.array([[204.0, 51.0, 51.0]], dtype=np.float32), (points.shape[0], 1))
    return np.hstack((points.astype(np.float32), colors))


def merge_cloud_with_gripper(
    original_cloud: np.ndarray,
    gripper_cloud: np.ndarray,
    rng: np.random.Generator,
    drop_strategy: str = "random",
    shuffle_points: bool = False,
) -> np.ndarray:
    total_points = original_cloud.shape[0]
    gripper_points = min(gripper_cloud.shape[0], total_points)
    keep_points = total_points - gripper_points

    if gripper_points == 0:
        merged = original_cloud.copy()
    elif keep_points == 0:
        idx = rng.choice(gripper_cloud.shape[0], total_points, replace=gripper_cloud.shape[0] < total_points)
        merged = gripper_cloud[idx]
    elif drop_strategy == "tail":
        merged = np.vstack((original_cloud[:keep_points], gripper_cloud[:gripper_points]))
    elif drop_strategy == "near_gripper":
        center = gripper_cloud[:gripper_points, :3].mean(axis=0)
        dist = np.linalg.norm(original_cloud[:, :3] - center, axis=1)
        keep_idx = np.argpartition(dist, gripper_points)[gripper_points:]
        merged = np.vstack((original_cloud[keep_idx], gripper_cloud[:gripper_points]))
    else:
        keep_idx = rng.choice(total_points, keep_points, replace=False)
        merged = np.vstack((original_cloud[keep_idx], gripper_cloud[:gripper_points]))

    if shuffle_points and merged.shape[0] > 1:
        rng.shuffle(merged, axis=0)
    return merged.astype(original_cloud.dtype, copy=False)


def add_reference_gripper_cloud_to_point_cloud(
    point_cloud_reference: np.ndarray,
    current_pose9_gripper_reference: np.ndarray,
    width_percent: float,
    *,
    total_points: int,
    gripper_points: int = 500,
    gripper_len: float = 0.06,
    gripper_template: str = "reap",
    gripper_max_width: float | None = None,
    gripper_opening_max_width: float | None = None,
    seed: int = 0,
    drop_strategy: str = "random",
    shuffle_points: bool = False,
) -> np.ndarray:
    """Merge a virtual gripper into a cloud without changing its fixed reference frame."""

    total_points = int(total_points)
    gripper_points = max(0, min(int(gripper_points), total_points))
    rng = np.random.default_rng(seed)
    point_cloud_reference = sample_or_repeat_points(point_cloud_reference, total_points, seed=seed)
    if gripper_points == 0:
        return point_cloud_reference

    current_traj6 = pose9_to_traj6_np(np.asarray(current_pose9_gripper_reference, dtype=np.float32))[..., :6]
    gripper_cloud_reference = create_gripper_cloud_rgb(
        width_percent,
        current_traj6,
        gripper_points,
        rng,
        gripper_len=float(gripper_len),
        gripper_template=str(gripper_template),
        reap_max_width=(
            float(gripper_max_width)
            if str(gripper_template) == "reap" and gripper_max_width is not None
            else 0.06
        ),
        reap_opening_max_width=gripper_opening_max_width,
    )
    return merge_cloud_with_gripper(
        point_cloud_reference,
        gripper_cloud_reference,
        rng,
        drop_strategy=drop_strategy,
        shuffle_points=shuffle_points,
    )


def add_reference_gripper_clouds_to_episode(
    point_clouds_reference: np.ndarray,
    current_pose9_grippers_reference: np.ndarray,
    gripper_widths: np.ndarray,
    *,
    total_points: int,
    gripper_points: int = 500,
    gripper_len: float = 0.06,
    gripper_template: str = "reap",
    seed: int = 0,
    drop_strategy: str = "random",
    shuffle_points: bool = False,
    widths_are_normalized: bool = False,
    gripper_max_width: float | None = None,
    gripper_opening_max_width: float | None = None,
) -> np.ndarray:
    """Add gripper geometry to every frame while preserving the shared reference frame."""

    point_clouds_reference = np.asarray(point_clouds_reference, dtype=np.float32)
    current_pose9_grippers_reference = np.asarray(current_pose9_grippers_reference, dtype=np.float32)
    widths = normalize_gripper_widths(
        gripper_widths,
        already_normalized=widths_are_normalized,
        max_physical_width=gripper_max_width,
    )
    if len(point_clouds_reference) != len(current_pose9_grippers_reference):
        raise ValueError(
            f"Point cloud frames {len(point_clouds_reference)} != pose frames "
            f"{len(current_pose9_grippers_reference)}"
        )
    if len(point_clouds_reference) != len(widths):
        raise ValueError(f"Point cloud frames {len(point_clouds_reference)} != gripper widths {len(widths)}")

    merged = np.empty((len(point_clouds_reference), int(total_points), 6), dtype=np.float32)
    for frame_idx, (cloud_reference, pose9, width) in enumerate(
        zip(point_clouds_reference, current_pose9_grippers_reference, widths, strict=True)
    ):
        merged[frame_idx] = add_reference_gripper_cloud_to_point_cloud(
            cloud_reference,
            pose9,
            float(width),
            total_points=total_points,
            gripper_points=gripper_points,
            gripper_len=gripper_len,
            gripper_template=gripper_template,
            gripper_opening_max_width=gripper_opening_max_width,
            seed=seed + frame_idx,
            drop_strategy=drop_strategy,
            shuffle_points=shuffle_points,
        )
    return merged


def add_local_gripper_cloud_to_point_cloud(
    point_cloud_eff: np.ndarray,
    width_percent: float,
    *,
    total_points: int,
    gripper_points: int = 500,
    gripper_len: float = 0.06,
    gripper_template: str = "reap",
    gripper_opening_max_width: float | None = None,
    seed: int = 0,
    drop_strategy: str = "random",
    shuffle_points: bool = False,
) -> np.ndarray:
    total_points = int(total_points)
    gripper_points = max(0, min(int(gripper_points), total_points))
    rng = np.random.default_rng(seed)
    point_cloud_eff = sample_or_repeat_points(point_cloud_eff, total_points, seed=seed)
    if gripper_points == 0:
        return point_cloud_eff
    gripper_pose_eff = np.zeros(6, dtype=np.float32)
    gripper_cloud = create_gripper_cloud_rgb(
        width_percent,
        gripper_pose_eff,
        gripper_points,
        rng,
        gripper_len=float(gripper_len),
        gripper_template=str(gripper_template),
        reap_opening_max_width=gripper_opening_max_width,
    )
    return merge_cloud_with_gripper(
        point_cloud_eff,
        gripper_cloud,
        rng,
        drop_strategy=drop_strategy,
        shuffle_points=shuffle_points,
    )


def add_local_gripper_clouds_to_episode(
    point_clouds_eff: np.ndarray,
    gripper_widths: np.ndarray,
    *,
    total_points: int,
    gripper_points: int = 500,
    gripper_len: float = 0.06,
    gripper_template: str = "reap",
    seed: int = 0,
    drop_strategy: str = "random",
    shuffle_points: bool = False,
    widths_are_normalized: bool = False,
    gripper_max_width: float | None = None,
    gripper_opening_max_width: float | None = None,
) -> np.ndarray:
    point_clouds_eff = np.asarray(point_clouds_eff, dtype=np.float32)
    widths = normalize_gripper_widths(
        gripper_widths,
        already_normalized=widths_are_normalized,
        max_physical_width=gripper_max_width,
    )
    if len(point_clouds_eff) != len(widths):
        raise ValueError(f"Point cloud frames {len(point_clouds_eff)} != gripper widths {len(widths)}")
    merged = np.empty((len(point_clouds_eff), int(total_points), 6), dtype=np.float32)
    for frame_idx, (cloud, width) in enumerate(zip(point_clouds_eff, widths)):
        merged[frame_idx] = add_local_gripper_cloud_to_point_cloud(
            cloud,
            float(width),
            total_points=total_points,
            gripper_points=gripper_points,
            gripper_len=gripper_len,
            gripper_template=gripper_template,
            gripper_opening_max_width=gripper_opening_max_width,
            seed=seed + frame_idx,
            drop_strategy=drop_strategy,
            shuffle_points=shuffle_points,
        )
    return merged


def rot6d_to_matrix_np(d6: np.ndarray) -> np.ndarray:
    d6 = np.asarray(d6, dtype=np.float32)
    a1 = d6[..., 0:3]
    a2 = d6[..., 3:6]
    b1 = a1 / np.clip(np.linalg.norm(a1, axis=-1, keepdims=True), 1e-6, None)
    b2 = a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = b2 / np.clip(np.linalg.norm(b2, axis=-1, keepdims=True), 1e-6, None)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=-1)


def pose9_to_homo_np(pose9: np.ndarray) -> np.ndarray:
    pose9 = np.asarray(pose9, dtype=np.float32)
    homo = np.zeros((*pose9.shape[:-1], 4, 4), dtype=np.float32)
    homo[..., 3, 3] = 1.0
    homo[..., :3, :3] = rot6d_to_matrix_np(pose9[..., 3:9])
    homo[..., :3, 3] = pose9[..., :3]
    return homo


def pose9_to_traj6_np(pose9: np.ndarray) -> np.ndarray:
    pose9 = np.asarray(pose9, dtype=np.float32)[..., :9]
    rot = rot6d_to_matrix_np(pose9[..., 3:9])
    flat_rot = rot.reshape(-1, 3, 3)
    euler = R.from_matrix(flat_rot).as_euler("zyx", degrees=False).astype(np.float32)
    euler = euler.reshape(*pose9.shape[:-1], 3)
    return np.concatenate([pose9[..., :3], euler], axis=-1).astype(np.float32)


def fast_inverse_homogeneous(T: np.ndarray) -> np.ndarray:
    T = np.asarray(T, dtype=np.float32)
    out = np.zeros_like(T)
    rot = T[..., :3, :3]
    trans = T[..., :3, 3]
    rot_inv = np.swapaxes(rot, -1, -2)
    out[..., :3, :3] = rot_inv
    out[..., :3, 3] = -(rot_inv @ trans[..., None])[..., 0]
    out[..., 3, 3] = 1.0
    return out


def traj6_to_pose9(traj6: np.ndarray) -> np.ndarray:
    traj6 = np.asarray(traj6, dtype=np.float32)
    if traj6.ndim == 1:
        rot = R.from_euler("zyx", traj6[3:6], degrees=False).as_matrix().astype(np.float32)
        return np.concatenate([traj6[:3], rot[:, 0], rot[:, 1]], axis=0).astype(np.float32)
    rot = R.from_euler("zyx", traj6[..., 3:6], degrees=False).as_matrix().astype(np.float32)
    rot6d = np.concatenate([rot[..., :, 0], rot[..., :, 1]], axis=-1)
    return np.concatenate([traj6[..., :3], rot6d], axis=-1).astype(np.float32)


def pose9_from_pos_quat(pos: np.ndarray, quat_xyzw: np.ndarray, gripper: float = 0.0) -> np.ndarray:
    rot = R.from_quat(np.asarray(quat_xyzw, dtype=np.float32)).as_matrix().astype(np.float32)
    rot6d = np.concatenate([rot[:, 0], rot[:, 1]], axis=0)
    return np.concatenate([np.asarray(pos, dtype=np.float32), rot6d, np.asarray([gripper], dtype=np.float32)])


def gripper_scalar(raw_obs: dict[str, Any]) -> float:
    qpos = np.asarray(raw_obs.get("robot0_gripper_qpos", [0.0, 0.0]), dtype=np.float32).reshape(-1)
    qpos = np.abs(qpos)  ### absolute pos
    if qpos.size == 0:
        return 0.0

    return float(np.sum(qpos))


def eef_pose9_gripper_from_obs(raw_obs: dict[str, Any]) -> np.ndarray:
    return pose9_from_pos_quat(
        np.asarray(raw_obs["robot0_eef_pos"], dtype=np.float32),
        np.asarray(raw_obs["robot0_eef_quat"], dtype=np.float32),
        gripper_scalar(raw_obs),
    )


def reference_point_cloud_to_current_eff(
    point_cloud_reference: np.ndarray,
    current_pose9_gripper_reference: np.ndarray,
) -> np.ndarray:
    """Transform a fixed-reference cloud into the current end-effector frame."""

    pc = np.asarray(point_cloud_reference, dtype=np.float32)
    squeeze = pc.ndim == 2
    if squeeze:
        pc = pc[None]
    if pc.ndim != 3 or pc.shape[-1] != 6:
        raise ValueError(
            f"Expected point_cloud shape (N, 6) or (B, N, 6), got {point_cloud_reference.shape}"
        )

    pose = np.asarray(current_pose9_gripper_reference, dtype=np.float32)
    if pose.ndim == 1:
        pose = pose[None]
    if pose.shape[0] == 1 and pc.shape[0] > 1:
        pose = np.repeat(pose, pc.shape[0], axis=0)
    if pose.shape[0] != pc.shape[0]:
        raise ValueError(f"Pose batch {pose.shape[0]} does not match point cloud batch {pc.shape[0]}.")

    eff_to_reference = pose9_to_homo_np(pose)
    reference_to_eff = fast_inverse_homogeneous(eff_to_reference)
    xyz_h = np.concatenate([pc[..., :3], np.ones((*pc.shape[:2], 1), dtype=np.float32)], axis=-1)
    eff_xyz_h = np.einsum("bij,bnj->bni", reference_to_eff, xyz_h)
    pc_eff = np.concatenate([eff_xyz_h[..., :3], pc[..., 3:]], axis=-1).astype(np.float32)
    return pc_eff[0] if squeeze else pc_eff


def world_point_cloud_to_current_eff(point_cloud_world: np.ndarray, current_pose9_gripper: np.ndarray) -> np.ndarray:
    """Backward-compatible alias for a generic fixed-reference transform."""

    return reference_point_cloud_to_current_eff(point_cloud_world, current_pose9_gripper)


def add_world_gripper_cloud_to_point_cloud(
    point_cloud_world: np.ndarray,
    current_pose9_gripper: np.ndarray,
    width_percent: float,
    *,
    total_points: int,
    gripper_points: int = 500,
    gripper_len: float = 0.06,
    gripper_template: str = "reap",
    gripper_max_width: float | None = None,
    gripper_opening_max_width: float | None = None,
    seed: int = 0,
    drop_strategy: str = "random",
    shuffle_points: bool = False,
) -> np.ndarray:
    merged_world = add_reference_gripper_cloud_to_point_cloud(
        point_cloud_world,
        current_pose9_gripper,
        width_percent,
        total_points=total_points,
        gripper_points=gripper_points,
        gripper_len=float(gripper_len),
        gripper_template=str(gripper_template),
        gripper_max_width=gripper_max_width,
        gripper_opening_max_width=gripper_opening_max_width,
        seed=seed,
        drop_strategy=drop_strategy,
        shuffle_points=shuffle_points,
    )
    return reference_point_cloud_to_current_eff(merged_world, current_pose9_gripper)


def add_world_gripper_clouds_to_episode(
    point_clouds_world: np.ndarray,
    current_pose9_grippers: np.ndarray,
    gripper_widths: np.ndarray,
    *,
    total_points: int,
    gripper_points: int = 500,
    gripper_len: float = 0.06,
    gripper_template: str = "reap",
    seed: int = 0,
    drop_strategy: str = "random",
    shuffle_points: bool = False,
    widths_are_normalized: bool = False,
    gripper_max_width: float | None = None,
    gripper_opening_max_width: float | None = None,
) -> np.ndarray:
    point_clouds_world = np.asarray(point_clouds_world, dtype=np.float32)
    current_pose9_grippers = np.asarray(current_pose9_grippers, dtype=np.float32)
    widths = normalize_gripper_widths(
        gripper_widths,
        already_normalized=widths_are_normalized,
        max_physical_width=gripper_max_width,
    )
    if len(point_clouds_world) != len(current_pose9_grippers):
        raise ValueError(
            f"Point cloud frames {len(point_clouds_world)} != pose frames {len(current_pose9_grippers)}"
        )
    if len(point_clouds_world) != len(widths):
        raise ValueError(f"Point cloud frames {len(point_clouds_world)} != gripper widths {len(widths)}")

    merged_eff = np.empty((len(point_clouds_world), int(total_points), 6), dtype=np.float32)
    for frame_idx, (cloud_world, pose9, width) in enumerate(
        zip(point_clouds_world, current_pose9_grippers, widths)
    ):
        merged_eff[frame_idx] = add_world_gripper_cloud_to_point_cloud(
            cloud_world,
            pose9,
            float(width),
            total_points=total_points,
            gripper_points=gripper_points,
            gripper_len=gripper_len,
            gripper_template=gripper_template,
            gripper_max_width=gripper_max_width,
            gripper_opening_max_width=gripper_opening_max_width,
            seed=seed + frame_idx,
            drop_strategy=drop_strategy,
            shuffle_points=shuffle_points,
        )
    return merged_eff


def backproject_camera(env: Any, raw_obs: dict[str, Any], camera_name: str, height: int, width: int) -> np.ndarray:
    """Back-project RGB-D pixels into that camera's own optical coordinate frame."""

    try:
        from robosuite.utils.camera_utils import (
            get_camera_intrinsic_matrix,
            get_real_depth_map,
        )
    except Exception as exc:  # pragma: no cover - optional LIBERO dependency path
        raise RuntimeError("robosuite camera utilities are required for LIBERO point-cloud generation.") from exc

    camera_name = normalize_camera_name(camera_name)
    image_key = f"{camera_name}_image"
    depth_key = f"{camera_name}_depth"
    if image_key not in raw_obs or depth_key not in raw_obs:
        available = ", ".join(sorted(raw_obs.keys()))
        raise KeyError(
            f"LIBERO observation is missing {image_key!r} or {depth_key!r}. "
            f"Available keys: {available}. Ensure OffScreenRenderEnv was created with camera_depths=True."
        )

    rgb = np.asarray(raw_obs[image_key], dtype=np.float32)
    depth = np.asarray(raw_obs[depth_key])
    depth = get_real_depth_map(env.sim, depth).astype(np.float32)
    if depth.ndim == 3:
        depth = depth[..., 0]

    depth = depth.reshape(height, width)
    vv, uu = np.meshgrid(np.arange(height, dtype=np.float32), np.arange(width, dtype=np.float32), indexing="ij")
    intrinsic = get_camera_intrinsic_matrix(env.sim, camera_name, height, width).astype(np.float32)
    z = depth
    x = (uu - intrinsic[0, 2]) * z / intrinsic[0, 0]
    # robosuite image arrays are indexed top-to-bottom, while the camera projection
    # matrix uses the opposite image-plane v convention.
    pixel_v = float(height - 1) - vv
    y = (pixel_v - intrinsic[1, 2]) * z / intrinsic[1, 1]
    camera_points = np.stack([x, y, z], axis=-1).reshape(-1, 3)

    colors = rgb.reshape(-1, 3)
    valid = np.isfinite(camera_points).all(axis=1) & np.isfinite(depth.reshape(-1)) & (depth.reshape(-1) > 0)
    return np.concatenate([camera_points[valid], colors[valid]], axis=1).astype(np.float32, copy=False)


def transform_point_cloud_reference(transform: np.ndarray, point_cloud: np.ndarray) -> np.ndarray:
    point_cloud = np.asarray(point_cloud, dtype=np.float32)
    output = point_cloud.copy()
    transform = np.asarray(transform, dtype=np.float32)
    output[..., :3] = point_cloud[..., :3] @ transform[:3, :3].T + transform[:3, 3]
    return output


def matrix_to_pose9_np(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    return np.concatenate(
        [matrix[..., :3, 3], matrix[..., :3, 0], matrix[..., :3, 1]],
        axis=-1,
    ).astype(np.float32)


def eef_pose9_world_to_reference_np(
    eef_pose_world: np.ndarray,
    reference_to_world: np.ndarray,
) -> np.ndarray:
    """Express a simulator-world EEF pose in a fixed reference frame.

    ``reference_to_world`` is the camera extrinsic used when the dataset point
    clouds were generated.  Any values after the pose9 prefix (for example the
    physical gripper width) are copied unchanged.
    """

    eef_pose_world = np.asarray(eef_pose_world, dtype=np.float32)
    reference_to_world = np.asarray(reference_to_world, dtype=np.float32)
    if eef_pose_world.ndim < 1 or eef_pose_world.shape[-1] < 9:
        raise ValueError(
            "eef_pose_world must end with at least 9 pose values, "
            f"got shape {eef_pose_world.shape}."
        )
    if reference_to_world.shape != (4, 4):
        raise ValueError(
            "reference_to_world must have shape (4, 4), "
            f"got {reference_to_world.shape}."
        )

    eef_to_world = pose9_to_homo_np(eef_pose_world[..., :9])
    world_to_reference = fast_inverse_homogeneous(reference_to_world)
    eef_to_reference = world_to_reference @ eef_to_world
    return np.concatenate(
        [matrix_to_pose9_np(eef_to_reference), eef_pose_world[..., 9:]],
        axis=-1,
    ).astype(np.float32)


def eef_pose9_world_to_reference_camera(
    env: Any,
    eef_pose_world: np.ndarray,
    reference_camera: str,
) -> np.ndarray:
    """Express a simulator-world EEF pose in a fixed LIBERO camera frame."""

    try:
        from robosuite.utils.camera_utils import get_camera_extrinsic_matrix
    except Exception as exc:  # pragma: no cover - optional LIBERO dependency path
        raise RuntimeError("robosuite camera utilities are required for LIBERO pose conversion.") from exc

    reference_camera = normalize_camera_name(reference_camera)
    reference_to_world = get_camera_extrinsic_matrix(env.sim, reference_camera).astype(np.float32)
    return eef_pose9_world_to_reference_np(eef_pose_world, reference_to_world)


def robot_base_to_world_matrix(env: Any) -> np.ndarray:
    """Return the calibrated robot-base pose in the simulator world frame."""

    robots = list(getattr(env, "robots", []))
    if len(robots) != 1:
        raise ValueError(
            "Robot-base WorldFlow currently requires exactly one robot, "
            f"got {len(robots)}."
        )
    root_body = getattr(getattr(robots[0], "robot_model", None), "root_body", None)
    if not root_body:
        raise ValueError("Could not resolve the robot model's root body.")
    position = np.asarray(env.sim.data.get_body_xpos(root_body), dtype=np.float32).reshape(3)
    rotation = np.asarray(env.sim.data.get_body_xmat(root_body), dtype=np.float32).reshape(3, 3)
    base_to_world = np.eye(4, dtype=np.float32)
    base_to_world[:3, :3] = rotation
    base_to_world[:3, 3] = position
    return base_to_world


def eef_pose9_world_to_robot_base(env: Any, eef_pose_world: np.ndarray) -> np.ndarray:
    """Express a simulator-world EEF pose in the calibrated robot-base frame."""

    return eef_pose9_world_to_reference_np(
        eef_pose_world,
        robot_base_to_world_matrix(env),
    )


def observation_to_camera_point_cloud(
    env: Any,
    raw_obs: dict[str, Any],
    camera_names: list[str],
    height: int,
    width: int,
    num_points: int,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return scene cloud and EEF pose in the first (Overview) camera frame.

    The third result keeps the simulator world-frame EEF pose for controller-only
    use during evaluation. It is not needed by dataset conversion or training.
    """

    try:
        from robosuite.utils.camera_utils import get_camera_extrinsic_matrix
    except Exception as exc:  # pragma: no cover - optional LIBERO dependency path
        raise RuntimeError("robosuite camera utilities are required for LIBERO point-cloud generation.") from exc

    normalized_names = [normalize_camera_name(name) for name in camera_names]
    if not normalized_names:
        raise ValueError("At least one fixed Overview camera is required.")

    reference_camera = normalized_names[0]
    reference_to_world = get_camera_extrinsic_matrix(env.sim, reference_camera).astype(np.float32)
    world_to_reference = fast_inverse_homogeneous(reference_to_world)
    clouds_reference = []
    for camera_name in normalized_names:
        cloud_camera = backproject_camera(env, raw_obs, camera_name, height, width)
        if camera_name != reference_camera:
            camera_to_world = get_camera_extrinsic_matrix(env.sim, camera_name).astype(np.float32)
            camera_to_reference = world_to_reference @ camera_to_world
            cloud_camera = transform_point_cloud_reference(camera_to_reference, cloud_camera)
        clouds_reference.append(cloud_camera)

    point_cloud_camera = sample_or_repeat_points(
        np.concatenate(clouds_reference, axis=0),
        num_points,
        seed=seed,
    )
    eef_pose_world = eef_pose9_gripper_from_obs(raw_obs)
    eef_to_world = pose9_to_homo_np(eef_pose_world[:9])
    eef_to_camera = world_to_reference @ eef_to_world
    eef_pose_camera = np.concatenate(
        [matrix_to_pose9_np(eef_to_camera), np.asarray([eef_pose_world[-1]], dtype=np.float32)]
    )
    return point_cloud_camera, eef_pose_camera.astype(np.float32), eef_pose_world.astype(np.float32)


def observation_to_point_clouds(
    env: Any,
    raw_obs: dict[str, Any],
    camera_names: list[str],
    height: int,
    width: int,
    num_points: int,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Backward-compatible helper returning current-EEF and world-frame clouds."""

    try:
        from robosuite.utils.camera_utils import get_camera_extrinsic_matrix
    except Exception as exc:  # pragma: no cover - optional LIBERO dependency path
        raise RuntimeError("robosuite camera utilities are required for LIBERO point-cloud generation.") from exc

    point_cloud_camera, eef_pose_camera, eef_pose_world = observation_to_camera_point_cloud(
        env,
        raw_obs,
        camera_names,
        height,
        width,
        num_points,
        seed=seed,
    )
    reference_camera = normalize_camera_name(camera_names[0])
    camera_to_world = get_camera_extrinsic_matrix(env.sim, reference_camera).astype(np.float32)
    point_cloud_world = transform_point_cloud_reference(camera_to_world, point_cloud_camera)
    point_cloud_eff = world_point_cloud_to_current_eff(point_cloud_world, eef_pose_world)
    return point_cloud_eff, point_cloud_world, eef_pose_world


def observation_to_model_point_cloud(
    env: Any,
    raw_obs: dict[str, Any],
    camera_views: str | list[str] | tuple[str, ...],
    height: int,
    width: int,
    num_points: int,
    *,
    add_gripper_cloud: bool = True,
    gripper_points: int = 500,
    gripper_len: float = 0.06,
    gripper_template: str = "reap",
    gripper_max_width: float = 0.08,
    gripper_drop_strategy: str = "tail",
    gripper_shuffle_points: bool = False,
    seed: int = 0,
    camera_view_weights: Any = None,
    camera_view_fusion: Any = "legacy_budget",
) -> tuple[np.ndarray, np.ndarray]:
    """Build the point cloud expected by the selected training fusion policy.

    Each selected camera first produces its own ``num_points`` cloud.  When the
    gripper cloud is enabled, every per-camera cloud keeps the training layout:
    scene points first and one addressable gripper tail.  The shared training
    composition helper then returns either the unchanged single-view cloud, a
    fixed-size legacy-budget cloud, or the full multi-view scene union consumed
    by the shared FPS sampler or primary-residual splitter. The first camera's
    gripper tail is appended once.

    Returns:
      model_point_cloud: xyzrgb in the current end-effector frame.
      eef_pose_world:    current simulator EEF pose9 + physical gripper width.
    """

    # Import lazily so this utility remains importable in lightweight LIBERO-only
    # tools that do not initialize the policy package.
    from lerobot.policies.smolvla.song_pointseg import (
        compose_point_cloud_views,
        parse_camera_view_fusion,
        parse_camera_views,
    )

    views = parse_camera_views(camera_views)
    stored_view_clouds: list[np.ndarray] = []
    eef_pose_world: np.ndarray | None = None
    total_points = int(num_points)
    gripper_points = int(gripper_points)

    for camera_index, camera_name in enumerate(views):
        per_view_seed = int(seed) + camera_index * 1_000_003
        point_cloud_eff, point_cloud_world, current_eef_pose_world = observation_to_point_clouds(
            env,
            raw_obs,
            [camera_name],
            int(height),
            int(width),
            total_points,
            seed=per_view_seed,
        )
        if bool(add_gripper_cloud):
            point_cloud_eff = add_world_gripper_cloud_to_point_cloud(
                point_cloud_world,
                current_eef_pose_world,
                gripper_width_percent_from_scalar(
                    float(current_eef_pose_world[-1]),
                    max_physical_width=float(gripper_max_width),
                ),
                total_points=total_points,
                gripper_points=gripper_points,
                gripper_len=float(gripper_len),
                gripper_template=str(gripper_template),
                # Dataset conversion uses one common gripper seed for all views.
                seed=int(seed),
                drop_strategy=str(gripper_drop_strategy),
                shuffle_points=bool(gripper_shuffle_points),
            )
        stored_view_clouds.append(np.ascontiguousarray(point_cloud_eff, dtype=np.float32))
        eef_pose_world = np.asarray(current_eef_pose_world, dtype=np.float32)

    if eef_pose_world is None:
        raise RuntimeError("No camera view was selected for point-cloud construction.")

    model_point_cloud = compose_point_cloud_views(
        stored_view_clouds,
        gripper_points=gripper_points if bool(add_gripper_cloud) else 0,
        seed=int(seed),
        view_weights=camera_view_weights,
        fusion=camera_view_fusion,
    )
    expected_points = total_points
    fusion_mode = parse_camera_view_fusion(camera_view_fusion)
    if fusion_mode in {
        "fps",
        "voxel_fps",
        "voxel_cover_fps",
        "novelty_union",
        "multiscale_novelty_union",
        "consensus_multiscale_novelty_union",
        "transport_novelty_union",
        "full_union",
        "primary_residual",
    } and len(views) > 1:
        scene_points = total_points - (gripper_points if bool(add_gripper_cloud) else 0)
        expected_points = len(views) * scene_points + (gripper_points if bool(add_gripper_cloud) else 0)
    expected_shape = (expected_points, 6)
    if model_point_cloud.shape != expected_shape:
        raise RuntimeError(
            f"Expected composed model point cloud shape {expected_shape}, got {model_point_cloud.shape}."
        )
    return np.ascontiguousarray(model_point_cloud, dtype=np.float32), eef_pose_world


def observation_to_world_point_cloud(
    env: Any,
    raw_obs: dict[str, Any],
    camera_names: list[str],
    height: int,
    width: int,
    num_points: int,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Backward-compatible world-frame wrapper around the camera-frame collector."""

    _point_cloud_eff, point_cloud_world, eef_pose_world = observation_to_point_clouds(
        env,
        raw_obs,
        camera_names,
        height,
        width,
        num_points,
        seed=seed,
    )
    return point_cloud_world, eef_pose_world


def action_pose9_to_libero(
    action_pose9_gripper: np.ndarray,
    trans_scale: float,
    rot_scale: float,
    gripper_threshold: float,
    *,
    gripper_max_width: float = 0.08,
    current_eef_pose9_gripper: np.ndarray | None = None,
) -> np.ndarray:
    action = np.asarray(action_pose9_gripper, dtype=np.float32).reshape(-1)
    if current_eef_pose9_gripper is None:
        delta_pos = action[:3]
        delta_rot = rot6d_to_matrix_np(action[3:9])
    else:
        current_world = pose9_to_homo_np(np.asarray(current_eef_pose9_gripper, dtype=np.float32)[..., :9])
        relative = pose9_to_homo_np(action[:9])
        target_world = current_world @ relative
        delta_pos = target_world[:3, 3] - current_world[:3, 3]
        delta_rot = target_world[:3, :3] @ current_world[:3, :3].T

    trans = np.clip(delta_pos / max(trans_scale, 1e-6), -1.0, 1.0)
    rotvec = R.from_matrix(delta_rot).as_rotvec().astype(np.float32)
    rotvec = np.clip(rotvec / max(rot_scale, 1e-6), -1.0, 1.0)
    # LIBERO / robosuite PandaGripper convention is -1=open, 1=closed.
    # The learned gripper label is stored as physical qpos width, so compare it
    # in normalized width space to keep config thresholds portable.
    gripper_width_percent = gripper_width_percent_from_scalar(
        float(action[-1]),
        max_physical_width=gripper_max_width,
    )
    gripper = -1.0 if gripper_width_percent + 1e-6 >= gripper_threshold else 1.0
    return np.concatenate([trans, rotvec, np.asarray([gripper], dtype=np.float32)]).astype(np.float32)


def get_task_init_states(task_suite: Any, task_id: int) -> np.ndarray:
    import torch

    from libero.libero import get_libero_path

    init_states_path = (
        Path(get_libero_path("init_states"))
        / task_suite.tasks[task_id].problem_folder
        / task_suite.tasks[task_id].init_states_file
    )
    return torch.load(init_states_path, weights_only=False)


def make_libero_env(
    task_suite: Any,
    task_id: int,
    height: int,
    width: int,
    camera_names: list[str],
    *,
    render_mode: str = "offscreen",
    render_camera: str | None = None,
    render_gpu_device_id: int = -1,
    control_delta: bool = True,
    control_freq=20,
    horizon: int = 1000,
    ignore_done: bool = False,
    env_seed: int | None = None,
):
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    from libero.libero.envs.env_wrapper import ControlEnv

    task = task_suite.get_task(task_id)
    bddl_file = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    cameras = [normalize_camera_name(camera_name) for camera_name in camera_names]
    render_mode = str(render_mode or "offscreen").lower()
    common_kwargs = {
        "bddl_file_name": str(bddl_file),
        "camera_heights": height,
        "camera_widths": width,
        "camera_names": cameras,
        "camera_depths": True,
        "control_freq": control_freq,
        "horizon": int(horizon),
        "ignore_done": bool(ignore_done),
    }
    viewer_camera = normalize_render_camera_name(render_camera, cameras[0])
    if render_mode in {"viewer3d", "mujoco"}:
        env = ControlEnv(
            **common_kwargs,
            has_renderer=False,
            has_offscreen_renderer=True,
            render_camera=viewer_camera,
            render_gpu_device_id=int(render_gpu_device_id),
        )
    elif render_mode in {"onscreen", "headed", "human"}:
        env = ControlEnv(
            **common_kwargs,
            has_renderer=True,
            has_offscreen_renderer=True,
            render_camera=viewer_camera,
            render_gpu_device_id=int(render_gpu_device_id),
        )
    elif render_mode in {"offscreen", "headless"}:
        env = OffScreenRenderEnv(**common_kwargs)
    else:
        raise ValueError(f"Unsupported LIBERO render_mode: {render_mode}")
    if env_seed is not None:
        env.seed(int(env_seed))
    env.reset()
    # Do not attach Viewer3D here.
    # run_episode() will reset/set_init_state again, which may replace sim/data.
    for robot in env.robots:
        robot.controller.use_delta = bool(control_delta)
    return env, task

def attach_mujoco_3d_viewer(env, render_camera="free", key_callback=None):
    inner_env = env

    while hasattr(inner_env, "env"):
        next_env = inner_env.env
        if next_env is inner_env:
            break
        inner_env = next_env

    sim = inner_env.sim

    # Current sim/model/data identity.
    sim_id = id(sim)
    model_obj = getattr(sim.model, "_model", sim.model)
    data_obj = getattr(sim.data, "_data", sim.data)
    model_id = id(model_obj)
    data_id = id(data_obj)

    existing_viewer = getattr(inner_env, "viewer", None)
    old_key = getattr(inner_env, "_viewer3d_key", None)
    # A passive-viewer key callback cannot be replaced after the viewer has
    # been launched. Include its identity so a new episode can install its own
    # callback even when reset() reuses the same MuJoCo model and data objects.
    callback_id = id(key_callback) if key_callback is not None else None
    new_key = (sim_id, model_id, data_id, callback_id)

    if existing_viewer is not None and hasattr(existing_viewer, "render") and old_key == new_key:
        return existing_viewer

    # If reset/set_init_state replaced sim/data, close old viewer and recreate.
    if existing_viewer is not None and hasattr(existing_viewer, "close"):
        try:
            existing_viewer.close()
        except Exception as e:
            print("[WARN] failed to close stale Viewer3D:", repr(e))
        inner_env.viewer = None

    print("[DEBUG] inner_env:", type(inner_env), type(inner_env).__module__)
    print("[DEBUG] sim:", type(sim), type(sim).__module__)
    print("[DEBUG] viewer key:", new_key)

    # New robosuite / DeepMind MuJoCo binding backend
    if type(sim).__module__ == "robosuite.utils.binding_utils":
        import mujoco
        import mujoco.viewer

        mj_model = model_obj
        mj_data = data_obj

        print("[DEBUG] mj_model:", type(mj_model), type(mj_model).__module__)
        print("[DEBUG] mj_data:", type(mj_data), type(mj_data).__module__)
        print("[DEBUG] Using mujoco.viewer.launch_passive")

        launch_kwargs = {
            "show_left_ui": True,
            "show_right_ui": True,
        }
        if key_callback is not None:
            launch_kwargs["key_callback"] = key_callback
        try:
            viewer = mujoco.viewer.launch_passive(mj_model, mj_data, **launch_kwargs)
        except TypeError as exc:
            # Keep compatibility with older MuJoCo releases. The evaluator
            # still has an immediate terminal-key fallback in this case.
            if "key_callback" not in launch_kwargs:
                raise
            print(f"[WARN] passive viewer does not support key_callback: {exc!r}")
            launch_kwargs.pop("key_callback")
            viewer = mujoco.viewer.launch_passive(mj_model, mj_data, **launch_kwargs)

        try:
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        except Exception as e:
            print("[WARN] failed to set free camera:", repr(e))

        class PassiveViewerWrapper:
            def __init__(self, viewer):
                self.viewer = viewer
                self.sync_count = 0

            def render(self):
                self.update()

            def update(self):
                if self.viewer is not None and self.viewer.is_running():
                    self.viewer.sync()
                    self.sync_count += 1
                    if self.sync_count % 100 == 0:
                        print("[DEBUG] Viewer3D synced", self.sync_count)

            def close(self):
                if self.viewer is not None:
                    self.viewer.close()
                    self.viewer = None

        inner_env.viewer = PassiveViewerWrapper(viewer)
        inner_env._viewer3d_key = new_key
        inner_env.has_renderer = True
        inner_env.renderer = "viewer3d"

        return inner_env.viewer

    # Old mujoco-py backend fallback
    try:
        import mujoco_py
        from robosuite.renderers.mujoco.mujoco_py_renderer import MujocoPyRenderer

        if isinstance(sim, mujoco_py.cymj.MjSim):
            print("[DEBUG] Using old MujocoPyRenderer")
            inner_env.viewer = MujocoPyRenderer(sim)
            inner_env._viewer3d_key = new_key
            return inner_env.viewer
    except Exception as e:
        print("[DEBUG] old MujocoPyRenderer unavailable:", repr(e))

    raise RuntimeError(
        f"Unsupported sim type for Viewer3D: {type(sim)} from {type(sim).__module__}"
    )
