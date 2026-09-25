#!/usr/bin/env python

from __future__ import annotations

import argparse
import json
import os
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

os.environ.setdefault("HF_DATASETS_CACHE", "/tmp/lerobot_hf_datasets_cache")

from lerobot.datasets.lerobot_dataset import LeRobotDataset


DEFAULT_ROOT = Path(
    os.environ.get(
        "SONG_SYNTHETIC_ROOT",
        "/home/liusong/ProgramFiles/Huggingface/lerobot/outputs/datasets/pick_place_synthetic_worldflow",
    )
)

POINT_CLOUD_DIR_NAME = "point_clouds"
POINT_CLOUD_KEY = "observation.point_cloud"

TASK_PREFIX = "Pick the red cube A and place it above the blue cube B."
PHASES = [
    ("approach_red_cube", 0.00, 0.15, f"{TASK_PREFIX} Phase: approach red cube A."),
    ("descend_to_grasp", 0.15, 0.26, f"{TASK_PREFIX} Phase: descend to grasp red cube A."),
    ("close_gripper", 0.26, 0.34, f"{TASK_PREFIX} Phase: close the two-finger gripper on red cube A."),
    ("lift_red_cube", 0.34, 0.50, f"{TASK_PREFIX} Phase: lift red cube A."),
    ("carry_above_blue", 0.50, 0.75, f"{TASK_PREFIX} Phase: carry red cube A above blue cube B."),
    ("lower_onto_blue", 0.75, 0.89, f"{TASK_PREFIX} Phase: lower red cube A onto blue cube B."),
    ("release_red_cube", 0.89, 0.95, f"{TASK_PREFIX} Phase: release red cube A above blue cube B."),
    ("retreat", 0.95, 1.00, f"{TASK_PREFIX} Phase: retreat after placing red cube A."),
]

POSE10_NAMES = ["x", "y", "z", "x1", "y1", "z1", "x2", "y2", "z2", "gripper"]
DATASET_FEATURES = {
    "action": {"dtype": "float32", "shape": (10,), "names": POSE10_NAMES},
    "observation.state": {"dtype": "float32", "shape": (10,), "names": POSE10_NAMES},
}


@dataclass(frozen=True)
class SyntheticPickPlaceConfig:
    root: str
    episodes: int = 100
    steps: int = 100
    points: int = 10000
    fps: int = 30
    seed: int = 20260525
    cube_size: float = 0.05
    table_z: float = 0.0
    place_gap: float = 0.002
    finger_length: float = 0.075
    finger_thickness: float = 0.007
    wrist_length: float = 0.070
    open_width: float = 0.085
    closed_width: float = 0.034
    debug_episode: int = 0
    debug_steps: tuple[int, ...] = (0, 25, 50, 75, 99)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic world-frame Pick A Place above B data.")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--episodes", type=int, default=int(os.environ.get("SONG_SYNTHETIC_EPISODES", "100")))
    parser.add_argument("--steps", type=int, default=int(os.environ.get("SONG_SYNTHETIC_STEPS", "100")))
    parser.add_argument("--points", type=int, default=int(os.environ.get("SONG_SYNTHETIC_POINTS", "10000")))
    parser.add_argument("--fps", type=int, default=int(os.environ.get("SONG_SYNTHETIC_FPS", "30")))
    parser.add_argument("--seed", type=int, default=int(os.environ.get("SONG_SYNTHETIC_SEED", "20260525")))
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--no-overwrite", action="store_true")
    parser.add_argument("--skip-debug-ply", action="store_true")
    return parser.parse_args()


def rotation_z(yaw: float) -> np.ndarray:
    c = np.cos(yaw)
    s = np.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)


def matrix_to_rot6d(matrix: np.ndarray) -> np.ndarray:
    return np.concatenate([matrix[:, 0], matrix[:, 1]], axis=0).astype(np.float32, copy=False)


def pose10(position: np.ndarray, rotation: np.ndarray, gripper_width: float) -> np.ndarray:
    return np.concatenate(
        [position.astype(np.float32), matrix_to_rot6d(rotation), np.array([gripper_width], dtype=np.float32)]
    )


def smoothstep(x: float) -> float:
    x = float(np.clip(x, 0.0, 1.0))
    return x * x * (3.0 - 2.0 * x)


def interpolate_piecewise(ratio: float, keypoints: list[tuple[float, np.ndarray]]) -> np.ndarray:
    if ratio <= keypoints[0][0]:
        return keypoints[0][1].copy()
    for (r0, p0), (r1, p1) in zip(keypoints[:-1], keypoints[1:], strict=True):
        if ratio <= r1:
            alpha = smoothstep((ratio - r0) / max(r1 - r0, 1e-6))
            return ((1.0 - alpha) * p0 + alpha * p1).astype(np.float32)
    return keypoints[-1][1].copy()


def interpolate_gripper_width(ratio: float, cfg: SyntheticPickPlaceConfig) -> float:
    if ratio < 0.26:
        return cfg.open_width
    if ratio < 0.34:
        alpha = smoothstep((ratio - 0.26) / 0.08)
        return float((1.0 - alpha) * cfg.open_width + alpha * cfg.closed_width)
    if ratio < 0.89:
        return cfg.closed_width
    if ratio < 0.95:
        alpha = smoothstep((ratio - 0.89) / 0.06)
        return float((1.0 - alpha) * cfg.closed_width + alpha * cfg.open_width)
    return cfg.open_width


def phase_for_step(step: int, steps: int) -> tuple[int, str, str]:
    ratio = (step + 0.5) / steps
    for idx, (name, start, end, task) in enumerate(PHASES):
        if start <= ratio < end or (idx == len(PHASES) - 1 and ratio <= end):
            return idx, name, task
    return len(PHASES) - 1, PHASES[-1][0], PHASES[-1][3]


def random_cube_centers(rng: np.random.Generator, cube_size: float) -> tuple[np.ndarray, np.ndarray]:
    half = cube_size / 2.0
    for _ in range(1000):
        red = np.array(
            [rng.uniform(-0.22, -0.08), rng.uniform(-0.13, 0.13), half],
            dtype=np.float32,
        )
        blue = np.array(
            [rng.uniform(0.08, 0.22), rng.uniform(-0.13, 0.13), half],
            dtype=np.float32,
        )
        if np.linalg.norm(red[:2] - blue[:2]) > 0.22:
            return red, blue
    return red, blue


def sample_box_surface(
    rng: np.random.Generator,
    n_points: int,
    size: tuple[float, float, float],
    center: tuple[float, float, float] | np.ndarray = (0.0, 0.0, 0.0),
) -> np.ndarray:
    size_arr = np.asarray(size, dtype=np.float32)
    center_arr = np.asarray(center, dtype=np.float32)
    points = rng.uniform(-0.5, 0.5, size=(n_points, 3)).astype(np.float32) * size_arr
    faces = rng.integers(0, 6, size=n_points)

    axis = faces // 2
    sign = (faces % 2) * 2 - 1
    for ax in range(3):
        mask = axis == ax
        points[mask, ax] = sign[mask].astype(np.float32) * size_arr[ax] * 0.5

    return points + center_arr


def sample_table_background(rng: np.random.Generator, n_points: int) -> np.ndarray:
    n_table = int(n_points * 0.78)
    n_wall = n_points - n_table

    table_xyz = np.empty((n_table, 3), dtype=np.float32)
    table_xyz[:, 0] = rng.uniform(-0.36, 0.36, size=n_table)
    table_xyz[:, 1] = rng.uniform(-0.26, 0.26, size=n_table)
    table_xyz[:, 2] = rng.normal(0.0, 0.0015, size=n_table)

    wall_xyz = np.empty((n_wall, 3), dtype=np.float32)
    wall_xyz[:, 0] = rng.uniform(-0.36, 0.36, size=n_wall)
    wall_xyz[:, 1] = rng.normal(0.27, 0.002, size=n_wall)
    wall_xyz[:, 2] = rng.uniform(0.0, 0.24, size=n_wall)

    xyz = np.concatenate([table_xyz, wall_xyz], axis=0)
    colors = np.empty((n_points, 3), dtype=np.float32)
    colors[:n_table] = np.array([142.0, 142.0, 136.0], dtype=np.float32)
    colors[n_table:] = np.array([112.0, 128.0, 128.0], dtype=np.float32)
    colors += rng.normal(0.0, 7.0, size=colors.shape).astype(np.float32)
    return np.concatenate([xyz, np.clip(colors, 0.0, 255.0)], axis=1)


def split_point_counts(total_points: int) -> dict[str, int]:
    n_red = int(round(total_points * 0.25))
    n_blue = int(round(total_points * 0.25))
    n_gripper = int(round(total_points * 0.18))
    n_background = total_points - n_red - n_blue - n_gripper
    if min(n_red, n_blue, n_gripper, n_background) <= 0:
        raise ValueError(f"Need enough points for all scene components, got {total_points}.")
    return {
        "red_cube": n_red,
        "blue_cube": n_blue,
        "gripper": n_gripper,
        "background": n_background,
    }


def build_gripper_local_points(
    rng: np.random.Generator,
    n_points: int,
    gripper_width: float,
    cfg: SyntheticPickPlaceConfig,
) -> np.ndarray:
    n_left = n_points // 4
    n_right = n_points // 4
    n_palm = n_points // 4
    n_wrist = n_points - n_left - n_right - n_palm
    thick = cfg.finger_thickness

    left = sample_box_surface(
        rng,
        n_left,
        (thick, thick, cfg.finger_length),
        (0.0, -gripper_width / 2.0, -cfg.finger_length / 2.0),
    )
    right = sample_box_surface(
        rng,
        n_right,
        (thick, thick, cfg.finger_length),
        (0.0, gripper_width / 2.0, -cfg.finger_length / 2.0),
    )
    palm = sample_box_surface(
        rng,
        n_palm,
        (thick, gripper_width + 2.0 * thick, thick),
        (0.0, 0.0, 0.0),
    )
    wrist = sample_box_surface(
        rng,
        n_wrist,
        (1.35 * thick, 1.35 * thick, cfg.wrist_length),
        (0.0, 0.0, cfg.wrist_length / 2.0),
    )
    return np.concatenate([left, right, palm, wrist], axis=0).astype(np.float32, copy=False)


def transform_points(points: np.ndarray, rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    return points @ rotation.T + translation


def shuffle_rows(rng: np.random.Generator, points: np.ndarray) -> np.ndarray:
    order = rng.permutation(points.shape[0])
    return np.ascontiguousarray(points[order], dtype=np.float32)


def create_episode_scene(
    episode_index: int,
    cfg: SyntheticPickPlaceConfig,
) -> dict[str, np.ndarray | list[str]]:
    rng = np.random.default_rng(cfg.seed + episode_index * 9973)
    counts = split_point_counts(cfg.points)
    half = cfg.cube_size / 2.0

    red_start, blue_center = random_cube_centers(rng, cfg.cube_size)
    red_final = blue_center + np.array([0.0, 0.0, cfg.cube_size + cfg.place_gap], dtype=np.float32)
    ee_offset = np.array([0.0, 0.0, cfg.finger_length + half], dtype=np.float32)
    yaw = float(rng.uniform(-0.35, 0.35))
    ee_rot = rotation_z(yaw)

    start_xy_offset = rng.uniform(-0.025, 0.025, size=2).astype(np.float32)
    start = red_start + np.array([start_xy_offset[0], start_xy_offset[1], 0.225], dtype=np.float32)
    pre_grasp = red_start + np.array([0.0, 0.0, 0.190], dtype=np.float32)
    grasp = red_start + ee_offset
    lift = red_start + np.array([0.0, 0.0, 0.170], dtype=np.float32) + ee_offset
    above_blue = red_final + np.array([0.0, 0.0, 0.145], dtype=np.float32) + ee_offset
    place = red_final + ee_offset
    retreat_xy_offset = rng.uniform(-0.030, 0.030, size=2).astype(np.float32)
    retreat = red_final + np.array([retreat_xy_offset[0], retreat_xy_offset[1], 0.215], dtype=np.float32)

    ee_keypoints = [
        (0.00, start),
        (0.15, pre_grasp),
        (0.26, grasp),
        (0.34, grasp),
        (0.50, lift),
        (0.75, above_blue),
        (0.89, place),
        (0.95, place),
        (1.00, retreat),
    ]

    red_local = sample_box_surface(rng, counts["red_cube"], (cfg.cube_size, cfg.cube_size, cfg.cube_size))
    blue_local = sample_box_surface(rng, counts["blue_cube"], (cfg.cube_size, cfg.cube_size, cfg.cube_size))
    background = sample_table_background(rng, counts["background"])

    blue_points = np.concatenate(
        [
            blue_local + blue_center,
            np.tile(np.array([[42.0, 86.0, 232.0]], dtype=np.float32), (counts["blue_cube"], 1)),
        ],
        axis=1,
    )

    actions = np.empty((cfg.steps, 10), dtype=np.float32)
    point_clouds = np.empty((cfg.steps, cfg.points, 6), dtype=np.float32)
    red_centers = np.empty((cfg.steps, 3), dtype=np.float32)
    ee_positions = np.empty((cfg.steps, 3), dtype=np.float32)
    gripper_widths = np.empty((cfg.steps,), dtype=np.float32)
    phase_indices = np.empty((cfg.steps,), dtype=np.int64)
    phase_names: list[str] = []
    tasks: list[str] = []

    attach_offset_world = ee_rot @ np.array([0.0, 0.0, -(cfg.finger_length + half)], dtype=np.float32)

    for step in range(cfg.steps):
        ratio = step / max(cfg.steps - 1, 1)
        frame_rng = np.random.default_rng(cfg.seed + episode_index * 100_000 + step)
        ee_position = interpolate_piecewise(ratio, ee_keypoints)
        gripper_width = interpolate_gripper_width(ratio, cfg)

        if ratio < 0.34:
            red_center = red_start
        elif ratio < 0.89:
            red_center = ee_position + attach_offset_world
        else:
            red_center = red_final

        red_points = np.concatenate(
            [
                red_local + red_center,
                np.tile(np.array([[226.0, 46.0, 36.0]], dtype=np.float32), (counts["red_cube"], 1)),
            ],
            axis=1,
        )
        gripper_local = build_gripper_local_points(frame_rng, counts["gripper"], gripper_width, cfg)
        gripper_xyz = transform_points(gripper_local, ee_rot, ee_position)
        gripper_points = np.concatenate(
            [
                gripper_xyz,
                np.tile(np.array([[218.0, 218.0, 204.0]], dtype=np.float32), (counts["gripper"], 1)),
            ],
            axis=1,
        )

        scene = np.concatenate([red_points, blue_points, gripper_points, background], axis=0)
        point_clouds[step] = shuffle_rows(frame_rng, scene)
        actions[step] = pose10(ee_position, ee_rot, gripper_width)
        red_centers[step] = red_center
        ee_positions[step] = ee_position
        gripper_widths[step] = gripper_width
        phase_idx, phase_name, task = phase_for_step(step, cfg.steps)
        phase_indices[step] = phase_idx
        phase_names.append(phase_name)
        tasks.append(task)

    return {
        "actions": actions,
        "point_clouds": point_clouds,
        "tasks": tasks,
        "phase_indices": phase_indices,
        "phase_names": np.array(phase_names, dtype=object),
        "red_centers": red_centers,
        "red_start": red_start,
        "red_final": red_final,
        "blue_center": blue_center,
        "ee_positions": ee_positions,
        "ee_rotation": ee_rot,
        "gripper_widths": gripper_widths,
        "counts": counts,
    }


def write_point_cloud_meta(root: Path, cfg: SyntheticPickPlaceConfig) -> None:
    pc_dir = root / POINT_CLOUD_DIR_NAME
    pc_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "key": POINT_CLOUD_KEY,
        "dtype": "float32",
        "shape": [cfg.points, 6],
        "layout": "episode_npy",
        "path_format": f"{POINT_CLOUD_DIR_NAME}/episode_{{episode_index:06d}}.npy",
        "components": split_point_counts(cfg.points),
    }
    with open(pc_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)


def make_episode_buffer(dataset: LeRobotDataset, episode: dict[str, np.ndarray | list[str]], cfg: SyntheticPickPlaceConfig) -> dict:
    actions = np.ascontiguousarray(episode["actions"], dtype=np.float32)
    episode_buffer = dataset.create_episode_buffer()
    episode_buffer["size"] = cfg.steps
    episode_buffer["task"] = list(episode["tasks"])
    episode_buffer["frame_index"] = np.arange(cfg.steps, dtype=np.int64)
    episode_buffer["timestamp"] = np.arange(cfg.steps, dtype=np.float32) / cfg.fps
    episode_buffer["action"] = actions
    episode_buffer["observation.state"] = actions.copy()
    return episode_buffer


def save_episode(dataset: LeRobotDataset, episode: dict[str, np.ndarray | list[str]], cfg: SyntheticPickPlaceConfig) -> None:
    episode_index = dataset.meta.total_episodes
    pc_dir = dataset.root / POINT_CLOUD_DIR_NAME
    pc_dir.mkdir(parents=True, exist_ok=True)
    np.save(pc_dir / f"episode_{episode_index:06d}.npy", np.ascontiguousarray(episode["point_clouds"], dtype=np.float32))
    np.savez_compressed(
        pc_dir / f"episode_{episode_index:06d}_annotations.npz",
        actions=episode["actions"],
        phase_indices=episode["phase_indices"],
        phase_names=episode["phase_names"],
        red_centers=episode["red_centers"],
        red_start=episode["red_start"],
        red_final=episode["red_final"],
        blue_center=episode["blue_center"],
        ee_positions=episode["ee_positions"],
        ee_rotation=episode["ee_rotation"],
        gripper_widths=episode["gripper_widths"],
    )
    dataset.save_episode(episode_data=make_episode_buffer(dataset, episode, cfg))


def write_ply(path: Path, cloud: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    xyz = cloud[:, :3]
    rgb = np.clip(cloud[:, 3:6], 0, 255).astype(np.uint8)
    with open(path, "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {cloud.shape[0]}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        for point, color in zip(xyz, rgb, strict=True):
            f.write(
                f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f} "
                f"{int(color[0])} {int(color[1])} {int(color[2])}\n"
            )


def write_debug_outputs(root: Path, episode_index: int, episode: dict[str, np.ndarray | list[str]], cfg: SyntheticPickPlaceConfig) -> None:
    if episode_index != cfg.debug_episode:
        return

    debug_dir = root / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)
    for step in cfg.debug_steps:
        if 0 <= step < cfg.steps:
            write_ply(debug_dir / f"episode_{episode_index:06d}_step_{step:03d}.ply", episode["point_clouds"][step])

    phase_summary = []
    last_phase = None
    for step, phase_name in enumerate(episode["phase_names"]):
        if phase_name != last_phase:
            phase_summary.append({"step": step, "phase": str(phase_name), "task": episode["tasks"][step]})
            last_phase = phase_name

    summary = {
        "episode_index": episode_index,
        "red_start": np.asarray(episode["red_start"]).tolist(),
        "red_final": np.asarray(episode["red_final"]).tolist(),
        "blue_center": np.asarray(episode["blue_center"]).tolist(),
        "phase_summary": phase_summary,
        "debug_steps": list(cfg.debug_steps),
    }
    with open(debug_dir / f"episode_{episode_index:06d}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)


def write_dataset_description(root: Path, cfg: SyntheticPickPlaceConfig) -> None:
    meta_dir = root / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    with open(meta_dir / "synthetic_worldflow_config.json", "w") as f:
        json.dump(asdict(cfg), f, indent=2)
    with open(meta_dir / "synthetic_phase_tasks.json", "w") as f:
        json.dump(
            [
                {"phase_index": idx, "phase": name, "start_ratio": start, "end_ratio": end, "task": task}
                for idx, (name, start, end, task) in enumerate(PHASES)
            ],
            f,
            indent=2,
        )


def verify_dataset(root: Path, expected: SyntheticPickPlaceConfig | None = None) -> dict:
    info_path = root / "meta" / "info.json"
    pc_meta_path = root / POINT_CLOUD_DIR_NAME / "meta.json"
    if not info_path.exists():
        raise FileNotFoundError(f"Missing LeRobot info file: {info_path}")
    if not pc_meta_path.exists():
        raise FileNotFoundError(f"Missing point cloud meta file: {pc_meta_path}")

    with open(info_path) as f:
        info = json.load(f)
    with open(pc_meta_path) as f:
        pc_meta = json.load(f)

    episodes = int(info["total_episodes"])
    frames = int(info["total_frames"])
    points = int(pc_meta["shape"][0])
    if expected is not None:
        assert episodes == expected.episodes, (episodes, expected.episodes)
        assert frames == expected.episodes * expected.steps, (frames, expected.episodes * expected.steps)
        assert points == expected.points, (points, expected.points)

    first_pc = np.load(root / POINT_CLOUD_DIR_NAME / "episode_000000.npy", mmap_mode="r")
    first_ann = np.load(root / POINT_CLOUD_DIR_NAME / "episode_000000_annotations.npz", allow_pickle=True)
    if first_pc.ndim != 3 or first_pc.shape[-1] != 6:
        raise AssertionError(f"Unexpected point cloud array shape: {first_pc.shape}")
    if first_pc.shape[1] != points:
        raise AssertionError(f"Point count mismatch: {first_pc.shape[1]} != {points}")

    phase_names = first_ann["phase_names"]
    if len(set(phase_names.tolist())) != len(PHASES):
        raise AssertionError(f"Expected {len(PHASES)} phases, got {sorted(set(phase_names.tolist()))}")

    red_final = first_ann["red_final"]
    blue_center = first_ann["blue_center"]
    cube_size = expected.cube_size if expected is not None else 0.05
    place_gap = expected.place_gap if expected is not None else 0.002
    expected_red_z = blue_center[2] + cube_size + place_gap
    if np.linalg.norm(red_final[:2] - blue_center[:2]) > 1e-5 or abs(float(red_final[2] - expected_red_z)) > 1e-5:
        raise AssertionError("Final red cube center is not above the blue cube.")

    actions = first_ann["actions"]
    red_centers = first_ann["red_centers"]
    phase_indices = first_ann["phase_indices"]
    gripper = actions[:, -1]
    if not (gripper.min() < gripper.max() and gripper[0] > gripper.min() and gripper[-1] > gripper.min()):
        raise AssertionError("Gripper width does not open-close-open across the episode.")

    attached = (phase_indices >= 3) & (phase_indices <= 5)
    if attached.any():
        cfg = expected or SyntheticPickPlaceConfig(root=str(root))
        offset = actions[attached, :3] - red_centers[attached]
        offset_norm = np.linalg.norm(offset, axis=1)
        expected_offset = cfg.finger_length + cfg.cube_size / 2.0
        if float(np.max(np.abs(offset_norm - expected_offset))) > 5e-4:
            raise AssertionError("Attached red cube does not track the end-effector offset.")

    fps = int(info["fps"])
    chunk_size = min(32, first_pc.shape[0])
    os.environ.setdefault("HF_DATASETS_CACHE", "/tmp/lerobot_hf_datasets_cache")
    dataset = LeRobotDataset(
        str(root),
        delta_timestamps={
            "action": [i / fps for i in range(chunk_size)],
            "observation.state": [0.0],
        },
    )
    sample0 = dataset[0]
    sample_mid = dataset[first_pc.shape[0] // 2]
    if tuple(sample0["action"].shape) != (chunk_size, 10):
        raise AssertionError(f"Unexpected action chunk shape from LeRobotDataset: {sample0['action'].shape}")
    if tuple(sample0["observation.state"].shape) != (1, 10):
        raise AssertionError(f"Unexpected state shape from LeRobotDataset: {sample0['observation.state'].shape}")

    return {
        "episodes": episodes,
        "frames": frames,
        "points_per_frame": points,
        "point_cloud_shape_ep0": list(first_pc.shape),
        "num_phase_tasks": len(PHASES),
        "phase_names": sorted(set(phase_names.tolist())),
        "red_final": red_final.tolist(),
        "blue_center": blue_center.tolist(),
        "sample_action_chunk_shape": list(sample0["action"].shape),
        "sample_state_shape": list(sample0["observation.state"].shape),
        "sample_task_step0": sample0["task"],
        "sample_task_mid": sample_mid["task"],
    }


def generate_dataset(cfg: SyntheticPickPlaceConfig, overwrite: bool = True, write_debug: bool = True) -> None:
    root = Path(cfg.root)
    if root.exists():
        if not overwrite:
            raise FileExistsError(f"{root} already exists. Remove it or omit --no-overwrite.")
        if root.is_dir():
            shutil.rmtree(root)
        else:
            root.unlink()
    dataset = LeRobotDataset.create(
        repo_id="pick_place_synthetic_worldflow",
        fps=cfg.fps,
        features=DATASET_FEATURES,
        robot_type="synthetic_two_finger",
        root=root,
        use_videos=False,
    )
    write_point_cloud_meta(dataset.root, cfg)
    write_dataset_description(dataset.root, cfg)

    for episode_index in range(cfg.episodes):
        episode = create_episode_scene(episode_index, cfg)
        save_episode(dataset, episode, cfg)
        if write_debug:
            write_debug_outputs(dataset.root, episode_index, episode, cfg)

    dataset.finalize()
    summary = verify_dataset(root, cfg)
    print(json.dumps(summary, indent=2))


def main() -> None:
    args = parse_args()
    cfg = SyntheticPickPlaceConfig(
        root=str(args.root),
        episodes=args.episodes,
        steps=args.steps,
        points=args.points,
        fps=args.fps,
        seed=args.seed,
    )
    if args.verify_only:
        print(json.dumps(verify_dataset(args.root, cfg), indent=2))
        return
    generate_dataset(cfg, overwrite=not args.no_overwrite, write_debug=not args.skip_debug_ply)


if __name__ == "__main__":
    main()
