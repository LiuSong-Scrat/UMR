#!/usr/bin/env python

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import h5py
import numpy as np

from lerobot.scripts.generate_pick_place_worldflow_dataset import (
    DEFAULT_ROOT,
    PHASES,
    TASK_PREFIX,
    SyntheticPickPlaceConfig,
    create_episode_scene,
    write_ply,
)


DEFAULT_HDF5_ROOT = Path(
    os.environ.get(
        "SONG_SYNTHETIC_HDF5_ROOT",
        "/home/liusong/ProgramFiles/Huggingface/lerobot/outputs/datasets/pick_place_synthetic_hdf5",
    )
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic Pick A Place B source HDF5 files.")
    parser.add_argument("--root", type=Path, default=DEFAULT_HDF5_ROOT)
    parser.add_argument("--episodes", type=int, default=int(os.environ.get("SONG_SYNTHETIC_EPISODES", "100")))
    parser.add_argument("--steps", type=int, default=int(os.environ.get("SONG_SYNTHETIC_STEPS", "100")))
    parser.add_argument("--points", type=int, default=int(os.environ.get("SONG_SYNTHETIC_POINTS", "10000")))
    parser.add_argument("--fps", type=int, default=int(os.environ.get("SONG_SYNTHETIC_FPS", "30")))
    parser.add_argument("--seed", type=int, default=int(os.environ.get("SONG_SYNTHETIC_SEED", "20260525")))
    parser.add_argument("--compression", choices=["none", "lzf", "gzip"], default="lzf")
    parser.add_argument("--gzip-level", type=int, default=1)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--no-overwrite", action="store_true")
    parser.add_argument("--skip-debug-ply", action="store_true")
    return parser.parse_args()


def hdf5_string_dtype():
    return h5py.string_dtype(encoding="utf-8")


def euler_zyx_from_episode(episode: dict[str, np.ndarray | list[str]]) -> np.ndarray:
    actions = np.asarray(episode["actions"], dtype=np.float32)
    rotation = np.asarray(episode["ee_rotation"], dtype=np.float32)
    yaw = np.arctan2(rotation[1, 0], rotation[0, 0]).astype(np.float32)
    pose_eular = np.zeros((actions.shape[0], 6), dtype=np.float32)
    pose_eular[:, :3] = actions[:, :3]
    pose_eular[:, 3] = yaw
    return pose_eular


def dataset_kwargs(compression: str, gzip_level: int) -> dict:
    if compression == "none":
        return {}
    if compression == "gzip":
        return {"compression": "gzip", "compression_opts": gzip_level}
    return {"compression": "lzf"}


def write_episode_hdf5(
    root: Path,
    episode_index: int,
    episode: dict[str, np.ndarray | list[str]],
    cfg: SyntheticPickPlaceConfig,
    compression: str,
    gzip_level: int,
) -> None:
    path = root / f"episode_{episode_index:06d}.hdf5"
    str_dtype = hdf5_string_dtype()
    kwargs = dataset_kwargs(compression, gzip_level)

    with h5py.File(path, "w") as f:
        f.attrs["source"] = "synthetic_pick_place_worldflow"
        f.attrs["episode_index"] = episode_index
        f.attrs["fps"] = cfg.fps
        f.attrs["points_per_frame"] = cfg.points
        f.attrs["steps"] = cfg.steps
        f.attrs["task"] = TASK_PREFIX
        f.create_dataset("episode_task_name", data=TASK_PREFIX, dtype=str_dtype)
        f.create_dataset("task_name", data=np.asarray(episode["tasks"], dtype=object), dtype=str_dtype)

        obs = f.create_group("observations")
        obs.create_dataset("pose_eular", data=euler_zyx_from_episode(episode), dtype="float32")
        obs.create_dataset(
            "eff_angular",
            data=(np.asarray(episode["gripper_widths"], dtype=np.float32) * 2.0).reshape(-1, 1),
            dtype="float32",
        )
        obs.create_dataset("phase_index", data=np.asarray(episode["phase_indices"], dtype=np.int64))
        obs.create_dataset("phase_name", data=np.asarray(episode["phase_names"], dtype=object), dtype=str_dtype)
        obs.create_dataset("red_cube_center", data=np.asarray(episode["red_centers"], dtype=np.float32))
        obs.create_dataset("ee_position", data=np.asarray(episode["ee_positions"], dtype=np.float32))
        obs.create_dataset("gripper_width", data=np.asarray(episode["gripper_widths"], dtype=np.float32))

        objects = f.create_group("objects")
        objects.create_dataset("red_cube_start", data=np.asarray(episode["red_start"], dtype=np.float32))
        objects.create_dataset("red_cube_final", data=np.asarray(episode["red_final"], dtype=np.float32))
        objects.create_dataset("blue_cube_center", data=np.asarray(episode["blue_center"], dtype=np.float32))

        cloud_group = obs.create_group("cloud_rgb")
        cloud_group.create_dataset(
            "overhead",
            data=np.asarray(episode["point_clouds"], dtype=np.float32),
            dtype="float32",
            chunks=(1, cfg.points, 6),
            **kwargs,
        )


def write_metadata(root: Path, cfg: SyntheticPickPlaceConfig) -> None:
    with open(root / "synthetic_hdf5_config.json", "w") as f:
        json.dump(
            {
                **cfg.__dict__,
                "lerobot_output_root": str(DEFAULT_ROOT),
                "phase_tasks": [
                    {"phase_index": idx, "phase": name, "start_ratio": start, "end_ratio": end, "task": task}
                    for idx, (name, start, end, task) in enumerate(PHASES)
                ],
            },
            f,
            indent=2,
        )


def write_debug(root: Path, episode_index: int, episode: dict[str, np.ndarray | list[str]], cfg: SyntheticPickPlaceConfig) -> None:
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
    with open(debug_dir / f"episode_{episode_index:06d}_summary.json", "w") as f:
        json.dump(
            {
                "episode_index": episode_index,
                "red_start": np.asarray(episode["red_start"]).tolist(),
                "red_final": np.asarray(episode["red_final"]).tolist(),
                "blue_center": np.asarray(episode["blue_center"]).tolist(),
                "phase_summary": phase_summary,
                "debug_steps": list(cfg.debug_steps),
            },
            f,
            indent=2,
        )


def verify_hdf5_root(root: Path, cfg: SyntheticPickPlaceConfig | None = None) -> dict:
    h5_paths = sorted(root.glob("*.hdf5"))
    if not h5_paths:
        raise FileNotFoundError(f"No .hdf5 files found in {root}")

    with h5py.File(h5_paths[0], "r") as f:
        cloud = f["observations/cloud_rgb/overhead"]
        pose = f["observations/pose_eular"]
        tasks = [task.decode("utf-8") if isinstance(task, bytes) else str(task) for task in f["task_name"][()]]
        phase_names = [
            phase.decode("utf-8") if isinstance(phase, bytes) else str(phase)
            for phase in f["observations/phase_name"][()]
        ]
        red_final = f["objects/red_cube_final"][()]
        blue_center = f["objects/blue_cube_center"][()]
        gripper_width = f["observations/gripper_width"][()]

        if cfg is not None:
            assert len(h5_paths) == cfg.episodes, (len(h5_paths), cfg.episodes)
            assert cloud.shape == (cfg.steps, cfg.points, 6), cloud.shape
            assert pose.shape == (cfg.steps, 6), pose.shape
            assert len(tasks) == cfg.steps, len(tasks)

        if len(set(phase_names)) != len(PHASES):
            raise AssertionError(f"Expected all {len(PHASES)} phases, got {sorted(set(phase_names))}")
        if not (float(gripper_width.min()) < float(gripper_width.max())):
            raise AssertionError("Gripper width does not change.")

        cube_size = cfg.cube_size if cfg is not None else 0.05
        place_gap = cfg.place_gap if cfg is not None else 0.002
        expected_red_z = blue_center[2] + cube_size + place_gap
        if np.linalg.norm(red_final[:2] - blue_center[:2]) > 1e-5 or abs(float(red_final[2] - expected_red_z)) > 1e-5:
            raise AssertionError("Final red cube center is not above the blue cube.")

        return {
            "hdf5_files": len(h5_paths),
            "first_file": str(h5_paths[0]),
            "cloud_shape": list(cloud.shape),
            "pose_eular_shape": list(pose.shape),
            "unique_phase_tasks": len(set(tasks)),
            "phase_names": sorted(set(phase_names)),
            "first_task": tasks[0],
            "middle_task": tasks[len(tasks) // 2],
            "red_final": red_final.tolist(),
            "blue_center": blue_center.tolist(),
        }


def generate_hdf5_root(
    cfg: SyntheticPickPlaceConfig,
    overwrite: bool,
    write_debug_ply: bool,
    compression: str,
    gzip_level: int,
) -> None:
    root = Path(cfg.root)
    if root.exists():
        if not overwrite:
            raise FileExistsError(f"{root} already exists. Remove it or omit --no-overwrite.")
        if root.is_dir():
            shutil.rmtree(root)
        else:
            root.unlink()
    root.mkdir(parents=True, exist_ok=True)
    write_metadata(root, cfg)

    for episode_index in range(cfg.episodes):
        episode = create_episode_scene(episode_index, cfg)
        write_episode_hdf5(root, episode_index, episode, cfg, compression, gzip_level)
        if write_debug_ply:
            write_debug(root, episode_index, episode, cfg)

    print(json.dumps(verify_hdf5_root(root, cfg), indent=2))


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
        print(json.dumps(verify_hdf5_root(args.root, cfg), indent=2))
        return
    generate_hdf5_root(
        cfg,
        overwrite=not args.no_overwrite,
        write_debug_ply=not args.skip_debug_ply,
        compression=args.compression,
        gzip_level=args.gzip_level,
    )


if __name__ == "__main__":
    main()
