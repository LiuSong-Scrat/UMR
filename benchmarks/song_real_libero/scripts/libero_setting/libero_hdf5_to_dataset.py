#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import re
import shutil
import sys
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.smolvla.song_pointseg import (
    episode_point_cloud_npy_path,
    episode_point_cloud_zarr_path,
    open_episode_point_clouds,
    save_episode_point_clouds_zarr,
    save_point_clouds_zarr,
)

if __package__ and __package__.startswith("benchmarks."):
    from .._paths import DEFAULT_LIBERO_CONFIG, LIBERO_DATA_ROOT, load_json_config
    from .libero_pointcloud_utils import (
        add_reference_gripper_clouds_to_episode,
        ensure_libero_config,
        eef_pose9_gripper_from_obs,
        fast_inverse_homogeneous,
        make_libero_env,
        normalize_camera_name,
        observation_to_camera_point_cloud,
        pointcloud_camera_names_from_config,
        pose9_to_homo_np,
        reference_point_cloud_to_current_eff,
        render_camera_names_from_config,
        robot_base_to_world_matrix,
    )
else:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from _paths import DEFAULT_LIBERO_CONFIG, LIBERO_DATA_ROOT, load_json_config
    from libero_setting.libero_pointcloud_utils import (
        add_reference_gripper_clouds_to_episode,
        ensure_libero_config,
        eef_pose9_gripper_from_obs,
        fast_inverse_homogeneous,
        make_libero_env,
        normalize_camera_name,
        observation_to_camera_point_cloud,
        pointcloud_camera_names_from_config,
        pose9_to_homo_np,
        reference_point_cloud_to_current_eff,
        render_camera_names_from_config,
        robot_base_to_world_matrix,
    )

from tqdm import tqdm

POINT_CLOUD_DIR_NAME = "point_clouds"
WORLD_EE_POSE_DIR_NAME = "world_ee_poses"
ACTION_TARGET_EE_POSE_DIR_NAME = "action_target_ee_poses"
WORLD_BASE_EE_POSE_DIR_NAME = "world_base_ee_poses"
WORLD_BASE_ACTION_TARGET_EE_POSE_DIR_NAME = "world_base_action_target_ee_poses"
POINT_CLOUD_CHANNELS = 6
ACTION_LABEL_SEMANTICS = (
    "causal_same_state_raw_delta_absolute_model_eef_target_"
    "with_next_achieved_gripper_width"
)
OBSERVATION_STATE_SEMANTICS = "pre_action_achieved_eef_pose_and_gripper_width"

DATASET_FEATURES = {
    "action": {
        "dtype": "float32",
        "shape": (10,),
        "names": ["x", "y", "z", "x1", "y1", "z1", "x2", "y2", "z2", "gripper"],
    },
    "observation.state": {
        "dtype": "float32",
        "shape": (10,),
        "names": ["x", "y", "z", "x1", "y1", "z1", "x2", "y2", "z2", "gripper"],
    },
}


def image_feature_key(camera: str) -> str:
    return f"observation.images.{str(camera).strip()}"


def selected_camera_names(cfg: dict[str, Any]) -> list[str]:
    value = cfg.get("selected_cameras")
    if value is None:
        value = pointcloud_camera_names_from_config(cfg)
    if isinstance(value, str):
        value = [value]
    cameras: list[str] = []
    for camera in value or []:
        name = normalize_camera_name(str(camera))
        if name and name not in cameras:
            cameras.append(name)
    if not cameras:
        raise ValueError("At least one camera must be selected.")
    return cameras


def image_feature_cameras(cfg: dict[str, Any]) -> list[str]:
    if not bool(cfg.get("save_rgb_images", True)):
        return []
    value = cfg.get("image_cameras")
    if value is None:
        value = selected_camera_names(cfg)
    if isinstance(value, str):
        value = [value]
    cameras: list[str] = []
    for camera in value or []:
        name = normalize_camera_name(str(camera))
        if name and name not in cameras:
            cameras.append(name)
    return cameras


def image_feature_camera(cfg: dict[str, Any]) -> str:
    return image_feature_cameras(cfg)[0]


def camera_token(camera: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", normalize_camera_name(camera))


def point_cloud_dir_name(camera: str, primary_camera: str) -> str:
    return POINT_CLOUD_DIR_NAME if normalize_camera_name(camera) == normalize_camera_name(primary_camera) else f"point_clouds_{camera_token(camera)}"


def point_cloud_artifact_name(camera: str, primary_camera: str, storage: str) -> str:
    suffix = "zarr" if storage == "zarr" else "npy"
    return f"{point_cloud_dir_name(camera, primary_camera)}.{suffix}"


def image_artifact_name(camera: str) -> str:
    return f"images_{camera_token(camera)}.npy"


def rendered_camera_names(cfg: dict[str, Any]) -> list[str]:
    names = [normalize_camera_name(name) for name in render_camera_names_from_config(cfg)]
    for camera in [*selected_camera_names(cfg), *image_feature_cameras(cfg)]:
        if camera not in names:
            names.append(camera)
    return names


def ensure_image_camera_rendered(cfg: dict[str, Any]) -> None:
    # Kept for compatibility with existing call sites. rendered_camera_names()
    # performs the actual de-duplicated resolution used by workers.
    if bool(cfg.get("save_rgb_images", True)):
        cfg["image_cameras"] = image_feature_cameras(cfg)


def dataset_features_with_image(cfg: dict[str, Any]) -> dict[str, Any]:
    features = dict(DATASET_FEATURES)
    shape = (
        int(cfg.get("observation_height", 128)),
        int(cfg.get("observation_width", 128)),
        3,
    )
    for camera in image_feature_cameras(cfg):
        features[image_feature_key(camera)] = {
            "dtype": "image",
            "shape": shape,
            "names": ["height", "width", "channels"],
        }
    return features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert LIBERO demonstrations into the Song point-cloud LeRobot format.")
    parser.add_argument("--config", type=Path, default=DEFAULT_LIBERO_CONFIG)
    parser.add_argument("--suite", action="append", default=None)
    parser.add_argument("--all-tasks", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--task-id", type=int, action="append", default=None)
    parser.add_argument("--demo-root", type=Path, default=None)
    parser.add_argument("--demo-file", type=Path, action="append", default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--repo-id", default=None)
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--max-frames-per-demo", type=int, default=None)
    parser.add_argument("--num-points", type=int, default=None)
    parser.add_argument(
        "--camera",
        action="append",
        default=None,
        help=(
            "Camera to store as a point cloud. Repeat for multiple views. Unless "
            "--image-camera is set, the same cameras are also stored as RGB. The "
            "first camera keeps the legacy point_clouds/ directory."
        ),
    )
    parser.add_argument("--point-cloud-storage", choices=("zarr", "npy"), default=None)
    parser.add_argument("--zarr-compression-level", type=int, default=None)
    parser.add_argument("--fps", type=int, default=None)
    parser.add_argument("--replay-mode", choices=("states", "step"), default=None)
    parser.add_argument(
        "--state-observation-offset",
        type=int,
        choices=(0, 1),
        default=None,
        help=(
            "Optional source-index offset in states replay mode. Output frame i uses "
            "observation states[i + offset] and source actions[i + offset], so the "
            "observation and arm action remain causally aligned. The default is 0."
        ),
    )
    parser.add_argument(
        "--restore-demo-model",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Restore every demonstration's HDF5 model_file before replay. This is required to "
            "preserve LIBERO fixture placements that are not stored in flattened simulator states."
        ),
    )
    parser.add_argument(
        "--require-source-fps-match",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Require the output dataset FPS to match the source LIBERO control frequency.",
    )
    parser.add_argument("--download-demos", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--download-use-huggingface", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--vis-dir", type=Path, default=None)
    parser.add_argument("--vis-count", type=int, default=None)
    parser.add_argument("--vis-stride", type=int, default=None)
    parser.add_argument("--save-video", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--save-rgb-images", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument(
        "--image-camera",
        action="append",
        default=None,
        help=(
            "Camera to store as RGB. Repeat for multiple RGB views. This is independent "
            "of --camera, so two point-cloud views can share one RGB model input."
        ),
    )
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument(
        "--worker-scope",
        choices=("task", "episode"),
        default=None,
        help=(
            "Parallel work unit. 'task' reuses one environment for all selected demos of a task; "
            "'episode' creates an isolated environment per demo and can use more workers when only "
            "a few tasks are selected."
        ),
    )
    parser.add_argument(
        "--resume-temp-artifacts",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Validate and reuse complete episode artifacts already present in --tmp-dir. "
            "Any existing artifact that fails strict validation aborts the run."
        ),
    )
    parser.add_argument("--tmp-dir", type=Path, default=None)
    parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=None)
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    return load_json_config(
        path,
        path_keys=("dataset_output_root", "libero_config_path", "demo_root", "vis_dir", "output_dir"),
    )


def cfg_get(cfg: dict[str, Any], cli_value: Any, key: str, default: Any = None) -> Any:
    return cli_value if cli_value is not None else cfg.get(key, default)


def resolve_suite_names(cli_suites: list[str] | None, cfg: dict[str, Any]) -> list[str]:
    suites = cli_suites if cli_suites is not None else cfg.get("suites", cfg.get("suite", "libero_object"))
    if isinstance(suites, str):
        suites = [suites]
    suites = [str(suite) for suite in suites]
    if not suites:
        raise ValueError("At least one LIBERO suite must be configured.")
    return suites


def resolve_task_ids_for_suite(
    *,
    suite_name: str,
    task_count: int,
    cli_task_ids: list[int] | None,
    cfg: dict[str, Any],
) -> list[int]:
    if bool(cfg.get("all_tasks", False)):
        return list(range(task_count))
    if cli_task_ids is not None:
        task_ids = cli_task_ids
    else:
        suite_task_ids = cfg.get("suite_task_ids", {})
        if isinstance(suite_task_ids, dict) and suite_name in suite_task_ids:
            task_ids = suite_task_ids[suite_name]
        else:
            task_ids = cfg.get("task_ids")
    if task_ids is None:
        return list(range(task_count))
    resolved = [int(task_id) for task_id in task_ids]
    invalid = [task_id for task_id in resolved if task_id < 0 or task_id >= task_count]
    if invalid:
        raise ValueError(f"Invalid task id(s) for {suite_name}: {invalid}; valid range is [0, {task_count - 1}]")
    return resolved


def camera_image_keys(camera_names: list[str]) -> list[str]:
    keys = []
    for camera_name in camera_names:
        name = str(camera_name)
        keys.append(name if name.endswith("_image") else f"{name}_image")
    return keys


def image_from_raw_obs(raw_obs: dict[str, Any], image_key: str) -> np.ndarray | None:
    if image_key not in raw_obs:
        return None
    image = np.asarray(raw_obs[image_key])
    if image.ndim != 3 or image.shape[-1] != 3:
        return None
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    return image


def dataset_image_from_raw_obs(raw_obs: dict[str, Any], camera_name: str) -> np.ndarray | None:
    image_key = camera_name if str(camera_name).endswith("_image") else f"{camera_name}_image"
    image = image_from_raw_obs(raw_obs, image_key)
    if image is None:
        return None
    if image_key == "agentview_image":
        image = np.ascontiguousarray(image[:, ::-1])
    return image.copy()


def append_video_frames(video_frames: dict[str, list[np.ndarray]], raw_obs: dict[str, Any], camera_names: list[str]) -> None:
    for image_key in camera_image_keys(camera_names):
        image = image_from_raw_obs(raw_obs, image_key)
        if image is not None:
            if image_key == "agentview_image":
                image = np.ascontiguousarray(image[:, ::-1])
            video_frames.setdefault(image_key, []).append(image.copy())


def write_video(path: Path, frames: list[np.ndarray], fps: int) -> None:
    if not frames:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import imageio.v3 as iio

        iio.imwrite(path, np.asarray(frames, dtype=np.uint8), fps=fps)
        return
    except Exception:
        pass

    import cv2

    first = np.asarray(frames[0], dtype=np.uint8)
    height, width = first.shape[:2]
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {path}")
    try:
        for frame in frames:
            frame = np.asarray(frame, dtype=np.uint8)
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()


def export_episode_videos(episode: dict[str, Any], video_dir: Path, record: dict[str, Any], cfg: dict[str, Any]) -> list[str]:
    if not cfg.get("save_video", False):
        return []
    video_frames = episode.get("video_frames") or {}
    if not video_frames:
        return []

    video_dir_name = record.get("video_dir_name")
    if video_dir_name is None:
        video_dir_name = f"episode_{int(record['episode_index']):06d}_{record['demo_name']}"

    episode_dir = video_dir / video_dir_name

    written = []
    for image_key, frames in video_frames.items():
        safe_key = re.sub(r"[^A-Za-z0-9_.-]+", "_", image_key)
        path = episode_dir / f"{safe_key}.mp4"
        write_video(path, frames, int(cfg["fps"]))
        written.append(str(path))
    return written


def normalized_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def natural_key(path: Path):
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", path.as_posix())]


def point_cloud_file(root: Path, episode_index: int) -> Path:
    return episode_point_cloud_npy_path(root / POINT_CLOUD_DIR_NAME, episode_index)


def point_cloud_storage_path(root: Path, episode_index: int, storage: str) -> Path:
    if storage == "zarr":
        return episode_point_cloud_zarr_path(root / POINT_CLOUD_DIR_NAME, episode_index)
    return point_cloud_file(root, episode_index)


def point_cloud_storage_path_for_camera(
    root: Path,
    camera: str,
    primary_camera: str,
    episode_index: int,
    storage: str,
) -> Path:
    directory = root / point_cloud_dir_name(camera, primary_camera)
    if storage == "zarr":
        return episode_point_cloud_zarr_path(directory, episode_index)
    return episode_point_cloud_npy_path(directory, episode_index)


def world_ee_pose_file(root: Path, episode_index: int) -> Path:
    return root / WORLD_EE_POSE_DIR_NAME / f"episode_{episode_index:06d}.npy"


def action_target_ee_pose_file(root: Path, episode_index: int) -> Path:
    return root / ACTION_TARGET_EE_POSE_DIR_NAME / f"episode_{episode_index:06d}.npy"


def world_base_ee_pose_file(root: Path, episode_index: int) -> Path:
    return root / WORLD_BASE_EE_POSE_DIR_NAME / f"episode_{episode_index:06d}.npy"


def world_base_action_target_ee_pose_file(root: Path, episode_index: int) -> Path:
    return root / WORLD_BASE_ACTION_TARGET_EE_POSE_DIR_NAME / f"episode_{episode_index:06d}.npy"


def write_point_cloud_meta(
    root: Path,
    storage: str = "zarr",
    *,
    cameras: list[str] | None = None,
    gripper_points: int = 500,
) -> None:
    cameras = list(cameras or ["agentview"])
    primary_camera = cameras[0]
    suffix = "zarr" if storage == "zarr" else "npy"
    view_dirs: dict[str, str] = {}
    for camera in cameras:
        directory_name = point_cloud_dir_name(camera, primary_camera)
        view_dirs[camera] = directory_name
        pc_dir = root / directory_name
        pc_dir.mkdir(parents=True, exist_ok=True)
        meta = {
            "key": (
                "observation.point_cloud"
                if camera == primary_camera
                else f"observation.point_cloud_views.{camera}"
            ),
            "camera": camera,
            "dtype": "float32",
            "shape": [None, POINT_CLOUD_CHANNELS],
            "variable_num_points": True,
            "layout": "episode_array",
            "storage_format": storage,
            "path_format": f"{directory_name}/episode_{{episode_index:06d}}.{suffix}",
            "coordinate_frame": "current_eff",
            "contains_gripper_template": True,
            "gripper_points": int(gripper_points),
            "gripper_at_tail": True,
        }
        if storage == "zarr":
            meta["zarr_encoding"] = "packed_xyz_float16_rgb_uint8"
        with open(pc_dir / "meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
    with open(root / "point_cloud_views.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "primary_camera": primary_camera,
                "camera_directories": view_dirs,
                "gripper_points": int(gripper_points),
                "fused_training": (
                    "split the non-gripper budget equally across selected cameras "
                    "and append one gripper tail"
                ),
            },
            f,
            indent=2,
        )


def write_worldflow_meta(root: Path) -> None:
    pose_dir = root / WORLD_EE_POSE_DIR_NAME
    pose_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "key": "worldflow.ee_poses",
        "dtype": "float32",
        "shape": [9],
        "layout": "episode_npy",
        "path_format": f"{WORLD_EE_POSE_DIR_NAME}/episode_{{episode_index:06d}}.npy",
        "coordinate_frame": "overview_camera",
        "legacy_directory_name": True,
        "sim_extrinsic_usage": "eef_world_to_overview_camera_only",
    }
    with open(pose_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def write_action_target_meta(root: Path) -> None:
    pose_dir = root / ACTION_TARGET_EE_POSE_DIR_NAME
    pose_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "key": "action.absolute_osc_target_pose",
        "dtype": "float32",
        "shape": [9],
        "layout": "episode_npy",
        "path_format": (
            f"{ACTION_TARGET_EE_POSE_DIR_NAME}/episode_{{episode_index:06d}}.npy"
        ),
        "coordinate_frame": "overview_camera",
        "source": "raw LIBERO HDF5 normalized OSC_POSE delta action",
        "source_state_anchor": "states[source_index]",
        "action_index_mapping": (
            "output[i] uses source actions[source_index], where "
            "source_index = i + state_observation_offset"
        ),
        "conversion": (
            "controller.set_goal(raw[:6]) with use_delta=True, including "
            "controller orientation-goal history; map the controller-site goal "
            "through the source-state controller-to-model EEF transform; no "
            "heuristic displacement or overshoot"
        ),
    }
    with open(pose_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def write_robot_base_worldflow_meta(root: Path) -> None:
    base_pose_dir = root / WORLD_BASE_EE_POSE_DIR_NAME
    target_pose_dir = root / WORLD_BASE_ACTION_TARGET_EE_POSE_DIR_NAME
    base_pose_dir.mkdir(parents=True, exist_ok=True)
    target_pose_dir.mkdir(parents=True, exist_ok=True)
    with open(base_pose_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "key": "worldflow.current_ee_pose",
                "dtype": "float32",
                "shape": [9],
                "layout": "episode_npy",
                "path_format": (
                    f"{WORLD_BASE_EE_POSE_DIR_NAME}/episode_{{episode_index:06d}}.npy"
                ),
                "coordinate_frame": "robot_base",
                "source": "exact MuJoCo robot root-body transform",
            },
            f,
            indent=2,
        )
    with open(target_pose_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "key": "worldflow.eef_trajectory",
                "dtype": "float32",
                "shape": [9],
                "layout": "episode_npy",
                "path_format": (
                    f"{WORLD_BASE_ACTION_TARGET_EE_POSE_DIR_NAME}/"
                    "episode_{episode_index:06d}.npy"
                ),
                "coordinate_frame": "robot_base",
                "target_semantics": "commanded_eef_pose",
                "source": "raw LIBERO OSC target mapped to model EEF and robot-base frame",
                "implicit_point_flow": (
                    "the EEF trajectory is a supervised task-relevant point-flow trace; "
                    "no simulator object pose or dense scene flow is used"
                ),
            },
            f,
            indent=2,
        )


def save_episode_point_clouds(
    root: Path,
    episode_index: int,
    point_clouds: np.ndarray,
    *,
    storage: str = "zarr",
    zarr_compression_level: int = 3,
) -> None:
    point_clouds = np.ascontiguousarray(point_clouds, dtype=np.float32)
    if point_clouds.ndim != 3 or point_clouds.shape[-1] != POINT_CLOUD_CHANNELS:
        raise ValueError(f"Expected point clouds shape (T, N, 6), got {point_clouds.shape}")
    if storage == "zarr":
        save_episode_point_clouds_zarr(
            root / POINT_CLOUD_DIR_NAME,
            episode_index,
            point_clouds,
            compression_level=int(zarr_compression_level),
        )
    else:
        path = point_cloud_file(root, episode_index)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, point_clouds)


def save_episode_worldflow(root: Path, episode_index: int, reference_ee_poses: np.ndarray) -> None:
    reference_ee_poses = np.ascontiguousarray(reference_ee_poses, dtype=np.float32)
    if reference_ee_poses.ndim != 2 or reference_ee_poses.shape[-1] != 9:
        raise ValueError(f"Expected reference-frame ee poses shape (T, 9), got {reference_ee_poses.shape}")
    path = world_ee_pose_file(root, episode_index)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, reference_ee_poses)


def save_episode_action_targets(
    root: Path,
    episode_index: int,
    action_target_ee_poses: np.ndarray,
) -> None:
    action_target_ee_poses = np.ascontiguousarray(action_target_ee_poses, dtype=np.float32)
    if action_target_ee_poses.ndim != 2 or action_target_ee_poses.shape[-1] != 9:
        raise ValueError(
            "Expected reference-frame absolute action targets shape (T, 9), "
            f"got {action_target_ee_poses.shape}"
        )
    path = action_target_ee_pose_file(root, episode_index)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, action_target_ee_poses)


def save_episode_images_to_paths(
    dataset: LeRobotDataset,
    images: np.ndarray,
    image_key: str,
    episode_index: int,
) -> list[str]:
    paths = []
    for frame_index, image in enumerate(np.asarray(images, dtype=np.uint8)):
        image_path = dataset._get_image_file_path(  # noqa: SLF001
            episode_index=episode_index,
            image_key=image_key,
            frame_index=frame_index,
        )
        if frame_index == 0:
            image_path.parent.mkdir(parents=True, exist_ok=True)
        dataset._save_image(image, image_path, compress_level=6)  # noqa: SLF001
        paths.append(str(image_path))
    return paths


def homo_to_pose9(H: np.ndarray) -> np.ndarray:
    H = np.asarray(H, dtype=np.float32)
    return np.concatenate([H[..., :3, 3], H[..., :3, 0], H[..., :3, 1]], axis=-1).astype(np.float32)


def from_reference_to_umi_tra_pose9(
    pose9_eff_to_reference: np.ndarray,
    *,
    origin_pose9_eff_to_reference: np.ndarray | None = None,
) -> np.ndarray:
    """Express poses in one explicit episode-origin EEF frame.

    Both achieved observation poses and commanded action targets must use the
    same origin. In particular, an action target is not allowed to become its
    own origin: doing that would erase the target displacement at frame zero.
    """

    reference_transforms = pose9_to_homo_np(pose9_eff_to_reference)
    if reference_transforms.ndim != 3 or reference_transforms.shape[-2:] != (4, 4):
        raise ValueError(
            f"Expected pose sequence with shape (T, 9), got {np.asarray(pose9_eff_to_reference).shape}"
        )
    if origin_pose9_eff_to_reference is None:
        origin_pose9_eff_to_reference = np.asarray(pose9_eff_to_reference)[0]
    origin_to_reference = pose9_to_homo_np(
        np.asarray(origin_pose9_eff_to_reference, dtype=np.float32)
    )
    reference_to_origin = fast_inverse_homogeneous(origin_to_reference)
    eff_to_origin = reference_to_origin @ reference_transforms
    return homo_to_pose9(eff_to_origin)


def make_episode_buffer(
    dataset: LeRobotDataset,
    task: str,
    actions: np.ndarray,
    observation_states: np.ndarray,
    timestamps: np.ndarray,
    images_by_camera: dict[str, np.ndarray] | None = None,
) -> dict[str, Any]:
    actions = np.ascontiguousarray(actions, dtype=np.float32)
    observation_states = np.ascontiguousarray(observation_states, dtype=np.float32)
    timestamps = np.asarray(timestamps, dtype=np.float32).reshape(-1)
    if observation_states.shape != actions.shape:
        raise ValueError(
            "observation.state must be frame-aligned with action and have the same "
            f"shape, got state={observation_states.shape}, action={actions.shape}."
        )
    episode_buffer = dataset.create_episode_buffer()
    episode_buffer["size"] = len(actions)
    episode_buffer["task"] = [task] * len(actions)
    episode_buffer["frame_index"] = np.arange(len(actions), dtype=np.int64)
    episode_buffer["timestamp"] = timestamps
    episode_buffer["action"] = actions
    episode_buffer["observation.state"] = observation_states
    for camera, images in (images_by_camera or {}).items():
        images = np.asarray(images, dtype=np.uint8)
        if len(images) != len(actions):
            raise ValueError(
                f"Image frame count for {camera!r} is {len(images)}, "
                f"but actions contain {len(actions)} frames."
            )
        key = image_feature_key(camera)
        episode_buffer[key] = save_episode_images_to_paths(
            dataset, images, key, int(episode_buffer["episode_index"])
        )
    return episode_buffer


def save_converted_episode(dataset: LeRobotDataset, episode: dict[str, Any]) -> None:
    episode_index = dataset.meta.total_episodes
    save_episode_point_clouds(dataset.root, episode_index, episode["point_clouds"])
    save_episode_worldflow(dataset.root, episode_index, episode["world_ee_poses"])
    save_episode_action_targets(
        dataset.root,
        episode_index,
        episode["action_target_ee_poses"],
    )
    world_base_path = world_base_ee_pose_file(dataset.root, episode_index)
    world_base_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(world_base_path, np.ascontiguousarray(episode["world_base_ee_poses"], dtype=np.float32))
    base_target_path = world_base_action_target_ee_pose_file(dataset.root, episode_index)
    base_target_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(
        base_target_path,
        np.ascontiguousarray(episode["world_base_action_target_ee_poses"], dtype=np.float32),
    )
    dataset.save_episode(
        episode_data=make_episode_buffer(
            dataset,
            episode["task"],
            episode["actions"],
            episode["observation_states"],
            episode["timestamps"],
            images_by_camera=episode.get("images_by_camera"),
        )
    )


def move_episode_array(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_dir():
        shutil.rmtree(dst)
    elif dst.exists():
        dst.unlink()
    shutil.move(str(src), str(dst))


def save_episode_artifact(artifact_dir: Path, episode: dict[str, Any], record: dict[str, Any], cfg: dict[str, Any]) -> None:
    if artifact_dir.exists():
        shutil.rmtree(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    storage = str(cfg.get("point_cloud_storage", "zarr"))
    primary_camera = selected_camera_names(cfg)[0]
    point_clouds_by_camera = dict(episode.get("point_clouds_by_camera") or {primary_camera: episode["point_clouds"]})
    for camera, point_clouds in point_clouds_by_camera.items():
        artifact_path = artifact_dir / point_cloud_artifact_name(camera, primary_camera, storage)
        if storage == "zarr":
            save_point_clouds_zarr(
                artifact_path,
                point_clouds,
                compression_level=int(cfg.get("zarr_compression_level", 3)),
            )
        else:
            np.save(artifact_path, np.ascontiguousarray(point_clouds, dtype=np.float32))
    np.save(artifact_dir / "world_ee_poses.npy", np.ascontiguousarray(episode["world_ee_poses"], dtype=np.float32))
    np.save(
        artifact_dir / "action_target_ee_poses.npy",
        np.ascontiguousarray(episode["action_target_ee_poses"], dtype=np.float32),
    )
    np.save(
        artifact_dir / "world_base_ee_poses.npy",
        np.ascontiguousarray(episode["world_base_ee_poses"], dtype=np.float32),
    )
    np.save(
        artifact_dir / "world_base_action_target_ee_poses.npy",
        np.ascontiguousarray(episode["world_base_action_target_ee_poses"], dtype=np.float32),
    )
    np.save(artifact_dir / "actions.npy", np.ascontiguousarray(episode["actions"], dtype=np.float32))
    np.save(
        artifact_dir / "observation_states.npy",
        np.ascontiguousarray(episode["observation_states"], dtype=np.float32),
    )
    np.save(artifact_dir / "timestamps.npy", np.asarray(episode["timestamps"], dtype=np.float32))
    for camera, images in (episode.get("images_by_camera") or {}).items():
        np.save(artifact_dir / image_artifact_name(camera), np.ascontiguousarray(images, dtype=np.uint8))
    with open(artifact_dir / "record.json", "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False)
    if cfg.get("save_video", False):
        video_record = {**record, "video_dir_name": "videos"}
        export_episode_videos(episode, artifact_dir, video_record, cfg)


def verify_episode_artifact(artifact_dir: Path, episode_job: dict[str, Any]) -> None:
    """Validate worker identity and frame alignment before the parent commits an episode."""
    required_files = (
        "record.json",
        "actions.npy",
        "observation_states.npy",
        "timestamps.npy",
        "world_ee_poses.npy",
        "action_target_ee_poses.npy",
        "world_base_ee_poses.npy",
        "world_base_action_target_ee_poses.npy",
    )
    missing = [name for name in required_files if not (artifact_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Incomplete worker artifact {artifact_dir}: missing {missing}")

    with open(artifact_dir / "record.json", "r", encoding="utf-8") as f:
        record = json.load(f)
    identity_errors = []
    for key in ("suite", "task_id", "demo_name"):
        if record.get(key) != episode_job.get(key):
            identity_errors.append(
                f"{key}: expected {episode_job.get(key)!r}, got {record.get(key)!r}"
            )
    if Path(record["demo_file"]).expanduser().resolve() != Path(
        episode_job["demo_file"]
    ).expanduser().resolve():
        identity_errors.append(
            f"demo_file: expected {episode_job['demo_file']!r}, got {record['demo_file']!r}"
        )
    restore_required = bool(episode_job["cfg"].get("restore_demo_model", True))
    if bool(record.get("model_restoration_verified", False)) != restore_required:
        identity_errors.append(
            "model_restoration_verified does not match restore_demo_model configuration"
        )
    if identity_errors:
        raise RuntimeError(
            f"Worker artifact identity mismatch for job {episode_job['job_index']}:\n  "
            + "\n  ".join(identity_errors)
        )

    actions = np.load(artifact_dir / "actions.npy", mmap_mode="r")
    observation_states = np.load(artifact_dir / "observation_states.npy", mmap_mode="r")
    timestamps = np.load(artifact_dir / "timestamps.npy", mmap_mode="r")
    world_ee_poses = np.load(artifact_dir / "world_ee_poses.npy", mmap_mode="r")
    action_target_ee_poses = np.load(
        artifact_dir / "action_target_ee_poses.npy", mmap_mode="r"
    )
    world_base_ee_poses = np.load(
        artifact_dir / "world_base_ee_poses.npy", mmap_mode="r"
    )
    world_base_action_target_ee_poses = np.load(
        artifact_dir / "world_base_action_target_ee_poses.npy", mmap_mode="r"
    )
    frames = int(actions.shape[0])
    shape_errors = []
    if actions.ndim != 2 or actions.shape[1] != 10:
        shape_errors.append(f"actions shape={actions.shape}, expected (T, 10)")
    if observation_states.shape != (frames, 10):
        shape_errors.append(
            f"observation_states shape={observation_states.shape}, expected ({frames}, 10)"
        )
    if timestamps.shape != (frames,):
        shape_errors.append(f"timestamps shape={timestamps.shape}, expected ({frames},)")
    if world_ee_poses.shape != (frames, 9):
        shape_errors.append(f"world_ee_poses shape={world_ee_poses.shape}, expected ({frames}, 9)")
    if action_target_ee_poses.shape != (frames, 9):
        shape_errors.append(
            "action_target_ee_poses shape="
            f"{action_target_ee_poses.shape}, expected ({frames}, 9)"
        )
    if world_base_ee_poses.shape != (frames, 9):
        shape_errors.append(
            f"world_base_ee_poses shape={world_base_ee_poses.shape}, expected ({frames}, 9)"
        )
    if world_base_action_target_ee_poses.shape != (frames, 9):
        shape_errors.append(
            "world_base_action_target_ee_poses shape="
            f"{world_base_action_target_ee_poses.shape}, expected ({frames}, 9)"
        )
    if int(record.get("frames", -1)) != frames:
        shape_errors.append(f"record frames={record.get('frames')}, actions frames={frames}")
    if record.get("action_label_semantics") != ACTION_LABEL_SEMANTICS:
        shape_errors.append(
            "record action_label_semantics does not identify causal raw-delta absolute targets"
        )
    if record.get("observation_state_semantics") != OBSERVATION_STATE_SEMANTICS:
        shape_errors.append(
            "record observation_state_semantics does not identify pre-action achieved EEF states"
        )
    if record.get("gripper_action_semantics") != "next_state_achieved_physical_width_metres":
        shape_errors.append(
            "record gripper_action_semantics does not identify the next-state physical-width label"
        )
    if not record.get("gripper_action_mapping"):
        shape_errors.append("record is missing gripper_action_mapping")
    finite_arrays = {
        "actions": actions,
        "observation_states": observation_states,
        "world_ee_poses": world_ee_poses,
        "action_target_ee_poses": action_target_ee_poses,
        "world_base_ee_poses": world_base_ee_poses,
        "world_base_action_target_ee_poses": world_base_action_target_ee_poses,
    }
    for name, array in finite_arrays.items():
        if not np.isfinite(np.asarray(array)).all():
            shape_errors.append(f"{name} contains non-finite values")

    expected_timestamps = np.arange(frames, dtype=np.float32) / float(episode_job["cfg"]["fps"])
    if timestamps.shape == expected_timestamps.shape and not np.array_equal(
        np.asarray(timestamps), expected_timestamps
    ):
        shape_errors.append("timestamps do not match frame_index / fps")

    expected_point_cloud_shape = (
        frames,
        int(episode_job["cfg"]["num_points"]),
        POINT_CLOUD_CHANNELS,
    )
    storage = str(episode_job["cfg"].get("point_cloud_storage", "zarr"))
    cameras = selected_camera_names(episode_job["cfg"])
    primary_camera = cameras[0]
    for camera in cameras:
        artifact_path = artifact_dir / point_cloud_artifact_name(camera, primary_camera, storage)
        if storage == "npy" and artifact_path.is_file():
            point_cloud_shape = tuple(np.load(artifact_path, mmap_mode="r").shape)
        elif storage == "zarr" and artifact_path.is_dir() and (artifact_path / ".zattrs").is_file():
            with open(artifact_path / ".zattrs", "r", encoding="utf-8") as f:
                point_cloud_shape = tuple(json.load(f).get("shape", ()))
        else:
            point_cloud_shape = ()
            shape_errors.append(f"point-cloud artifact for camera {camera!r} is missing")
        if point_cloud_shape and point_cloud_shape != expected_point_cloud_shape:
            shape_errors.append(
                f"point cloud {camera!r} shape={point_cloud_shape}, expected {expected_point_cloud_shape}"
            )

    save_rgb_images = bool(episode_job["cfg"].get("save_rgb_images", True))
    expected_image_shape = (
        frames,
        int(episode_job["cfg"]["observation_height"]),
        int(episode_job["cfg"]["observation_width"]),
        3,
    )
    for camera in image_feature_cameras(episode_job["cfg"]):
        images_path = artifact_dir / image_artifact_name(camera)
        if save_rgb_images:
            if not images_path.is_file():
                shape_errors.append(f"RGB image artifact for camera {camera!r} is missing")
            else:
                images = np.load(images_path, mmap_mode="r")
                if images.shape != expected_image_shape:
                    shape_errors.append(
                        f"images {camera!r} shape={images.shape}, expected {expected_image_shape}"
                    )
        elif images_path.exists():
            shape_errors.append(
                f"unexpected RGB image artifact for camera {camera!r} while saving is disabled"
            )

    if shape_errors:
        raise RuntimeError(
            f"Worker artifact validation failed for job {episode_job['job_index']}:\n  "
            + "\n  ".join(shape_errors)
        )


def move_episode_videos(artifact_dir: Path, vis_dir: Path, record: dict[str, Any], cfg: dict[str, Any]) -> list[str]:
    if not cfg.get("save_video", False):
        return []
    source_dir = artifact_dir / "videos"
    if not source_dir.exists():
        return []
    dest_dir = vis_dir / f"episode_{int(record['episode_index']):06d}_{record['demo_name']}"
    dest_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for source_path in sorted(source_dir.iterdir(), key=lambda path: path.name):
        if not source_path.is_file():
            continue
        dest_path = dest_dir / source_path.name
        if dest_path.exists():
            dest_path.unlink()
        shutil.move(str(source_path), str(dest_path))
        written.append(str(dest_path))
    return written


def rot6d_to_matrix(d6: np.ndarray) -> np.ndarray:
    d6 = np.asarray(d6, dtype=np.float32)
    a1 = d6[..., 0:3]
    a2 = d6[..., 3:6]
    b1 = a1 / np.clip(np.linalg.norm(a1, axis=-1, keepdims=True), 1e-6, None)
    b2 = a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = b2 / np.clip(np.linalg.norm(b2, axis=-1, keepdims=True), 1e-6, None)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=-1)


def write_ascii_ply_points(path: Path, xyzrgb: np.ndarray) -> None:
    xyzrgb = np.asarray(xyzrgb, dtype=np.float32)
    if xyzrgb.ndim != 2 or xyzrgb.shape[1] != 6:
        raise ValueError(f"Expected xyzrgb shape (N, 6), got {xyzrgb.shape}")
    colors = np.clip(xyzrgb[:, 3:6], 0, 255).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {xyzrgb.shape[0]}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for xyz, rgb in zip(xyzrgb[:, :3], colors):
            f.write(f"{xyz[0]:.7f} {xyz[1]:.7f} {xyz[2]:.7f} {int(rgb[0])} {int(rgb[1])} {int(rgb[2])}\n")


def write_ascii_ply_lines(path: Path, points: np.ndarray, colors: np.ndarray | None = None) -> None:
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"Expected line points shape (N, 3), got {points.shape}")
    lines = [(idx, idx + 1) for idx in range(max(0, len(points) - 1))]
    if colors is None:
        colors = np.tile(np.asarray([[0, 180, 255]], dtype=np.uint8), (len(lines), 1))
    colors = np.asarray(colors, dtype=np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write(f"element edge {len(lines)}\n")
        f.write("property int vertex1\nproperty int vertex2\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for xyz in points:
            f.write(f"{xyz[0]:.7f} {xyz[1]:.7f} {xyz[2]:.7f}\n")
        for (start, end), rgb in zip(lines, colors):
            f.write(f"{start} {end} {int(rgb[0])} {int(rgb[1])} {int(rgb[2])}\n")


def write_ascii_ply_frame(path: Path, pose9: np.ndarray, scale: float = 0.04) -> None:
    points = make_frame_points(pose9, scale=scale)
    lines = [(0, 1), (0, 2), (0, 3)]
    colors = np.asarray([[255, 40, 40], [40, 220, 40], [60, 120, 255]], dtype=np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write(f"element edge {len(lines)}\n")
        f.write("property int vertex1\nproperty int vertex2\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for xyz in points:
            f.write(f"{xyz[0]:.7f} {xyz[1]:.7f} {xyz[2]:.7f}\n")
        for (start, end), rgb in zip(lines, colors):
            f.write(f"{start} {end} {int(rgb[0])} {int(rgb[1])} {int(rgb[2])}\n")


def make_frame_points(pose9: np.ndarray, scale: float = 0.04) -> np.ndarray:
    pose9 = np.asarray(pose9, dtype=np.float32)
    origin = pose9[:3]
    rot = rot6d_to_matrix(pose9[3:9])
    endpoints = origin[None] + rot.T * scale
    return np.concatenate([origin[None], endpoints], axis=0)


def export_episode_preview(episode: dict[str, Any], vis_dir: Path, record: dict[str, Any], cfg: dict[str, Any]) -> None:
    vis_count = int(cfg.get("vis_count", 0) or 0)
    if vis_count <= 0:
        return
    stride = max(1, int(cfg.get("vis_stride", 1) or 1))
    episode_dir = vis_dir / f"episode_{int(record['episode_index']):06d}_{record['demo_name']}"
    episode_dir.mkdir(parents=True, exist_ok=True)

    point_clouds = np.asarray(episode["point_clouds"], dtype=np.float32)
    actions = np.asarray(episode["actions"], dtype=np.float32)
    observation_states = np.asarray(episode["observation_states"], dtype=np.float32)
    reference_ee_poses = np.asarray(episode["world_ee_poses"], dtype=np.float32)
    action_target_ee_poses = np.asarray(
        episode["action_target_ee_poses"], dtype=np.float32
    )
    candidate_indices = list(range(0, len(point_clouds), stride))
    if len(candidate_indices) > vis_count:
        pick = np.linspace(0, len(candidate_indices) - 1, vis_count).round().astype(int)
        frame_indices = [candidate_indices[int(i)] for i in pick]
    else:
        frame_indices = candidate_indices

    for frame_idx in frame_indices:
        write_ascii_ply_points(episode_dir / f"frame_{frame_idx:04d}_point_cloud_eff.ply", point_clouds[frame_idx])
        write_ascii_ply_frame(episode_dir / f"frame_{frame_idx:04d}_umi_action_frame.ply", actions[frame_idx, :9])
        write_ascii_ply_frame(
            episode_dir / f"frame_{frame_idx:04d}_umi_observation_state_frame.ply",
            observation_states[frame_idx, :9],
        )

    write_ascii_ply_lines(episode_dir / "umi_action_trajectory.ply", actions[:, :3])
    write_ascii_ply_lines(
        episode_dir / "umi_observation_state_trajectory.ply",
        observation_states[:, :3],
    )
    write_ascii_ply_lines(episode_dir / "reference_ee_trajectory.ply", reference_ee_poses[:, :3])
    write_ascii_ply_lines(
        episode_dir / "reference_action_target_trajectory.ply",
        action_target_ee_poses[:, :3],
    )
    preview = {
        **record,
        "frame_indices": [int(idx) for idx in frame_indices],
        "camera_names": list(cfg.get("camera_names", [])),
        "pointcloud_camera_names": selected_camera_names(cfg),
        "render_camera_names": rendered_camera_names(cfg),
        "add_gripper_cloud": bool(cfg.get("add_gripper_cloud", True)),
        "gripper_points": int(cfg.get("gripper_points", 500)),
        "gripper_template": str(cfg.get("gripper_template", "reap")),
        "files": {
            "umi_action_trajectory": "umi_action_trajectory.ply",
            "umi_observation_state_trajectory": "umi_observation_state_trajectory.ply",
            "reference_ee_trajectory": "reference_ee_trajectory.ply",
            "reference_action_target_trajectory": "reference_action_target_trajectory.ply",
            "point_cloud_pattern": "frame_XXXX_point_cloud_eff.ply",
            "action_frame_pattern": "frame_XXXX_umi_action_frame.ply",
            "observation_state_frame_pattern": (
                "frame_XXXX_umi_observation_state_frame.ply"
            ),
        },
    }
    with open(episode_dir / "preview.json", "w", encoding="utf-8") as f:
        json.dump(preview, f, indent=2, ensure_ascii=False)


def get_libero_dataset_root(cli_demo_root: Path | None, cfg: dict[str, Any]) -> Path:
    if cli_demo_root is not None:
        return cli_demo_root.expanduser().resolve()
    if cfg.get("demo_root"):
        return Path(cfg["demo_root"]).expanduser().resolve()
    from libero.libero import get_libero_path

    return Path(get_libero_path("datasets")).expanduser().resolve()


def downloadable_suite_name(suite_name: str) -> str | None:
    if suite_name in {"libero_object", "libero_goal", "libero_spatial"}:
        return suite_name
    if suite_name in {"libero_10", "libero_90", "libero_100"}:
        return "libero_100"
    return None


def maybe_download_demos(cfg: dict[str, Any], demo_root: Path) -> None:
    if not cfg.get("download_demos", False):
        return
    dataset_name = downloadable_suite_name(str(cfg["suite"]))
    if dataset_name is None:
        raise ValueError(f"Do not know which LIBERO demo package to download for suite {cfg['suite']!r}.")
    from libero.libero.utils.download_utils import libero_dataset_download


    print(f"No local demos found. Downloading {dataset_name} demos to {demo_root} ...")
    libero_dataset_download(
        datasets=dataset_name,
        download_dir=str(demo_root),
        check_overwrite=False,
        use_huggingface=bool(cfg.get("download_use_huggingface", True)),
    )


def find_demo_file(task: Any, demo_root: Path, explicit_files: list[Path] | None, cfg: dict[str, Any]) -> Path:
    if explicit_files:
        if len(explicit_files) == 1:
            path = explicit_files[0].expanduser().resolve()
            if not path.exists():
                raise FileNotFoundError(f"Explicit demo file does not exist: {path}")
            return path
        task_key = normalized_name(task.name)
        for file_path in explicit_files:
            path = file_path.expanduser().resolve()
            if task_key in normalized_name(path.as_posix()):
                return path
        raise FileNotFoundError(f"No explicit --demo-file matches task {task.name!r}.")

    candidates = sorted(list(demo_root.rglob("*.hdf5")) + list(demo_root.rglob("*.h5")), key=natural_key)
    if not candidates:
        maybe_download_demos(cfg, demo_root)
        candidates = sorted(list(demo_root.rglob("*.hdf5")) + list(demo_root.rglob("*.h5")), key=natural_key)
    if not candidates:
        download_hint = (
            f"No .hdf5/.h5 demo files found under {demo_root}.\n"
            "Download LIBERO demos first, or rerun with --download-demos. Example:\n"
            f"  python {Path(__file__).name} --config <config> --suite {cfg['suite']} --task-id <id> --episodes 1 --download-demos\n"
            f"Or pass --demo-root /path/to/libero/datasets / --demo-file /path/to/demo.hdf5."
        )
        raise FileNotFoundError(download_hint)

    task_keys = [
        normalized_name(task.name),
        normalized_name(getattr(task, "problem_folder", "")),
        normalized_name(Path(getattr(task, "bddl_file", "")).stem),
    ]
    scored: list[tuple[int, Path]] = []
    for candidate in candidates:
        key = normalized_name(candidate.as_posix())
        score = sum(1 for task_key in task_keys if task_key and task_key in key)
        if score > 0:
            scored.append((score, candidate))
    if not scored:
        raise FileNotFoundError(
            f"Could not find a demo file for task {task.name!r} under {demo_root}. "
            "Pass --demo-file explicitly, or verify that the downloaded suite matches --suite."
        )
    scored.sort(key=lambda item: (-item[0], natural_key(item[1])))
    return scored[0][1]


def iter_demo_groups(demo_file: Path):
    with h5py.File(demo_file, "r") as h5_file:
        root = h5_file["data"] if "data" in h5_file else h5_file
        names = sorted(
            [name for name in root.keys() if isinstance(root[name], h5py.Group)],
            key=lambda value: [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", value)],
        )
        for name in names:
            group = root[name]
            if "states" not in group:
                continue
            yield name, group["states"][:], group["actions"][:] if "actions" in group else None


def iter_demo_group_names(demo_file: Path) -> list[str]:
    with h5py.File(demo_file, "r") as h5_file:
        root = h5_file["data"] if "data" in h5_file else h5_file
        return sorted(
            [
                name
                for name in root.keys()
                if isinstance(root[name], h5py.Group) and "states" in root[name]
            ],
            key=lambda value: [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", value)],
        )


def _decode_hdf5_text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def source_control_frequency(h5_file: h5py.File) -> float | None:
    root = h5_file["data"] if "data" in h5_file else h5_file
    env_args = root.attrs.get("env_args")
    if env_args is None:
        return None
    try:
        metadata = json.loads(_decode_hdf5_text(env_args))
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    env_kwargs = metadata.get("env_kwargs", {})
    value = env_kwargs.get("control_freq")
    return float(value) if value is not None else None


def load_demo_group(
    demo_file: Path,
    demo_name: str,
) -> tuple[np.ndarray, np.ndarray | None, str | None, float | None]:
    with h5py.File(demo_file, "r") as h5_file:
        root = h5_file["data"] if "data" in h5_file else h5_file
        if demo_name not in root:
            raise KeyError(f"Demo group {demo_name!r} not found in {demo_file}")
        group = root[demo_name]
        if "states" not in group:
            raise KeyError(f"Demo group {demo_name!r} in {demo_file} is missing 'states'.")
        states = group["states"][:]
        actions = group["actions"][:] if "actions" in group else None
        model_value = group.attrs.get("model_file")
        model_xml = _decode_hdf5_text(model_value) if model_value is not None else None
        return states, actions, model_xml, source_control_frequency(h5_file)


def postprocess_demo_model_xml(model_xml: str) -> str:
    """Resolve asset paths in an official LIBERO demonstration model XML."""
    import robosuite
    from libero.libero import get_assets_path

    root = ET.fromstring(model_xml)
    libero_assets = Path(get_assets_path()).expanduser().resolve()
    robosuite_root = Path(robosuite.__file__).resolve().parent
    unresolved: list[str] = []

    for element in root.findall(".//*[@file]"):
        original = str(element.attrib["file"])
        normalized = original.replace("\\", "/")
        path = Path(original).expanduser()
        replacement: Path | None = None
        if path.is_file():
            replacement = path.resolve()
        elif "chiliocosm/assets/" in normalized:
            replacement = libero_assets / normalized.split("chiliocosm/assets/", 1)[1]
        elif "/libero/libero/assets/" in normalized:
            replacement = libero_assets / normalized.split("/libero/libero/assets/", 1)[1]
        elif "/libero/assets/" in normalized:
            replacement = libero_assets / normalized.split("/libero/assets/", 1)[1]
        elif "/robosuite/" in normalized:
            replacement = robosuite_root / normalized.rsplit("/robosuite/", 1)[1]
        elif normalized.startswith("robosuite/"):
            replacement = robosuite_root / normalized.split("robosuite/", 1)[1]

        if replacement is not None:
            element.set("file", str(replacement))
            if not replacement.is_file():
                unresolved.append(f"{original!r} -> {str(replacement)!r}")
        elif Path(normalized).is_absolute():
            unresolved.append(repr(original))

    if unresolved:
        preview = "\n  ".join(unresolved[:10])
        suffix = f"\n  ... and {len(unresolved) - 10} more" if len(unresolved) > 10 else ""
        raise FileNotFoundError(
            "Could not resolve asset paths from the demonstration model XML:\n  "
            f"{preview}{suffix}"
        )
    return ET.tostring(root, encoding="unicode")


def ensure_demo_model_assets_ready() -> None:
    """Resolve/download shared assets once in the parent before workers are spawned."""
    from libero.libero import get_assets_path

    assets_path = Path(get_assets_path()).expanduser().resolve()
    if not assets_path.is_dir():
        raise FileNotFoundError(f"LIBERO asset directory is unavailable: {assets_path}")


def verify_restored_demo_model(env: Any, model_xml: str, *, atol: float = 1e-10) -> None:
    """Fail fast if a worker is not using the model belonging to its current demo."""
    root = ET.fromstring(model_xml)
    model = env.sim.model
    mismatches: list[str] = []

    for body in root.findall(".//body[@name]"):
        name = body.attrib["name"]
        try:
            body_id = int(model.body_name2id(name))
        except Exception:
            mismatches.append(f"missing body {name!r}")
            continue

        if "pos" in body.attrib:
            expected_pos = np.fromstring(body.attrib["pos"], sep=" ", dtype=np.float64)
            actual_pos = np.asarray(model.body_pos[body_id], dtype=np.float64)
            error = float(np.max(np.abs(actual_pos - expected_pos)))
            if error > atol:
                mismatches.append(f"body {name!r} position error={error:.3e}")

        if "quat" in body.attrib:
            expected_quat = np.fromstring(body.attrib["quat"], sep=" ", dtype=np.float64)
            expected_quat /= max(float(np.linalg.norm(expected_quat)), np.finfo(np.float64).eps)
            actual_quat = np.asarray(model.body_quat[body_id], dtype=np.float64)
            # q and -q represent the same rotation.
            error = min(
                float(np.max(np.abs(actual_quat - expected_quat))),
                float(np.max(np.abs(actual_quat + expected_quat))),
            )
            if error > atol:
                mismatches.append(f"body {name!r} quaternion error={error:.3e}")

    if mismatches:
        preview = "\n  ".join(mismatches[:10])
        suffix = f"\n  ... and {len(mismatches) - 10} more" if len(mismatches) > 10 else ""
        raise RuntimeError(
            "The restored MuJoCo model does not match the current demonstration XML:\n  "
            f"{preview}{suffix}"
        )


def restore_demo_model(env: Any, model_xml: str | None, *, required: bool) -> str | None:
    """Restore model-level scene state omitted by MuJoCo flattened states."""
    if model_xml is None:
        if required:
            raise KeyError(
                "The demonstration is missing the model_file attribute required for exact LIBERO replay."
            )
        return None
    processed_xml = postprocess_demo_model_xml(model_xml)
    env.reset_from_xml_string(processed_xml)
    env.sim.reset()
    env.sim.forward()
    verify_restored_demo_model(env, processed_xml)
    return hashlib.sha256(model_xml.encode("utf-8")).hexdigest()


def validate_source_fps(
    *,
    demo_file: Path,
    demo_name: str,
    source_fps: float | None,
    output_fps: int,
    required: bool,
) -> None:
    if source_fps is None:
        if required:
            raise ValueError(
                f"Cannot determine source control frequency for {demo_file}:{demo_name}."
            )
        return
    if required and not np.isclose(float(source_fps), float(output_fps)):
        raise ValueError(
            f"FPS mismatch for {demo_file}:{demo_name}: source LIBERO control frequency is "
            f"{source_fps:g} Hz, but output dataset FPS is {output_fps}. Use --fps "
            f"{int(source_fps)} for official timing, or --no-require-source-fps-match only "
            "for an intentional temporal reinterpretation."
        )


def set_env_state_and_get_obs(env: Any, state: np.ndarray) -> dict[str, Any]:
    try:
        return env.set_init_state(state)
    except Exception:
        env.sim.set_state_from_flattened(state)
        env.sim.forward()
        if hasattr(env, "_get_observations"):
            return env._get_observations()
        return env.env._get_observations()


def current_controller_eef_world(env: Any) -> np.ndarray:
    """Return the actual world pose of the EEF site controlled by OSC."""

    if len(getattr(env, "robots", [])) != 1:
        raise ValueError("EEF-frame conversion currently requires one LIBERO robot.")
    controller = env.robots[0].controller
    controller.update(force=True)
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = np.asarray(controller.ee_ori_mat, dtype=np.float64)
    out[:3, 3] = np.asarray(controller.ee_pos, dtype=np.float64)
    return out


def current_controller_goal_world(env: Any) -> np.ndarray:
    """Return the current absolute OSC goal without modifying controller state."""

    if len(getattr(env, "robots", [])) != 1:
        raise ValueError("OSC-goal extraction currently requires one LIBERO robot.")
    controller = env.robots[0].controller
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = np.asarray(controller.goal_ori, dtype=np.float64)
    out[:3, 3] = np.asarray(controller.goal_pos, dtype=np.float64)
    return out


def controller_target_to_model_target_world(
    target_controller_world: np.ndarray,
    source_model_world: np.ndarray,
    source_controller_world: np.ndarray,
) -> np.ndarray:
    """Map an OSC-controller-site goal to the dataset/model EEF frame.

    Evaluation applies the forward relation

        target_controller_world = target_model_world @ model_to_controller

    where ``model_to_controller`` is measured at the source state. Dataset
    generation must therefore apply the exact inverse relation before storing
    the target as a model-space action label.
    """

    target_controller_world = np.asarray(target_controller_world, dtype=np.float64)
    source_model_world = np.asarray(source_model_world, dtype=np.float64)
    source_controller_world = np.asarray(source_controller_world, dtype=np.float64)
    for name, value in (
        ("target_controller_world", target_controller_world),
        ("source_model_world", source_model_world),
        ("source_controller_world", source_controller_world),
    ):
        if value.shape != (4, 4) or not np.isfinite(value).all():
            raise ValueError(f"{name} must be a finite 4x4 transform, got {value.shape}.")

    model_to_controller = (
        fast_inverse_homogeneous(source_model_world) @ source_controller_world
    )
    target_model_world = (
        target_controller_world @ fast_inverse_homogeneous(model_to_controller)
    )
    reconstructed_controller_world = target_model_world @ model_to_controller
    roundtrip_error = float(
        np.max(np.abs(reconstructed_controller_world - target_controller_world))
    )
    # source_model_world is reconstructed from float32 pose9 observations, while
    # the OSC controller goal is read as float64. A mathematically exact rigid
    # transform round trip therefore normally accumulates errors around 1e-8 to
    # 1e-7. Keep a strict sanity check, but use a tolerance appropriate for this
    # mixed-precision conversion instead of rejecting harmless round-off noise.
    roundtrip_atol = 1e-6
    if not np.allclose(
        reconstructed_controller_world,
        target_controller_world,
        rtol=1e-7,
        atol=roundtrip_atol,
    ):
        raise RuntimeError(
            "Controller/model EEF target conversion failed its rigid-transform "
            f"round trip: max error={roundtrip_error:.3e}, "
            f"atol={roundtrip_atol:.1e}."
        )
    return np.asarray(target_model_world, dtype=np.float64)


def reset_source_delta_controller_goal(env: Any) -> None:
    """Initialize source-action goal history from the restored episode state."""

    if len(getattr(env, "robots", [])) != 1:
        raise ValueError("Controller reset currently requires one LIBERO robot.")
    controller = env.robots[0].controller
    if not bool(getattr(controller, "use_delta", False)):
        raise RuntimeError(
            "Raw LIBERO action conversion requires an OSC controller with use_delta=True."
        )
    controller.update(force=True)
    reset_goal = getattr(controller, "reset_goal", None)
    if callable(reset_goal):
        reset_goal()
    else:
        # Compatibility with older robosuite controllers without reset_goal().
        controller.goal_pos = np.asarray(controller.ee_pos, dtype=np.float64).copy()
        controller.goal_ori = np.asarray(controller.ee_ori_mat, dtype=np.float64).copy()


def libero_delta_action_to_absolute_target_world(
    env: Any,
    source_action: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Reconstruct the OSC setpoint encoded by one raw LIBERO delta action.

    The returned pose is the controller goal, not the pose reached after one
    environment step. Contact, interpolation and tracking error may make those
    poses differ. This function mirrors robosuite's delta branch exactly and
    deliberately adds no heuristic offset.
    """

    raw_action = np.asarray(source_action, dtype=np.float64).reshape(-1)
    if raw_action.shape != (7,):
        raise ValueError(
            f"Expected one raw LIBERO action with shape (7,), got {raw_action.shape}."
        )
    if not np.isfinite(raw_action).all():
        raise ValueError("Raw LIBERO action contains non-finite values.")
    if len(getattr(env, "robots", [])) != 1:
        raise ValueError("Absolute OSC target reconstruction requires one LIBERO robot.")

    controller = env.robots[0].controller
    if not bool(getattr(controller, "use_delta", False)):
        raise RuntimeError(
            "Raw LIBERO action conversion requires an OSC controller with use_delta=True."
        )
    if str(getattr(controller, "impedance_mode", "fixed")) != "fixed":
        raise RuntimeError(
            "Raw 7D LIBERO action conversion currently requires fixed OSC impedance mode."
        )
    scaled_delta = np.asarray(
        controller.scale_action(raw_action[:6]), dtype=np.float64
    )
    # Calling the controller itself also preserves robosuite's orientation
    # history: an all-zero rotation delta keeps the previous goal_ori instead
    # of silently replacing it with the currently achieved orientation.
    controller.update(force=True)
    controller.set_goal(raw_action[:6])
    target_position = np.asarray(controller.goal_pos, dtype=np.float64).copy()
    target_orientation = np.asarray(controller.goal_ori, dtype=np.float64).copy()

    target_to_world = np.eye(4, dtype=np.float64)
    target_to_world[:3, :3] = np.asarray(target_orientation, dtype=np.float64)
    target_to_world[:3, 3] = np.asarray(target_position, dtype=np.float64)
    return target_to_world, scaled_delta


def world_target_to_reference_pose9(
    env: Any,
    target_to_world: np.ndarray,
    reference_camera: str,
) -> np.ndarray:
    """Express a model-EEF world target in the fixed observation camera."""

    from robosuite.utils.camera_utils import get_camera_extrinsic_matrix

    reference_to_world = np.asarray(
        get_camera_extrinsic_matrix(env.sim, normalize_camera_name(reference_camera)),
        dtype=np.float64,
    )
    world_to_reference = fast_inverse_homogeneous(reference_to_world)
    target_to_reference = world_to_reference @ np.asarray(
        target_to_world, dtype=np.float64
    )
    return homo_to_pose9(target_to_reference).astype(np.float32)


def collect_demo_episode(
    *,
    env: Any,
    states: np.ndarray,
    actions: np.ndarray | None,
    task_language: str,
    cfg: dict[str, Any],
    max_frames: int | None,
    episode_seed: int,
) -> dict[str, Any]:
    """Convert one LIBERO demonstration into causally aligned training frames.

    In states replay mode, output frame i uses one common source index k:

        observation[i]      <- render(states[k])
        arm action[i]       <- decode(actions[k], anchored at states[k])
        gripper action[i]   <- achieved width in states[k + 1]
        k                    = i + state_observation_offset

    The default offset is zero. A nonzero offset only skips leading source
    transitions; it never shifts the observation relative to its arm action.
    """

    if states.ndim != 2:
        raise ValueError(f"Expected demo states shape (T, D), got {states.shape}")
    if actions is None:
        raise KeyError(
            "Raw HDF5 actions are required: action labels must be reconstructed "
            "from source OSC commands and may not fall back to achieved states."
        )
    actions = np.asarray(actions)
    if actions.ndim != 2 or actions.shape[1] != 7:
        raise ValueError(f"Expected raw LIBERO actions shape (T, 7), got {actions.shape}")
    if not np.isfinite(actions).all():
        raise ValueError("Raw LIBERO actions contain non-finite values.")

    configured_state_observation_offset = int(cfg.get("state_observation_offset", 0))
    if configured_state_observation_offset not in (0, 1):
        raise ValueError(
            "state_observation_offset must be 0 or 1, got "
            f"{configured_state_observation_offset}."
        )
    state_observation_offset = (
        configured_state_observation_offset if cfg["replay_mode"] == "states" else 0
    )

    if cfg["replay_mode"] == "states":
        # A causal frame needs states[k] for the observation / arm-action anchor
        # and states[k + 1] for the physical-width gripper target.
        frame_count = min(
            len(actions) - state_observation_offset,
            len(states) - state_observation_offset - 1,
        )
    else:
        # Step replay obtains the post-action gripper width directly from next_obs.
        frame_count = min(len(states), len(actions))

    if max_frames is not None and max_frames > 0:
        frame_count = min(frame_count, int(max_frames))
    if frame_count <= 1:
        raise ValueError(
            "A collected LIBERO episode needs at least two causal frames. "
            f"Got states={len(states)}, actions={len(actions)}, "
            f"offset={state_observation_offset}, replay_mode={cfg['replay_mode']!r}."
        )

    num_points = int(cfg["num_points"])
    observation_height = int(cfg["observation_height"])
    observation_width = int(cfg["observation_width"])
    save_video = bool(cfg.get("save_video", False))
    pc_camera_names = selected_camera_names(cfg)
    reference_camera = pc_camera_names[0]
    point_clouds_reference_by_camera = {
        camera: np.empty(
            (frame_count, num_points, POINT_CLOUD_CHANNELS), dtype=np.float32
        )
        for camera in pc_camera_names
    }
    reference_ee_poses_by_camera = {
        camera: np.empty((frame_count, 9), dtype=np.float32)
        for camera in pc_camera_names
    }
    reference_ee_poses = reference_ee_poses_by_camera[reference_camera]
    base_ee_poses = np.empty((frame_count, 9), dtype=np.float32)
    action_target_reference_ee_poses = np.empty((frame_count, 9), dtype=np.float32)
    action_target_base_ee_poses = np.empty((frame_count, 9), dtype=np.float32)
    observation_grippers = np.empty((frame_count, 1), dtype=np.float32)
    action_grippers = np.empty((frame_count, 1), dtype=np.float32)
    video_frames: dict[str, list[np.ndarray]] = {} if save_video else {}
    save_rgb_images = bool(cfg.get("save_rgb_images", True))
    image_frames_by_camera: dict[str, list[np.ndarray]] = {
        camera: [] for camera in image_feature_cameras(cfg)
    }
    def collect_observation_frame(
        frame_idx: int,
        raw_obs: dict[str, Any],
    ) -> np.ndarray:
        """Store one pre-action observation and return its model-EEF world pose."""

        if save_video:
            append_video_frames(video_frames, raw_obs, rendered_camera_names(cfg))
        if save_rgb_images:
            for image_camera, frames_for_camera in image_frames_by_camera.items():
                image = dataset_image_from_raw_obs(raw_obs, image_camera)
                if image is None:
                    raise KeyError(
                        f"Missing RGB image for camera {image_camera!r} in LIBERO observation."
                    )
                frames_for_camera.append(image)

        primary_world_pose = None
        for camera_index, camera in enumerate(pc_camera_names):
            point_cloud_reference, pose9_gripper_reference, pose9_gripper_sim_world = (
                observation_to_camera_point_cloud(
                    env,
                    raw_obs,
                    [camera],
                    observation_height,
                    observation_width,
                    num_points,
                    seed=episode_seed + frame_idx + camera_index * 1_000_003,
                )
            )
            point_clouds_reference_by_camera[camera][frame_idx] = point_cloud_reference
            reference_ee_poses_by_camera[camera][frame_idx] = pose9_gripper_reference[:9]
            if camera == reference_camera:
                observation_grippers[frame_idx, 0] = pose9_gripper_reference[-1]
                primary_world_pose = pose9_gripper_sim_world
        assert primary_world_pose is not None
        base_to_world = robot_base_to_world_matrix(env)
        base_ee_poses[frame_idx] = homo_to_pose9(
            fast_inverse_homogeneous(base_to_world)
            @ pose9_to_homo_np(np.asarray(primary_world_pose, dtype=np.float32)[:9])
        )
        return pose9_to_homo_np(
            np.asarray(primary_world_pose, dtype=np.float32)[:9]
        ).astype(np.float64)

    def gripper_width_from_obs(raw_obs: dict[str, Any]) -> float:
        pose9_gripper = np.asarray(
            eef_pose9_gripper_from_obs(raw_obs), dtype=np.float32
        ).reshape(-1)
        if pose9_gripper.shape[0] < 10:
            raise ValueError(
                "Expected eef_pose9_gripper_from_obs() to return at least 10 values, "
                f"got shape {pose9_gripper.shape}."
            )
        width = float(pose9_gripper[-1])
        if not np.isfinite(width):
            raise ValueError("Reconstructed gripper width is not finite.")
        return width

    def store_model_action_target(frame_idx: int, target_model_world: np.ndarray) -> None:
        action_target_reference_ee_poses[frame_idx] = world_target_to_reference_pose9(
            env,
            target_model_world,
            reference_camera,
        )
        action_target_base_ee_poses[frame_idx] = homo_to_pose9(
            fast_inverse_homogeneous(robot_base_to_world_matrix(env))
            @ np.asarray(target_model_world, dtype=np.float64)
        )

    # Initialize the delta controller once at the beginning of the source
    # sequence. Its orientation-goal history must then advance in source-action
    # order, even when a nonzero source-index offset skips leading output frames.
    initial_obs = set_env_state_and_get_obs(env, states[0])
    reset_source_delta_controller_goal(env)

    if cfg["replay_mode"] == "step":
        raw_obs = initial_obs
        for frame_idx in range(frame_count):
            # The model observes the state immediately before actions[i].
            source_model_world = collect_observation_frame(frame_idx, raw_obs)
            source_controller_world = current_controller_eef_world(env)

            # env.step() executes the raw action exactly once and updates both the
            # controller goal and the achieved post-action gripper width.
            next_obs, _, done, _ = env.step(actions[frame_idx])
            target_controller_world = current_controller_goal_world(env)
            target_model_world = controller_target_to_model_target_world(
                target_controller_world,
                source_model_world,
                source_controller_world,
            )
            store_model_action_target(frame_idx, target_model_world)
            action_grippers[frame_idx, 0] = gripper_width_from_obs(next_obs)
            raw_obs = next_obs
            if bool(done) and frame_idx + 1 < frame_count:
                raise RuntimeError(
                    f"Source action replay terminated at frame {frame_idx} before "
                    f"the expected {frame_count} converted frames."
                )
    else:
        # Advance controller goal history through skipped source actions. This is
        # needed because robosuite's orientation delta semantics depend on the
        # previous goal orientation, not only on the current achieved state.
        for skipped_idx in range(state_observation_offset):
            set_env_state_and_get_obs(env, states[skipped_idx])
            libero_delta_action_to_absolute_target_world(env, actions[skipped_idx])

        for frame_idx in range(frame_count):
            source_idx = frame_idx + state_observation_offset
            next_state_idx = source_idx + 1

            # Observation and arm action share the same pre-action source state.
            source_obs = set_env_state_and_get_obs(env, states[source_idx])
            source_model_world = collect_observation_frame(frame_idx, source_obs)
            source_controller_world = current_controller_eef_world(env)
            target_controller_world, _scaled_delta = (
                libero_delta_action_to_absolute_target_world(
                    env,
                    actions[source_idx],
                )
            )
            target_model_world = controller_target_to_model_target_world(
                target_controller_world,
                source_model_world,
                source_controller_world,
            )
            store_model_action_target(frame_idx, target_model_world)

            # The gripper label remains a physical width, but now represents the
            # achieved width after the aligned source action instead of copying
            # the current observation width.
            next_obs = set_env_state_and_get_obs(env, states[next_state_idx])
            action_grippers[frame_idx, 0] = gripper_width_from_obs(next_obs)

    if cfg.get("add_gripper_cloud", True):
        for camera in pc_camera_names:
            point_clouds_reference_by_camera[camera] = (
                add_reference_gripper_clouds_to_episode(
                    point_clouds_reference_by_camera[camera],
                    reference_ee_poses_by_camera[camera],
                    observation_grippers.reshape(-1),
                    total_points=int(cfg["num_points"]),
                    gripper_points=int(cfg.get("gripper_points", 500)),
                    gripper_len=float(cfg.get("gripper_len", 0.06)),
                    gripper_template=str(cfg.get("gripper_template", "reap")),
                    seed=episode_seed,
                    # Multi-view composition relies on one addressable gripper tail.
                    drop_strategy="tail",
                    shuffle_points=False,
                    widths_are_normalized=False,
                    gripper_max_width=float(cfg.get("gripper_qpos_max_width", 0.08)),
                )
            )

    point_clouds_by_camera = {
        camera: reference_point_cloud_to_current_eff(
            point_clouds_reference_by_camera[camera],
            reference_ee_poses_by_camera[camera],
        )
        for camera in pc_camera_names
    }
    point_clouds = point_clouds_by_camera[reference_camera]
    episode_origin_pose = reference_ee_poses[0]
    observation_umi_poses = from_reference_to_umi_tra_pose9(
        reference_ee_poses,
        origin_pose9_eff_to_reference=episode_origin_pose,
    )
    action_target_umi_poses = from_reference_to_umi_tra_pose9(
        action_target_reference_ee_poses,
        origin_pose9_eff_to_reference=episode_origin_pose,
    )
    observation_states = np.concatenate(
        [observation_umi_poses, observation_grippers], axis=-1
    ).astype(np.float32)
    episode_actions = np.concatenate(
        [action_target_umi_poses, action_grippers], axis=-1
    ).astype(np.float32)
    timestamps = np.arange(len(episode_actions), dtype=np.float32) / float(cfg["fps"])
    target_residual_m = np.linalg.norm(
        action_target_reference_ee_poses[:, :3] - reference_ee_poses[:, :3],
        axis=-1,
    )
    if cfg["replay_mode"] == "states":
        source_index_expr = f"i + {state_observation_offset}"
        action_source_index_mapping = (
            f"output[i] uses source actions[{source_index_expr}]"
        )
        action_source_state_mapping = (
            f"source actions[{source_index_expr}] is anchored at "
            f"states[{source_index_expr}]"
        )
        observation_state_mapping = (
            f"output observation[i] is reconstructed from states[{source_index_expr}]"
        )
        gripper_action_mapping = (
            f"output action gripper[i] is achieved physical width from "
            f"states[{source_index_expr} + 1]"
        )
    else:
        action_source_index_mapping = "output[i] uses source actions[i]"
        action_source_state_mapping = (
            "source actions[i] is anchored at the pre-action replay state"
        )
        observation_state_mapping = (
            "output observation[i] is the pre-action replay state for actions[i]"
        )
        gripper_action_mapping = (
            "output action gripper[i] is achieved physical width from next_obs "
            "after actions[i]"
        )

    episode = {
        "task": task_language,
        "actions": episode_actions,
        "observation_states": observation_states,
        "point_clouds": point_clouds,
        "point_clouds_by_camera": point_clouds_by_camera,
        # Legacy key/path retained for the existing WorldFlow dataset wrapper.
        # Values are expressed in the fixed Overview-camera reference frame.
        "world_ee_poses": reference_ee_poses,
        "world_base_ee_poses": base_ee_poses,
        "world_base_action_target_ee_poses": action_target_base_ee_poses,
        "action_target_ee_poses": action_target_reference_ee_poses,
        "timestamps": timestamps,
        "video_frames": video_frames,
        "state_observation_offset": state_observation_offset,
        "action_label_semantics": ACTION_LABEL_SEMANTICS,
        "observation_state_semantics": OBSERVATION_STATE_SEMANTICS,
        "action_source_index_mapping": action_source_index_mapping,
        "action_source_state_mapping": action_source_state_mapping,
        "action_target_eef_mapping": (
            "source controller goal mapped through the aligned source-state "
            "controller-to-model EEF transform"
        ),
        "observation_state_mapping": observation_state_mapping,
        "gripper_action_mapping": gripper_action_mapping,
        "target_residual_translation_m_mean": float(target_residual_m.mean()),
        "target_residual_translation_m_max": float(target_residual_m.max()),
    }
    if save_rgb_images:
        images_by_camera: dict[str, np.ndarray] = {}
        for camera, frames_for_camera in image_frames_by_camera.items():
            if len(frames_for_camera) != len(episode_actions):
                raise ValueError(
                    f"Collected {len(frames_for_camera)} RGB frames for camera {camera!r}, "
                    f"but there are {len(episode_actions)} actions."
                )
            images_by_camera[camera] = np.asarray(frames_for_camera, dtype=np.uint8)
        episode["images_by_camera"] = images_by_camera
    return episode


def make_episode_record(
    *,
    suite_name: str,
    task_id: int,
    task: Any,
    demo_file: Path,
    demo_name: str,
    frames: int,
    model_sha256: str | None,
    source_fps: float | None,
    state_observation_offset: int,
    episode: dict[str, Any],
) -> dict[str, Any]:
    return {
        "suite": suite_name,
        "task_id": int(task_id),
        "task_name": task.name,
        "task_language": task.language,
        "problem_folder": getattr(task, "problem_folder", ""),
        "bddl_file": getattr(task, "bddl_file", ""),
        "demo_file": str(demo_file),
        "demo_name": demo_name,
        "frames": int(frames),
        "model_sha256": model_sha256,
        "model_restoration_verified": model_sha256 is not None,
        "source_fps": source_fps,
        "state_observation_offset": int(state_observation_offset),
        "action_label_semantics": str(episode["action_label_semantics"]),
        "observation_state_semantics": str(
            episode["observation_state_semantics"]
        ),
        "action_source_index_mapping": str(
            episode["action_source_index_mapping"]
        ),
        "action_source_state_mapping": str(
            episode["action_source_state_mapping"]
        ),
        "action_target_eef_mapping": str(episode["action_target_eef_mapping"]),
        "observation_state_mapping": str(episode["observation_state_mapping"]),
        "gripper_action_mapping": str(episode["gripper_action_mapping"]),
        "action_pose_coordinate_frame": "episode_origin_eef",
        "observation_state_coordinate_frame": "episode_origin_eef",
        "absolute_action_target_sidecar_coordinate_frame": "overview_camera",
        "worldflow_reference_frame": "robot_base",
        "worldflow_target_semantics": "commanded_eef_pose",
        "gripper_action_semantics": "next_state_achieved_physical_width_metres",
        "heuristic_action_target_offset": False,
        "target_residual_translation_m_mean": float(
            episode["target_residual_translation_m_mean"]
        ),
        "target_residual_translation_m_max": float(
            episode["target_residual_translation_m_max"]
        ),
    }


def collect_episode_worker(job: dict[str, Any]) -> dict[str, Any]:
    cfg = dict(job["cfg"])
    suite_name = str(job["suite"])
    cfg["suite"] = suite_name
    ensure_libero_config(cfg.get("libero_config_path"), cfg.get("demo_root"))

    from libero.libero import benchmark

    suite_cls = benchmark.get_benchmark_dict()[suite_name]
    suite = suite_cls()
    task_id = int(job["task_id"])
    demo_file = Path(job["demo_file"])
    demo_name = str(job["demo_name"])
    states, actions, model_xml, source_fps = load_demo_group(demo_file, demo_name)
    validate_source_fps(
        demo_file=demo_file,
        demo_name=demo_name,
        source_fps=source_fps,
        output_fps=int(cfg["fps"]),
        required=bool(cfg.get("require_source_fps_match", True)),
    )

    env, task = make_libero_env(
        suite,
        task_id,
        int(cfg["observation_height"]),
        int(cfg["observation_width"]),
        rendered_camera_names(cfg),
    )
    try:
        model_sha256 = (
            restore_demo_model(env, model_xml, required=True)
            if bool(cfg.get("restore_demo_model", True))
            else None
        )
        episode = collect_demo_episode(
            env=env,
            states=states,
            actions=actions,
            task_language=task.language,
            cfg=cfg,
            max_frames=job.get("max_frames"),
            episode_seed=int(job["episode_seed"]),
        )
    finally:
        env.close()

    record = make_episode_record(
        suite_name=suite_name,
        task_id=task_id,
        task=task,
        demo_file=demo_file,
        demo_name=demo_name,
        frames=len(episode["actions"]),
        model_sha256=model_sha256,
        source_fps=source_fps,
        state_observation_offset=int(episode["state_observation_offset"]),
        episode=episode,
    )
    artifact_dir = Path(job["tmp_path"])
    save_episode_artifact(artifact_dir, episode, record, cfg)
    return {"job_index": int(job["job_index"]), "tmp_path": str(artifact_dir)}


def collect_task_worker(job: dict[str, Any]) -> list[dict[str, Any]]:
    """Collect multiple demos for one LIBERO task while reusing a single environment."""
    cfg = dict(job["cfg"])
    suite_name = str(job["suite"])
    cfg["suite"] = suite_name
    ensure_libero_config(cfg.get("libero_config_path"), cfg.get("demo_root"))

    from libero.libero import benchmark

    suite_cls = benchmark.get_benchmark_dict()[suite_name]
    suite = suite_cls()
    task_id = int(job["task_id"])
    env, task = make_libero_env(
        suite,
        task_id,
        int(cfg["observation_height"]),
        int(cfg["observation_width"]),
        rendered_camera_names(cfg),
    )

    results: list[dict[str, Any]] = []
    try:
        for episode_job in job["episodes"]:
            demo_file = Path(episode_job["demo_file"])
            demo_name = str(episode_job["demo_name"])
            states, actions, model_xml, source_fps = load_demo_group(demo_file, demo_name)
            validate_source_fps(
                demo_file=demo_file,
                demo_name=demo_name,
                source_fps=source_fps,
                output_fps=int(cfg["fps"]),
                required=bool(cfg.get("require_source_fps_match", True)),
            )
            model_sha256 = (
                restore_demo_model(env, model_xml, required=True)
                if bool(cfg.get("restore_demo_model", True))
                else None
            )
            episode = collect_demo_episode(
                env=env,
                states=states,
                actions=actions,
                task_language=task.language,
                cfg=cfg,
                max_frames=episode_job.get("max_frames"),
                episode_seed=int(episode_job["episode_seed"]),
            )
            record = make_episode_record(
                suite_name=suite_name,
                task_id=task_id,
                task=task,
                demo_file=demo_file,
                demo_name=demo_name,
                frames=len(episode["actions"]),
                model_sha256=model_sha256,
                source_fps=source_fps,
                state_observation_offset=int(episode["state_observation_offset"]),
                episode=episode,
            )
            artifact_dir = Path(episode_job["tmp_path"])
            save_episode_artifact(artifact_dir, episode, record, cfg)
            results.append({"job_index": int(episode_job["job_index"]), "tmp_path": str(artifact_dir)})
    finally:
        env.close()

    return results


def save_collected_temp_episode(
    *,
    temp_path: Path,
    dataset: LeRobotDataset,
    vis_dir: Path,
    cfg: dict[str, Any],
) -> dict[str, Any]:
    artifact_dir = temp_path
    with open(artifact_dir / "record.json", "r", encoding="utf-8") as f:
        record = json.load(f)
    actions = np.load(artifact_dir / "actions.npy")
    observation_states = np.load(artifact_dir / "observation_states.npy")
    timestamps = np.load(artifact_dir / "timestamps.npy")
    images_by_camera = {
        camera: np.load(artifact_dir / image_artifact_name(camera), mmap_mode="r")
        for camera in image_feature_cameras(cfg)
        if (artifact_dir / image_artifact_name(camera)).exists()
    }
    episode_index = int(dataset.meta.total_episodes)

    point_cloud_storage = str(cfg.get("point_cloud_storage", "zarr"))
    cameras = selected_camera_names(cfg)
    primary_camera = cameras[0]
    for camera in cameras:
        final_path = point_cloud_storage_path_for_camera(
            dataset.root, camera, primary_camera, episode_index, point_cloud_storage
        )
        artifact_path = artifact_dir / point_cloud_artifact_name(
            camera, primary_camera, point_cloud_storage
        )
        if point_cloud_storage == "zarr":
            if artifact_path.exists():
                move_episode_array(artifact_path, final_path)
            else:
                npy_fallback = artifact_dir / point_cloud_artifact_name(camera, primary_camera, "npy")
                if not npy_fallback.exists():
                    raise FileNotFoundError(
                        f"Missing point-cloud artifact for camera {camera!r} under {artifact_dir}"
                    )
                save_episode_point_clouds_zarr(
                    dataset.root / point_cloud_dir_name(camera, primary_camera),
                    episode_index,
                    np.load(npy_fallback, mmap_mode="r"),
                    compression_level=int(cfg.get("zarr_compression_level", 3)),
                )
                npy_fallback.unlink(missing_ok=True)
        else:
            move_episode_array(artifact_path, final_path)

    final_world_ee_pose_path = world_ee_pose_file(dataset.root, episode_index)
    final_action_target_ee_pose_path = action_target_ee_pose_file(dataset.root, episode_index)
    move_episode_array(artifact_dir / "world_ee_poses.npy", final_world_ee_pose_path)
    move_episode_array(
        artifact_dir / "action_target_ee_poses.npy",
        final_action_target_ee_pose_path,
    )
    move_episode_array(
        artifact_dir / "world_base_ee_poses.npy",
        world_base_ee_pose_file(dataset.root, episode_index),
    )
    move_episode_array(
        artifact_dir / "world_base_action_target_ee_poses.npy",
        world_base_action_target_ee_pose_file(dataset.root, episode_index),
    )
    dataset.save_episode(
        episode_data=make_episode_buffer(
            dataset,
            str(record["task_language"]),
            actions,
            observation_states,
            timestamps,
            images_by_camera=images_by_camera,
        )
    )
    record["episode_index"] = episode_index
    if int(cfg.get("vis_count", 0) or 0) > 0:
        preview_episode = {
            "actions": actions,
            "observation_states": observation_states,
            "point_clouds": open_episode_point_clouds(
                dataset.root / POINT_CLOUD_DIR_NAME, episode_index
            ),
            "world_ee_poses": np.load(final_world_ee_pose_path, mmap_mode="r"),
            "action_target_ee_poses": np.load(
                final_action_target_ee_pose_path, mmap_mode="r"
            ),
        }
        export_episode_preview(preview_episode, vis_dir, record, cfg)
    record["videos"] = move_episode_videos(artifact_dir, vis_dir, record, cfg)
    return record


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    suite_names = resolve_suite_names(args.suite, cfg)
    cfg["suites"] = suite_names
    cfg["all_tasks"] = bool(cfg_get(cfg, args.all_tasks, "all_tasks", False))
    cfg["task_ids"] = args.task_id if args.task_id is not None else cfg.get("task_ids")
    cfg["episodes"] = int(cfg_get(cfg, args.episodes, "episodes", 1))
    cfg["num_points"] = int(cfg_get(cfg, args.num_points, "num_points", 10000))
    if args.camera is not None:
        cfg["selected_cameras"] = [normalize_camera_name(camera) for camera in args.camera]
    else:
        cfg["selected_cameras"] = selected_camera_names(cfg)
    cfg["point_cloud_storage"] = str(cfg_get(cfg, args.point_cloud_storage, "point_cloud_storage", "zarr"))
    cfg["zarr_compression_level"] = int(cfg_get(cfg, args.zarr_compression_level, "zarr_compression_level", 3))
    cfg["fps"] = int(cfg_get(cfg, args.fps, "fps", 20))
    cfg["replay_mode"] = cfg_get(cfg, args.replay_mode, "replay_mode", "states")
    cfg["state_observation_offset"] = int(
        cfg_get(
            cfg,
            args.state_observation_offset,
            "state_observation_offset",
            0,
        )
    )
    cfg["restore_demo_model"] = bool(
        cfg_get(cfg, args.restore_demo_model, "restore_demo_model", True)
    )
    cfg["require_source_fps_match"] = bool(
        cfg_get(
            cfg,
            args.require_source_fps_match,
            "require_source_fps_match",
            True,
        )
    )
    cfg["download_demos"] = bool(cfg_get(cfg, args.download_demos, "download_demos", True))
    cfg["download_use_huggingface"] = bool(cfg_get(cfg, args.download_use_huggingface, "download_use_huggingface", True))
    cfg["overwrite_dataset"] = bool(cfg_get(cfg, args.overwrite, "overwrite_dataset", True))
    cfg["vis_count"] = int(cfg_get(cfg, args.vis_count, "vis_count", 0) or 0)
    cfg["vis_stride"] = int(cfg_get(cfg, args.vis_stride, "vis_stride", 1) or 1)
    cfg["save_video"] = bool(cfg_get(cfg, args.save_video, "save_video", False))
    cfg["save_rgb_images"] = bool(cfg_get(cfg, args.save_rgb_images, "save_rgb_images", True))
    image_camera_value = args.image_camera
    if image_camera_value is None:
        image_camera_value = cfg.get("image_cameras", cfg.get("image_camera"))
    if image_camera_value is not None:
        values = image_camera_value if isinstance(image_camera_value, (list, tuple)) else [image_camera_value]
        cfg["image_cameras"] = [normalize_camera_name(str(value)) for value in values]
    elif args.camera is not None:
        cfg["image_cameras"] = list(cfg["selected_cameras"])
    else:
        cfg["image_cameras"] = image_feature_cameras(cfg)
    ensure_image_camera_rendered(cfg)
    cfg["num_workers"] = int(cfg_get(cfg, args.num_workers, "num_workers", 1) or 1)
    cfg["worker_scope"] = str(cfg_get(cfg, args.worker_scope, "worker_scope", "task"))
    cfg["resume_temp_artifacts"] = bool(
        cfg_get(cfg, args.resume_temp_artifacts, "resume_temp_artifacts", False)
    )
    if cfg["num_workers"] < 1:
        raise ValueError(f"num_workers must be at least 1, got {cfg['num_workers']}.")
    if cfg["worker_scope"] not in {"task", "episode"}:
        raise ValueError(
            f"worker_scope must be 'task' or 'episode', got {cfg['worker_scope']!r}."
        )
    max_frames = cfg_get(cfg, args.max_frames_per_demo, "max_frames_per_demo")
    max_frames = int(max_frames) if max_frames is not None else None
    ensure_libero_config(cfg.get("libero_config_path"), args.demo_root or cfg.get("demo_root"))
    if cfg["restore_demo_model"]:
        ensure_demo_model_assets_ready()

    output_root = Path(
        cfg_get(cfg, args.output_root, "dataset_output_root", LIBERO_DATA_ROOT / "libero_lerobot_dataset")
    ).expanduser().resolve()
    repo_id = cfg_get(cfg, args.repo_id, "dataset_repo_id", "song_libero_pointcloud")
    demo_root = get_libero_dataset_root(args.demo_root, cfg)
    cfg["demo_root"] = str(demo_root)
    cfg["image_feature_keys"] = [
        image_feature_key(camera) for camera in image_feature_cameras(cfg)
    ]
    cfg["image_feature_key"] = cfg["image_feature_keys"][0] if cfg["image_feature_keys"] else None
    vis_dir_value = cfg_get(cfg, args.vis_dir, "vis_dir")
    vis_dir = Path(vis_dir_value).expanduser().resolve() if vis_dir_value else output_root / "visualizations"
    tmp_dir = (
        args.tmp_dir.expanduser().resolve()
        if args.tmp_dir is not None
        else output_root.parent / f".{output_root.name}_tmp"
    )

    from libero.libero import benchmark

    if output_root.exists():
        if not cfg["overwrite_dataset"]:
            raise FileExistsError(f"Output dataset already exists: {output_root}")
        if output_root.is_dir():
            shutil.rmtree(output_root)
        else:
            output_root.unlink()

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=int(cfg["fps"]),
        features=dataset_features_with_image(cfg),
        robot_type="libero",
        root=output_root,
        use_videos=False,
    )
    write_point_cloud_meta(
        dataset.root,
        storage=str(cfg["point_cloud_storage"]),
        cameras=selected_camera_names(cfg),
        gripper_points=int(cfg.get("gripper_points", 500)),
    )
    write_worldflow_meta(dataset.root)
    write_action_target_meta(dataset.root)
    write_robot_base_worldflow_meta(dataset.root)

    summary: dict[str, Any] = {
        "created_unix_s": time.time(),
        "suites": suite_names,
        "demo_root": str(demo_root),
        "output_root": str(output_root),
        "camera_names": list(cfg.get("camera_names", [])),
        "pointcloud_camera_names": selected_camera_names(cfg),
        "reference_frame": "overview_camera",
        "reference_camera": selected_camera_names(cfg)[0],
        "sim_extrinsic_usage": "eef_world_to_overview_camera_only",
        "robot_base_worldflow": {
            "coordinate_frame": "robot_base",
            "target": "commanded model-EEF trajectory",
            "implicit_point_flow": True,
            "explicit_object_pose_supervision": False,
        },
        "render_camera_names": rendered_camera_names(cfg),
        "image_cameras": image_feature_cameras(cfg),
        "image_feature_keys": cfg.get("image_feature_keys", []),
        "add_gripper_cloud": bool(cfg.get("add_gripper_cloud", True)),
        "gripper_points": int(cfg.get("gripper_points", 500)),
        "gripper_template": str(cfg.get("gripper_template", "reap")),
        "point_cloud_storage": str(cfg.get("point_cloud_storage", "zarr")),
        "point_cloud_zarr_encoding": "packed_xyz_float16_rgb_uint8",
        "zarr_compression_level": int(cfg.get("zarr_compression_level", 3)),
        "num_workers": int(cfg["num_workers"]),
        "parallel_collection_protocol": {
            "execution_mode": (
                "spawn_process_pool" if int(cfg["num_workers"]) > 1 else "in_process_serial"
            ),
            "worker_scope": (
                "one isolated environment per episode"
                if str(cfg["worker_scope"]) == "episode"
                else "one isolated environment per task"
            ),
            "worker_output": "unique temporary directory per episode job",
            "dataset_commit": "single parent process in deterministic job order",
            "episode_seed": "independent of worker count and completion order",
            "demo_model_runtime_verification": bool(cfg["restore_demo_model"]),
            "resume_temp_artifacts": bool(cfg["resume_temp_artifacts"]),
        },
        "fps": int(cfg["fps"]),
        "state_observation_alignment": {
            "replay_mode": str(cfg["replay_mode"]),
            "source_index_offset": int(cfg["state_observation_offset"]),
            "causal_default": "state_observation_offset=0",
            "mapping": (
                "output observation[i] = render(states[k]); output action[i] "
                "uses source actions[k]; k = i + state_observation_offset"
                if str(cfg["replay_mode"]) == "states"
                else "output observation[i] is captured immediately before source actions[i]"
            ),
            "terminal_policy": (
                "drop the final source transition because states[k + 1] is required "
                "for the achieved physical-width gripper label"
                if str(cfg["replay_mode"]) == "states"
                else "next_obs supplies the post-action gripper width"
            ),
        },
        "action_label_semantics": {
            "pose_label": ACTION_LABEL_SEMANTICS,
            "observation_state": OBSERVATION_STATE_SEMANTICS,
            "source_action_mapping": (
                f"output[i] uses source HDF5 actions[i + {int(cfg['state_observation_offset'])}]"
                if str(cfg["replay_mode"]) == "states"
                else "output[i] uses source HDF5 actions[i]"
            ),
            "source_state_anchor": (
                f"source actions[i + {int(cfg['state_observation_offset'])}] is converted "
                f"at states[i + {int(cfg['state_observation_offset'])}]"
                if str(cfg["replay_mode"]) == "states"
                else "actions[i] is converted at its pre-action replay state"
            ),
            "observation_mapping": (
                f"output observation[i] = render(states[i + {int(cfg['state_observation_offset'])}])"
                if str(cfg["replay_mode"]) == "states"
                else "output observation[i] is captured immediately before actions[i]"
            ),
            "absolute_target_conversion": (
                "robosuite controller.set_goal(raw[:6]) with use_delta=True; "
                "the resulting controller-site target is mapped into the "
                "model/data EEF frame measured at the same pre-action source state"
            ),
            "heuristic_target_offset": False,
            "contact_semantics": (
                "arm action remains the commanded OSC setpoint even when contact "
                "prevents the achieved observation pose from reaching it"
            ),
            "stored_action_frame": "episode_origin_eef",
            "absolute_target_audit_sidecar_frame": "overview_camera",
            "gripper_label": (
                "achieved physical width after the aligned source action; "
                "states[k + 1] in states replay or next_obs in step replay"
            ),
        },
        "demo_model_restoration": (
            "per_demo_model_file" if bool(cfg["restore_demo_model"]) else "disabled"
        ),
        "require_source_fps_match": bool(cfg["require_source_fps_match"]),
        "episodes": [],
    }

    episode_jobs: list[dict[str, Any]] = []
    task_jobs: list[dict[str, Any]] = []
    job_index = 0
    task_job_index = 0
    benchmark_dict = benchmark.get_benchmark_dict()
    for suite_index, suite_name in enumerate(suite_names):
        suite_cls = benchmark_dict[suite_name]
        suite = suite_cls()
        task_ids = resolve_task_ids_for_suite(
            suite_name=suite_name,
            task_count=len(suite.tasks),
            cli_task_ids=args.task_id,
            cfg=cfg,
        )
        suite_cfg = dict(cfg)
        suite_cfg["suite"] = suite_name
        for task_id_value in task_ids:
            task_id = int(task_id_value)
            task = suite.get_task(task_id)
            demo_file = find_demo_file(task, demo_root, args.demo_file, suite_cfg)
            demo_names = iter_demo_group_names(demo_file)[: int(cfg["episodes"])]
            if not demo_names:
                raise RuntimeError(f"No usable demos with states found in {demo_file}")
            task_episode_jobs = []
            for converted_for_task, demo_name in enumerate(demo_names):
                episode_job = {
                    "job_index": job_index,
                    "suite": suite_name,
                    "task_id": task_id,
                    "demo_file": str(demo_file),
                    "demo_name": demo_name,
                    "episode_seed": suite_index * 10_000_000 + task_id * 100000 + converted_for_task * 1000,
                    "max_frames": max_frames,
                    "cfg": suite_cfg,
                    "tmp_path": str(tmp_dir / f"episode_job_{job_index:06d}_{suite_name}_task_{task_id:03d}"),
                }
                episode_jobs.append(episode_job)
                task_episode_jobs.append(episode_job)
                job_index += 1
            task_jobs.append(
                {
                    "task_job_index": task_job_index,
                    "suite": suite_name,
                    "task_id": task_id,
                    "cfg": suite_cfg,
                    "episodes": task_episode_jobs,
                }
            )
            task_job_index += 1

    if tmp_dir.exists() and not bool(cfg["resume_temp_artifacts"]):
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    expected_jobs = {int(job["job_index"]): job for job in episode_jobs}
    results: dict[int, Path] = {}

    def register_result(result: dict[str, Any]) -> None:
        result_index = int(result["job_index"])
        if result_index not in expected_jobs:
            raise RuntimeError(f"Worker returned unknown episode job index {result_index}.")
        if result_index in results:
            raise RuntimeError(f"Worker returned duplicate episode job index {result_index}.")
        expected_job = expected_jobs[result_index]
        actual_path = Path(result["tmp_path"]).expanduser().resolve()
        expected_path = Path(expected_job["tmp_path"]).expanduser().resolve()
        if actual_path != expected_path:
            raise RuntimeError(
                f"Worker returned the wrong artifact path for job {result_index}: "
                f"expected {expected_path}, got {actual_path}."
            )
        verify_episode_artifact(actual_path, expected_job)
        results[result_index] = actual_path

    if bool(cfg["resume_temp_artifacts"]):
        for episode_job in episode_jobs:
            artifact_path = Path(episode_job["tmp_path"])
            if not artifact_path.exists():
                continue
            register_result(
                {
                    "job_index": int(episode_job["job_index"]),
                    "tmp_path": str(artifact_path),
                }
            )
        if results:
            print(
                f"[info] resumed {len(results)}/{len(episode_jobs)} strictly validated "
                f"temporary episode artifact(s) from {tmp_dir}"
            )

    pending_indices = set(expected_jobs) - set(results)
    pending_episode_jobs = [
        job for job in episode_jobs if int(job["job_index"]) in pending_indices
    ]
    pending_task_jobs = []
    for task_job in task_jobs:
        pending_for_task = [
            job
            for job in task_job["episodes"]
            if int(job["job_index"]) in pending_indices
        ]
        if pending_for_task:
            pending_task_jobs.append({**task_job, "episodes": pending_for_task})

    if not pending_episode_jobs:
        print("[info] all requested episodes were restored from validated temporary artifacts")
    elif int(cfg["num_workers"]) <= 1:
        for task_job in tqdm(pending_task_jobs, desc="Collecting LIBERO tasks", unit="task"):
            for result in collect_task_worker(task_job):
                register_result(result)
    elif str(cfg["worker_scope"]) == "episode":
        worker_count = min(int(cfg["num_workers"]), len(pending_episode_jobs))
        print(
            f"[info] collecting {len(pending_episode_jobs)} remaining LIBERO episode(s) "
            f"with {worker_count} episode-scoped worker(s)"
        )
        with ProcessPoolExecutor(
            max_workers=worker_count,
            mp_context=mp.get_context("spawn"),
        ) as executor:
            futures = [
                executor.submit(collect_episode_worker, episode_job)
                for episode_job in pending_episode_jobs
            ]
            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Collecting LIBERO episodes",
                unit="episode",
            ):
                register_result(future.result())
    else:
        print(
            f"[info] collecting {len(pending_episode_jobs)} remaining LIBERO episode(s) "
            f"across {len(pending_task_jobs)} task job(s) "
            f"with {cfg['num_workers']} worker(s)"
        )
        with ProcessPoolExecutor(
            max_workers=int(cfg["num_workers"]),
            mp_context=mp.get_context("spawn"),
        ) as executor:
            futures = [
                executor.submit(collect_task_worker, task_job)
                for task_job in pending_task_jobs
            ]
            for future in tqdm(as_completed(futures), total=len(futures), desc="Collecting LIBERO tasks", unit="task"):
                for result in future.result():
                    register_result(result)

    missing_results = sorted(set(expected_jobs) - set(results))
    if missing_results:
        raise RuntimeError(f"Collection finished with missing episode job(s): {missing_results}")

    for episode_job in episode_jobs:
        temp_path = results[int(episode_job["job_index"])]
        record = save_collected_temp_episode(
            temp_path=temp_path,
            dataset=dataset,
            vis_dir=vis_dir,
            cfg=cfg,
        )
        summary["episodes"].append(record)
        print(
            f"[OK] suite={record['suite']} task={record['task_id']} demo={record['demo_name']} "
            f"frames={record['frames']} episode_index={record['episode_index']}"
        )

    shutil.rmtree(tmp_dir, ignore_errors=True)

    dataset.finalize()
    summary["num_episodes"] = len(summary["episodes"])
    summary_path = output_root / "libero_collect_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"[done] collected {summary['num_episodes']} episode(s) at {output_root}")
    print(f"[done] summary saved to {summary_path}")


if __name__ == "__main__":
    main()
