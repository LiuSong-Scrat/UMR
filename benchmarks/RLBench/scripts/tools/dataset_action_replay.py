#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Restore one recorded RLBench scene and replay its episode actions.

New collector artifacts save RLBench's initial task configuration tree. This
program restores that tree before calculating EEF0 and executing actions:

* eef0: the converted 10-D dataset action with Jacobian IK; or
* eef0_planning: the converted 10-D dataset action as a waypoint, using
  RLBench's EndEffectorPoseViaPlanning (IK + RRTConnect path execution); or
* raw_joint: the original 7-D absolute Panda joint target plus gripper bit.

The raw_joint result is a control group for deciding whether a failure comes
from the joint-FK-to-EEF0 conversion. Old artifacts without an initial task
state are rejected unless --allow-fresh-reset is passed explicitly.
"""

import argparse
from io import BytesIO
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial.transform import Rotation


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]
RL_BENCH_ROOT = REPO_ROOT / "benchmarks" / "RLBench"
sys.path.insert(0, str(RL_BENCH_ROOT))

from collect_lerobot_pointcloud import (
    capture_initial_object_states,
    configure_front_camera,
    make_observation_config,
    restore_initial_object_states_from_arrays,
    task_class_from_name,
)
from gripper_control import (
    ABSOLUTE_WIDTH,
    DELTA_WIDTH_INITIAL_SYNC,
    GRIPPER_CONTROL_MODES,
    initial_delta_reference_for_chunk,
    libero_style_gripper_target,
    set_gripper_absolute_width_position_target,
)
from video_utils import annotate_final_task_result_frames


DEFAULT_ARTIFACT_ROOT = (
    RL_BENCH_ROOT
    / "datasets/rlbench_water_plants_sweep_to_dustpan_150episodes_pointcloud_lerobot_artifacts"
)
DEFAULT_DATASET_ROOT = (
    RL_BENCH_ROOT
    / "datasets/rlbench_water_plants_sweep_to_dustpan_150episodes_pointcloud_lerobot"
)
RUN_DATE = time.strftime("%Y%m%d_%H%M%S")
DEFAULT_OUTPUT_ROOT = None

def parse_args():
    parser = argparse.ArgumentParser(description="Replay actions from one RLBench dataset episode.")
    parser.add_argument(
        "--task",
        choices=[
            "close_box",
            "close_fridge",
            "close_laptop_lid",
            "phone_on_base",
            "stack_wine",
            "water_plants",
            "sweep_to_dustpan",
            "take_frame_off_hanger",
            "take_umbrella_out_of_umbrella_stand",
            "toilet_seat_down",
        ],
        required=True,
    )
    parser.add_argument(
        "--episode",
        default="0",
        help=(
            "Local episode index, or comma-separated indices such as "
            "1,2,3,134. Each episode is replayed independently."
        ),
    )
    parser.add_argument(
        "--mode",
        choices=["eef0_planning", "eef0", "raw_joint"],
        required=True,
        help="eef0_planning matches RLBench's waypoint/path planner.",
    )
    parser.add_argument("--variation", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument(
        "--front-camera-position-m",
        type=float,
        nargs=3,
        default=None,
        metavar=("X", "Y", "Z"),
        help="Optional world-frame cam_front position override.",
    )
    parser.add_argument(
        "--front-camera-look-at-m",
        type=float,
        nargs=3,
        default=None,
        metavar=("X", "Y", "Z"),
        help="Optional world-frame point aimed at by cam_front local +Z.",
    )
    parser.add_argument(
        "--disable-point-cloud",
        action="store_true",
        help="Disable replay-time point-cloud rendering; RGB and task control remain unchanged.",
    )
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument(
        "--artifact-only-input",
        action="store_true",
        help=(
            "Load actions, states, initial robot pose, RNG state, and object "
            "snapshot directly from arrays.npz. This preserves a raw artifact "
            "trajectory when no packed LeRobot dataset is available."
        ),
    )
    parser.add_argument(
        "--action-source",
        choices=["parquet", "artifact", "observation"],
        default="parquet",
        help=(
            "parquet uses the action labels seen during training; artifact uses "
            "the converted artifact actions; observation ignores both action "
            "sources and commands the recorded observation.state trajectory."
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=(
            "Replay output root. By default, use the canonical eval/replays "
            "directory and include timestamp, task, and replay mode."
        ),
    )
    parser.add_argument("--video-fps", type=int, default=20)
    parser.add_argument(
        "--video-policy",
        choices=["all", "failures", "none"],
        default="all",
        help="Save replay video for all episodes, failed episodes only, or no episodes.",
    )
    parser.add_argument(
        "--continue-after-action-error",
        action="store_true",
        help=(
            "Record a frame and continue with the next dataset action after an individual "
            "planner/controller exception. Also continue after the first success so the "
            "saved video covers every action label."
        ),
    )
    parser.add_argument(
        "--controller-profile",
        choices=["single_step", "pointact_eval"],
        default="single_step",
        help=(
            "single_step preserves the historical one-task.step replay. "
            "pointact_eval uses the same Mover retries, deferred gripper, workspace "
            "clipping, and PyRep compatibility patch as formal evaluation."
        ),
    )
    parser.add_argument("--mover-max-tries", type=int, default=10)
    parser.add_argument("--mover-position-tolerance", type=float, default=0.05)
    parser.add_argument(
        "--mover-rotation-tolerance",
        type=float,
        default=0.0,
        help="Rotation tolerance in radians; 0 disables the rotation gate.",
    )
    parser.add_argument("--mover-gripper-position-tolerance", type=float, default=0.02)
    parser.add_argument(
        "--mover-gripper-rotation-tolerance",
        type=float,
        default=0.0,
        help="Rotation tolerance before a deferred gripper change; 0 disables it.",
    )
    parser.add_argument("--waypoint-position-tolerance", type=float, default=0.002)
    parser.add_argument("--waypoint-rotation-tolerance", type=float, default=0.03)
    parser.add_argument(
        "--clip-within-workspace",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--gripper-after-reach",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--pointact-pyrep-compat",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--log-control-details",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--gripper-open-width", type=float, default=0.04)
    parser.add_argument(
        "--gripper-mode",
        choices=GRIPPER_CONTROL_MODES,
        default=DELTA_WIDTH_INITIAL_SYNC,
        help=(
            "delta_width_initial_sync synchronizes action[9] once before replay "
            "and self-references the first executed row; libero_delta keeps the "
            "historical measured-width anchor; absolute_width thresholds each row."
        ),
    )
    parser.add_argument(
        "--gripper-delta-threshold",
        type=float,
        default=0.002,
        help="LIBERO width-change deadband in metres.",
    )
    parser.add_argument(
        "--gripper-delta-alignment",
        choices=["current_minus_previous", "next_minus_current"],
        default="current_minus_previous",
    )
    parser.add_argument("--seed", type=int, default=123, help="Fix the fresh-reset scene for fair mode comparison.")
    parser.add_argument(
        "--allow-fresh-reset",
        action="store_true",
        help=(
            "Allow legacy replay without a recorded initial task state. "
            "This does not reproduce the recorded object layout."
        ),
    )
    parser.add_argument(
        "--water-plant-collision",
        choices=["enabled", "disabled"],
        default="enabled",
        help="Match the water_plants task physics used during collection.",
    )
    parser.add_argument(
        "--water-drop-collision",
        choices=["original", "disabled"],
        default="original",
        help="Match the water-drop physics used during collection.",
    )
    return parser.parse_args()


def pose7_to_matrix(pose):
    """Convert RLBench [x,y,z,qx,qy,qz,qw] into a 4x4 matrix."""
    pose = np.asarray(pose, dtype=np.float64)
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = Rotation.from_quat(pose[3:7]).as_matrix()
    matrix[:3, 3] = pose[:3]
    return matrix


def pose9_to_matrix(pose):
    """Convert Song [xyz, rotation column 0, rotation column 1] to 4x4."""
    pose = np.asarray(pose, dtype=np.float64)
    first = pose[3:6]
    first = first / np.linalg.norm(first)
    second = pose[6:9] - np.dot(pose[6:9], first) * first
    second = second / np.linalg.norm(second)
    third = np.cross(first, second)
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = np.column_stack((first, second, third))
    matrix[:3, 3] = pose[:3]
    return matrix


def matrix_to_pose7(matrix):
    """Convert a 4x4 matrix into RLBench [x,y,z,qx,qy,qz,qw]."""
    quaternion = Rotation.from_matrix(matrix[:3, :3]).as_quat()
    return np.concatenate((matrix[:3, 3], quaternion)).astype(np.float32)


def resize_rgb_for_comparison(image, target_shape):
    """Resize only the diagnostic image so 128 and 512 camera outputs can be compared."""
    target_height = int(target_shape[0])
    target_width = int(target_shape[1])
    if image.shape[0] == target_height and image.shape[1] == target_width:
        return image
    resized = Image.fromarray(np.asarray(image, dtype=np.uint8)).resize(
        (target_width, target_height), Image.Resampling.BILINEAR
    )
    return np.asarray(resized, dtype=np.uint8)


def annotate_replay_video_frames(
    frames,
    execution_trace,
    actions,
    task_name,
    episode_index,
    mode,
    success,
):
    """Overlay exact video-frame to dataset-action mapping on replay frames."""
    trace_by_video_frame = {
        int(item["video_frame"]): item
        for item in execution_trace
        if item.get("video_frame") is not None
    }
    action_array = np.asarray(actions)
    total_video_index = max(len(frames) - 1, 0)
    annotated = []
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"

    for video_frame_index, frame in enumerate(frames):
        image = Image.fromarray(np.asarray(frame, dtype=np.uint8)).convert("RGB")
        draw = ImageDraw.Draw(image)
        trace = trace_by_video_frame.get(video_frame_index, {})
        action_index = trace.get("action_index")
        phase = str(trace.get("phase", "unmapped"))
        mover_attempt = int(trace.get("mover_attempt", 0) or 0)
        lines = [
            f"task={task_name} ep={int(episode_index):03d} mode={mode}",
            f"video_frame={video_frame_index:04d}/{total_video_index:04d} "
            f"result={'SUCCESS' if success else 'FAILURE'}",
        ]
        if action_index is None:
            lines.extend(
                [
                    "dataset_frame=RESET action_index=none",
                    f"phase={phase} mover_attempt={mover_attempt}",
                ]
            )
        else:
            action_index = int(action_index)
            lines.extend(
                [
                    f"dataset_frame={action_index:04d} action_index={action_index:04d}",
                    f"phase={phase} mover_attempt={mover_attempt}",
                ]
            )
            if 0 <= action_index < len(action_array) and action_array.shape[1] >= 10:
                action = action_array[action_index]
                lines.append(
                    "label xyz=({:+.3f},{:+.3f},{:+.3f}) grip={:.4f}".format(
                        float(action[0]),
                        float(action[1]),
                        float(action[2]),
                        float(action[9]),
                    )
                )

        padding = max(3, int(round(image.width / 96.0)))
        font_size = max(8, min(14, int(round(image.width / 20.0))))
        while True:
            try:
                font = ImageFont.truetype(font_path, font_size)
            except OSError:
                font = ImageFont.load_default()
                break
            widths = [draw.textbbox((0, 0), line, font=font)[2] for line in lines]
            if max(widths, default=0) + 2 * padding <= image.width or font_size <= 7:
                break
            font_size -= 1
        text_boxes = [draw.textbbox((0, 0), line, font=font) for line in lines]
        line_height = max((box[3] - box[1] for box in text_boxes), default=9)
        line_gap = max(1, padding // 2)
        bar_height = (
            2 * padding + len(lines) * line_height + max(0, len(lines) - 1) * line_gap
        )
        overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
        overlay_draw = ImageDraw.Draw(overlay)
        overlay_draw.rectangle(
            (0, 0, image.width, bar_height), fill=(0, 0, 0, 190)
        )
        image = Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")
        draw = ImageDraw.Draw(image)
        y = padding
        for line_index, line in enumerate(lines):
            color = (255, 230, 80) if line_index in {1, 2} else (255, 255, 255)
            draw.text((padding, y), line, fill=color, font=font)
            y += line_height + line_gap
        annotated.append(np.asarray(image, dtype=np.uint8))
    return annotated


def artifact_path(root, task, episode):
    name = task + "__episode_" + str(episode).zfill(5)
    return root.expanduser().resolve() / name / "arrays.npz"


def initial_task_state_path(dataset_root, global_episode):
    name = "episode_" + str(global_episode).zfill(6) + ".npz"
    return dataset_root.expanduser().resolve() / "initial_task_states" / name


def initial_object_state_path(dataset_root, global_episode):
    name = "episode_" + str(global_episode).zfill(6) + ".npz"
    return dataset_root.expanduser().resolve() / "initial_object_states" / name


def compare_initial_object_states(task, state_path):
    if not state_path.is_file():
        return None
    with np.load(state_path) as recorded_file:
        recorded = {key: recorded_file[key] for key in recorded_file.files}
    current = capture_initial_object_states(task)
    current_by_name = {
        str(name): index for index, name in enumerate(current["initial_object_names"])
    }
    position_errors = []
    rotation_errors = []
    position_errors_by_name = []
    rotation_errors_by_name = []
    missing_names = []
    for index, name in enumerate(recorded["initial_object_names"]):
        current_index = current_by_name.get(str(name))
        if current_index is None:
            missing_names.append(str(name))
            continue
        recorded_pose = recorded["initial_object_poses"][index]
        current_pose = current["initial_object_poses"][current_index]
        if np.isfinite(recorded_pose).all() and np.isfinite(current_pose).all():
            position_error = float(np.linalg.norm(recorded_pose[:3] - current_pose[:3]))
            position_errors.append(position_error)
            position_errors_by_name.append((position_error, str(name)))
            recorded_rotation = Rotation.from_quat(recorded_pose[3:7])
            current_rotation = Rotation.from_quat(current_pose[3:7])
            rotation_error = float((recorded_rotation.inv() * current_rotation).magnitude())
            rotation_errors.append(rotation_error)
            rotation_errors_by_name.append((rotation_error, str(name)))
    return {
        "source": str(state_path),
        "recorded_object_count": int(len(recorded["initial_object_names"])),
        "current_object_count": int(len(current["initial_object_names"])),
        "missing_names": missing_names,
        "position_error_max_m": max(position_errors) if position_errors else None,
        "position_error_max_object": (
            max(position_errors_by_name)[1] if position_errors_by_name else None
        ),
        "rotation_error_max_rad": max(rotation_errors) if rotation_errors else None,
        "rotation_error_max_object": (
            max(rotation_errors_by_name)[1] if rotation_errors_by_name else None
        ),
    }


def restore_initial_object_states(task, state_path):
    """Restore the readable per-object snapshot after configuration-tree restore.

    CoppeliaSim configuration trees do not reliably restore every dynamic child
    pose in this RLBench/PyRep combination (notably the umbrella task).  The
    collection sidecar contains the absolute pose of every task object, so use
    it as the final authoritative scene-state restore.
    """
    if not state_path.is_file():
        return False
    with np.load(state_path) as recorded_file:
        recorded = {key: recorded_file[key] for key in recorded_file.files}
    objects = task.get_base().get_objects_in_tree(
        exclude_base=False, first_generation_only=False
    )
    current_by_name = {obj.get_name(): obj for obj in objects}
    recorded_names = [str(name) for name in recorded["initial_object_names"]]
    if len(set(recorded_names)) != len(recorded_names):
        raise RuntimeError("Cannot restore object snapshot with duplicate object names.")
    missing = [name for name in recorded_names if name not in current_by_name]
    if missing:
        raise RuntimeError("Cannot restore missing task objects: " + ", ".join(missing))

    # get_objects_in_tree() is parent-first. Absolute poses are applied in that
    # order so moving a parent cannot invalidate an already restored child.
    for index, name in enumerate(recorded_names):
        obj = current_by_name[name]
        pose = np.asarray(recorded["initial_object_poses"][index], dtype=np.float64)
        if np.isfinite(pose).all():
            obj.set_pose(pose.tolist(), reset_dynamics=True)
        joint_position = float(recorded["initial_object_joint_positions"][index])
        if np.isfinite(joint_position) and hasattr(obj, "set_joint_position"):
            obj.set_joint_position(joint_position, disable_dynamics=True)
        joint_target = float(recorded["initial_object_joint_target_positions"][index])
        if np.isfinite(joint_target) and hasattr(obj, "set_joint_target_position"):
            obj.set_joint_target_position(joint_target)
        joint_target_velocity = float(
            recorded["initial_object_joint_target_velocities"][index]
        )
        if np.isfinite(joint_target_velocity) and hasattr(obj, "set_joint_target_velocity"):
            obj.set_joint_target_velocity(joint_target_velocity)
    # Joint setters can internally step CoppeliaSim. Re-apply absolute poses so
    # the final state, rather than the intermediate setter state, is exact.
    for index, name in enumerate(recorded_names):
        pose = np.asarray(recorded["initial_object_poses"][index], dtype=np.float64)
        if np.isfinite(pose).all():
            current_by_name[name].set_pose(pose.tolist(), reset_dynamics=True)
    return True


def collect_task_specific_diagnostics(task):
    """Collect success-condition internals while the simulator is still live."""
    diagnostics = {}
    if task.get_name() != "water_plants":
        return diagnostics
    diagnostics["reached_pour_point"] = bool(getattr(task, "reached", False))
    diagnostics["reached_pour_point_once"] = bool(
        getattr(task, "reachedOnce", False)
    )
    drops = list(getattr(task, "drops", []))
    diagnostics["water_drop_count"] = len(drops)
    diagnostics["water_drop_detected"] = []
    diagnostics["water_drop_positions"] = []
    success_sensor = getattr(task, "success_sensor", None)
    for drop in drops:
        try:
            diagnostics["water_drop_detected"].append(
                bool(success_sensor.is_detected(drop))
            )
        except Exception:
            diagnostics["water_drop_detected"].append(None)
        try:
            diagnostics["water_drop_positions"].append(
                np.asarray(drop.get_position(), dtype=np.float64).tolist()
            )
        except Exception:
            diagnostics["water_drop_positions"].append(None)
    for key, obj in (
        ("waterer_position", getattr(task, "waterer", None)),
        ("head_position", getattr(task, "head", None)),
        ("pour_point_position", getattr(task, "pour_point", None)),
        ("success_sensor_position", success_sensor),
    ):
        try:
            diagnostics[key] = np.asarray(
                obj.get_position(), dtype=np.float64
            ).tolist()
        except Exception:
            diagnostics[key] = None
    return diagnostics


def read_initial_task_state(dataset_root, global_episode, artifact_arrays, artifact_file):
    """Load configuration bytes and object count, preferring the dataset sidecar."""
    sidecar = initial_task_state_path(dataset_root, global_episode)
    if sidecar.is_file():
        with np.load(sidecar) as state_file:
            configuration_bytes = np.asarray(
                state_file["configuration_bytes"], dtype=np.uint8
            ).tobytes()
            object_count = int(state_file["object_count"])
        return (configuration_bytes, object_count), str(sidecar)

    required = {"initial_task_state_bytes", "initial_task_state_object_count"}
    if artifact_arrays is not None and required.issubset(set(artifact_arrays.files)):
        configuration_bytes = np.asarray(
            artifact_arrays["initial_task_state_bytes"], dtype=np.uint8
        ).tobytes()
        object_count = int(artifact_arrays["initial_task_state_object_count"])
        return (configuration_bytes, object_count), str(artifact_file)
    return None, None


def read_demo_reset_state(dataset_root, global_episode, artifact_arrays, artifact_file):
    """Load the random state used by RLBench's official reset_to_demo path."""
    sidecar = initial_task_state_path(dataset_root, global_episode)
    source = sidecar if sidecar.is_file() else artifact_file
    required = {
        "demo_random_seed_state",
        "demo_random_seed_position",
        "demo_random_seed_has_gauss",
        "demo_random_seed_cached_gaussian",
        "demo_num_reset_attempts",
    }
    try:
        with np.load(source) as state_file:
            if not required.issubset(set(state_file.files)):
                return None, None
            random_state = (
                "MT19937",
                np.asarray(state_file["demo_random_seed_state"], dtype=np.uint32),
                int(state_file["demo_random_seed_position"]),
                int(state_file["demo_random_seed_has_gauss"]),
                float(state_file["demo_random_seed_cached_gaussian"]),
            )
            reset_attempts = int(state_file["demo_num_reset_attempts"])
    except (OSError, ValueError, KeyError):
        return None, None
    return random_state, reset_attempts


def load_parquet_episode(dataset_root, task, local_episode):
    """Read one episode from the parquet files used by the training loader."""
    import pyarrow.dataset as pyarrow_dataset

    parquet = pyarrow_dataset.dataset(
        str(dataset_root.expanduser().resolve() / "data"),
        format="parquet",
    )
    # The original ten-task dataset used the fixed mapping below, while
    # standalone task subsets are reindexed from zero.  Prefer the conversion
    # metadata so replay always follows the dataset that is actually opened.
    fallback_task_indices = {
        "close_box": 0,
        "close_fridge": 1,
        "close_laptop_lid": 2,
        "phone_on_base": 3,
        "stack_wine": 4,
        "water_plants": 9,
        "sweep_to_dustpan": 5,
        "take_frame_off_hanger": 6,
        "take_umbrella_out_of_umbrella_stand": 7,
        "toilet_seat_down": 8,
    }
    task_index = fallback_task_indices[str(task)]
    conversion_path = dataset_root.expanduser().resolve() / "meta" / "rlbench_conversion.json"
    if conversion_path.is_file():
        with open(conversion_path, "r", encoding="utf-8") as file:
            conversion = json.load(file)
        task_names = [str(name) for name in conversion.get("tasks", [])]
        if str(task) in task_names:
            task_index = task_names.index(str(task))
    episode_table = parquet.to_table(columns=["episode_index", "task_index"])
    episode_pairs = np.asarray(
        episode_table.select(["episode_index", "task_index"]).to_pydict()["episode_index"],
        dtype=np.int64,
    )
    task_pairs = np.asarray(
        episode_table.select(["episode_index", "task_index"]).to_pydict()["task_index"],
        dtype=np.int64,
    )
    task_episodes = np.unique(episode_pairs[task_pairs == task_index])
    task_episodes.sort()
    if len(task_episodes) == 0:
        raise RuntimeError(
            f"No episodes found for task={task!r}, resolved task_index={task_index}."
        )
    if int(local_episode) < 0 or int(local_episode) >= len(task_episodes):
        raise IndexError(
            f"{task} episode {local_episode} is out of range; "
            f"dataset contains {len(task_episodes)} episodes for task_index={task_index}."
        )
    global_episode = int(task_episodes[int(local_episode)])
    table = parquet.to_table(
        filter=pyarrow_dataset.field("episode_index") == global_episode,
        columns=["action", "observation.state", "frame_index"],
    )
    actions = np.asarray(table["action"].to_pylist(), dtype=np.float32)
    states = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)
    frame_indices = np.asarray(table["frame_index"].to_pylist(), dtype=np.int64)
    order = np.argsort(frame_indices)
    return actions[order], states[order], global_episode


def load_dataset_episode_sidecars(dataset_root, global_episode):
    """Load artifacts that are stored directly in the converted dataset."""
    root = dataset_root.expanduser().resolve()
    action_path = root / "raw_expert_actions" / ("episode_" + str(global_episode).zfill(6) + ".npy")
    image_path = root / "point_clouds" / ("episode_" + str(global_episode).zfill(6) + ".zarr")
    if not action_path.is_file():
        raise FileNotFoundError("Dataset raw expert action file is missing: " + str(action_path))
    raw_actions = np.asarray(np.load(action_path), dtype=np.float32)
    if raw_actions.ndim != 2 or raw_actions.shape[1] != 8:
        raise RuntimeError("Expected raw expert actions with shape (T, 8), got " + str(raw_actions.shape))
    return raw_actions, image_path


def load_recorded_first_robot_state(dataset_root, global_episode, parquet_states):
    """Load the recorded absolute first EEF pose and physical gripper width."""
    root = dataset_root.expanduser().resolve()
    pose_path = root / "world_ee_poses" / ("episode_" + str(global_episode).zfill(6) + ".npy")
    if not pose_path.is_file():
        raise FileNotFoundError("Dataset world EEF pose file is missing: " + str(pose_path))
    poses = np.asarray(np.load(pose_path), dtype=np.float32)
    if poses.ndim != 2 or poses.shape[1] != 9 or len(poses) != len(parquet_states):
        raise RuntimeError(
            "Expected world EEF poses with shape "
            + str((len(parquet_states), 9))
            + ", got "
            + str(poses.shape)
        )
    return poses[0], float(parquet_states[0, 9]), str(pose_path)


def restore_recorded_first_robot_state(task, recorded_pose9, recorded_gripper_width):
    """Put the Panda at the recorded first EEF pose before replaying actions."""
    target_matrix = pose9_to_matrix(recorded_pose9)
    target_pose7 = matrix_to_pose7(target_matrix)
    arm = task._robot.arm
    joints = arm.solve_ik_via_jacobian(
        target_pose7[:3], quaternion=target_pose7[3:7], relative_to=None
    )
    arm.set_joint_positions(list(joints), disable_dynamics=True)
    arm.set_joint_target_positions(list(joints))

    # Both tasks in this dataset begin fully open.  Fail loudly rather than
    # silently replaying from a different discrete gripper state.
    recorded_open = bool(float(recorded_gripper_width) > 0.04)
    current_open = all(value > 0.9 for value in task._robot.gripper.get_open_amount())
    if current_open != recorded_open:
        done = False
        while not done:
            done = task._robot.gripper.actuate(1.0 if recorded_open else 0.0, velocity=0.2)
            task._scene.step()

    observation = task.get_observation()
    actual_matrix = pose7_to_matrix(observation.gripper_pose)
    position_error = float(
        np.linalg.norm(actual_matrix[:3, 3] - target_matrix[:3, 3])
    )
    rotation_delta = target_matrix[:3, :3].T @ actual_matrix[:3, :3]
    rotation_error = float(Rotation.from_matrix(rotation_delta).magnitude())
    actual_open = bool(float(observation.gripper_open) > 0.5)
    if position_error > 5e-4 or rotation_error > np.deg2rad(0.1) or actual_open != recorded_open:
        raise RuntimeError(
            "Could not restore recorded first robot state exactly enough: "
            f"position_error={position_error:.6g} m, "
            f"rotation_error={np.rad2deg(rotation_error):.6g} deg, "
            f"gripper_open={actual_open} expected={recorded_open}."
        )
    return observation, {
        "source": "world_ee_poses plus observation.state gripper width",
        "target_pose9": np.asarray(recorded_pose9, dtype=np.float64).tolist(),
        "recorded_gripper_width_m": float(recorded_gripper_width),
        "position_error_m": position_error,
        "rotation_error_rad": rotation_error,
        "gripper_open": actual_open,
    }


def load_parquet_first_image(dataset_root, global_episode):
    """Decode the recorded first front-camera frame for replay diagnostics."""
    import pyarrow.dataset as pyarrow_dataset

    parquet = pyarrow_dataset.dataset(
        str(dataset_root.expanduser().resolve() / "data"),
        format="parquet",
    )
    table = parquet.to_table(
        filter=pyarrow_dataset.field("episode_index") == int(global_episode),
        columns=["frame_index", "observation.images.front"],
    )
    frame_indices = np.asarray(table["frame_index"].to_pylist(), dtype=np.int64)
    if len(frame_indices) == 0:
        raise RuntimeError("No rows found for episode " + str(global_episode))
    first_index = int(np.argmin(frame_indices))
    encoded = table["observation.images.front"][first_index].as_py()
    image_bytes = encoded.get("bytes", b"") if isinstance(encoded, dict) else b""
    if not image_bytes:
        raise RuntimeError("Recorded front image bytes are missing for episode " + str(global_episode))
    with Image.open(BytesIO(image_bytes)) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def dataset_action_alignment(dataset_root):
    """Read the conversion contract; old datasets used post-step alignment."""
    metadata_path = dataset_root.expanduser().resolve() / "meta" / "rlbench_conversion.json"
    if not metadata_path.is_file():
        return "observation"
    try:
        with open(metadata_path, "r", encoding="utf-8") as file:
            value = json.load(file).get("action_alignment", "observation")
    except (OSError, ValueError, TypeError):
        return "observation"
    return str(value) if value in {"transition", "observation"} else "observation"


def replay_observation_config(image_size, include_point_cloud):
    if include_point_cloud:
        return make_observation_config(image_size)
    from rlbench import CameraConfig, ObservationConfig

    config = ObservationConfig()
    config.set_all(False)
    config.front_camera = CameraConfig(
        rgb=True, depth=False, point_cloud=False, mask=False, image_size=(image_size, image_size)
    )
    config.wrist_camera = CameraConfig(
        rgb=True, depth=False, point_cloud=False, mask=False, image_size=(image_size, image_size)
    )
    config.gripper_open = True
    config.gripper_pose = True
    config.record_gripper_closing = True
    return config


def make_environment(mode, image_size, include_point_cloud=True):
    from rlbench import Environment
    from rlbench.action_modes.action_mode import MoveArmThenGripper
    from rlbench.action_modes.arm_action_modes import (
        EndEffectorPoseViaIK,
        EndEffectorPoseViaPlanning,
        JointPosition,
        RelativeFrame,
    )
    from rlbench.action_modes.gripper_action_modes import Discrete

    if mode in {"eef0", "eef0_planning"}:
        if mode == "eef0_planning":
            arm_mode = EndEffectorPoseViaPlanning(
                absolute_mode=True,
                frame=RelativeFrame.WORLD,
                collision_checking=False,
            )
        else:
            arm_mode = EndEffectorPoseViaIK(
                absolute_mode=True,
                frame=RelativeFrame.WORLD,
                collision_checking=False,
            )
    else:
        arm_mode = JointPosition(absolute_mode=True)
    return Environment(
        MoveArmThenGripper(arm_mode, Discrete()),
        obs_config=replay_observation_config(image_size, include_point_cloud),
        headless=True,
        static_positions=False,
    )


def run_replay(args):
    # water_plants reads these at task-module import time. Set them before
    # task_class_from_name() imports that module so replay matches collection.
    os.environ["RLBENCH_WATER_PLANT_COLLISION"] = str(args.water_plant_collision)
    os.environ["RLBENCH_WATER_DROP_COLLISION"] = str(args.water_drop_collision)
    source_path = artifact_path(args.artifact_root, args.task, args.episode)
    artifact_snapshot_source = None
    if args.artifact_only_input:
        if not source_path.is_file():
            raise FileNotFoundError(
                "Artifact-only replay source is missing: " + str(source_path)
            )
        global_episode = int(args.episode)
        with np.load(source_path, allow_pickle=False) as source:
            artifact_actions = np.asarray(source["actions"], dtype=np.float32)
            artifact_states = np.asarray(source["states"], dtype=np.float32)
            raw_actions = np.asarray(
                source["raw_expert_actions"], dtype=np.float32
            )
            recorded_first_image = np.asarray(
                source["images"][0], dtype=np.uint8
            )
            recorded_first_pose9 = np.asarray(
                source["world_ee_poses"][0], dtype=np.float32
            )
            recorded_first_gripper_width = float(artifact_states[0, 9])
            initial_task_state, initial_task_state_source = (
                read_initial_task_state(
                    args.dataset_root, global_episode, source, source_path
                )
            )
            demo_random_state, demo_reset_attempts = read_demo_reset_state(
                args.dataset_root, global_episode, source, source_path
            )
        artifact_snapshot_source = source_path
        recorded_robot_state_source = str(source_path)
        parquet_actions = artifact_actions
        parquet_states = artifact_states
    else:
        parquet_actions, parquet_states, global_episode = load_parquet_episode(
            args.dataset_root, args.task, args.episode
        )
        (
            recorded_first_pose9,
            recorded_first_gripper_width,
            recorded_robot_state_source,
        ) = load_recorded_first_robot_state(
            args.dataset_root, global_episode, parquet_states
        )
    parquet_action_state_equal = bool(
        np.array_equal(parquet_actions, parquet_states)
    )
    parquet_action_state_max_abs_difference = float(
        np.max(np.abs(parquet_actions - parquet_states))
    )
    dataset_direct = not source_path.is_file()
    if args.artifact_only_input:
        dataset_direct = False
    elif dataset_direct:
        # New converted datasets keep replay inputs in sidecar directories and
        # keep the recorded RGB frame in parquet instead of duplicating an
        # artifact-level arrays.npz file.
        raw_actions, _ = load_dataset_episode_sidecars(args.dataset_root, global_episode)
        recorded_first_image = load_parquet_first_image(args.dataset_root, global_episode)
        artifact_actions = parquet_actions
        artifact_states = parquet_states
        initial_task_state, initial_task_state_source = read_initial_task_state(
            args.dataset_root, global_episode, None, source_path
        )
        demo_random_state, demo_reset_attempts = read_demo_reset_state(
            args.dataset_root, global_episode, None, source_path
        )
    else:
        with np.load(source_path) as source:
            artifact_actions = np.asarray(source["actions"], dtype=np.float32)
            artifact_states = np.asarray(source["states"], dtype=np.float32)
            raw_actions = np.asarray(source["raw_expert_actions"], dtype=np.float32)
            recorded_first_image = np.asarray(source["images"][0], dtype=np.uint8)
            initial_task_state, initial_task_state_source = read_initial_task_state(
                args.dataset_root, global_episode, source, source_path
            )
            demo_random_state, demo_reset_attempts = read_demo_reset_state(
                args.dataset_root, global_episode, source, source_path
            )
    if initial_task_state is None and not args.allow_fresh_reset:
        raise RuntimeError(
            "This episode has no recorded RLBench initial task state. Exact action replay "
            "is impossible because task.reset() creates a different object layout. "
            "Recollect the dataset with collect_lerobot_pointcloud.py. "
            "Use --allow-fresh-reset only for the old non-exact diagnostic behavior."
        )
    if args.action_source == "parquet":
        actions = parquet_actions
        states = parquet_states
    elif args.action_source == "artifact":
        actions = artifact_actions
        states = artifact_states
    else:
        # observation.state stores the measured EEF pose9 relative to the
        # episode's first EEF frame plus physical gripper width.  Copy it into
        # the command buffer so this mode never reads an action label.
        actions = parquet_states.copy()
        states = parquet_states
    if len(actions) != len(raw_actions):
        raise RuntimeError(
            "Dataset action and raw expert action lengths do not match: "
            + str(len(actions)) + " versus " + str(len(raw_actions))
        )
    if args.artifact_only_input:
        record_path = source_path.parent / "record.json"
        try:
            action_alignment = str(
                json.loads(record_path.read_text(encoding="utf-8")).get(
                    "action_alignment", "transition"
                )
            )
        except (OSError, ValueError, TypeError):
            action_alignment = "transition"
    else:
        action_alignment = dataset_action_alignment(args.dataset_root)

    # Use the same seed in separate eef0/raw_joint processes.  RLBench task
    # placement uses NumPy random sampling; Python's seed is fixed as well so
    # future task code cannot silently make the two control groups differ.
    np.random.seed(args.seed)
    random.seed(args.seed)
    environment = make_environment(args.mode, args.image_size, not args.disable_point_cloud)
    frames = []
    errors = []
    position_errors = []
    rotation_errors = []
    reward = 0.0
    terminate = False
    executed = 0
    attempted = 0
    ever_succeeded = False
    execution_trace = [
        {
            "video_frame": 0,
            "action_index": None,
            "phase": "reset",
            "mover_attempt": 0,
        }
    ]
    mover_attempt_count = 0
    mover_retry_count = 0
    mover_unreached_count = 0
    controller_continue_errors = 0
    object_state_validation = None
    robot_state_validation = None
    task_specific_diagnostics = {}
    try:
        environment.launch()
        configure_front_camera(
            position_m=args.front_camera_position_m,
            look_at_m=args.front_camera_look_at_m,
        )
        if args.controller_profile == "pointact_eval" and args.pointact_pyrep_compat:
            from official_eval import apply_pointact_pyrep_compatibility_patch

            apply_pointact_pyrep_compatibility_patch()
        task = environment.get_task(task_class_from_name(args.task))
        task.set_variation(args.variation)
        scene_restore_method = "fixed_seed_reset"
        if demo_random_state is not None:
            # This mirrors Demo.restore_state() followed by TaskEnvironment.reset(demo).
            np.random.set_state(demo_random_state)

            class ReplayDemo:
                num_reset_attempts = demo_reset_attempts

            descriptions, observation = task.reset(ReplayDemo())
            exact_recorded_scene_restored = True
            scene_restore_method = "official_demo_random_state"
        else:
            descriptions, observation = task.reset()
            exact_recorded_scene_restored = initial_task_state is not None
        if initial_task_state is not None:
            # reset() must run first so RLBench creates the correct task object
            # tree and initializes this variation's success conditions. Even
            # the official demo RNG path can settle physics differently by a
            # few centimetres, so always apply the recorded configuration tree
            # afterwards instead of treating the RNG reset as exact by itself.
            task._task.restore_state(initial_task_state)
            observation = task.get_observation()
            exact_recorded_scene_restored = True
            scene_restore_method += "_plus_configuration_tree"
        if artifact_snapshot_source is not None:
            with np.load(
                artifact_snapshot_source, allow_pickle=False
            ) as artifact_source:
                restore_initial_object_states_from_arrays(
                    task._task, artifact_source
                )
            exact_recorded_scene_restored = True
            scene_restore_method += "_plus_artifact_object_snapshot"
        else:
            object_snapshot_path = initial_object_state_path(
                args.dataset_root, global_episode
            )
            if restore_initial_object_states(task._task, object_snapshot_path):
                exact_recorded_scene_restored = True
                scene_restore_method += "_plus_per_object_snapshot"
            if exact_recorded_scene_restored:
                object_state_validation = compare_initial_object_states(
                    task._task, object_snapshot_path,
                )
        observation, robot_state_validation = restore_recorded_first_robot_state(
            task,
            recorded_first_pose9,
            recorded_first_gripper_width,
        )
        robot_state_validation["recorded_pose_source"] = recorded_robot_state_source
        anchor_world = pose7_to_matrix(observation.gripper_pose)
        reset_image = np.asarray(observation.front_rgb, dtype=np.uint8)
        recorded_image_for_mae = resize_rgb_for_comparison(
            recorded_first_image, reset_image.shape
        )
        first_image_mae = float(
            np.mean(
                np.abs(
                    reset_image.astype(np.float32)
                    - recorded_image_for_mae.astype(np.float32)
                )
            )
        )
        frames.append(reset_image)

        # Transition-aligned datasets place the first expert command at row 0.
        # Legacy post-step datasets contain an identity placeholder at row 0.
        # For observation replay, row 0 is the robot state already restored
        # above, so the first meaningful measured-state waypoint is row 1.
        start_index = (
            1
            if args.action_source == "observation"
            else (0 if action_alignment == "transition" else 1)
        )
        gripper_command_open = float(observation.gripper_open) > 0.5
        previous_predicted_width = (
            float(np.clip(observation.gripper_open, 0.0, 1.0)) * 0.08
        )
        gripper_initial_sync_info = None
        if (
            args.mode in {"eef0", "eef0_planning"}
            and args.gripper_mode == DELTA_WIDTH_INITIAL_SYNC
        ):
            gripper_initial_sync_info = set_gripper_absolute_width_position_target(
                task, float(actions[0, 9]), 0.08
            )
            observation = task.get_observation()
            gripper_command_open = bool(
                gripper_initial_sync_info["command_gripper_open"]
            )
            previous_predicted_width = initial_delta_reference_for_chunk(
                actions, start_index
            )
            print(
                "[gripper-initial-sync] "
                + json.dumps(gripper_initial_sync_info, separators=(",", ":")),
                flush=True,
            )
        gripper_events = []
        for frame_index in range(start_index, len(actions)):
            attempted += 1
            try:
                relative_target = pose9_to_matrix(actions[frame_index, :9])
                target_world = anchor_world @ relative_target
                if args.mode in {"eef0", "eef0_planning"}:
                    previous_command_open = bool(gripper_command_open)
                    if args.gripper_mode != ABSOLUTE_WIDTH:
                        gripper_command_open, gripper_event, width_change = (
                            libero_style_gripper_target(
                                previous_predicted_width,
                                actions[frame_index, 9],
                                gripper_command_open,
                                args.gripper_delta_threshold,
                                alignment=args.gripper_delta_alignment,
                                next_width=(
                                    actions[frame_index + 1, 9]
                                    if frame_index + 1 < len(actions)
                                    else None
                                ),
                            )
                        )
                        gripper_events.append(
                            {
                                "frame": int(frame_index),
                                "event": gripper_event,
                                "previous_width_m": float(previous_predicted_width),
                                "predicted_width_m": float(actions[frame_index, 9]),
                                "width_change_m": float(width_change),
                                "protocol": args.gripper_mode,
                            }
                        )
                        previous_predicted_width = float(actions[frame_index, 9])
                        gripper = 1.0 if gripper_command_open else 0.0
                    else:
                        gripper = 1.0 if actions[frame_index, 9] > args.gripper_open_width else 0.0
                    command = np.concatenate((matrix_to_pose7(target_world), [gripper]))
                else:
                    raw = raw_actions[frame_index]
                    if raw.shape != (8,) or not np.isfinite(raw).all():
                        continue
                    command = raw.copy()
                    command[7] = 1.0 if command[7] > 0.5 else 0.0

                if (
                    args.controller_profile == "pointact_eval"
                    and args.mode in {"eef0", "eef0_planning"}
                ):
                    from official_eval import (
                        clip_world_target_to_pointact_workspace,
                        execute_dataset_target_with_pointact_mover,
                    )

                    target_world, workspace_clipped, _raw_requested_xyz = (
                        clip_world_target_to_pointact_workspace(
                            target_world, args.clip_within_workspace
                        )
                    )
                    step_result = execute_dataset_target_with_pointact_mover(
                        task_env=task,
                        observation=observation,
                        command_world=target_world,
                        target_gripper_open=bool(gripper_command_open),
                        previous_gripper_open=previous_command_open,
                        was_limited=workspace_clipped,
                        args=args,
                        diagnostic_context={
                            "task_name": args.task,
                            "episode_index": int(args.episode),
                            "model_call": int(frame_index),
                            "chunk_row_index": int(frame_index),
                        },
                    )
                    mover_attempt_count += int(step_result.get("mover_attempts", 0))
                    mover_retry_count += int(step_result.get("mover_retries", 0))
                    mover_unreached_count += int(
                        not step_result.get("mover_reached", True)
                        and not step_result.get("success", False)
                        and not step_result.get("termination", False)
                    )
                    for mover_observation_index, mover_observation in enumerate(
                        step_result["observations"]
                    ):
                        observation = mover_observation
                        executed += 1
                        frames.append(
                            np.asarray(observation.front_rgb, dtype=np.uint8)
                        )
                        mover_attempt = min(
                            mover_observation_index + 1,
                            int(step_result.get("mover_attempts", 0)),
                        )
                        phase = (
                            "gripper_after_reach"
                            if mover_observation_index
                            >= int(step_result.get("mover_attempts", 0))
                            else "move"
                        )
                        execution_trace.append(
                            {
                                "video_frame": len(frames) - 1,
                                "action_index": int(frame_index),
                                "phase": phase,
                                "mover_attempt": int(mover_attempt),
                            }
                        )
                        actual_world = pose7_to_matrix(observation.gripper_pose)
                        position_errors.append(
                            float(
                                np.linalg.norm(
                                    actual_world[:3, 3] - target_world[:3, 3]
                                )
                            )
                        )
                        rotation_delta = (
                            target_world[:3, :3].T @ actual_world[:3, :3]
                        )
                        rotation_errors.append(
                            float(Rotation.from_matrix(rotation_delta).magnitude())
                        )
                    observation = step_result["observation"]
                    reward = 1.0 if step_result["success"] else 0.0
                    terminate = bool(step_result["termination"])
                    if (
                        not step_result.get("mover_reached", True)
                        and not step_result["success"]
                        and not step_result["termination"]
                    ):
                        # Formal eval would discard the stale action chunk and
                        # ask the model for a fresh chunk. Dataset replay has no
                        # model, so the next expert label is the closest
                        # teacher-forced replacement from the current state.
                        continue
                else:
                    observation, reward, terminate = task.step(
                        command.astype(np.float32)
                    )
                    executed += 1
                    frames.append(np.asarray(observation.front_rgb, dtype=np.uint8))
                    execution_trace.append(
                        {
                            "video_frame": len(frames) - 1,
                            "action_index": int(frame_index),
                            "phase": "single_step",
                            "mover_attempt": 1,
                        }
                    )
                    actual_world = pose7_to_matrix(observation.gripper_pose)
                    position_errors.append(
                        float(np.linalg.norm(actual_world[:3, 3] - target_world[:3, 3]))
                    )
                    rotation_delta = target_world[:3, :3].T @ actual_world[:3, :3]
                    rotation_errors.append(float(Rotation.from_matrix(rotation_delta).magnitude()))
                if float(reward) > 0.0 or bool(terminate):
                    ever_succeeded = ever_succeeded or bool(float(reward) > 0.0)
                    if (
                        args.controller_profile == "pointact_eval"
                        or not args.continue_after_action_error
                    ):
                        break
            except Exception as error:
                errors.append({"frame": frame_index, "error": repr(error)})
                if (
                    args.controller_profile != "pointact_eval"
                    and not args.continue_after_action_error
                ):
                    break
                # Preserve a strict one-attempt/one-video-frame mapping. A
                # failed planning call may have left the robot unchanged or
                # partially moved it, so capture the simulator state that
                # actually exists before attempting the next label.
                observation = task.get_observation()
                frames.append(np.asarray(observation.front_rgb, dtype=np.uint8))
                execution_trace.append(
                    {
                        "video_frame": len(frames) - 1,
                        "action_index": int(frame_index),
                        "phase": "controller_error_reinfer_or_next_label",
                        "mover_attempt": 0,
                        "error": repr(error),
                    }
                )
                controller_continue_errors += 1
                step_success, _ = task._task.success()
                ever_succeeded = ever_succeeded or bool(step_success)
                continue

        success = ever_succeeded or bool(float(reward) > 0.0)
        if not success:
            success, _ = task._task.success()
            success = bool(success)
        task_specific_diagnostics = collect_task_specific_diagnostics(task._task)
    finally:
        environment.shutdown()

    stamp = time.strftime("%Y%m%d_%H%M%S")
    run_name = args.task + "_episode_" + str(args.episode).zfill(3) + "_" + args.mode + "_" + stamp
    run_dir = args.output_root.expanduser().resolve() / run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    action_labels_path = run_dir / "action_labels.npy"
    np.save(action_labels_path, np.asarray(actions, dtype=np.float32))
    save_video = args.video_policy == "all" or (
        args.video_policy == "failures" and not success
    )
    video_path = run_dir / "replay.mp4" if save_video else None
    if video_path is not None:
        video_frames = annotate_replay_video_frames(
            frames=frames,
            execution_trace=execution_trace,
            actions=actions,
            task_name=args.task,
            episode_index=args.episode,
            mode=args.mode,
            success=success,
        )
        video_frames = annotate_final_task_result_frames(video_frames, success)
        imageio.mimsave(
            video_path,
            video_frames,
            fps=args.video_fps,
            macro_block_size=1,
        )
    result = {
        "task": args.task,
        "episode": args.episode,
        "global_episode": global_episode,
        "mode": args.mode,
        "action_source": args.action_source,
        "action_labels": str(action_labels_path),
        "action_labels_shape": list(np.asarray(actions).shape),
        "action_labels_dtype": str(np.asarray(actions, dtype=np.float32).dtype),
        "gripper_mode": args.gripper_mode,
        "gripper_protocol": (
            "one_time_physical_width_sync_then_first_executed_row_self_reference"
            if args.gripper_mode == DELTA_WIDTH_INITIAL_SYNC
            else (
                "measured_episode_anchor_then_continuous_prediction_carry"
                if args.gripper_mode != ABSOLUTE_WIDTH
                else "per_row_absolute_width_threshold"
            )
        ),
        "gripper_initial_sync_applied": bool(
            gripper_initial_sync_info is not None
        ),
        "gripper_initial_sync": gripper_initial_sync_info,
        "gripper_delta_threshold_m": float(args.gripper_delta_threshold),
        "gripper_delta_alignment": args.gripper_delta_alignment,
        "water_plant_collision": args.water_plant_collision,
        "water_drop_collision": args.water_drop_collision,
        "planner_max_time_ms": int(
            os.environ.get("RLBENCH_PLANNER_MAX_TIME_MS", "1000")
        ),
        "gripper_events": gripper_events,
        "gripper_transition_count": int(
            sum(item["event"] in {"delta_open", "delta_close"} for item in gripper_events)
        ),
        "observation_replay_start_index": (
            int(start_index) if args.action_source == "observation" else None
        ),
        "observation_replay_uses_action_labels": False
        if args.action_source == "observation"
        else None,
        "dataset_direct_mode": dataset_direct,
        "action_alignment": action_alignment,
        "fresh_reset_seed": args.seed,
        "success": success,
        "recorded_frames": int(len(actions)),
        "attempted_actions": attempted,
        "executed_actions": executed,
        "continue_after_action_error": bool(args.continue_after_action_error),
        "controller_profile": args.controller_profile,
        "mover_position_tolerance_m": float(args.mover_position_tolerance),
        "mover_rotation_tolerance_rad": float(args.mover_rotation_tolerance),
        "mover_gripper_position_tolerance_m": float(
            args.mover_gripper_position_tolerance
        ),
        "mover_gripper_rotation_tolerance_rad": float(
            args.mover_gripper_rotation_tolerance
        ),
        "pointact_pyrep_compat": bool(args.pointact_pyrep_compat),
        "clip_within_workspace": bool(args.clip_within_workspace),
        "mover_max_tries": int(args.mover_max_tries),
        "mover_attempts": int(mover_attempt_count),
        "mover_retries": int(mover_retry_count),
        "mover_unreached_targets": int(mover_unreached_count),
        "controller_continue_errors": int(controller_continue_errors),
        "gripper_after_reach": bool(args.gripper_after_reach),
        "execution_trace": execution_trace,
        "dataset_action_equals_state": bool(np.array_equal(actions, states)),
        "dataset_action_state_max_abs_difference": float(np.max(np.abs(actions - states))),
        "parquet_action_equals_observation_state": parquet_action_state_equal,
        "parquet_action_observation_state_max_abs_difference": (
            parquet_action_state_max_abs_difference
        ),
        "eef_tracking_position_error_median_m": (
            float(np.median(position_errors)) if position_errors else None
        ),
        "eef_tracking_position_error_max_m": (
            float(np.max(position_errors)) if position_errors else None
        ),
        "eef_tracking_rotation_error_median_rad": (
            float(np.median(rotation_errors)) if rotation_errors else None
        ),
        "eef_tracking_rotation_error_max_rad": (
            float(np.max(rotation_errors)) if rotation_errors else None
        ),
        "restored_or_reset_vs_recorded_first_rgb_mae": first_image_mae,
        "recorded_first_rgb_shape": list(recorded_first_image.shape),
        "replay_front_rgb_shape": list(reset_image.shape),
        "recorded_first_rgb_resized_for_mae": bool(
            recorded_first_image.shape != reset_image.shape
        ),
        "exact_recorded_scene_restored": exact_recorded_scene_restored,
        "scene_restore_method": scene_restore_method,
        "initial_task_state_source": initial_task_state_source,
        "initial_object_state_validation": object_state_validation,
        "initial_robot_state_validation": robot_state_validation,
        "task_specific_diagnostics": task_specific_diagnostics,
        "errors": errors,
        "video_policy": args.video_policy,
        "video": str(video_path) if video_path is not None else None,
        "video_frame_overlay": {
            "enabled": bool(video_path is not None),
            "schema": "rlbench_replay_frame_action_v1",
            "mapping_source": "execution_trace",
            "fields": [
                "task",
                "episode",
                "mode",
                "video_frame",
                "result",
                "dataset_frame",
                "action_index",
                "phase",
                "mover_attempt",
                "label_xyz",
                "label_gripper",
            ],
        },
        "language": descriptions[0] if descriptions else args.task.replace("_", " "),
    }
    with open(run_dir / "result.json", "w", encoding="utf-8") as file:
        json.dump(result, file, indent=2, ensure_ascii=False)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return success


def parse_episode_list(value):
    """Convert --episode text into a list so one command can replay many episodes."""
    episodes = []
    for item in str(value).split(","):
        item = item.strip()
        if not item:
            continue
        episodes.append(int(item))
    if not episodes:
        raise ValueError("--episode must contain at least one integer.")
    return episodes


def completed_result_path(output_root, task, episode, mode):
    """Return an existing result so parallel replay shards can resume safely."""
    pattern = (
        str(task)
        + "_episode_"
        + str(episode).zfill(3)
        + "_"
        + str(mode)
        + "_*/result.json"
    )
    matches = sorted(output_root.expanduser().resolve().glob(pattern))
    return matches[-1] if matches else None


def replay_episode_in_child_process(args, episode):
    """Run one episode in a fresh process so CoppeliaSim gets a fresh GL context."""
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--task", str(args.task),
        "--episode", str(episode),
        "--mode", str(args.mode),
        "--variation", str(args.variation),
        "--image-size", str(args.image_size),
        "--disable-point-cloud" if args.disable_point_cloud else "",
        "--artifact-root", str(args.artifact_root),
        "--dataset-root", str(args.dataset_root),
        "--action-source", str(args.action_source),
        "--output-root", str(args.output_root),
        "--video-fps", str(args.video_fps),
        "--video-policy", str(args.video_policy),
        "--continue-after-action-error" if args.continue_after_action_error else "",
        "--controller-profile", str(args.controller_profile),
        "--mover-max-tries", str(args.mover_max_tries),
        "--mover-position-tolerance", str(args.mover_position_tolerance),
        "--mover-rotation-tolerance", str(args.mover_rotation_tolerance),
        "--mover-gripper-position-tolerance", str(args.mover_gripper_position_tolerance),
        "--mover-gripper-rotation-tolerance", str(args.mover_gripper_rotation_tolerance),
        "--waypoint-position-tolerance", str(args.waypoint_position_tolerance),
        "--waypoint-rotation-tolerance", str(args.waypoint_rotation_tolerance),
        "--clip-within-workspace" if args.clip_within_workspace else "--no-clip-within-workspace",
        "--gripper-after-reach" if args.gripper_after_reach else "--no-gripper-after-reach",
        "--pointact-pyrep-compat" if args.pointact_pyrep_compat else "--no-pointact-pyrep-compat",
        "--log-control-details" if args.log_control_details else "--no-log-control-details",
        "--gripper-open-width", str(args.gripper_open_width),
        "--gripper-mode", str(args.gripper_mode),
        "--gripper-delta-threshold", str(args.gripper_delta_threshold),
        "--gripper-delta-alignment", str(args.gripper_delta_alignment),
        "--seed", str(args.seed),
        "--water-plant-collision", str(args.water_plant_collision),
        "--water-drop-collision", str(args.water_drop_collision),
    ]
    command = [item for item in command if item]
    if args.front_camera_position_m is not None:
        command.extend(
            ["--front-camera-position-m"]
            + [str(value) for value in args.front_camera_position_m]
        )
    if args.front_camera_look_at_m is not None:
        command.extend(
            ["--front-camera-look-at-m"]
            + [str(value) for value in args.front_camera_look_at_m]
        )
    if args.artifact_only_input:
        command.append("--artifact-only-input")
    if args.allow_fresh_reset:
        command.append("--allow-fresh-reset")
    print("[child-start] episode=" + str(episode), flush=True)
    completed = subprocess.run(command, check=False)
    if completed.returncode != 0:
        print(
            "[child-error] episode=" + str(episode)
            + " return_code=" + str(completed.returncode),
            flush=True,
        )
    # Let Qt/CoppeliaSim release the X11 connection before starting the next one.
    time.sleep(1.0)
    return completed.returncode == 0


def main():
    args = parse_args()
    if args.output_root is None:
        args.output_root = (
            RL_BENCH_ROOT
            / "eval"
            / "replays"
            / f"{RUN_DATE}__{args.task}__{args.mode}_dataset_action_replay"
        )
    if args.artifact_only_input and args.action_source != "artifact":
        raise ValueError(
            "--artifact-only-input requires --action-source artifact"
        )
    if (
        args.gripper_mode == DELTA_WIDTH_INITIAL_SYNC
        and args.gripper_delta_alignment != "current_minus_previous"
    ):
        raise ValueError(
            "delta_width_initial_sync requires --gripper-delta-alignment "
            "current_minus_previous."
        )
    episodes = parse_episode_list(args.episode)
    for episode in episodes:
        if episode < 0:
            raise ValueError("Every --episode value must be non-negative.")

    if len(episodes) > 1:
        failed_episodes = []
        for episode in episodes:
            if not replay_episode_in_child_process(args, episode):
                failed_episodes.append(episode)
        if failed_episodes:
            print("[failed-episodes] " + ",".join(str(item) for item in failed_episodes))
            raise SystemExit(1)
        return

    failed_episodes = []
    for episode in episodes:
        completed = completed_result_path(
            args.output_root, args.task, episode, args.mode
        )
        if completed is not None:
            print(
                "[episode-skip-complete] episode="
                + str(episode)
                + " result="
                + str(completed),
                flush=True,
            )
            continue
        args.episode = episode
        try:
            success = run_replay(args)
            if not success:
                failed_episodes.append(episode)
        except Exception as error:
            failed_episodes.append(episode)
            print(
                "[episode-error] episode=" + str(episode) + " error=" + repr(error),
                flush=True,
            )

    if failed_episodes:
        print("[failed-episodes] " + ",".join(str(item) for item in failed_episodes))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
