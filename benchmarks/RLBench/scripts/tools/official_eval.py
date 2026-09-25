#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Evaluate one trained Song/SmolVLA policy on one official RLBench task.

This is an online RLBench evaluation, not an offline action-error test:

1. RLBench resets the requested task once for every episode.
2. The policy receives the live front RGB image and front-camera point cloud,
   identity current-EEF state, and RLBench language description.
3. The policy predicts 10-D targets relative to the current model EEF at the
   start of that model call. For REAP this can be a virtual TCP shifted from
   Panda_tip when the selected gripper length differs from the calibrated
   0.09 m training length.
4. Each virtual-TCP target is converted back to a physical Panda_tip world
   target and executed through RLBench's official absolute EEF action mode.
5. RLBench's task.success() result is the episode result. Failed episodes are
   not reset or retried.

After the training UMI processor, the policy sees 10-D rows:
    [x, y, z, rotation_column_1(3), rotation_column_2(3), gripper_width]
where the first nine values are T_eef_t_target. The conversion used here is:
    T_world_target = T_world_eef_t_at_model_call @ T_eef_t_target
"""

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path

# Set this before importing the policy stack (and therefore torch/CUDA).  The
# evaluator exposes fixed Flow-Matching noise seeds, so allowing cuBLAS to pick
# non-deterministic kernels would still make identical observations produce
# millimetre-scale action differences across otherwise identical runs.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial.transform import Rotation


SCRIPT_DIR = Path(__file__).resolve().parent
RL_BENCH_ROOT = Path(
    os.environ.get("RLBENCH_ROOT", str(SCRIPT_DIR.parents[1]))
).expanduser().resolve()
REPO_ROOT = Path(
    os.environ.get("LEROBOT_ROOT", str(SCRIPT_DIR.parents[3]))
).expanduser().resolve()
SONG_SCRIPTS = Path(
    os.environ.get(
        "SONG_SCRIPTS",
        str(REPO_ROOT / "benchmarks" / "song_real_libero" / "scripts"),
    )
).expanduser().resolve()
LEROBOT_SRC = Path(
    os.environ.get("LEROBOT_SRC", str(REPO_ROOT / "src"))
).expanduser().resolve()
EXPECTED_RLBENCH_REAP_TCP_CALIBRATION_LEN_M = 0.09
sys.path.insert(0, str(LEROBOT_SRC))
sys.path.insert(0, str(RL_BENCH_ROOT))
sys.path.insert(0, str(SONG_SCRIPTS))

from collect_lerobot_pointcloud import (
    RLBENCH_SCENE_BOUNDS,
    make_observation_config,
    observation_cloud,
    pose7_to_pose9,
    task_class_from_name,
)
from worldflow_sidecars import (
    homo_to_pose9 as worldflow_homo_to_pose9,
    rlbench_panda_link0_to_world_matrix,
)
from gripper_control import (
    ABSOLUTE_WIDTH,
    DELTA_WIDTH_INITIAL_SYNC,
    GRIPPER_CONTROL_MODES,
    LEGACY_LIBERO_DELTA,
    absolute_width_gripper_target,
    initial_delta_reference_for_chunk,
    libero_style_gripper_target,
    recover_discrete_gripper_command_after_control_failure,
    resolve_pending_delta_gripper_target,
    set_gripper_absolute_width_position_target,
)
from video_utils import annotate_final_task_result_frames
from reap_gripper import (
    LIBERO_REAP_GRIPPER_LEN,
    LIBERO_REAP_OPENING_MAX_WIDTH,
    RLBENCH_REAP_ALIGNED_GRIPPER_LEN,
    canonical_reap_metadata,
    libero_reap_width_percent_from_physical,
    create_rlbench_reap_points_from_physical_width,
)
from libero_setting.libero_pointcloud_utils import (
    RLBENCH_MINIMAL_TWO_FINGER_TEMPLATE,
    RLBENCH_PANDA_GRIPPER_TEMPLATE,
    RLBENCH_PANDA_MAX_WIDTH,
    add_world_gripper_cloud_to_point_cloud,
    create_rlbench_minimal_two_finger_points,
    create_rlbench_panda_gripper_points,
    reference_point_cloud_to_current_eff,
    rlbench_panda_gripper_local_boxes,
    pose9_to_homo_np,
    sample_or_repeat_points,
)
from smolvla_model_inference import SmolVLA_ModelInference
from lerobot.policies.smolvla.inference_diagnostics import foreground_score_colors
from dataset_action_replay import (
    compare_initial_object_states,
    restore_recorded_first_robot_state,
)


def array_sha256(value):
    """Hash an array together with its dtype and shape for exact comparisons."""
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("utf-8"))
    digest.update(b"\0")
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("utf-8"))
    digest.update(b"\0")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def configure_torch_determinism(enabled):
    """Make seeded policy inference repeatable across evaluator processes."""
    if not bool(enabled):
        return {
            "enabled": False,
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        }

    import torch

    # LitePT contains CUDA operations for which PyTorch can only warn about a
    # missing deterministic implementation. Keep deterministic alternatives
    # enabled everywhere else without aborting a full benchmark run.
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    return {
        "enabled": True,
        "warn_only": True,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "deterministic_algorithms": bool(
            torch.are_deterministic_algorithms_enabled()
        ),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        "matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "float32_matmul_precision": str(torch.get_float32_matmul_precision()),
    }


def object_pose7_or_none(obj):
    if obj is None:
        return None
    try:
        return np.asarray(obj.get_pose(), dtype=np.float64).tolist()
    except Exception:
        return None


def finite_json_value(value):
    """Convert NumPy values and replace non-finite diagnostics with null."""
    if isinstance(value, dict):
        return {str(key): finite_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite_json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return finite_json_value(value.tolist())
    if isinstance(value, np.generic):
        return finite_json_value(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


DEFAULT_POLICY = (
    Path(os.environ.get("EVAL_POLICY_PATH", str(RL_BENCH_ROOT / "outputs" / "policy")))
)
DEFAULT_OUTPUT = Path(
    os.environ.get("EVAL_OUTPUT_DIR", str(RL_BENCH_ROOT / "eval"))
)
CONTROL_LOG_PATH = None

# PointACT's released RLBench evaluator clips absolute EEF targets to these
# bounds before sending them to the planner.  Keep them separate from the
# point-cloud crop bounds: this box is an action-controller constraint.
POINTACT_WORKSPACE_MIN = np.asarray([-0.274, -0.655, 0.752], dtype=np.float32)
POINTACT_WORKSPACE_MAX = np.asarray([0.774, 0.655, 1.751], dtype=np.float32)

EXECUTION_PHASE_CODES = {
    0: "move",
    1: "gripper_after_reach",
    2: "adaptive_controller",
    3: "gripper_after_control_failure",
}


def rgb_to_uint8(rgb):
    """Normalize RLBench RGB observations without truncating [0, 1] floats."""
    image = np.asarray(rgb)
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError("Expected an HxWx3 RGB observation, got " + str(image.shape))
    if np.issubdtype(image.dtype, np.floating):
        image = np.nan_to_num(image, nan=0.0, posinf=255.0, neginf=0.0)
        if image.size and float(np.max(image)) <= 1.0 + 1e-6:
            image = image * 255.0
    return np.clip(image, 0.0, 255.0).astype(np.uint8)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run one official RLBench task with a trained Song/SmolVLA policy."
    )
    parser.add_argument("--task", required=True, help="RLBench task name, for example water_plants.")
    parser.add_argument("--policy-path", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument(
        "--simulation-timestep",
        type=float,
        default=0.0,
        help="Optional CoppeliaSim simulation timestep in seconds; 0 keeps the scene default.",
    )
    parser.add_argument(
        "--planner-max-time-ms",
        type=int,
        default=None,
        help=(
            "Per-task RLBench path-planner time budget in milliseconds. "
            "When omitted, use RLBENCH_PLANNER_MAX_TIME_MS from the launcher."
        ),
    )
    parser.add_argument("--variation", type=int, default=0)
    parser.add_argument("--max-model-calls", type=int, default=50)
    parser.add_argument(
        "--model-call-compensation-min-executed-rows",
        type=int,
        default=3,
        help=(
            "Compensate one model call when a predicted chunk successfully "
            "executes fewer than this many policy rows; 0 disables this condition."
        ),
    )
    parser.add_argument(
        "--model-call-compensation-consecutive-planning-failures",
        type=int,
        default=1,
        help=(
            "Compensate one model call after this many consecutive model calls "
            "encounter a planning/controller failure; the default 1 treats one "
            "exhausted planner/controller attempt sequence as a failure. Set 0 "
            "to disable this condition."
        ),
    )
    parser.add_argument(
        "--max-model-call-compensations",
        type=int,
        default=-1,
        help=(
            "Maximum extra model calls granted per episode; -1 uses the original "
            "--max-model-calls value and 0 disables all compensation."
        ),
    )
    parser.add_argument(
        "--max-policy-action-steps",
        type=int,
        default=0,
        help=(
            "Maximum predicted policy rows attempted in one episode across all "
            "model calls; 0 disables this independent budget. The last chunk is "
            "truncated so the budget is never exceeded."
        ),
    )
    parser.add_argument(
        "--controller-error-retries",
        type=int,
        default=-1,
        help="Additional attempts after a controller rejection; -1 retries until the retry timeout.",
    )
    parser.add_argument(
        "--controller-error-retry-timeout-seconds",
        type=float,
        default=300.0,
        help="Wall-clock budget per episode for controller-error retries.",
    )
    parser.add_argument(
        "--controller-error-mode",
        choices=["retry_episode", "continue_episode"],
        default="retry_episode",
        help=(
            "retry_episode resets the episode after a controller error; "
            "continue_episode discards the remaining action-chunk rows and "
            "calls the policy again from the current state."
        ),
    )
    parser.add_argument(
        "--controller-continue-max-errors",
        type=int,
        default=0,
        help="Maximum controller errors skipped in one episode; 0 means unlimited in continue_episode mode.",
    )
    parser.add_argument("--action-index", type=int, default=0)
    parser.add_argument("--exec-action-steps", type=int, default=8)
    parser.add_argument(
        "--execution-mode",
        choices=["adaptive", "dataset_step", "bounded_step"],
        default="adaptive",
        help=(
            "adaptive uses midpoint/hold control; dataset_step sends the complete target once; "
            "bounded_step sends one bounded target per predicted waypoint without extra holds."
        ),
    )
    parser.add_argument(
        "--arm-action-mode",
        choices=[
            "ik", "planning", "linear_planning", "joint_velocity_waypoint", "franka_ik_servo",
            "franka_cartesian_servo",
        ],
        default="ik",
        help=(
            "RLBench end-effector controller: ik uses Jacobian IK; planning uses "
            "RLBench EndEffectorPoseViaPlanning; linear_planning prefers a "
            "straight Cartesian IK path and only accepts a bounded non-linear "
            "fallback that passes joint-detour limits; "
            "joint_velocity_waypoint solves "
            "each EEF target to a joint waypoint and tracks it with joint velocity; "
            "franka_ik_servo uses Pinocchio Panda URDF IK followed by simulator joint-position servo."
            " franka_cartesian_servo uses a damped Pinocchio Jacobian velocity servo and does not require exact IK."
        ),
    )
    parser.add_argument(
        "--clip-within-workspace",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Clip every predicted world XYZ target to the PointACT RLBench "
            "workspace before controller execution."
        ),
    )
    parser.add_argument(
        "--mover-max-tries",
        type=int,
        default=None,
        help=(
            "Maximum executions of the same absolute target when its measured "
            "pose remains outside the Mover tolerances. When omitted, the SONG, "
            "legacy PointACT, and raw profiles use 2, 10, and 1 respectively. "
            "An explicit value overrides the selected profile."
        ),
    )
    parser.add_argument(
        "--mover-position-tolerance",
        type=float,
        default=0.05,
        help="Position tolerance in metres for a normal PointACT Mover target.",
    )
    parser.add_argument(
        "--mover-rotation-tolerance",
        type=float,
        default=0.0,
        help=(
            "Rotation tolerance in radians for a normal PointACT Mover target. "
            "The target is reached only when both position and rotation pass. "
            "0 disables the rotation gate for historical-result compatibility."
        ),
    )
    parser.add_argument(
        "--mover-gripper-position-tolerance",
        type=float,
        default=0.02,
        help=(
            "Position tolerance in metres before a deferred gripper state "
            "change is allowed."
        ),
    )
    parser.add_argument(
        "--mover-gripper-rotation-tolerance",
        type=float,
        default=0.0,
        help=(
            "Rotation tolerance in radians before a deferred gripper state "
            "change is allowed. Position and rotation must both pass. 0 "
            "disables the rotation gate for historical-result compatibility."
        ),
    )
    parser.add_argument(
        "--reinfer-on-control-error",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "When a PointACT Mover target is still unreached after all tries, "
            "discard the rest of the current action chunk and call the policy "
            "again from the latest observation. Controller exceptions also "
            "re-infer immediately when --controller-error-mode=continue_episode."
        ),
    )
    parser.add_argument(
        "--continue-chunk-on-mover-unreached",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "When task_env.step() returns normally but the measured EEF pose is "
            "still outside the Mover tolerance, keep executing the remaining "
            "rows of the current policy chunk instead of discarding them and "
            "re-inferring. Deferred gripper transitions remain pending until a "
            "later waypoint is reached. Controller/planner exceptions still "
            "follow --controller-error-mode."
        ),
    )
    parser.add_argument(
        "--gripper-after-reach",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Move with the previous gripper state, then issue the requested "
            "gripper state only after reaching the target tolerance."
        ),
    )
    parser.add_argument(
        "--gripper-close-require-reach",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Require the EEF target tolerance before executing an open-to-close "
            "gripper transition. When omitted, inherit --gripper-after-reach."
        ),
    )
    parser.add_argument(
        "--gripper-open-require-reach",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Require the EEF target tolerance before executing a close-to-open "
            "gripper transition. Disable this for non-blocking release after "
            "the arm controller has executed the requested motion. When "
            "omitted, inherit --gripper-after-reach."
        ),
    )
    parser.add_argument(
        "--pointact-pyrep-compat",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Apply PointACT's two legacy PyRep trajectory-interpolation changes "
            "at runtime without modifying the installed PyRep package."
        ),
    )
    collision_group = parser.add_mutually_exclusive_group()
    collision_group.add_argument(
        "--collision-checking",
        dest="collision_checking",
        action="store_true",
        help="Enable RLBench planner collision checking.",
    )
    collision_group.add_argument(
        "--no-collision-checking",
        dest="collision_checking",
        action="store_false",
        help="Use the official RLBench planning behavior without scene collision checking.",
    )
    # RLBench tasks such as closing, pushing, and pouring may require the
    # planned end-effector path to touch task geometry. Keep the official
    # action-mode default; collision-free planning is an explicit diagnostic.
    parser.set_defaults(collision_checking=False)
    parser.add_argument("--num-points", type=int, default=10000)
    parser.add_argument("--gripper-points", type=int, default=500)
    parser.add_argument(
        "--gripper-len",
        "--virtual-gripper-len",
        dest="gripper_len",
        type=float,
        default=LIBERO_REAP_GRIPPER_LEN,
        help=(
            "Selected REAP virtual-gripper length in metres. The calibrated "
            "training value is 0.09 m. With virtual-TCP synchronization enabled, "
            "changing this value shifts the model EEF and is inverted before "
            "sending a physical Panda_tip target to RLBench."
        ),
    )
    parser.add_argument(
        "--sync-virtual-gripper-tcp",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Treat the selected REAP fingertip midpoint as the model EEF. Relative "
            "to the aligned 0.09 m contract, the model TCP shifts by "
            "0.09-gripper_len along Panda_tip local +Z. Disable only to reproduce "
            "the legacy behavior that moved gripper points without moving state."
        ),
    )
    parser.add_argument(
        "--add-gripper-cloud",
        dest="add_gripper_cloud",
        action="store_true",
        help="Merge virtual gripper points into the model input point cloud.",
    )
    parser.add_argument(
        "--no-add-gripper-cloud",
        dest="add_gripper_cloud",
        action="store_false",
        help="Keep the model input as scene points only; gripper points are not inserted.",
    )
    parser.set_defaults(add_gripper_cloud=True)
    parser.add_argument(
        "--gripper-template",
        choices=["reap", RLBENCH_PANDA_GRIPPER_TEMPLATE, RLBENCH_MINIMAL_TWO_FINGER_TEMPLATE],
        default="reap",
        help=(
            "Virtual gripper point-cloud geometry. reap is the canonical RLBench-aligned REAP v4 template; "
            "rlbench_panda keeps fingers and palm; rlbench_minimal_two_finger keeps only "
            "the two Panda-sized finger bars."
        ),
    )
    parser.add_argument(
        "--gripper-delta-threshold",
        type=float,
        default=0.0025,
        help=(
            "LIBERO-style physical width-change threshold in metres. "
            "A larger predicted width opens, a smaller width closes, and a "
            "change inside the deadband keeps the measured RLBench state."
        ),
    )
    parser.add_argument(
        "--gripper-delta-open-threshold",
        type=float,
        default=None,
        help=(
            "Optional opening-only physical width-change threshold in metres. "
            "When omitted, --gripper-delta-threshold is used."
        ),
    )
    parser.add_argument(
        "--gripper-delta-close-threshold",
        type=float,
        default=None,
        help=(
            "Optional closing-only physical width-change threshold in metres. "
            "When omitted, --gripper-delta-threshold is used."
        ),
    )
    parser.add_argument(
        "--gripper-delta-alignment",
        choices=["current_minus_previous", "next_minus_current"],
        default="current_minus_previous",
        help=(
            "LIBERO delta convention: use current_width-previous_width, or "
            "the legacy next_width-current_width alignment."
        ),
    )
    parser.add_argument(
        "--gripper-mode",
        choices=GRIPPER_CONTROL_MODES,
        default=DELTA_WIDTH_INITIAL_SYNC,
        help=(
            "delta_width_initial_sync physically synchronizes action[9] once at "
            "episode start and decodes deltas only within each newly predicted "
            "chunk; libero_delta preserves the historical measured/cross-chunk "
            "decoder; absolute_width thresholds every row independently."
        ),
    )
    parser.add_argument(
        "--gripper-open-threshold",
        type=float,
        default=RLBENCH_PANDA_MAX_WIDTH / 2.0,
        help=(
            "Absolute-width mode opens when predicted width is strictly above "
            "this value in metres. RLBench labels use 0.00 m=closed and "
            "0.08 m=open, so the default is the 0.04 m midpoint."
        ),
    )
    parser.add_argument(
        "--close-laptop-contact-z-offset-m",
        type=float,
        default=0.0,
        help=(
            "Diagnostic close_laptop_lid-only world-Z offset applied to the "
            "first open-to-closed gripper target. This changes the policy target "
            "and is disabled by default; use it only to test a suspected vertical "
            "contact bias."
        ),
    )
    parser.add_argument(
        "--close-laptop-contact-seek-z-offset-m",
        type=float,
        default=0.0,
        help=(
            "Diagnostic close_laptop_lid contact-seeking correction. After the "
            "first close command, add this world-Z offset to every closed target "
            "until the lid joint has moved by --close-laptop-contact-established-rad. "
            "Disabled by default and reported as controller assistance."
        ),
    )
    parser.add_argument(
        "--close-laptop-contact-established-rad",
        type=float,
        default=0.05,
        help="Lid displacement at which diagnostic contact-seeking assistance stops.",
    )
    parser.add_argument(
        "--gripper-lock-after-close",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "After the first closed gripper command in an episode, force all "
            "later gripper commands to remain closed."
        ),
    )
    parser.add_argument(
        "--max-eef-position-step",
        type=float,
        default=0.0,
        help="Optional hard cap for one model waypoint; 0 keeps the complete predicted target.",
    )
    parser.add_argument(
        "--max-eef-rotation-step",
        type=float,
        default=0.0,
        help="Optional hard rotation cap for one model waypoint; 0 keeps the complete predicted target.",
    )
    parser.add_argument("--jacobian-position-threshold", type=float, default=0.008)
    parser.add_argument("--jacobian-rotation-threshold", type=float, default=0.05)
    parser.add_argument("--jacobian-midpoint-max-depth", type=int, default=4)
    parser.add_argument("--waypoint-position-tolerance", type=float, default=0.002)
    parser.add_argument("--waypoint-rotation-tolerance", type=float, default=0.03)
    parser.add_argument("--waypoint-max-control-steps", type=int, default=64)
    parser.add_argument(
        "--linear-planning-max-joint-goal-distance",
        type=float,
        default=2.2,
        help=(
            "Maximum L2 distance in radians from the measured joints to a "
            "linear_planning path goal; 0 disables this limit."
        ),
    )
    parser.add_argument(
        "--linear-planning-max-joint-path-length",
        type=float,
        default=3.5,
        help=(
            "Maximum cumulative L2 joint path length in radians for "
            "linear_planning; 0 disables this limit."
        ),
    )
    parser.add_argument(
        "--linear-planning-max-joint-path-ratio",
        type=float,
        default=2.0,
        help=(
            "Maximum cumulative path length divided by direct joint-goal "
            "distance; 0 disables this limit."
        ),
    )
    parser.add_argument(
        "--linear-planning-max-single-joint-travel",
        type=float,
        default=1.5,
        help=(
            "Maximum cumulative travel of any one joint in radians for "
            "linear_planning; 0 disables this limit."
        ),
    )
    parser.add_argument(
        "--joint-velocity-kp",
        type=float,
        default=2.0,
        help="Joint-velocity waypoint proportional gain in rad/s per rad.",
    )
    parser.add_argument(
        "--joint-velocity-max-speed",
        type=float,
        default=1.0,
        help="Per-joint velocity limit in rad/s for joint_velocity_waypoint.",
    )
    parser.add_argument(
        "--joint-velocity-joint-tolerance",
        type=float,
        default=0.01,
        help="Joint waypoint tolerance in radians for joint_velocity_waypoint.",
    )
    parser.add_argument(
        "--joint-velocity-stall-steps",
        type=int,
        default=8,
        help="Stop a waypoint after this many physics steps with negligible joint motion.",
    )
    parser.add_argument("--franka-ik-max-iterations", type=int, default=300)
    parser.add_argument("--franka-ik-tolerance", type=float, default=0.01)
    parser.add_argument("--franka-ik-damping", type=float, default=1e-4)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument(
        "--seed",
        type=int,
        default=100,
        help=(
            "Base seed for RLBench test-scene initialization. Episode i uses "
            "seed + i for ordinary resets; the default 100 keeps evaluation "
            "placements separate from the seed-0 training collection."
        ),
    )
    parser.add_argument(
        "--model-noise-seed",
        type=int,
        default=None,
        help="Fix Flow Matching noise per episode/model call; unset keeps stochastic sampling.",
    )
    parser.add_argument(
        "--deterministic-torch",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Use deterministic PyTorch/cuDNN/cuBLAS inference settings. Enabled "
            "by default so a fixed observation and --model-noise-seed reproduce "
            "the same action chunk."
        ),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--water-plant-collision",
        choices=["disabled", "enabled"],
        default="enabled",
        help=(
            "Whether the water_plants plant object participates in collision. "
            "Defaults to the official RLBench/training-collection behavior. "
            "This affects physical blocking only; success is still judged by sensors."
        ),
    )
    parser.add_argument(
        "--water-drop-collision",
        choices=["disabled", "original"],
        default="original",
        help=(
            "Reproduce the official RLBench/training-collection drop physics by "
            "default; disabled is retained only as an explicit diagnostic ablation."
        ),
    )
    parser.add_argument(
        "--pointseg-device",
        default=None,
        help="Optional separate device for CUDA-only LitePT/PointSeg, for example cuda:0.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help=(
            "Use this exact directory for the task result. Intended for a multi-task "
            "evaluation run; without it, create task_timestamp under --output-dir."
        ),
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help=(
            "Use the recorded RLBench reset state from this LeRobot dataset "
            "instead of task.reset() random placement."
        ),
    )
    parser.add_argument(
        "--dataset-episodes",
        default=None,
        help=(
            "Comma-separated task-local dataset episode indices, for example "
            "0,1,2,3. If omitted, use 0..episodes-1. Requires --dataset-root."
        ),
    )
    parser.add_argument(
        "--episode-indices",
        default=None,
        help=(
            "Comma-separated sparse random-reset episode indices, for example "
            "4,7,11. This preserves the same reset, point-sampling, and model-noise "
            "streams as those episode ids in a full 0..N-1 evaluation. Cannot be "
            "combined with --dataset-root or --dataset-episodes."
        ),
    )
    parser.add_argument(
        "--action-replay-parity",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Force the controller, gripper, observation, task-physics, and "
            "recorded-reset settings to match dataset_action_replay.py. "
            "Action chunks then come from the policy and are refreshed at "
            "policy.n_action_steps. Simulator robustness workarounds are controlled "
            "independently by --simulator-robustness-optimizations."
        ),
    )
    parser.add_argument(
        "--simulator-robustness-optimizations",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Atomically enable the retained PointACT/RLBench simulator robustness "
            "workarounds: action-workspace clipping, up to 10 sends of an unreached "
            "target, changing the gripper only after reaching the target, the legacy "
            "PyRep trajectory compatibility patch, and policy re-inference after a "
            "controller exception or unreached target. Disable only for strict raw "
            "controller diagnostics. This switch does not change model-input point "
            "counts, virtual-gripper geometry, or the gripper threshold."
        ),
    )
    parser.add_argument(
        "--simulator-robustness-optimizations-song",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Use the SONG RLBench execution profile. This preserves the "
            "training action domain (no PointACT action-workspace clipping), "
            "retries each normally-returning target up to 10 times, continues "
            "past failed rows instead of discarding the rest of the chunk, "
            "executes a requested gripper transition even when arm control is "
            "blocked, re-infers from the latest observation after the chunk, "
            "and never restarts an episode. This profile takes precedence over "
            "--simulator-robustness-optimizations."
        ),
    )
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument(
        "--failure-artifacts-only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Retain action visualizations only for failed episodes. Videos are "
            "controlled independently: when --save-video is set, both successful "
            "and failed episodes are encoded and retained."
        ),
    )
    parser.add_argument(
        "--save-action-records",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Save per-episode executed actions, predicted model chunks, and "
            "executed-action alignment arrays. Disable for lightweight success-rate "
            "evaluation that must not persist model or controller actions."
        ),
    )
    parser.add_argument(
        "--draw-pour-point",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Overlay the water_plants pour-point detection volume on saved front-camera "
            "video frames. Defaults to enabled for water_plants."
        ),
    )
    parser.add_argument(
        "--draw-task-stages",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Overlay water_plants stage goals, sensor states, and drop counts "
            "on saved video frames. Defaults to enabled for water_plants."
        ),
    )
    parser.add_argument(
        "--draw-phone-success-sensor",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Overlay the phone_on_base success-sensor detection volume, phone "
            "position, and simulator condition state on saved video frames. "
            "Defaults to enabled for phone_on_base."
        ),
    )
    parser.add_argument("--video-fps", type=int, default=20)
    parser.add_argument("--video-width", type=int, default=512)
    parser.add_argument("--video-height", type=int, default=512)
    parser.add_argument(
        "--video-refresh-rgb",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Before saving each video frame, explicitly refresh the front RGB "
            "camera. Disabled by default because the explicit sensor render can "
            "change the next policy observation on legacy CoppeliaSim."
        ),
    )
    parser.add_argument("--log-control-details", action="store_true")
    parser.add_argument(
        "--save-control-log",
        action="store_true",
        help="Save detailed control JSON to control.log without printing it to the terminal.",
    )
    parser.add_argument(
        "--save-determinism-diagnostics",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Save exact reset/model-input arrays and hashes, the first predicted "
            "action chunk and hash, and simulator state after every successful "
            "task_env.step() for paired nondeterminism diagnosis."
        ),
    )
    parser.add_argument("--visualize-foreground", action="store_true")
    parser.add_argument(
        "--save-action-visualizations",
        action="store_true",
        help=(
            "Save an EEF-frame PLY containing LitePT foreground-probability colors and the full "
            "future action chunk, plus a front-RGB action overlay, at the configured frame interval."
        ),
    )
    parser.add_argument(
        "--save-frame-pointclouds",
        action="store_true",
        help=(
            "Save the exact raw model-input point cloud for reset frame 0 and selected "
            "subsequent video/control frames as compact binary PLY files under "
            "frame_pointclouds/episode_XXX/. These per-frame PLY files do not contain "
            "PointSeg probabilities because the policy is not called at every frame."
        ),
    )
    parser.add_argument(
        "--frame-pointcloud-every-n-frames",
        type=int,
        default=2,
        help=(
            "When --save-frame-pointclouds is enabled, save frames whose zero-based "
            "video frame index is divisible by this interval. Use 2 to save frames "
            "0,2,4,... while keeping every frame in the MP4."
        ),
    )
    parser.add_argument(
        "--save-action-chunks",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Save every postprocessed model action chunk as an individual .npy "
            "file under action_chunks/episode_XXX/. Enabled by default."
        ),
    )
    parser.add_argument("--action-vis-every-n-frames", type=int, default=32)
    parser.add_argument("--action-vis-max-points", type=int, default=50000)
    parser.add_argument("--action-vis-image-width", type=int, default=768)
    parser.add_argument(
        "--action-vis-point-mode",
        choices=["full", "prob"],
        default="prob",
        help=(
            "full keeps the complete scene cloud in its original RGB colors; prob colors the same "
            "scene points with the LitePT operation_prob heat map. Both modes include the future action chunk."
        ),
    )
    return parser.parse_args()


DATASET_TASK_DESCRIPTIONS = {
    "close_box": "close box",
    "close_fridge": "close fridge",
    "close_laptop_lid": "close laptop lid",
    "phone_on_base": "put the phone on the base",
    "stack_wine": "stack wine bottle",
    "sweep_to_dustpan": "sweep dirt to dustpan",
    "take_frame_off_hanger": "take frame off hanger",
    "take_umbrella_out_of_umbrella_stand": (
        "take umbrella out of umbrella stand"
    ),
    "toilet_seat_down": "toilet seat down",
    "water_plants": "water plant",
}


def parse_dataset_episode_indices(value):
    """Parse a stable, shell-friendly list of local dataset episode indices."""
    if value is None:
        return None
    tokens = str(value).replace(" ", "").split(",")
    if not tokens or any(token == "" for token in tokens):
        raise ValueError("--dataset-episodes must be comma-separated integers.")
    try:
        indices = [int(token) for token in tokens]
    except ValueError as exc:
        raise ValueError(
            "--dataset-episodes must be comma-separated integers, for example 0,1,2."
        ) from exc
    if any(index < 0 for index in indices):
        raise ValueError("--dataset-episodes cannot contain negative indices.")
    if len(set(indices)) != len(indices):
        raise ValueError("--dataset-episodes cannot contain duplicates.")
    return indices


def _normalized_dataset_task_name(value):
    return " ".join(str(value).strip().lower().replace("_", " ").split())


def load_dataset_reset_specs(dataset_root, task_name, requested_indices, episode_count):
    """Resolve task-local episodes and load the exact RLBench reset state."""
    import pyarrow.dataset as pyarrow_dataset

    root = dataset_root.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError("Dataset root does not exist: " + str(root))
    episodes_root = root / "meta" / "episodes"
    if not episodes_root.is_dir():
        raise FileNotFoundError("Dataset episode metadata does not exist: " + str(episodes_root))

    episode_table = pyarrow_dataset.dataset(
        str(episodes_root), format="parquet"
    ).to_table(columns=["episode_index", "tasks"])
    expected_name = _normalized_dataset_task_name(
        DATASET_TASK_DESCRIPTIONS.get(task_name, task_name)
    )
    matching_global_episodes = []
    for global_episode, task_values in zip(
        episode_table["episode_index"].to_pylist(),
        episode_table["tasks"].to_pylist(),
    ):
        recorded_name = task_values[0] if task_values else ""
        if _normalized_dataset_task_name(recorded_name) == expected_name:
            matching_global_episodes.append(int(global_episode))
    matching_global_episodes.sort()
    if not matching_global_episodes:
        raise ValueError(
            "No dataset episodes for task "
            + task_name
            + " in "
            + str(root)
        )

    data_table = pyarrow_dataset.dataset(
        str(root / "data"), format="parquet"
    ).to_table(
        filter=pyarrow_dataset.field("frame_index") == 0,
        columns=["episode_index", "observation.state"],
    )
    first_state_by_episode = {
        int(global_episode): np.asarray(state, dtype=np.float32)
        for global_episode, state in zip(
            data_table["episode_index"].to_pylist(),
            data_table["observation.state"].to_pylist(),
        )
    }

    local_indices = parse_dataset_episode_indices(requested_indices)
    if local_indices is None:
        local_indices = list(range(int(episode_count)))
    if not local_indices:
        raise ValueError("At least one dataset episode is required.")
    specs = []
    for local_index in local_indices:
        if local_index >= len(matching_global_episodes):
            raise IndexError(
                task_name
                + " dataset episode "
                + str(local_index)
                + " is out of range; dataset contains "
                + str(len(matching_global_episodes))
                + " episodes for this task."
            )
        global_episode = matching_global_episodes[local_index]
        state_path = root / "initial_task_states" / (
            "episode_" + str(global_episode).zfill(6) + ".npz"
        )
        if not state_path.is_file():
            raise FileNotFoundError(
                "Recorded initial task state does not exist: " + str(state_path)
            )
        with np.load(state_path, allow_pickle=False) as state_file:
            required = {
                "configuration_bytes",
                "object_count",
                "demo_random_seed_state",
                "demo_random_seed_position",
                "demo_random_seed_has_gauss",
                "demo_random_seed_cached_gaussian",
                "demo_num_reset_attempts",
            }
            missing = sorted(required.difference(state_file.files))
            if missing:
                raise ValueError(
                    "Initial task state is incomplete: "
                    + str(state_path)
                    + " missing "
                    + ", ".join(missing)
                )
            random_state = (
                "MT19937",
                np.asarray(state_file["demo_random_seed_state"], dtype=np.uint32),
                int(state_file["demo_random_seed_position"]),
                int(state_file["demo_random_seed_has_gauss"]),
                float(state_file["demo_random_seed_cached_gaussian"]),
            )
            initial_task_state = (
                np.asarray(state_file["configuration_bytes"], dtype=np.uint8).tobytes(),
                int(state_file["object_count"]),
            )
            reset_attempts = int(state_file["demo_num_reset_attempts"])
        first_state = first_state_by_episode.get(global_episode)
        if first_state is None or first_state.shape != (10,):
            raise RuntimeError(
                "Dataset first observation.state is missing or invalid for episode "
                + str(global_episode)
            )
        pose_path = root / "world_ee_poses" / (
            "episode_" + str(global_episode).zfill(6) + ".npy"
        )
        if not pose_path.is_file():
            raise FileNotFoundError(
                "Recorded first world EEF trajectory does not exist: " + str(pose_path)
            )
        recorded_world_poses = np.asarray(np.load(pose_path), dtype=np.float32)
        if (
            recorded_world_poses.ndim != 2
            or recorded_world_poses.shape[1] != 9
            or len(recorded_world_poses) == 0
        ):
            raise RuntimeError(
                "Expected a non-empty (T, 9) world EEF trajectory, got "
                + str(recorded_world_poses.shape)
            )
        object_state_path = root / "initial_object_states" / (
            "episode_" + str(global_episode).zfill(6) + ".npz"
        )
        specs.append(
            {
                "local_episode_index": int(local_index),
                "global_episode_index": int(global_episode),
                "state_path": str(state_path),
                "random_state": random_state,
                "initial_task_state": initial_task_state,
                "reset_attempts": reset_attempts,
                "recorded_first_pose9": recorded_world_poses[0],
                "recorded_first_gripper_width_m": float(first_state[9]),
                "recorded_world_pose_path": str(pose_path),
                "initial_object_state_path": str(object_state_path),
            }
        )
    return specs


def reset_task_for_episode(task_env, args, episode_index, dataset_reset_spec):
    """Reset randomly or reproduce one recorded dataset initial state."""
    seed_episode_index = (
        int(episode_index)
        if dataset_reset_spec is None
        else int(dataset_reset_spec["local_episode_index"])
    )
    np.random.seed(args.seed + seed_episode_index)
    if dataset_reset_spec is None:
        descriptions, observation = task_env.reset()
        return descriptions, observation, None, None

    np.random.set_state(dataset_reset_spec["random_state"])

    class RecordedDemoReset:
        num_reset_attempts = dataset_reset_spec["reset_attempts"]

    descriptions, observation = task_env.reset(RecordedDemoReset())
    object_validation = compare_initial_object_states(
        task_env._task, Path(dataset_reset_spec["initial_object_state_path"])
    )
    observation, robot_validation = restore_recorded_first_robot_state(
        task_env,
        dataset_reset_spec["recorded_first_pose9"],
        dataset_reset_spec["recorded_first_gripper_width_m"],
    )
    # Restoring a mismatched discrete gripper state advances physics until the
    # fingers reach the recorded state.  Validate the task objects again after
    # that operation; the earlier check alone can hide reset-time object drift.
    object_validation_after_robot_restore = compare_initial_object_states(
        task_env._task, Path(dataset_reset_spec["initial_object_state_path"])
    )
    robot_validation["recorded_pose_source"] = dataset_reset_spec[
        "recorded_world_pose_path"
    ]
    validation = {
        "initial_object_state": object_validation,
        "initial_object_state_after_robot_restore": (
            object_validation_after_robot_restore
        ),
        "initial_robot_state": robot_validation,
    }
    return descriptions, observation, "official_demo_random_state", validation


def world_target_to_rlbench_action(world_target, gripper_open):
    """Make RLBench's [xyz, qx, qy, qz, qw, discrete_gripper] action."""
    matrix = np.asarray(world_target, dtype=np.float32)
    quaternion = Rotation.from_matrix(matrix[:3, :3]).as_quat().astype(np.float32)
    discrete_gripper = 1.0 if bool(gripper_open) else 0.0
    return np.concatenate((matrix[:3, 3], quaternion, [discrete_gripper])).astype(np.float32)


def observed_gripper_width(observation):
    """Map RLBench's continuous opening fraction to dataset width in metres."""
    return float(np.clip(observation.gripper_open, 0.0, 1.0)) * RLBENCH_PANDA_MAX_WIDTH


def matrix_to_pose7(matrix):
    """Convert a homogeneous matrix to RLBench [xyz, qx, qy, qz, qw]."""
    matrix = np.asarray(matrix, dtype=np.float32)
    quaternion = Rotation.from_matrix(matrix[:3, :3]).as_quat().astype(np.float32)
    return np.concatenate((matrix[:3, 3], quaternion)).astype(np.float32)


def round_log_floats(value):
    """Round every floating-point value in nested log data to two decimals."""
    if isinstance(value, (float, np.floating)):
        return round(float(value), 2)
    if isinstance(value, dict):
        return {key: round_log_floats(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [round_log_floats(item) for item in value]
    return value


def print_control_log(label, values, enabled):
    """Print one machine-readable JSON line for control debugging."""
    if not enabled and CONTROL_LOG_PATH is None:
        return
    line = label + " " + json.dumps(values, separators=(",", ":"))
    if enabled:
        print(line, flush=True)
    if CONTROL_LOG_PATH is not None:
        saved_values = round_log_floats(values)
        saved_line = label + " " + json.dumps(saved_values, separators=(",", ":"))
        with open(CONTROL_LOG_PATH, "a", encoding="utf-8") as file:
            file.write(saved_line + "\n")


def collect_close_laptop_diagnostics(task_env, task_name):
    """Read simulator-side lid and gripper contact state for close_laptop_lid."""
    if task_name != "close_laptop_lid":
        return {}

    diagnostics = {}
    try:
        from pyrep.const import ObjectType
        from pyrep.objects.joint import Joint

        laptop_joint = Joint("joint")
        joint_position = float(laptop_joint.get_joint_position())
        diagnostics["laptop_joint_position_rad"] = joint_position

        task = getattr(task_env, "_task", None)
        success_conditions = getattr(task, "_success_conditions", [])
        if success_conditions:
            condition = success_conditions[0]
            original_position = getattr(condition, "_original_pos", None)
            threshold = getattr(condition, "_pos", None)
            if original_position is not None:
                diagnostics["laptop_joint_displacement_rad"] = abs(
                    joint_position - float(original_position)
                )
            if threshold is not None:
                diagnostics["laptop_success_threshold_rad"] = float(threshold)
        diagnostics["laptop_success_condition_met"] = bool(
            task_env._task.success()[0]
        )

        robot = getattr(task_env, "_robot", None)
        gripper = getattr(robot, "gripper", None)
        if gripper is None:
            return diagnostics
        diagnostics["gripper_open_amount"] = [
            float(value) for value in gripper.get_open_amount()
        ]
        diagnostics["gripper_joint_positions"] = [
            float(value) for value in gripper.get_joint_positions()
        ]
        diagnostics["gripper_grasped_objects"] = [
            str(obj.get_name()) for obj in gripper.get_grasped_objects()
        ]
        try:
            diagnostics["gripper_touch_sensor_forces"] = [
                [float(force) for force in forces]
                for forces in gripper.get_touch_sensor_forces()
            ]
        except Exception:
            diagnostics["gripper_touch_sensor_forces"] = None

        laptop_shapes = laptop_joint.get_objects_in_tree(
            object_type=ObjectType.SHAPE,
            exclude_base=False,
        )
        gripper_shapes = gripper.get_objects_in_tree(
            object_type=ObjectType.SHAPE,
            exclude_base=False,
        )
        diagnostics["laptop_shape_names"] = [
            str(shape.get_name()) for shape in laptop_shapes
        ]
        diagnostics["gripper_shape_names"] = [
            str(shape.get_name()) for shape in gripper_shapes
        ]

        pairs = []
        min_distance = None
        min_pair = None
        for gripper_shape in gripper_shapes:
            for laptop_shape in laptop_shapes:
                pair = {
                    "gripper_shape": str(gripper_shape.get_name()),
                    "laptop_shape": str(laptop_shape.get_name()),
                    "collision": bool(gripper_shape.check_collision(laptop_shape)),
                }
                try:
                    distance = float(gripper_shape.check_distance(laptop_shape))
                    if np.isfinite(distance) and distance >= 0.0:
                        pair["distance_m"] = distance
                        if min_distance is None or distance < min_distance:
                            min_distance = distance
                            min_pair = (
                                pair["gripper_shape"]
                                + "->"
                                + pair["laptop_shape"]
                            )
                except Exception:
                    pair["distance_m"] = None
                if pair["collision"]:
                    pairs.append(pair)
        diagnostics["gripper_laptop_collision_pairs"] = pairs
        diagnostics["gripper_laptop_collision"] = bool(pairs)
        diagnostics["gripper_laptop_min_distance_m"] = min_distance
        diagnostics["gripper_laptop_min_distance_pair"] = min_pair
    except Exception as error:
        diagnostics["diagnostic_error"] = repr(error)
    return diagnostics


def print_close_laptop_diagnostics(task_env, task_name, context, enabled):
    """Append simulator-side contact state without changing the action."""
    diagnostics = collect_close_laptop_diagnostics(task_env, task_name)
    if not diagnostics:
        return
    payload = dict(context)
    payload.update(diagnostics)
    print_control_log("[sim-diagnostic]", payload, enabled)


def limit_absolute_eef_target(current_world, requested_world, max_position_step, max_rotation_step):
    """Bound one absolute EEF command around the actually observed EEF pose."""
    current_world = np.asarray(current_world, dtype=np.float32)
    requested_world = np.asarray(requested_world, dtype=np.float32)
    limited_world = requested_world.copy()
    limited = False

    position_delta = requested_world[:3, 3] - current_world[:3, 3]
    position_distance = float(np.linalg.norm(position_delta))
    if max_position_step > 0.0 and position_distance > max_position_step:
        limited_world[:3, 3] = current_world[:3, 3] + position_delta * (
            max_position_step / position_distance
        )
        limited = True

    current_rotation = Rotation.from_matrix(current_world[:3, :3])
    requested_rotation = Rotation.from_matrix(requested_world[:3, :3])
    rotation_delta = (requested_rotation * current_rotation.inv()).as_rotvec()
    rotation_distance = float(np.linalg.norm(rotation_delta))
    if max_rotation_step > 0.0 and rotation_distance > max_rotation_step:
        limited_rotation = Rotation.from_rotvec(
            rotation_delta * (max_rotation_step / rotation_distance)
        ) * current_rotation
        limited_world[:3, :3] = limited_rotation.as_matrix().astype(np.float32)
        limited = True
    return limited_world, limited


def rotation_error_radians(first_rotation, second_rotation):
    first = Rotation.from_matrix(np.asarray(first_rotation, dtype=np.float64))
    second = Rotation.from_matrix(np.asarray(second_rotation, dtype=np.float64))
    return float(np.linalg.norm((second * first.inv()).as_rotvec()))


def interpolate_absolute_pose(start_world, end_world, fraction):
    """Interpolate translation and rotation between two absolute EEF poses."""
    fraction = float(np.clip(fraction, 0.0, 1.0))
    start_world = np.asarray(start_world, dtype=np.float32)
    end_world = np.asarray(end_world, dtype=np.float32)
    result = np.eye(4, dtype=np.float32)
    result[:3, 3] = (
        start_world[:3, 3]
        + fraction * (end_world[:3, 3] - start_world[:3, 3])
    )
    start_rotation = Rotation.from_matrix(start_world[:3, :3])
    end_rotation = Rotation.from_matrix(end_world[:3, :3])
    rotation_delta = (end_rotation * start_rotation.inv()).as_rotvec()
    result[:3, :3] = (
        Rotation.from_rotvec(rotation_delta * fraction) * start_rotation
    ).as_matrix().astype(np.float32)
    return result


def pose_error(current_world, target_world):
    """Return translation and rotation error from the measured EEF to a target."""
    position_error = float(
        np.linalg.norm(target_world[:3, 3] - current_world[:3, 3])
    )
    rotation_error = rotation_error_radians(
        current_world[:3, :3], target_world[:3, :3]
    )
    return position_error, rotation_error


def clip_world_target_to_pointact_workspace(target_world, enabled):
    """Clip only XYZ to the action workspace used by PointACT's evaluator."""
    clipped_world = np.asarray(target_world, dtype=np.float32).copy()
    requested_xyz = clipped_world[:3, 3].copy()
    if enabled:
        clipped_world[:3, 3] = np.clip(
            requested_xyz, POINTACT_WORKSPACE_MIN, POINTACT_WORKSPACE_MAX
        )
    clipped = bool(not np.array_equal(requested_xyz, clipped_world[:3, 3]))
    return clipped_world, clipped, requested_xyz


def apply_pointact_pyrep_compatibility_patch():
    """Install PointACT's legacy PyRep path interpolation changes in-process.

    This intentionally patches only the evaluator process.  It does not edit
    the conda environment and only changes the legacy RML path implementation.
    """
    from pyrep.backend import sim
    from pyrep.robots.configuration_paths.arm_configuration_path import (
        ArmConfigurationPath,
    )

    if getattr(ArmConfigurationPath, "_pointact_compat_applied", False):
        return {
            "requested": True,
            "applied": True,
            "implementation": "legacy_rml_runtime_monkeypatch",
            "already_applied": True,
        }
    if not hasattr(ArmConfigurationPath, "_get_rml_handle") or not hasattr(
        ArmConfigurationPath, "_step_motion"
    ):
        return {
            "requested": True,
            "applied": False,
            "implementation": "not_applicable_non_rml_configuration_path",
            "already_applied": False,
        }

    def pointact_get_rml_handle(self):
        dt = sim.simGetSimulationTimeStep()
        limits = np.asarray(self._arm.get_joint_upper_velocity_limits())
        velocity_correction = 1.0
        max_velocity = self._arm.max_velocity
        max_acceleration = self._arm.max_acceleration
        max_jerk = self._arm.max_jerk
        lengths = self._get_path_point_lengths()
        target_position_velocity = [lengths[-1], 0]
        previous_configuration = self._path_points[0 : len(self._arm.joints)]

        # PointACT's source uses an unbounded correction loop here. Certain
        # close_box paths can keep increasing the velocity correction forever,
        # leaving task_env.step() unable to return. Bound both loops so the
        # evaluator can treat the path as a controller error and re-infer.
        # One direct ratio correction normally converges on the following
        # pass. Four passes leave margin without letting a bad path burn CPU
        # for minutes before the evaluator can re-infer.
        maximum_correction_passes = 4
        maximum_rml_steps_per_pass = 10000
        for correction_pass in range(maximum_correction_passes):
            position_velocity_acceleration = [0, 0, 0]
            maximum_ratio = 0
            rml_handle = sim.simRMLPos(
                1,
                0.0001,
                -1,
                position_velocity_acceleration,
                [
                    max_velocity * velocity_correction,
                    max_acceleration,
                    max_jerk,
                ],
                [1],
                target_position_velocity,
            )
            state = 0
            rml_steps = 0
            while state == 0:
                rml_steps += 1
                if rml_steps > maximum_rml_steps_per_pass:
                    sim.simRMLRemove(rml_handle)
                    raise RuntimeError(
                        "PointACT RML preview exceeded "
                        + str(maximum_rml_steps_per_pass)
                        + " steps in correction pass "
                        + str(correction_pass + 1)
                        + "."
                    )
                state, position_velocity_acceleration = sim.simRMLStep(
                    rml_handle, dt, 1
                )
                if state >= 0:
                    position = position_velocity_acceleration[0]
                    for index in range(len(lengths) - 1):
                        if lengths[index] <= position <= lengths[index + 1]:
                            fraction = (position - lengths[index]) / np.maximum(
                                lengths[index + 1] - lengths[index], 1e-10
                            )
                            offset = len(self._arm.joints) * index
                            first = self._path_points[
                                offset : offset + self._num_joints
                            ]
                            offset = len(self._arm.joints) * (index + 1)
                            second = self._path_points[
                                offset : offset + self._num_joints
                            ]
                            configuration = first + (second - first) * fraction
                            delta = configuration - previous_configuration
                            previous_configuration = configuration
                            ratio = np.abs(delta / dt) / limits
                            maximum_ratio = max(maximum_ratio, float(np.max(ratio)))
                            break
            sim.simRMLRemove(rml_handle)
            if maximum_ratio > 1.001:
                velocity_correction = velocity_correction / maximum_ratio
            else:
                break
        else:
            raise RuntimeError(
                "PointACT RML velocity correction did not converge after "
                + str(maximum_correction_passes)
                + " passes."
            )
        return sim.simRMLPos(
            1,
            0.0001,
            -1,
            [0, 0, 0],
            [
                max_velocity * velocity_correction,
                max_acceleration,
                max_jerk,
            ],
            [1],
            target_position_velocity,
        )

    def pointact_step_motion(self):
        self._joint_position_action = None
        dt = sim.simGetSimulationTimeStep()
        lengths = self._get_path_point_lengths()
        state, position_velocity_acceleration = sim.simRMLStep(
            self._rml_handle, dt, 1
        )
        if state >= 0:
            position = position_velocity_acceleration[0]
            for index in range(len(lengths) - 1):
                # PointACT deliberately does not force an overshooting RML
                # position into the final path segment.
                if lengths[index] <= position <= lengths[index + 1]:
                    fraction = (position - lengths[index]) / np.maximum(
                        lengths[index + 1] - lengths[index], 1e-10
                    )
                    offset = len(self._arm.joints) * index
                    first = self._path_points[
                        offset : offset + len(self._arm.joints)
                    ]
                    offset = self._arm._num_joints * (index + 1)
                    second = self._path_points[
                        offset : offset + len(self._arm.joints)
                    ]
                    configuration = first + (second - first) * fraction
                    self._joint_position_action = configuration
                    self._arm.set_joint_target_positions(configuration)
                    break
        if state == 1:
            sim.simRMLRemove(self._rml_handle)
        return state

    ArmConfigurationPath._get_rml_handle = pointact_get_rml_handle
    ArmConfigurationPath._step_motion = pointact_step_motion
    ArmConfigurationPath._pointact_compat_applied = True
    return {
        "requested": True,
        "applied": True,
        "implementation": "legacy_rml_runtime_monkeypatch",
        "already_applied": False,
        "changes": [
            "zero_length_segment_denominator_guard",
            "no_forced_final_segment_on_rml_overshoot",
            "bounded_rml_velocity_correction_search",
        ],
    }


def execute_dataset_target_with_pointact_mover(
    task_env,
    observation,
    command_world,
    target_gripper_open,
    previous_gripper_open,
    was_limited,
    args,
    diagnostic_context,
    execution_recorder=None,
):
    """Execute one chunk row with PointACT's same-target Mover semantics.

    This function never calls the policy.  Retry attempts repeat the exact
    same world target, so the caller's action-chunk/replanning cadence remains
    unchanged.
    """
    target_gripper_open = bool(target_gripper_open)
    previous_gripper_open = bool(previous_gripper_open)
    gripper_change = target_gripper_open != previous_gripper_open
    transition_requires_reach = bool(
        args.gripper_open_require_reach
        if target_gripper_open
        else args.gripper_close_require_reach
    )
    defer_gripper = bool(gripper_change and transition_requires_reach)
    movement_gripper_open = (
        previous_gripper_open if defer_gripper else target_gripper_open
    )
    mover_position_tolerance = float(
        args.mover_gripper_position_tolerance
        if defer_gripper
        else args.mover_position_tolerance
    )
    mover_rotation_tolerance = float(
        args.mover_gripper_rotation_tolerance
        if defer_gripper
        else args.mover_rotation_tolerance
    )
    observations = []
    actions = []
    current_observation = observation
    success = False
    termination = False
    mover_reached = False
    final_position_error = float("inf")
    final_rotation_error = float("inf")

    for attempt_index in range(int(args.mover_max_tries)):
        command = world_target_to_rlbench_action(
            command_world, movement_gripper_open
        )
        print_control_log(
            "[control-before]",
            {
                **diagnostic_context,
                "pointact_mover_phase": "move",
                "pointact_mover_attempt": int(attempt_index + 1),
                "pointact_mover_max_tries": int(args.mover_max_tries),
                "pointact_mover_tolerance_m": mover_position_tolerance,
                "pointact_mover_position_tolerance_m": (
                    mover_position_tolerance
                ),
                "pointact_mover_rotation_tolerance_rad": (
                    mover_rotation_tolerance
                ),
                "pointact_mover_rotation_gate_enabled": bool(
                    mover_rotation_tolerance > 0.0
                ),
                "actual_state_world_pose7": np.asarray(
                    current_observation.gripper_pose, dtype=np.float32
                ).tolist(),
                "actual_gripper_open": float(current_observation.gripper_open),
                "command_gripper_discrete": int(command[-1] > 0.5),
                "actual_joint_positions": np.asarray(
                    getattr(current_observation, "joint_positions", []),
                    dtype=np.float32,
                ).tolist(),
                "command_action_world8": command.tolist(),
            },
            args.log_control_details,
        )
        next_observation, reward, termination_value = task_env.step(command)
        if execution_recorder is not None:
            execution_recorder(
                command, 0, int(attempt_index + 1), next_observation
            )
        actions.append(command)
        observations.append(next_observation)
        current_observation = next_observation
        success = bool(float(reward) > 0.0)
        termination = bool(termination_value)
        actual_world = pose9_to_homo_np(
            pose7_to_pose9(current_observation.gripper_pose)
        )
        final_position_error, final_rotation_error = pose_error(
            actual_world, command_world
        )
        mover_position_reached = bool(
            final_position_error < mover_position_tolerance
        )
        mover_rotation_reached = bool(
            mover_rotation_tolerance <= 0.0
            or final_rotation_error < mover_rotation_tolerance
        )
        mover_reached = bool(
            mover_position_reached and mover_rotation_reached
        )
        print_control_log(
            "[control-after]",
            {
                **diagnostic_context,
                "pointact_mover_phase": "move",
                "pointact_mover_attempt": int(attempt_index + 1),
                "actual_state_world_pose7": np.asarray(
                    current_observation.gripper_pose, dtype=np.float32
                ).tolist(),
                "actual_gripper_open": float(current_observation.gripper_open),
                "command_gripper_discrete": int(command[-1] > 0.5),
                "actual_joint_positions": np.asarray(
                    getattr(current_observation, "joint_positions", []),
                    dtype=np.float32,
                ).tolist(),
                "position_error_m": float(final_position_error),
                "rotation_error_rad": float(final_rotation_error),
                "pointact_mover_position_reached": bool(
                    mover_position_reached
                ),
                "pointact_mover_rotation_reached": bool(
                    mover_rotation_reached
                ),
                "pointact_mover_reached": bool(mover_reached),
                "reward": float(reward),
                "termination": bool(termination),
            },
            args.log_control_details,
        )
        print_close_laptop_diagnostics(
            task_env,
            diagnostic_context.get("task_name", ""),
            {**diagnostic_context, "command_action_world8": command.tolist()},
            args.log_control_details,
        )
        if success or termination or mover_reached:
            break

    gripper_action_executed = False
    if defer_gripper and mover_reached and not success and not termination:
        command = world_target_to_rlbench_action(
            command_world, target_gripper_open
        )
        print_control_log(
            "[control-before]",
            {
                **diagnostic_context,
                "pointact_mover_phase": "gripper_after_reach",
                "actual_state_world_pose7": np.asarray(
                    current_observation.gripper_pose, dtype=np.float32
                ).tolist(),
                "actual_gripper_open": float(current_observation.gripper_open),
                "command_gripper_discrete": int(command[-1] > 0.5),
                "command_action_world8": command.tolist(),
            },
            args.log_control_details,
        )
        next_observation, reward, termination_value = task_env.step(command)
        if execution_recorder is not None:
            execution_recorder(command, 1, 0, next_observation)
        actions.append(command)
        observations.append(next_observation)
        current_observation = next_observation
        success = bool(float(reward) > 0.0)
        termination = bool(termination_value)
        gripper_action_executed = True
        actual_world = pose9_to_homo_np(
            pose7_to_pose9(current_observation.gripper_pose)
        )
        final_position_error, final_rotation_error = pose_error(
            actual_world, command_world
        )
        print_control_log(
            "[control-after]",
            {
                **diagnostic_context,
                "pointact_mover_phase": "gripper_after_reach",
                "actual_state_world_pose7": np.asarray(
                    current_observation.gripper_pose, dtype=np.float32
                ).tolist(),
                "actual_gripper_open": float(current_observation.gripper_open),
                "command_gripper_discrete": int(command[-1] > 0.5),
                "position_error_m": float(final_position_error),
                "rotation_error_rad": float(final_rotation_error),
                "reward": float(reward),
                "termination": bool(termination),
            },
            args.log_control_details,
        )
        print_close_laptop_diagnostics(
            task_env,
            diagnostic_context.get("task_name", ""),
            {**diagnostic_context, "command_action_world8": command.tolist()},
            args.log_control_details,
        )

    measured_gripper_open = float(current_observation.gripper_open) > 0.5
    gripper_reached = measured_gripper_open == target_gripper_open
    print_control_log(
        "[waypoint-error]",
        {
            **diagnostic_context,
            "position_error_m": float(final_position_error),
            "rotation_error_rad": float(final_rotation_error),
            "gripper_reached": bool(gripper_reached),
            "target_was_limited": bool(was_limited),
            "pointact_mover_reached": bool(mover_reached),
            "pointact_mover_attempts": int(
                len(actions) - int(gripper_action_executed)
            ),
            "gripper_after_reach_executed": bool(gripper_action_executed),
        },
        args.log_control_details,
    )
    return {
        "was_limited": bool(was_limited),
        "recursive_midpoints": 0,
        "threshold_midpoints": 0,
        "jacobian_intermediate_actions": 0,
        "reached": bool(
            final_position_error <= args.waypoint_position_tolerance
            and final_rotation_error <= args.waypoint_rotation_tolerance
            and gripper_reached
        ),
        "actions": actions,
        "observations": observations,
        "observation": current_observation,
        "success": bool(success),
        "termination": bool(termination),
        "ok": True,
        "mover_attempts": int(len(actions) - int(gripper_action_executed)),
        "mover_retries": int(
            max(len(actions) - int(gripper_action_executed) - 1, 0)
        ),
        "mover_reached": bool(mover_reached),
        "final_position_error": float(final_position_error),
        "final_rotation_error": float(final_rotation_error),
        "error": (
            None
            if mover_reached or success or termination
            else "pointact_mover_unreached"
        ),
        "gripper_after_reach_executed": bool(gripper_action_executed),
    }


def execute_gripper_only_after_control_failure(
    task_env,
    observation,
    target_gripper_open,
    diagnostic_context,
    execution_recorder=None,
    log_control_details=False,
):
    """Execute the gripper half of an action without invoking arm planning.

    RLBench's normal ``MoveArmThenGripper`` action runs the arm controller
    first.  A planning exception therefore prevents the gripper half from
    running at all.  SONG uses this fallback only after arm control has either
    raised or exhausted all Mover attempts, and only for a decoded gripper
    transition.  The current measured EEF pose is recorded as the controller
    pose because no new arm target is sent by this fallback.
    """
    target_gripper_open = bool(target_gripper_open)
    current_world = pose9_to_homo_np(pose7_to_pose9(observation.gripper_pose))
    command = world_target_to_rlbench_action(
        current_world, target_gripper_open
    )
    print_control_log(
        "[control-before]",
        {
            **diagnostic_context,
            "pointact_mover_phase": "gripper_after_control_failure",
            "actual_state_world_pose7": np.asarray(
                observation.gripper_pose, dtype=np.float32
            ).tolist(),
            "actual_gripper_open": float(observation.gripper_open),
            "command_gripper_discrete": int(target_gripper_open),
            "command_action_world8": command.tolist(),
        },
        log_control_details,
    )
    action_mode = getattr(task_env, "_action_mode", None)
    gripper_action_mode = getattr(action_mode, "gripper_action_mode", None)
    scene = getattr(task_env, "_scene", None)
    if gripper_action_mode is None or scene is None:
        raise RuntimeError(
            "SONG gripper fallback could not resolve the RLBench gripper action mode."
        )
    gripper_action_mode.action(
        scene,
        np.asarray([1.0 if target_gripper_open else 0.0], dtype=np.float32),
    )
    next_observation = task_env.get_observation()
    success_value, termination_value = task_env._task.success()
    if execution_recorder is not None:
        execution_recorder(command, 3, 0, next_observation)
    print_control_log(
        "[control-after]",
        {
            **diagnostic_context,
            "pointact_mover_phase": "gripper_after_control_failure",
            "actual_state_world_pose7": np.asarray(
                next_observation.gripper_pose, dtype=np.float32
            ).tolist(),
            "actual_gripper_open": float(next_observation.gripper_open),
            "command_gripper_discrete": int(target_gripper_open),
            "reward": float(bool(success_value)),
            "termination": bool(termination_value),
        },
        log_control_details,
    )
    return {
        "was_limited": False,
        "recursive_midpoints": 0,
        "threshold_midpoints": 0,
        "jacobian_intermediate_actions": 0,
        "reached": False,
        "actions": [command],
        "observations": [next_observation],
        "observation": next_observation,
        "success": bool(success_value),
        "termination": bool(termination_value),
        "ok": True,
        "error": None,
        "gripper_after_reach_executed": False,
        "gripper_after_control_failure_executed": True,
    }


def nearest_safe_midpoint(current_world, target_world, args):
    """Repeatedly bisect a target until one Jacobian command is sufficiently small."""
    safe_target = np.asarray(target_world, dtype=np.float32).copy()
    midpoint_count = 0
    while True:
        position_error, rotation_error = pose_error(current_world, safe_target)
        position_too_large = (
            args.jacobian_position_threshold > 0.0
            and position_error > args.jacobian_position_threshold
        )
        rotation_too_large = (
            args.jacobian_rotation_threshold > 0.0
            and rotation_error > args.jacobian_rotation_threshold
        )
        if not position_too_large and not rotation_too_large:
            return safe_target, midpoint_count
        safe_target = interpolate_absolute_pose(current_world, safe_target, 0.5)
        midpoint_count += 1
        if midpoint_count >= 32:
            raise RuntimeError("Could not reduce a Jacobian target after 32 midpoint insertions.")


def execute_jacobian_segment(
    task_env,
    observation,
    target_world,
    gripper_open,
    args,
    depth=0,
    diagnostic_context=None,
    execution_recorder=None,
):
    """Execute one target; recursively insert a midpoint if Jacobian IK fails."""
    from rlbench.backend.exceptions import InvalidActionError

    action = world_target_to_rlbench_action(target_world, gripper_open)
    print_control_log(
        "[control-before]",
        {
            "recursion_depth": int(depth),
            "actual_state_world_pose7": np.asarray(
                observation.gripper_pose, dtype=np.float32
            ).tolist(),
            "actual_gripper_open": float(observation.gripper_open),
            "command_action_world8": action.tolist(),
        },
        args.log_control_details,
    )
    try:
        next_observation, reward, termination = task_env.step(action)
        if execution_recorder is not None:
            execution_recorder(action, 2, 0, next_observation)
        print_control_log(
            "[control-after]",
            {
                "recursion_depth": int(depth),
                "actual_state_world_pose7": np.asarray(
                    next_observation.gripper_pose, dtype=np.float32
                ).tolist(),
                "actual_gripper_open": float(next_observation.gripper_open),
                "reward": float(reward),
                "termination": bool(termination),
            },
            args.log_control_details,
        )
        if diagnostic_context is not None:
            print_close_laptop_diagnostics(
                task_env,
                args.task,
                diagnostic_context,
                args.log_control_details,
            )
        return {
            "ok": True,
            "observation": next_observation,
            "observations": [next_observation],
            "actions": [action],
            "success": bool(reward > 0.5),
            "termination": bool(termination),
            "recursive_midpoints": 0,
            "error": None,
        }
    except InvalidActionError as error:
        print_control_log(
            "[control-ik-failed]",
            {
                "recursion_depth": int(depth),
                "actual_state_world_pose7": np.asarray(
                    observation.gripper_pose, dtype=np.float32
                ).tolist(),
                "failed_action_world8": action.tolist(),
                "error": repr(error),
            },
            args.log_control_details,
        )
        if depth >= int(args.jacobian_midpoint_max_depth):
            return {
                "ok": False,
                "observation": observation,
                "observations": [],
                "actions": [],
                "success": False,
                "termination": False,
                "recursive_midpoints": 0,
                "error": repr(error),
            }

    current_world = pose9_to_homo_np(pose7_to_pose9(observation.gripper_pose))
    midpoint_world = interpolate_absolute_pose(current_world, target_world, 0.5)
    first_half = execute_jacobian_segment(
        task_env,
        observation,
        midpoint_world,
        gripper_open,
        args,
        depth=depth + 1,
        diagnostic_context=diagnostic_context,
        execution_recorder=execution_recorder,
    )
    first_half["recursive_midpoints"] += 1
    if not first_half["ok"] or first_half["success"] or first_half["termination"]:
        return first_half

    second_half = execute_jacobian_segment(
        task_env,
        first_half["observation"],
        target_world,
        gripper_open,
        args,
        depth=depth + 1,
        diagnostic_context=diagnostic_context,
        execution_recorder=execution_recorder,
    )
    return {
        "ok": bool(second_half["ok"]),
        "observation": second_half["observation"],
        "observations": first_half["observations"] + second_half["observations"],
        "actions": first_half["actions"] + second_half["actions"],
        "success": bool(first_half["success"] or second_half["success"]),
        "termination": bool(first_half["termination"] or second_half["termination"]),
        "recursive_midpoints": int(
            first_half["recursive_midpoints"] + second_half["recursive_midpoints"]
        ),
        "error": second_half["error"],
    }


def execute_absolute_target_with_midpoints(
    task_env,
    observation,
    requested_world,
    gripper_open,
    args,
    diagnostic_context=None,
    execution_recorder=None,
):
    """Track one model waypoint to tolerance before allowing the next waypoint."""
    current_world = pose9_to_homo_np(pose7_to_pose9(observation.gripper_pose))
    requested_position_distance, requested_rotation_distance = pose_error(
        current_world, requested_world
    )
    final_world, was_limited = limit_absolute_eef_target(
        current_world,
        requested_world,
        args.max_eef_position_step,
        args.max_eef_rotation_step,
    )

    result = {
        "ok": True,
        "observation": observation,
        "observations": [],
        "actions": [],
        "success": False,
        "termination": False,
        "recursive_midpoints": 0,
        "threshold_midpoints": 0,
        "segments": 0,
        "was_limited": bool(was_limited),
        "reached": False,
        "requested_position_distance": requested_position_distance,
        "requested_rotation_distance": requested_rotation_distance,
        "final_position_error": requested_position_distance,
        "final_rotation_error": requested_rotation_distance,
        "error": None,
    }
    desired_gripper_open = bool(gripper_open)
    while len(result["actions"]) < int(args.waypoint_max_control_steps):
        current_world = pose9_to_homo_np(
            pose7_to_pose9(result["observation"].gripper_pose)
        )
        position_error, rotation_error = pose_error(current_world, final_world)
        result["final_position_error"] = position_error
        result["final_rotation_error"] = rotation_error
        pose_reached = (
            position_error <= args.waypoint_position_tolerance
            and rotation_error <= args.waypoint_rotation_tolerance
        )
        current_gripper_open = float(result["observation"].gripper_open) > 0.5
        gripper_reached = current_gripper_open == desired_gripper_open
        print_control_log(
            "[waypoint-error]",
            {
                "control_actions": int(len(result["actions"])),
                "actual_state_world_pose7": np.asarray(
                    result["observation"].gripper_pose, dtype=np.float32
                ).tolist(),
                "target_state_world_pose7": matrix_to_pose7(final_world).tolist(),
                "position_error_m": float(position_error),
                "rotation_error_rad": float(rotation_error),
                "actual_gripper_open": bool(current_gripper_open),
                "target_gripper_open": bool(desired_gripper_open),
            },
            args.log_control_details,
        )
        if pose_reached and gripper_reached:
            result["reached"] = True
            return result

        target_world, midpoint_count = nearest_safe_midpoint(
            current_world, final_world, args
        )
        result["threshold_midpoints"] += midpoint_count
        segment_result = execute_jacobian_segment(
            task_env,
            result["observation"],
            target_world,
            desired_gripper_open,
            args,
            diagnostic_context=diagnostic_context,
            execution_recorder=execution_recorder,
        )
        result["observation"] = segment_result["observation"]
        result["observations"].extend(segment_result["observations"])
        result["actions"].extend(segment_result["actions"])
        result["recursive_midpoints"] += segment_result["recursive_midpoints"]
        result["segments"] += 1
        result["success"] = bool(result["success"] or segment_result["success"])
        result["termination"] = bool(
            result["termination"] or segment_result["termination"]
        )
        if not segment_result["ok"]:
            result["ok"] = False
            result["error"] = segment_result["error"]
            return result
        if result["success"] or result["termination"]:
            return result

    current_world = pose9_to_homo_np(
        pose7_to_pose9(result["observation"].gripper_pose)
    )
    position_error, rotation_error = pose_error(current_world, final_world)
    result["final_position_error"] = position_error
    result["final_rotation_error"] = rotation_error
    result["ok"] = False
    result["error"] = (
        "Waypoint did not reach tolerance after "
        + str(args.waypoint_max_control_steps)
        + " control actions: position_error="
        + str(position_error)
        + " rotation_error="
        + str(rotation_error)
    )
    return result


def physical_eef_to_model_tcp(args):
    """Return ``T_physical_eef_model_tcp`` for the selected REAP length.

    The collected 0.09 m REAP geometry defines the identity calibration: its
    fingertip midpoint coincides with Panda_tip.  Shortening the selected
    virtual gripper moves that midpoint forward along local +Z; lengthening it
    moves the midpoint backward.  Non-REAP and legacy-unsynchronised modes keep
    the physical and model EEF frames identical.
    """

    transform = np.eye(4, dtype=np.float64)
    if (
        str(args.gripper_template) == "reap"
        and bool(args.sync_virtual_gripper_tcp)
    ):
        transform[2, 3] = float(RLBENCH_REAP_ALIGNED_GRIPPER_LEN) - float(
            args.gripper_len
        )
    return transform


def physical_eef_world_to_model_tcp_world(physical_eef_world, args):
    """Map a physical Panda_tip world pose to the model's virtual-TCP pose."""

    return np.asarray(physical_eef_world, dtype=np.float64) @ physical_eef_to_model_tcp(
        args
    )


def model_tcp_world_to_physical_eef_world(model_tcp_world, args):
    """Map a model virtual-TCP world target back to a Panda_tip controller target."""

    return np.asarray(model_tcp_world, dtype=np.float64) @ np.linalg.inv(
        physical_eef_to_model_tcp(args)
    )


def observation_model_tcp_world(observation, args):
    """Return the current model EEF/TCP in the RLBench world frame."""

    physical_eef_world = pose9_to_homo_np(pose7_to_pose9(observation.gripper_pose))
    return physical_eef_world_to_model_tcp_world(physical_eef_world, args)


def gripper_len_in_model_tcp_frame(args):
    """Length offset used to draw REAP geometry in the selected model frame."""

    if (
        str(args.gripper_template) == "reap"
        and bool(args.sync_virtual_gripper_tcp)
    ):
        return float(RLBENCH_REAP_ALIGNED_GRIPPER_LEN)
    return float(args.gripper_len)


def live_model_observation(observation, args, seed):
    """Build exactly the modalities used by the RLBench LeRobot training set."""
    current_model_tcp_world = observation_model_tcp_world(observation, args)
    current_world_pose9 = worldflow_homo_to_pose9(current_model_tcp_world)
    # Collection stores the continuous RLBench opening fraction multiplied by
    # Panda's 0.08 m maximum width.  Preserve that value for state and virtual
    # gripper geometry; only predicted actions are discretized for the RLBench
    # controller.
    current_gripper_width = observed_gripper_width(observation)
    state = np.zeros(10, dtype=np.float32)
    state[3] = 1.0
    state[7] = 1.0
    state[9] = current_gripper_width

    cloud_world = observation_cloud(
        observation, fallback_to_all_finite=True
    )
    cloud_world = sample_or_repeat_points(
        cloud_world, args.num_points, seed=seed
    )
    current_world_pose10 = np.concatenate((current_world_pose9, [current_gripper_width])).astype(np.float32)
    if args.add_gripper_cloud:
        if args.gripper_template == "reap":
            width_percent = float(
                libero_reap_width_percent_from_physical(current_gripper_width)
            )
            # With TCP sync, the whole observation is expressed in the shifted
            # virtual-TCP frame. Keeping the calibrated 0.09 m local geometry
            # here places the gripper at the requested physical-world length:
            # T_W_E @ T_E_V(L) @ geometry(0.09) == T_W_E @ geometry(L).
            gripper_len = gripper_len_in_model_tcp_frame(args)
        else:
            width_percent = float(observation.gripper_open > 0.5)
            gripper_len = 0.0
        cloud_eef = add_world_gripper_cloud_to_point_cloud(
            cloud_world,
            current_world_pose10,
            width_percent=width_percent,
            total_points=args.num_points,
            gripper_points=args.gripper_points,
            gripper_template=args.gripper_template,
            gripper_len=gripper_len,
            gripper_opening_max_width=(
                LIBERO_REAP_OPENING_MAX_WIDTH
                if args.gripper_template == "reap"
                else None
            ),
            seed=seed,
            drop_strategy="tail",
            shuffle_points=False,
        )
    else:
        cloud_eef = reference_point_cloud_to_current_eff(cloud_world, current_world_pose10)
    front_rgb = rgb_to_uint8(observation.front_rgb)
    model_observation = {
        "point_cloud": cloud_eef,
        "state": state,
        # Keep the legacy RLBench key while exposing the standard feature name
        # expected by v043 frozen-VLM checkpoints. All aliases share one RGB
        # array and therefore do not change the rendered observation.
        "front": front_rgb,
        "agentview": front_rgb,
        "observation.images.agentview": front_rgb,
    }
    # WorldFlow's scene branch consumes the same current-EEF cloud as Ego, but
    # transports it into the fixed Panda-link0 frame.  The Ego/UMI state must
    # remain identity-normalized, so expose the physical pose through its
    # dedicated side-channel rather than changing observation.state. With
    # virtual-TCP sync this is T_base_virtual_tcp, not physical Panda_tip.
    t_base_world = getattr(args, "worldflow_t_base_world", None)
    if t_base_world is not None:
        t_base_eef = np.asarray(t_base_world, dtype=np.float64) @ pose9_to_homo_np(
            current_world_pose9
        )
        model_observation["worldflow.current_ee_pose"] = worldflow_homo_to_pose9(
            t_base_eef
        )
    return model_observation


def resize_video_frames(frames, width, height):
    """Resize saved video frames without changing the policy's RGB observation."""
    resized_frames = []
    for frame in frames:
        image = Image.fromarray(np.asarray(frame, dtype=np.uint8))
        image = image.resize((int(width), int(height)), Image.Resampling.BILINEAR)
        resized_frames.append(np.asarray(image, dtype=np.uint8))
    return resized_frames


def video_front_rgb(task_env, fallback_rgb, refresh):
    """Capture one final front RGB image without advancing the simulation."""
    fallback = rgb_to_uint8(fallback_rgb)
    if not refresh:
        return fallback
    try:
        sensor = task_env._scene._cam_front
        sensor.handle_explicitly()
        return rgb_to_uint8(sensor.capture_rgb())
    except Exception:
        return fallback


def annotate_video_frame(
    frame,
    task_name,
    episode_index,
    frame_index,
    physics_frame_index,
    model_call,
    chunk_row=None,
):
    """Add readable control-frame metadata to the top of a saved video frame."""
    image = Image.fromarray(np.asarray(frame, dtype=np.uint8))
    draw = ImageDraw.Draw(image)
    padding = max(4, int(round(image.width / 128.0)))
    font_size = max(9, min(20, int(round(image.width / 32.0))))
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    while True:
        try:
            font = ImageFont.truetype(font_path, font_size)
        except OSError:
            font = ImageFont.load_default()
            break
        lines = [
            "task="
            + str(task_name)
            + " episode="
            + str(episode_index).zfill(3)
            + " frame="
            + str(frame_index).zfill(6),
            "physics_frame="
            + str(physics_frame_index).zfill(6)
            + " model_call="
            + str(model_call).zfill(4),
        ]
        if chunk_row is not None:
            lines.append("chunk_row=" + str(chunk_row).zfill(2))
        widths = [draw.textbbox((0, 0), line, font=font)[2] for line in lines]
        if max(widths) + 2 * padding <= image.width or font_size <= 8:
            break
        font_size -= 1

    line_height = max(draw.textbbox((0, 0), line, font=font)[3] for line in lines)
    line_gap = max(2, padding // 2)
    bar_height = 2 * padding + len(lines) * line_height + (len(lines) - 1) * line_gap
    draw.rectangle((0, 0, image.width, bar_height), fill=(0, 0, 0))
    y = padding
    for line in lines:
        draw.text((padding, y), line, fill=(255, 255, 255), font=font)
        y += line_height + line_gap
    return np.asarray(image, dtype=np.uint8)


def action_time_colors(count):
    """Return LIBERO-style blue/cyan -> green -> red chunk colors."""
    if count <= 0:
        return np.empty((0, 3), dtype=np.uint8)
    if count == 1:
        return np.asarray([[26, 217, 64]], dtype=np.uint8)
    values = np.linspace(0.0, 1.0, count, dtype=np.float32)[:, None]
    start = np.asarray([13.0, 140.0, 255.0], dtype=np.float32)
    middle = np.asarray([26.0, 217.0, 64.0], dtype=np.float32)
    end = np.asarray([255.0, 46.0, 13.0], dtype=np.float32)
    first_half = (1.0 - 2.0 * values) * start + (2.0 * values) * middle
    second_half = (2.0 - 2.0 * values) * middle + (2.0 * values - 1.0) * end
    colors = np.where(values <= 0.5, first_half, second_half)
    return np.clip(colors, 0.0, 255.0).astype(np.uint8)


def sample_colored_line(start, end, start_color, end_color, spacing=0.0015):
    """Represent one line segment as dense colored PLY points."""
    distance = float(np.linalg.norm(end - start))
    count = max(2, int(np.ceil(distance / max(float(spacing), 1e-5))) + 1)
    alpha = np.linspace(0.0, 1.0, count, dtype=np.float32)[:, None]
    points = (1.0 - alpha) * start + alpha * end
    colors = (1.0 - alpha) * start_color + alpha * end_color
    return points.astype(np.float32), colors.astype(np.uint8)


def _foreground_scores_for_cloud(point_cloud, snapshot):
    """Return LitePT foreground scores in the same order as the model input cloud."""
    cloud = np.asarray(point_cloud, dtype=np.float32)
    scores = np.full(len(cloud), np.nan, dtype=np.float32)
    if not isinstance(snapshot, dict):
        return scores

    snapshot_cloud = snapshot.get("point_cloud")
    operation_prob = snapshot.get("operation_prob")
    point_is_pad = snapshot.get("point_is_pad")
    if snapshot_cloud is None or operation_prob is None:
        return scores

    if hasattr(snapshot_cloud, "detach"):
        snapshot_cloud = snapshot_cloud.detach().float().cpu().numpy()
    if hasattr(operation_prob, "detach"):
        operation_prob = operation_prob.detach().float().cpu().numpy()
    if hasattr(point_is_pad, "detach"):
        point_is_pad = point_is_pad.detach().bool().cpu().numpy()

    snapshot_cloud = np.asarray(snapshot_cloud, dtype=np.float32)
    operation_prob = np.asarray(operation_prob, dtype=np.float32).reshape(-1)
    if snapshot_cloud.ndim == 3:
        snapshot_cloud = snapshot_cloud[0]
    if point_is_pad is not None:
        point_is_pad = np.asarray(point_is_pad, dtype=bool).reshape(-1)

    # build_model_batch sends this exact point order to LitePT.  Refuse an
    # accidental positional mismatch instead of painting unrelated points.
    if snapshot_cloud.ndim != 2 or len(snapshot_cloud) != len(cloud):
        return scores
    if not np.allclose(snapshot_cloud[:, :3], cloud[:, :3], atol=1e-5, rtol=1e-5):
        return scores
    if len(operation_prob) != len(cloud):
        return scores
    valid = np.isfinite(operation_prob)
    if point_is_pad is not None and len(point_is_pad) == len(cloud):
        valid = valid & ~point_is_pad
    scores[valid] = np.clip(operation_prob[valid], 0.0, 1.0)
    return scores


def action_execution_window(action_count, args):
    """Return the rows sent during the current model call and future rows."""
    if action_count <= 0:
        return 0, 0
    start = min(max(int(args.action_index), 0), action_count - 1)
    stop = min(action_count, start + max(int(args.exec_action_steps), 1))
    return start, stop


def target_gripper_points_in_eef_frame(action_row, count, seed, gripper_template):
    """Sample target gripper geometry in the current-call EEF reference frame."""
    row = np.asarray(action_row, dtype=np.float32)
    target_pose = pose9_to_homo_np(row[:9])
    target_euler = Rotation.from_matrix(target_pose[:3, :3]).as_euler("zyx")
    target_pose6 = np.concatenate(
        (target_pose[:3, 3], target_euler.astype(np.float32))
    )
    width_percent = (
        float(libero_reap_width_percent_from_physical(row[9]))
        if gripper_template == "reap"
        else float(np.clip(row[9] / max(RLBENCH_PANDA_MAX_WIDTH, 1e-6), 0.0, 1.0))
    )
    rng = np.random.default_rng(int(seed))
    create_points = (
        create_rlbench_minimal_two_finger_points
        if gripper_template == RLBENCH_MINIMAL_TWO_FINGER_TEMPLATE
        else create_rlbench_panda_gripper_points
    )
    if gripper_template == "reap":
        return create_rlbench_reap_points_from_physical_width(
            float(row[9]),
            target_pose6,
            int(count),
            rng,
        ).astype(np.float32)
    return create_points(
        width_percent,
        target_pose6,
        int(count),
        rng,
    ).astype(np.float32)


def gripper_box_corners_in_target_frame(
    action_row, gripper_template, reap_gripper_len=LIBERO_REAP_GRIPPER_LEN
):
    """Return selected gripper box corners before applying the target pose."""
    row = np.asarray(action_row, dtype=np.float32)
    width_percent = (
        float(libero_reap_width_percent_from_physical(row[9]))
        if gripper_template == "reap"
        else float(np.clip(row[9] / max(RLBENCH_PANDA_MAX_WIDTH, 1e-6), 0.0, 1.0))
    )
    corners = []
    if gripper_template == "reap":
        width = width_percent * LIBERO_REAP_OPENING_MAX_WIDTH
        boxes = [
            (np.array([width / 2.0, 0.01, 0.0]), np.array([0.01, 0.08, 0.01])),
            (np.array([-width / 2.0 - 0.01, 0.01, 0.0]), np.array([0.01, 0.08, 0.01])),
            (np.array([-0.035, 0.0, 0.0]), np.array([0.07, 0.01, 0.01])),
            (np.array([-0.005, -0.05, 0.0]), np.array([0.01, 0.05, 0.01])),
        ]
        static_rotation = Rotation.from_euler("zyx", [np.pi / 2.0, np.pi / 2.0, 0.0]).as_matrix()
        for min_corner, size in boxes:
            max_corner = min_corner + size
            box_corners = np.asarray(
                [
                    [min_corner[0], min_corner[1], min_corner[2]],
                    [max_corner[0], min_corner[1], min_corner[2]],
                    [max_corner[0], max_corner[1], min_corner[2]],
                    [min_corner[0], max_corner[1], min_corner[2]],
                    [min_corner[0], min_corner[1], max_corner[2]],
                    [max_corner[0], min_corner[1], max_corner[2]],
                    [max_corner[0], max_corner[1], max_corner[2]],
                    [min_corner[0], max_corner[1], max_corner[2]],
                ],
                dtype=np.float32,
            )
            corners.append(
                box_corners @ static_rotation.T
                + np.asarray(
                    [0.0, 0.0, -float(reap_gripper_len)], dtype=np.float32
                )
            )
        return corners
    local_boxes = rlbench_panda_gripper_local_boxes(width_percent)
    if gripper_template == RLBENCH_MINIMAL_TWO_FINGER_TEMPLATE:
        left_finger = local_boxes["left_finger"]
        right_finger = local_boxes["right_finger"]
        left_min, finger_size = left_finger
        right_min, _ = right_finger
        local_boxes = {
            "left_finger": left_finger,
            "right_finger": right_finger,
            "bridge": (
                np.array(
                    [left_min[0], left_min[1], left_min[2]], dtype=np.float64
                ),
                np.array(
                    [
                        finger_size[0],
                        (right_min[1] + finger_size[1]) - left_min[1],
                        0.012,
                    ],
                    dtype=np.float64,
                ),
            ),
        }
    for min_corner, size in local_boxes.values():
        max_corner = min_corner + size
        box_corners = np.asarray(
            [
                [min_corner[0], min_corner[1], min_corner[2]],
                [max_corner[0], min_corner[1], min_corner[2]],
                [max_corner[0], max_corner[1], min_corner[2]],
                [min_corner[0], max_corner[1], min_corner[2]],
                [min_corner[0], min_corner[1], max_corner[2]],
                [max_corner[0], min_corner[1], max_corner[2]],
                [max_corner[0], max_corner[1], max_corner[2]],
                [min_corner[0], max_corner[1], max_corner[2]],
            ],
            dtype=np.float32,
        )
        corners.append(box_corners)
    return corners


def current_virtual_gripper_world_geometry(observation, model_observation, args):
    """Return the exact current virtual-gripper points and box edges in world frame."""
    if not args.add_gripper_cloud or int(args.gripper_points) <= 0:
        return np.empty((0, 3), dtype=np.float32), []
    cloud_eef = np.asarray(model_observation["point_cloud"], dtype=np.float32)
    if cloud_eef.ndim != 2 or cloud_eef.shape[1] < 3:
        raise ValueError("Model point cloud must have shape (N, >=3).")
    gripper_count = int(args.gripper_points)
    if gripper_count > len(cloud_eef):
        raise ValueError(
            "Configured virtual-gripper point count exceeds model point cloud: "
            + str(gripper_count)
            + " > "
            + str(len(cloud_eef))
        )
    gripper_points_eef = cloud_eef[-gripper_count:, :3]

    t_world_eef = observation_model_tcp_world(observation, args)
    rotation_world_eef = t_world_eef[:3, :3]
    translation_world_eef = t_world_eef[:3, 3]
    gripper_points_world = (
        gripper_points_eef @ rotation_world_eef.T + translation_world_eef
    ).astype(np.float32)

    identity_with_width = np.zeros(10, dtype=np.float32)
    identity_with_width[3] = 1.0
    identity_with_width[7] = 1.0
    identity_with_width[9] = (
        observed_gripper_width(observation)
        if args.gripper_template == "reap"
        else (
            RLBENCH_PANDA_MAX_WIDTH
            if float(observation.gripper_open) > 0.5
            else 0.0
        )
    )
    boxes_world = [
        (
            np.asarray(box_eef, dtype=np.float32) @ rotation_world_eef.T
            + translation_world_eef
        ).astype(np.float32)
        for box_eef in gripper_box_corners_in_target_frame(
            identity_with_width,
            args.gripper_template,
            reap_gripper_len=gripper_len_in_model_tcp_frame(args),
        )
    ]
    return gripper_points_world, boxes_world


def build_action_chunk_ply_cloud(
    point_cloud,
    action_chunk,
    max_points,
    foreground_snapshot=None,
    point_mode="prob",
    execution_start=0,
    execution_stop=0,
):
    """Add EEF paths and orientation axes to the current EEF cloud."""
    cloud = np.asarray(point_cloud, dtype=np.float32)
    actions = np.asarray(action_chunk, dtype=np.float32)
    if cloud.ndim != 2 or cloud.shape[1] < 3:
        raise ValueError("Point cloud must have shape (N, >=3).")
    if actions.ndim != 2 or actions.shape[1] < 9:
        raise ValueError("Action chunk must have shape (T, >=9).")

    valid_cloud = np.isfinite(cloud[:, :3]).all(axis=1)
    scene_xyz = cloud[valid_cloud, :3]
    scene_source_indices = np.flatnonzero(valid_cloud).astype(np.int32)
    if point_mode not in {"full", "prob"}:
        raise ValueError("point_mode must be 'full' or 'prob'.")
    foreground_scores = _foreground_scores_for_cloud(cloud, foreground_snapshot)
    if cloud.shape[1] >= 6:
        scene_rgb = cloud[valid_cloud, 3:6]
        if scene_rgb.size and float(np.max(scene_rgb)) <= 1.0:
            scene_rgb = scene_rgb * 255.0
        scene_rgb = np.clip(scene_rgb, 0.0, 255.0).astype(np.uint8)
    else:
        scene_rgb = np.full((len(scene_xyz), 3), 128, dtype=np.uint8)
    scene_scores = foreground_scores[valid_cloud]
    score_valid = np.isfinite(scene_scores)
    if point_mode == "prob" and np.any(score_valid):
        colored = foreground_score_colors(scene_scores[score_valid], scene_rgb[score_valid])
        scene_rgb[score_valid] = np.rint(colored * 255.0).astype(np.uint8)

    valid_action_indices = np.flatnonzero(
        np.isfinite(actions[:, :9]).all(axis=1)
    ).astype(np.int32)
    poses = pose9_to_homo_np(actions[valid_action_indices, :9])
    positions = poses[:, :3, 3]
    colors = action_time_colors(len(positions))
    overlay_xyz = []
    overlay_rgb = []
    overlay_kind = []
    overlay_action_index = []
    overlay_phase = []

    def append_overlay(points, point_colors, kind, action_index):
        points = np.asarray(points, dtype=np.float32)
        point_colors = np.array(point_colors, dtype=np.uint8, copy=True)
        phase = 1 if execution_start <= action_index < execution_stop else 2
        if kind == 2 and phase == 2:
            point_colors[:] = np.asarray([150, 150, 150], dtype=np.uint8)
        overlay_xyz.append(points)
        overlay_rgb.append(point_colors)
        overlay_kind.append(np.full(len(points), kind, dtype=np.uint8))
        overlay_action_index.append(
            np.full(len(points), int(action_index), dtype=np.int32)
        )
        overlay_phase.append(np.full(len(points), phase, dtype=np.uint8))

    for local_index in range(max(0, len(positions) - 1)):
        action_index = int(valid_action_indices[local_index])
        next_action_index = int(valid_action_indices[local_index + 1])
        points, point_colors = sample_colored_line(
            positions[local_index],
            positions[local_index + 1],
            colors[local_index].astype(np.float32),
            colors[local_index + 1].astype(np.float32),
        )
        append_overlay(points, point_colors, 1, action_index)
        if next_action_index != action_index + 1:
            raise ValueError("Action chunk contains a non-contiguous valid row.")

    # Every action gets a colored cross and orientation triad.
    marker_offsets = np.linspace(-0.004, 0.004, 9, dtype=np.float32)
    axis_offsets = np.linspace(0.0, 0.018, 8, dtype=np.float32)
    axis_colors = np.asarray(
        [[255, 30, 30], [30, 255, 30], [30, 100, 255]], dtype=np.uint8
    )
    for local_index, action_index_value in enumerate(valid_action_indices):
        action_index = int(action_index_value)
        marker_points = []
        for axis in range(3):
            points = np.repeat(positions[local_index][None, :], len(marker_offsets), axis=0)
            points[:, axis] += marker_offsets
            marker_points.append(points)
        marker_points = np.concatenate(marker_points, axis=0)
        append_overlay(
            marker_points,
            np.repeat(colors[local_index][None, :], len(marker_points), axis=0),
            1,
            action_index,
        )

        for axis in range(3):
            direction = poses[local_index, :3, axis]
            axis_points = positions[local_index][None, :] + axis_offsets[:, None] * direction[None, :]
            append_overlay(
                axis_points,
                np.repeat(axis_colors[axis][None, :], len(axis_points), axis=0),
                1,
                action_index,
            )

    if overlay_xyz:
        trajectory_xyz = np.concatenate(overlay_xyz, axis=0).astype(np.float32)
        trajectory_rgb = np.concatenate(overlay_rgb, axis=0).astype(np.uint8)
        trajectory_kind = np.concatenate(overlay_kind, axis=0)
        trajectory_action_index = np.concatenate(overlay_action_index, axis=0)
        trajectory_phase = np.concatenate(overlay_phase, axis=0)
    else:
        trajectory_xyz = np.empty((0, 3), dtype=np.float32)
        trajectory_rgb = np.empty((0, 3), dtype=np.uint8)
        trajectory_kind = np.empty((0,), dtype=np.uint8)
        trajectory_action_index = np.empty((0,), dtype=np.int32)
        trajectory_phase = np.empty((0,), dtype=np.uint8)

    max_points = max(int(max_points), len(trajectory_xyz))
    scene_limit = max_points - len(trajectory_xyz)
    if len(scene_xyz) > scene_limit:
        indices = np.linspace(0, len(scene_xyz) - 1, scene_limit, dtype=np.int64)
        scene_xyz = scene_xyz[indices]
        scene_rgb = scene_rgb[indices]
        scene_scores = scene_scores[indices]
        scene_source_indices = scene_source_indices[indices]
    xyz = np.concatenate((scene_xyz, trajectory_xyz), axis=0)
    rgb = np.concatenate((scene_rgb, trajectory_rgb), axis=0)
    scores = np.concatenate(
        (scene_scores, np.full(len(trajectory_xyz), np.nan, dtype=np.float32)), axis=0
    )
    source_indices = np.concatenate(
        (
            scene_source_indices,
            np.full(len(trajectory_xyz), -1, dtype=np.int32),
        ),
        axis=0,
    )
    point_kind = np.concatenate(
        (
            np.zeros(len(scene_xyz), dtype=np.uint8),
            trajectory_kind,
        ),
        axis=0,
    )
    action_indices = np.concatenate(
        (
            np.full(len(scene_xyz), -1, dtype=np.int32),
            trajectory_action_index,
        ),
        axis=0,
    )
    action_phase = np.concatenate(
        (
            np.zeros(len(scene_xyz), dtype=np.uint8),
            trajectory_phase,
        ),
        axis=0,
    )
    return xyz, rgb, scores, source_indices, point_kind, action_indices, action_phase


def write_colored_ply(
    path,
    xyz,
    rgb,
    foreground_scores=None,
    source_indices=None,
    point_kind=None,
    action_indices=None,
    action_phase=None,
    point_mode="prob",
    header_comments=(),
):
    """Write action RGB plus LitePT score fields in an Open3D/MeshLab PLY."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        file.write("ply\n")
        file.write("format ascii 1.0\n")
        file.write("comment coordinate_frame current_virtual_tcp_at_model_call\n")
        file.write("comment coordinate_origin virtual_gripper_fingertip_midpoint\n")
        for comment in header_comments:
            normalized_comment = str(comment).replace("\n", " ").replace("\r", " ")
            file.write("comment " + normalized_comment + "\n")
        file.write("comment action_path blue_green_red means chunk_1_to_chunk_32\n")
        file.write(
            "comment point_kind 0=scene_or_current_gripper "
            "1=eef_path_and_axes\n"
        )
        file.write(
            "comment action_phase 0=scene "
            "1=current_execution_window 2=future_forecast\n"
        )
        file.write("comment scene_point_mode " + str(point_mode) + "\n")
        file.write("element vertex " + str(len(xyz)) + "\n")
        file.write("property float x\nproperty float y\nproperty float z\n")
        file.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        if foreground_scores is not None:
            file.write("property float operation_prob\n")
            file.write("property float selection_score\n")
            file.write("property int source_point_index\n")
            file.write("property uchar point_kind\n")
            file.write("property int action_index\n")
            file.write("property uchar action_phase\n")
        file.write("end_header\n")
        for index, (point, color) in enumerate(zip(xyz, rgb)):
            line = "%.7f %.7f %.7f %d %d %d" % (
                point[0], point[1], point[2], color[0], color[1], color[2]
            )
            if foreground_scores is not None:
                score = float(foreground_scores[index])
                source = int(source_indices[index]) if source_indices is not None else index
                kind = int(point_kind[index]) if point_kind is not None else 0
                action = int(action_indices[index]) if action_indices is not None else -1
                phase = int(action_phase[index]) if action_phase is not None else 0
                line += " %.7f %.7f %d %d %d %d" % (
                    score,
                    score,
                    source,
                    kind,
                    action,
                    phase,
                )
            file.write(line + "\n")


def write_model_input_ply_binary(path, point_cloud, comments=()):
    """Write one raw XYZRGB model-input cloud as a compact binary PLY."""
    cloud = np.asarray(point_cloud, dtype=np.float32)
    if cloud.ndim != 2 or cloud.shape[1] < 6:
        raise ValueError("Expected point cloud shape (N, >=6), got " + str(cloud.shape))
    xyz = cloud[:, :3]
    rgb = cloud[:, 3:6]
    if rgb.size and float(np.nanmax(rgb)) <= 1.0 + 1e-6:
        rgb = rgb * 255.0
    rgb = np.clip(
        np.rint(np.nan_to_num(rgb, nan=0.0, posinf=255.0, neginf=0.0)),
        0.0,
        255.0,
    ).astype(np.uint8)
    valid = np.isfinite(xyz).all(axis=1)
    xyz = np.asarray(xyz[valid], dtype="<f4")
    rgb = rgb[valid]
    vertex_dtype = np.dtype(
        [
            ("x", "<f4"),
            ("y", "<f4"),
            ("z", "<f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
        ],
        align=False,
    )
    vertices = np.empty(len(xyz), dtype=vertex_dtype)
    vertices["x"], vertices["y"], vertices["z"] = xyz.T
    vertices["red"], vertices["green"], vertices["blue"] = rgb.T
    header = [
        "ply",
        "format binary_little_endian 1.0",
        *["comment " + str(comment) for comment in comments],
        "element vertex " + str(len(vertices)),
        "property float x",
        "property float y",
        "property float z",
        "property uchar red",
        "property uchar green",
        "property uchar blue",
        "end_header",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as file:
        file.write("\n".join(header).encode("ascii"))
        file.write(vertices.tobytes(order="C"))


def project_world_points_to_front_image(world_points, observation):
    """Project world XYZ through RLBench's front camera calibration."""
    extrinsics = np.asarray(observation.misc["front_camera_extrinsics"], dtype=np.float64)
    intrinsics = np.asarray(observation.misc["front_camera_intrinsics"], dtype=np.float64)
    points = np.asarray(world_points, dtype=np.float64)
    homogeneous = np.concatenate((points, np.ones((len(points), 1))), axis=1)
    camera_points = (np.linalg.inv(extrinsics) @ homogeneous.T).T[:, :3]
    projected = (intrinsics @ camera_points.T).T
    pixels = projected[:, :2] / projected[:, 2:3]
    height, width = np.asarray(observation.front_rgb).shape[:2]
    valid = np.isfinite(pixels).all(axis=1)
    # PyRep uses positive camera Z as metric depth. Its focal lengths are
    # negative because of CoppeliaSim's image-axis convention.
    valid = valid & (camera_points[:, 2] > 1e-6)
    valid = valid & (pixels[:, 0] >= 0.0) & (pixels[:, 0] < width)
    valid = valid & (pixels[:, 1] >= 0.0) & (pixels[:, 1] < height)
    return pixels.astype(np.float32), valid


def _pour_point_box_corners(sensor):
    """Return the sensor's local bounding-box corners in world coordinates."""
    bbox = np.asarray(sensor.get_bounding_box(), dtype=np.float64)
    if bbox.shape != (6,) or not np.isfinite(bbox).all():
        raise ValueError("Invalid pour_point sensor bounding box: " + str(bbox))
    min_x, max_x, min_y, max_y, min_z, max_z = bbox.tolist()
    corners_local = np.asarray(
        [
            [min_x, min_y, min_z],
            [max_x, min_y, min_z],
            [max_x, max_y, min_z],
            [min_x, max_y, min_z],
            [min_x, min_y, max_z],
            [max_x, min_y, max_z],
            [max_x, max_y, max_z],
            [min_x, max_y, max_z],
        ],
        dtype=np.float64,
    )
    homogeneous = np.concatenate(
        [corners_local, np.ones((len(corners_local), 1), dtype=np.float64)],
        axis=1,
    )
    return (np.asarray(sensor.get_matrix(), dtype=np.float64) @ homogeneous.T).T[:, :3]


def _draw_projected_sensor_box(
    image,
    sensor,
    observation,
    color,
    label,
    label_position=None,
    fill_alpha=0,
):
    """Draw a proximity-sensor volume approximation and return its center."""
    if sensor is None:
        return None, False
    corners = _pour_point_box_corners(sensor)
    pixels, valid = project_world_points_to_front_image(corners, observation)
    line_width = max(2, int(round(image.width / 128.0)))
    edges = (
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    )
    if fill_alpha > 0:
        faces = (
            (0, 1, 2, 3),
            (4, 5, 6, 7),
            (0, 1, 5, 4),
            (1, 2, 6, 5),
            (2, 3, 7, 6),
            (3, 0, 4, 7),
        )
        fill = Image.new("RGBA", image.size, (0, 0, 0, 0))
        fill_draw = ImageDraw.Draw(fill)
        rgba_color = tuple(int(value) for value in color) + (
            int(np.clip(fill_alpha, 0, 255)),
        )
        for face in faces:
            if all(valid[index] for index in face):
                fill_draw.polygon(
                    [tuple(pixels[index]) for index in face],
                    fill=rgba_color,
                )
        image.paste(fill, (0, 0), fill)
    draw = ImageDraw.Draw(image)
    for start, end in edges:
        if valid[start] and valid[end]:
            draw.line(
                [tuple(pixels[start]), tuple(pixels[end])],
                fill=color,
                width=line_width,
            )

    center = corners.mean(axis=0, keepdims=True)
    center_pixels, center_valid = project_world_points_to_front_image(
        center, observation
    )
    if center_valid[0]:
        x, y = center_pixels[0]
        radius = max(3, line_width + 1)
        draw.ellipse(
            (int(x - radius), int(y - radius), int(x + radius), int(y + radius)),
            fill=color,
            outline="white",
            width=1,
        )
        if label_position is None:
            label_position = (int(x) + radius + 3, int(y) - radius - 3)
    elif label_position is None:
        label_position = (8, 8)
    if label:
        draw.text(
            label_position,
            label,
            fill=color,
            stroke_width=max(1, line_width // 2),
            stroke_fill=(0, 0, 0),
        )
    return center_pixels[0], bool(center_valid[0])


def draw_pour_point_on_front_image(frame, observation, task_env):
    """Overlay the pour-point sensor envelope without changing the observation."""
    task = getattr(task_env, "_task", None)
    sensor = getattr(task, "pour_point", None)
    head = getattr(task, "head", None)
    if sensor is None or head is None:
        return frame
    misc = getattr(observation, "misc", None)
    if not isinstance(misc, dict) or "front_camera_extrinsics" not in misc:
        return frame

    image = Image.fromarray(np.asarray(frame, dtype=np.uint8))
    detected = bool(sensor.is_detected(head))
    color = (45, 225, 80) if detected else (255, 70, 35)
    center_pixels, center_valid = _draw_projected_sensor_box(
        image,
        sensor,
        observation,
        color,
        "pour_point: DETECTED" if detected else "pour_point: not detected",
    )
    line_width = max(2, int(round(image.width / 128.0)))
    draw = ImageDraw.Draw(image)
    head_pixels, head_valid = project_world_points_to_front_image(
        np.asarray(head.get_position(), dtype=np.float64).reshape(1, 3), observation
    )
    if head_valid[0]:
        x, y = head_pixels[0]
        radius = max(3, line_width + 1)
        draw.ellipse(
            (int(x - radius), int(y - radius), int(x + radius), int(y + radius)),
            outline=(40, 190, 255),
            width=line_width,
        )
    return np.asarray(image, dtype=np.uint8)


def _phone_on_base_snapshot(task_env, success=False, termination=False):
    """Read phone_on_base's simulator-side success conditions for diagnostics."""
    task = getattr(task_env, "_task", None)
    sensor = getattr(task, "success_sensor", None)
    phone = getattr(task, "phone", None)
    if task is None or sensor is None or phone is None:
        return None

    try:
        sensor_detected = bool(sensor.is_detected(phone))
    except Exception:
        sensor_detected = False

    gripper_empty = None
    robot = getattr(task_env, "_robot", None)
    gripper = getattr(robot, "gripper", None)
    if gripper is not None:
        try:
            gripper_empty = len(gripper.get_grasped_objects()) == 0
        except Exception:
            gripper_empty = None

    if success:
        result = "SUCCESS"
    elif termination:
        result = "FAIL / TERMINATED"
    else:
        result = "RUNNING"
    return {
        "sensor_detected": sensor_detected,
        "gripper_empty": gripper_empty,
        "success": bool(success),
        "result": result,
    }


def draw_phone_success_sensor_overlay(
    frame,
    observation,
    task_env,
    success=False,
    termination=False,
):
    """Overlay phone_on_base's success sensor volume and live condition state."""
    snapshot = _phone_on_base_snapshot(
        task_env, success=success, termination=termination
    )
    if snapshot is None:
        return frame
    misc = getattr(observation, "misc", None)
    if not isinstance(misc, dict) or "front_camera_extrinsics" not in misc:
        return frame

    image = Image.fromarray(np.asarray(frame, dtype=np.uint8)).convert("RGB")
    task = getattr(task_env, "_task", None)
    sensor = getattr(task, "success_sensor", None)
    phone = getattr(task, "phone", None)
    line_width = max(2, int(round(image.width / 128.0)))
    sensor_color = (
        (45, 225, 80) if snapshot["sensor_detected"] else (255, 70, 35)
    )
    _draw_projected_sensor_box(
        image,
        sensor,
        observation,
        sensor_color,
        "success sensor: DETECTED"
        if snapshot["sensor_detected"]
        else "success sensor: not detected",
        fill_alpha=42,
    )

    draw = ImageDraw.Draw(image)
    phone_position = np.asarray(phone.get_position(), dtype=np.float64).reshape(1, 3)
    phone_pixels, phone_valid = project_world_points_to_front_image(
        phone_position, observation
    )
    if phone_valid[0]:
        x, y = phone_pixels[0]
        radius = max(4, line_width + 2)
        draw.ellipse(
            (int(x - radius), int(y - radius), int(x + radius), int(y + radius)),
            fill=(40, 190, 255),
            outline="white",
            width=1,
        )
        draw.text(
            (int(x) + radius + 3, int(y) + radius + 2),
            "phone",
            fill=(40, 190, 255),
            stroke_width=max(1, line_width // 2),
            stroke_fill=(0, 0, 0),
        )

    padding = max(4, int(round(image.width / 128.0)))
    panel_x = padding
    panel_y = max(8, int(round(image.height * 0.16)))
    panel_width = min(image.width - 2 * padding, int(round(image.width * 0.64)))
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    font_size = max(8, min(16, int(round(image.width / 32.0))))
    try:
        font = ImageFont.truetype(font_path, font_size)
    except OSError:
        font = ImageFont.load_default()
    gripper_state = (
        "EMPTY"
        if snapshot["gripper_empty"] is True
        else "GRASPED"
        if snapshot["gripper_empty"] is False
        else "UNKNOWN"
    )
    lines = [
        "PHONE ON BASE",
        "success sensor: %s"
        % ("DETECTED" if snapshot["sensor_detected"] else "NOT DETECTED"),
        "gripper: " + gripper_state,
        "task result: " + snapshot["result"],
    ]
    line_height = max(draw.textbbox((0, 0), line, font=font)[3] for line in lines)
    line_gap = max(1, padding // 2)
    panel_height = 2 * padding + len(lines) * line_height + (len(lines) - 1) * line_gap
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    overlay_draw.rectangle(
        (panel_x, panel_y, panel_x + panel_width, panel_y + panel_height),
        fill=(0, 0, 0, 190),
        outline=(255, 255, 255, 180),
        width=1,
    )
    image = Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")
    draw = ImageDraw.Draw(image)
    y = panel_y + padding
    for index, line in enumerate(lines):
        if index == 0:
            text_color = (255, 255, 255)
        elif index == 1:
            text_color = sensor_color
        elif "EMPTY" in line or "SUCCESS" in line:
            text_color = (70, 235, 100)
        elif "GRASPED" in line or "NOT DETECTED" in line or "FAIL" in line:
            text_color = (255, 100, 80)
        else:
            text_color = (220, 220, 220)
        draw.text((panel_x + padding, y), line, fill=text_color, font=font)
        y += line_height + line_gap
    return np.asarray(image, dtype=np.uint8)


def _water_plants_snapshot(task_env, success=False, termination=False):
    """Read simulator-side water_plants conditions for video diagnostics."""
    task = getattr(task_env, "_task", None)
    if task is None or not hasattr(task, "pour_point"):
        return None

    pour_point = getattr(task, "pour_point", None)
    head = getattr(task, "head", None)
    success_sensor = getattr(task, "success_sensor", None)
    waterer = getattr(task, "waterer", None)
    drops = list(getattr(task, "drops", []) or [])

    def detected(sensor, obj):
        if sensor is None or obj is None:
            return False
        try:
            return bool(sensor.is_detected(obj))
        except Exception:
            return False

    grasped = False
    robot = getattr(task_env, "_robot", None)
    gripper = getattr(robot, "gripper", None)
    if gripper is not None and waterer is not None:
        try:
            waterer_handle = waterer.get_handle()
            grasped = any(
                obj.get_handle() == waterer_handle
                for obj in gripper.get_grasped_objects()
            )
        except Exception:
            grasped = False

    pour_detected = detected(pour_point, head)
    drop_detected = [detected(success_sensor, drop) for drop in drops]
    drop_count = int(sum(drop_detected))
    reached_once = bool(getattr(task, "reachedOnce", False))
    drops_created = len(drops) >= 5

    if success:
        stage_index = 5
        stage_name = "task success"
    elif not grasped:
        stage_index = 1
        stage_name = "grasp watering can"
    elif not reached_once and not pour_detected:
        stage_index = 2
        stage_name = "spout -> pour point"
    elif not reached_once and pour_detected:
        stage_index = 3
        stage_name = "pour point triggered"
    elif drop_count < len(drops) or not drops_created:
        stage_index = 4
        stage_name = "drops -> pot sensor"
    else:
        stage_index = 5
        stage_name = "verify task success"

    if termination and not success:
        stage_name = "environment terminated"

    return {
        "stage_index": stage_index,
        "stage_name": stage_name,
        "grasped": grasped,
        "pour_detected": pour_detected,
        "drops_created": drops_created,
        "drop_count": drop_count,
        "drop_total": len(drops),
        "drop_detected": drop_detected,
        "success": bool(success),
    }


def draw_water_plants_stage_overlay(
    frame,
    observation,
    task_env,
    success=False,
    termination=False,
    result_override=None,
):
    """Overlay water_plants goals and simulator-side condition state."""
    snapshot = _water_plants_snapshot(
        task_env, success=success, termination=termination
    )
    if snapshot is None:
        return frame
    misc = getattr(observation, "misc", None)
    if not isinstance(misc, dict) or "front_camera_extrinsics" not in misc:
        return frame

    image = Image.fromarray(np.asarray(frame, dtype=np.uint8)).convert("RGB")
    task = getattr(task_env, "_task", None)
    pour_point = getattr(task, "pour_point", None)
    head = getattr(task, "head", None)
    success_sensor = getattr(task, "success_sensor", None)
    line_width = max(2, int(round(image.width / 128.0)))
    draw = ImageDraw.Draw(image)

    # Keep both 3-D detection volumes visible: red/green for the spout target,
    # blue for the pot target. The circles mark the actual simulated objects.
    pour_color = (45, 225, 80) if snapshot["pour_detected"] else (255, 70, 35)
    pour_center, pour_valid = _draw_projected_sensor_box(
        image,
        pour_point,
        observation,
        pour_color,
        "pour target",
    )
    success_color = (45, 225, 80) if snapshot["drop_count"] == snapshot["drop_total"] and snapshot["drop_total"] > 0 else (60, 150, 255)
    _draw_projected_sensor_box(
        image,
        success_sensor,
        observation,
        success_color,
        "pot sensor",
    )

    head_pixels, head_valid = project_world_points_to_front_image(
        np.asarray(head.get_position(), dtype=np.float64).reshape(1, 3), observation
    ) if head is not None else (np.empty((1, 2)), np.asarray([False]))
    if head_valid[0]:
        x, y = head_pixels[0]
        radius = max(3, line_width + 1)
        draw.ellipse(
            (int(x - radius), int(y - radius), int(x + radius), int(y + radius)),
            outline=(40, 190, 255),
            width=line_width,
        )
    if pour_valid and head_valid[0]:
        draw.line(
            [tuple(head_pixels[0]), tuple(pour_center)],
            fill=(45, 225, 80) if snapshot["pour_detected"] else (255, 190, 40),
            width=max(1, line_width // 2),
        )

    for index, drop in enumerate(getattr(task, "drops", []) or []):
        drop_pixels, drop_valid = project_world_points_to_front_image(
            np.asarray(drop.get_position(), dtype=np.float64).reshape(1, 3), observation
        )
        if not drop_valid[0]:
            continue
        x, y = drop_pixels[0]
        detected = snapshot["drop_detected"][index]
        radius = max(2, line_width)
        drop_color = (45, 225, 80) if detected else (255, 210, 40)
        draw.ellipse(
            (int(x - radius), int(y - radius), int(x + radius), int(y + radius)),
            fill=drop_color,
            outline="white",
            width=1,
        )

    # Put the checklist below the existing control metadata bar.
    padding = max(4, int(round(image.width / 128.0)))
    panel_x = padding
    panel_y = max(8, int(round(image.height * 0.16)))
    panel_width = min(image.width - 2 * padding, int(round(image.width * 0.78)))
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    font_size = max(8, min(16, int(round(image.width / 32.0))))
    lines = [
        "WATER PLANTS",
        "stage %d/5: %s" % (snapshot["stage_index"], snapshot["stage_name"]),
        "%s grasp watering can" % ("[OK]" if snapshot["grasped"] else "[  ]"),
        "%s spout -> pour point" % ("[OK]" if snapshot["pour_detected"] or getattr(task, "reachedOnce", False) else "[GO]"),
        "%s water drops created" % ("[OK]" if snapshot["drops_created"] else "[  ]"),
        "%s drops -> pot (%d/%d)" % (
            "[OK]" if snapshot["drop_count"] == snapshot["drop_total"] and snapshot["drop_total"] > 0 else "[  ]",
            snapshot["drop_count"],
            max(snapshot["drop_total"], 5),
        ),
        "%s task success" % ("[OK]" if snapshot["success"] else "[  ]"),
    ]
    if result_override:
        lines.append(str(result_override))
    while True:
        try:
            font = ImageFont.truetype(font_path, font_size)
        except OSError:
            font = ImageFont.load_default()
            break
        widths = [draw.textbbox((0, 0), line, font=font)[2] for line in lines]
        if max(widths) + 2 * padding <= panel_width or font_size <= 8:
            break
        font_size -= 1
    line_height = max(draw.textbbox((0, 0), line, font=font)[3] for line in lines)
    line_gap = max(1, padding // 2)
    panel_height = 2 * padding + len(lines) * line_height + (len(lines) - 1) * line_gap
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    overlay_draw.rectangle(
        (panel_x, panel_y, panel_x + panel_width, panel_y + panel_height),
        fill=(0, 0, 0, 190),
        outline=(255, 255, 255, 180),
        width=1,
    )
    image = Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")
    draw = ImageDraw.Draw(image)
    y = panel_y + padding
    for index, line in enumerate(lines):
        if index == 0:
            color = (255, 255, 255)
        elif index == 1:
            color = pour_color if snapshot["stage_index"] in (2, 3) else (255, 210, 80)
        elif "[OK]" in line:
            color = (70, 235, 100)
        elif "[GO]" in line:
            color = (255, 210, 60)
        elif result_override:
            color = (255, 90, 70)
        else:
            color = (220, 220, 220)
        draw.text((panel_x + padding, y), line, fill=color, font=font)
        y += line_height + line_gap
    return np.asarray(image, dtype=np.uint8)


def draw_action_chunk_on_front_image(
    path,
    observation,
    world_targets,
    image_width,
    execution_start,
    execution_stop,
    current_gripper_points_world=None,
    current_gripper_boxes_world=None,
):
    """Draw the target path and current virtual gripper on the front RGB image."""
    source = Image.fromarray(rgb_to_uint8(observation.front_rgb))
    target_width = int(image_width)
    target_height = max(1, int(round(source.height * target_width / source.width)))
    image = source.resize((target_width, target_height), Image.Resampling.BILINEAR)
    draw = ImageDraw.Draw(image)
    colors = action_time_colors(len(world_targets))
    positions = world_targets[:, :3, 3]
    direction_ends = positions + world_targets[:, :3, 0] * 0.025
    position_pixels, position_valid = project_world_points_to_front_image(positions, observation)
    direction_pixels, direction_valid = project_world_points_to_front_image(direction_ends, observation)
    scale = np.asarray([target_width / source.width, target_height / source.height], dtype=np.float32)
    scaled = position_pixels * scale
    scaled_direction = direction_pixels * scale

    def phase_color(index):
        if execution_start <= index < execution_stop:
            return tuple(int(value) for value in colors[index])
        return (150, 150, 150)

    def draw_dashed_line(start_point, end_point, color, width):
        start_point = np.asarray(start_point, dtype=np.float32)
        end_point = np.asarray(end_point, dtype=np.float32)
        distance = float(np.linalg.norm(end_point - start_point))
        segments = max(1, int(np.ceil(distance / 8.0)))
        for segment in range(0, segments, 2):
            alpha_start = segment / segments
            alpha_end = min(segment + 1, segments) / segments
            point_start = start_point + (end_point - start_point) * alpha_start
            point_end = start_point + (end_point - start_point) * alpha_end
            draw.line(
                [tuple(point_start), tuple(point_end)],
                fill=color,
                width=width,
            )

    for index in range(len(scaled) - 1):
        if position_valid[index] and position_valid[index + 1]:
            if execution_start <= index < execution_stop:
                draw.line(
                    [tuple(scaled[index]), tuple(scaled[index + 1])],
                    fill=phase_color(index),
                    width=4,
                )
            else:
                draw_dashed_line(
                    scaled[index],
                    scaled[index + 1],
                    phase_color(index),
                    width=2,
                )

    for index in range(len(scaled)):
        if not position_valid[index]:
            continue
        x, y = scaled[index]
        color = phase_color(index)
        radius = 7
        draw.ellipse(
            (x - radius, y - radius, x + radius, y + radius),
            fill=color,
            outline="white",
            width=2,
        )
        if direction_valid[index]:
            draw.line([tuple(scaled[index]), tuple(scaled_direction[index])], fill="white", width=2)
        draw.text(
            (x + radius + 2, y - radius - 2),
            str(index + 1),
            fill="white",
            stroke_width=2,
            stroke_fill="black",
        )

    # The point cloud passed to the model is in the current EEF frame. Its
    # virtual-gripper tail is transformed back to world above and projected
    # here, so this is the exact current gripper input rather than a target-pose
    # approximation. The wireframe makes the four REAP boxes readable even
    # when most surface samples project to the same pixels.
    current_gripper_points_world = np.asarray(
        current_gripper_points_world
        if current_gripper_points_world is not None
        else np.empty((0, 3)),
        dtype=np.float32,
    ).reshape(-1, 3)
    if len(current_gripper_points_world):
        gripper_pixels, gripper_valid = project_world_points_to_front_image(
            current_gripper_points_world, observation
        )
        gripper_pixels = gripper_pixels * scale
        for x, y in gripper_pixels[gripper_valid]:
            draw.ellipse(
                (x - 1.5, y - 1.5, x + 1.5, y + 1.5),
                fill=(0, 235, 255),
            )

    box_edges = (
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    )
    valid_box_centers = []
    for box_world in current_gripper_boxes_world or []:
        box_pixels, box_valid = project_world_points_to_front_image(
            np.asarray(box_world, dtype=np.float32), observation
        )
        box_pixels = box_pixels * scale
        for start_index, end_index in box_edges:
            if box_valid[start_index] and box_valid[end_index]:
                segment = [
                    tuple(box_pixels[start_index]),
                    tuple(box_pixels[end_index]),
                ]
                draw.line(segment, fill=(0, 0, 0), width=5)
                draw.line(segment, fill=(0, 235, 255), width=3)
        if np.any(box_valid):
            valid_box_centers.append(box_pixels[box_valid].mean(axis=0))
    if valid_box_centers:
        label_position = np.mean(valid_box_centers, axis=0)
        draw.text(
            (float(label_position[0]) + 6, float(label_position[1]) + 6),
            "current virtual gripper",
            fill=(0, 235, 255),
            stroke_width=2,
            stroke_fill="black",
        )

    draw.rectangle((0, 0, target_width, 62), fill=(0, 0, 0))
    draw.text((8, 5), "Solid path = current execution window", fill="white")
    draw.text((8, 23), "Dashed gray path = future forecast", fill="white")
    draw.text((8, 41), "Cyan wireframe = current virtual gripper", fill=(0, 235, 255))
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
    return position_pixels, position_valid


def save_action_chunk_visualization(
    run_dir,
    episode_index,
    frame_index,
    model_call,
    observation,
    model_observation,
    chunk_anchor_world,
    chunk,
    foreground_snapshot,
    args,
):
    """Save one PLY and front-image overlay for a predicted action chunk."""
    output_dir = run_dir / "action_visualizations" / ("episode_" + str(episode_index).zfill(3))
    stem = "frame_" + str(frame_index).zfill(6) + "_model_call_" + str(model_call).zfill(4)
    ply_path = output_dir / (stem + "_" + args.action_vis_point_mode + ".ply")
    image_path = output_dir / (stem + ".png")

    execution_start, execution_stop = action_execution_window(len(chunk), args)
    (
        xyz,
        rgb,
        foreground_scores,
        source_indices,
        point_kind,
        action_indices,
        action_phase,
    ) = build_action_chunk_ply_cloud(
        model_observation["point_cloud"],
        chunk,
        args.action_vis_max_points,
        foreground_snapshot=foreground_snapshot,
        point_mode=args.action_vis_point_mode,
        execution_start=execution_start,
        execution_stop=execution_stop,
    )
    write_colored_ply(
        ply_path,
        xyz,
        rgb,
        foreground_scores=foreground_scores,
        source_indices=source_indices,
        point_kind=point_kind,
        action_indices=action_indices,
        action_phase=action_phase,
        point_mode=args.action_vis_point_mode,
        header_comments=(
            "selected_gripper_len_m " + format(float(args.gripper_len), ".9g"),
            "aligned_gripper_len_m "
            + format(float(RLBENCH_REAP_ALIGNED_GRIPPER_LEN), ".9g"),
            "physical_eef_to_virtual_tcp_local_translation_m "
            + " ".join(
                format(float(value), ".9g")
                for value in physical_eef_to_model_tcp(args)[:3, 3]
            ),
            "virtual_tcp_sync " + str(bool(args.sync_virtual_gripper_tcp)).lower(),
        ),
    )

    relative_targets = pose9_to_homo_np(chunk[:, :9])
    world_targets = chunk_anchor_world @ relative_targets
    current_gripper_points_world, current_gripper_boxes_world = (
        current_virtual_gripper_world_geometry(
            observation, model_observation, args
        )
    )
    draw_action_chunk_on_front_image(
        image_path,
        observation,
        world_targets,
        args.action_vis_image_width,
        execution_start,
        execution_stop,
        current_gripper_points_world=current_gripper_points_world,
        current_gripper_boxes_world=current_gripper_boxes_world,
    )
    valid_scores = foreground_scores[np.isfinite(foreground_scores)]
    return {
        "frame_index": int(frame_index),
        "ply": str(ply_path),
        "image": str(image_path),
        "point_mode": str(args.action_vis_point_mode),
        "foreground_scores": int(len(valid_scores)),
        "foreground_score_mean": float(valid_scores.mean()) if len(valid_scores) else None,
        "foreground_score_p90": float(np.quantile(valid_scores, 0.90)) if len(valid_scores) else None,
        "current_execution_window": [int(execution_start), int(execution_stop)],
        "target_gripper_geometry": False,
        "current_gripper_geometry": bool(len(current_gripper_points_world)),
        "current_gripper_points": int(len(current_gripper_points_world)),
        "current_gripper_template": str(args.gripper_template),
        "current_gripper_length_m": (
            float(args.gripper_len)
            if args.gripper_template == "reap"
            else 0.0
        ),
        "model_tcp_sync": bool(
            args.gripper_template == "reap" and args.sync_virtual_gripper_tcp
        ),
        "physical_eef_to_model_tcp_translation_m": (
            physical_eef_to_model_tcp(args)[:3, 3].tolist()
        ),
    }


def _execution_array(values, columns, dtype, label):
    """Return a stable two-dimensional array, including for zero-step episodes."""
    array = np.asarray(values, dtype=dtype)
    if array.size == 0:
        return np.empty((0, int(columns)), dtype=dtype)
    if array.ndim != 2 or array.shape[1] != int(columns):
        raise ValueError(
            "Expected " + str(label) + " to have shape (N, "
            + str(columns) + "), got " + str(array.shape)
        )
    return array


def save_episode_executed_action_alignment(
    run_dir,
    episode_index,
    model_actions,
    controller_actions,
    execution_indices,
):
    """Save exact model/controller pairs for every successful simulator step."""
    model_array = _execution_array(
        model_actions, 10, np.float32, "executed model actions"
    )
    controller_array = _execution_array(
        controller_actions, 8, np.float32, "executed controller actions"
    )
    index_array = _execution_array(
        execution_indices, 6, np.int64, "executed action indices"
    )
    if not len(model_array) == len(controller_array) == len(index_array):
        raise RuntimeError(
            "Executed-action alignment length mismatch for episode "
            + str(episode_index)
            + ": model=" + str(len(model_array))
            + " controller=" + str(len(controller_array))
            + " index=" + str(len(index_array))
        )

    output_dir = run_dir / "executed_action_alignment"
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = "episode_" + str(episode_index).zfill(3)
    model_path = output_dir / (
        prefix + "_executed_model_actions_relative10.npy"
    )
    controller_path = output_dir / (
        prefix + "_executed_controller_actions_world8.npy"
    )
    pair_path = output_dir / (prefix + "_executed_model10_controller8.npy")
    index_path = output_dir / (prefix + "_execution_index.npy")
    pair_array = np.concatenate((model_array, controller_array), axis=1)
    np.save(model_path, model_array)
    np.save(controller_path, controller_array)
    np.save(pair_path, pair_array)
    np.save(index_path, index_array)
    return {
        "executed_model_actions_relative10": str(model_path.relative_to(run_dir)),
        "executed_controller_actions_world8": str(
            controller_path.relative_to(run_dir)
        ),
        "executed_model10_controller8": str(pair_path.relative_to(run_dir)),
        "execution_index": str(index_path.relative_to(run_dir)),
    }


def finalize_executed_action_alignment(run_dir, results):
    """Write task-level concatenations and documentation for automatic exports."""
    output_dir = run_dir / "executed_action_alignment"
    output_dir.mkdir(parents=True, exist_ok=True)
    all_model = []
    all_controller = []
    all_pairs = []
    all_indices = []
    episode_summaries = []
    for result in sorted(results, key=lambda item: int(item["episode_index"])):
        episode_index = int(result["episode_index"])
        prefix = "episode_" + str(episode_index).zfill(3)
        model_array = np.asarray(
            np.load(
                output_dir
                / (prefix + "_executed_model_actions_relative10.npy")
            ),
            dtype=np.float32,
        )
        controller_array = np.asarray(
            np.load(
                output_dir
                / (prefix + "_executed_controller_actions_world8.npy")
            ),
            dtype=np.float32,
        )
        pair_array = np.asarray(
            np.load(output_dir / (prefix + "_executed_model10_controller8.npy")),
            dtype=np.float32,
        )
        index_array = np.asarray(
            np.load(output_dir / (prefix + "_execution_index.npy")),
            dtype=np.int64,
        )
        if not (
            len(model_array)
            == len(controller_array)
            == len(pair_array)
            == len(index_array)
        ):
            raise RuntimeError(
                "Cannot aggregate misaligned executed actions for episode "
                + str(episode_index)
            )
        all_model.append(model_array)
        all_controller.append(controller_array)
        all_pairs.append(pair_array)
        all_indices.append(index_array)
        phase_counts = {
            name: int(np.sum(index_array[:, 3] == code))
            for code, name in EXECUTION_PHASE_CODES.items()
        }
        episode_summaries.append(
            {
                "episode_index": episode_index,
                "executed_environment_steps": int(len(controller_array)),
                "model_calls": int(result["model_calls"]),
                "execution_phase_counts": phase_counts,
            }
        )

    combined_model = (
        np.concatenate(all_model, axis=0)
        if all_model
        else np.empty((0, 10), dtype=np.float32)
    )
    combined_controller = (
        np.concatenate(all_controller, axis=0)
        if all_controller
        else np.empty((0, 8), dtype=np.float32)
    )
    combined_pairs = (
        np.concatenate(all_pairs, axis=0)
        if all_pairs
        else np.empty((0, 18), dtype=np.float32)
    )
    combined_indices = (
        np.concatenate(all_indices, axis=0)
        if all_indices
        else np.empty((0, 6), dtype=np.int64)
    )
    np.save(output_dir / "all_executed_model_actions_relative10.npy", combined_model)
    np.save(
        output_dir / "all_executed_controller_actions_world8.npy",
        combined_controller,
    )
    np.save(output_dir / "all_executed_model10_controller8.npy", combined_pairs)
    np.save(output_dir / "all_execution_index.npy", combined_indices)

    manifest = {
        "source_run_dir": str(run_dir),
        "episodes": int(len(episode_summaries)),
        "executed_environment_steps": int(len(combined_model)),
        "alignment": (
            "one row per successful controller action; phase 3 is a SONG "
            "gripper-only fallback that intentionally bypasses arm planning"
        ),
        "capture_method": (
            "captured online in the evaluator immediately after each simulator step; "
            "control.log floating-point text is not used"
        ),
        "model_action_columns": [
            "relative_x",
            "relative_y",
            "relative_z",
            "rotation_column_1_x",
            "rotation_column_1_y",
            "rotation_column_1_z",
            "rotation_column_2_x",
            "rotation_column_2_y",
            "rotation_column_2_z",
            "predicted_gripper_width_m",
        ],
        "controller_action_columns": [
            "world_x",
            "world_y",
            "world_z",
            "qx",
            "qy",
            "qz",
            "qw",
            "gripper_open_discrete",
        ],
        "execution_index_columns": [
            "episode_index",
            "model_call_1based",
            "chunk_row_index_0based",
            "execution_phase_code",
            "mover_attempt_1based_or_0",
            "environment_step_index_0based",
        ],
        "execution_phase_codes": {
            str(code): name for code, name in EXECUTION_PHASE_CODES.items()
        },
        "precision_note": (
            "Both action arrays are exact float32 values captured in memory. "
            "A model row is repeated for mover retries, deferred gripper steps, "
            "and adaptive-controller intermediate steps so rows remain aligned."
        ),
        "compatibility_outputs": {
            "actions": "actions/ contains the same simulator world8 arrays",
            "model_chunks": "model_chunks/ contains every predicted chunk, including unexecuted rows",
        },
        "episode_summaries": episode_summaries,
    }
    with open(output_dir / "manifest.json", "w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2, ensure_ascii=False)
        file.write("\n")
    readme = """# Executed model-action alignment

这里保存的是逐个仿真执行步严格对齐的两版 action：

- `*_executed_model_actions_relative10.npy`：产生该执行步的原始模型 10 维输出。
- `*_executed_controller_actions_world8.npy`：实际传给 `task_env.step()` 的 8 维世界坐标命令。
- `*_executed_model10_controller8.npy`：上述两者按列拼接，便于直接对照。
- `*_execution_index.npy`：episode/model_call/chunk_row/执行阶段/重试序号/执行步序号。
- `all_*`：按 episode 顺序拼接的任务级数组。

同一模型行因 Mover 重试、到位后单独操作夹爪或自适应控制中间步而执行多次时，
模型行会对应重复，从而保证两版 action 的第 i 行始终描述同一个仿真执行步。
完整字段定义见 `manifest.json`。
"""
    (output_dir / "README.md").write_text(readme, encoding="utf-8")
    return manifest


def run_episode(
    task_env,
    infer,
    args,
    episode_index,
    task_name,
    run_dir,
    dataset_reset_spec=None,
    completed_successes=0,
    completed_episodes=0,
):
    """Run exactly one reset-to-terminal RLBench rollout with no retries."""
    # Dataset-backed evaluation must be invariant to process sharding.  Use the
    # dataset-local episode id for every per-episode random stream; otherwise a
    # 50--99 shard silently reuses the noise and point-sampling seeds of 0--49.
    seed_episode_index = (
        int(episode_index)
        if dataset_reset_spec is None
        else int(dataset_reset_spec["local_episode_index"])
    )
    (
        descriptions,
        observation,
        dataset_reset_method,
        dataset_initial_state_validation,
    ) = reset_task_for_episode(task_env, args, episode_index, dataset_reset_spec)
    if args.simulation_timestep > 0.0:
        task_env._pyrep.set_simulation_timestep(args.simulation_timestep)
        if episode_index == 0:
            print(
                "[simulation] timestep_seconds="
                + str(task_env._pyrep.get_simulation_timestep()),
                flush=True,
            )
    language = descriptions[0] if descriptions else task_name.replace("_", " ")
    draw_pour_point = (
        args.draw_pour_point
        if args.draw_pour_point is not None
        else task_name == "water_plants"
    )
    draw_task_stages = (
        args.draw_task_stages
        if args.draw_task_stages is not None
        else task_name == "water_plants"
    )
    draw_phone_success_sensor = (
        args.draw_phone_success_sensor
        if args.draw_phone_success_sensor is not None
        else task_name == "phone_on_base"
    )
    initial_frame = video_front_rgb(
        task_env,
        observation.front_rgb,
        args.save_video and args.video_refresh_rgb,
    )
    if draw_pour_point and not draw_task_stages:
        initial_frame = draw_pour_point_on_front_image(
            initial_frame, observation, task_env
        )
    if draw_task_stages:
        initial_frame = draw_water_plants_stage_overlay(
            initial_frame, observation, task_env
        )
    if draw_phone_success_sensor:
        initial_frame = draw_phone_success_sensor_overlay(
            initial_frame, observation, task_env
        )
    raw_frames = [initial_frame.copy()]
    frames = [
        annotate_video_frame(
            initial_frame,
            task_name,
            episode_index,
            frame_index=0,
            physics_frame_index=0,
            model_call=0,
        )
    ]
    physics_frame_count = 0

    executed_actions = []
    # These three arrays are the canonical, online record of successful
    # task_env.step(command) calls.  Keep them separate from video bookkeeping
    # and from complete predicted chunks, which may contain unexecuted rows.
    executed_simulator_actions = []
    executed_model_actions = []
    executed_action_indices = []
    predicted_chunks = []
    model_calls = 0
    base_model_call_budget = int(args.max_model_calls)
    effective_model_call_budget = int(base_model_call_budget)
    max_model_call_compensations = (
        int(base_model_call_budget)
        if int(args.max_model_call_compensations) < 0
        else int(args.max_model_call_compensations)
    )
    model_call_compensations = 0
    model_call_compensation_events = []
    consecutive_planning_failure_calls = 0
    policy_action_steps_attempted = 0
    termination = False
    success = False
    error = None
    end_reason = None
    limited_action_count = 0
    jacobian_intermediate_action_count = 0
    jacobian_recursive_midpoint_count = 0
    threshold_midpoint_count = 0
    reached_waypoint_count = 0
    workspace_clipped_action_count = 0
    mover_target_count = 0
    mover_reached_target_count = 0
    mover_attempt_count = 0
    mover_retry_count = 0
    mover_unreached_target_count = 0
    mover_unreached_continued_count = 0
    mover_unreached_continue_events = []
    gripper_after_reach_action_count = 0
    song_failed_chunk_row_count = 0
    song_all_rows_failed_call_count = 0
    song_gripper_fallback_count = 0
    song_gripper_fallback_failure_count = 0
    song_gripper_fallback_events = []
    rejected_chunk_count = 0
    jacobian_failures = []
    controller_continue_errors = 0
    skipped_controller_waypoints = 0
    controller_continue_failures = []
    control_error_reinferences = 0
    discarded_chunk_rows = 0
    control_error_reinference_events = []
    action_visualizations = []
    action_chunk_paths = []
    frame_pointcloud_count = 0
    frame_pointcloud_dir = (
        run_dir
        / "frame_pointclouds"
        / ("episode_" + str(episode_index).zfill(3))
    )
    determinism_dir = (
        run_dir
        / "determinism_diagnostics"
        / ("episode_" + str(episode_index).zfill(3))
    )
    determinism_state_log = determinism_dir / "executed_states.jsonl"
    determinism_manifest_path = determinism_dir / "manifest.json"
    determinism_state_count = 0
    determinism_manifest = {
        "task": task_name,
        "episode_index": int(episode_index),
        "seed_episode_index": int(seed_episode_index),
        "point_sampling_seed_first_call": (
            int(args.seed) * 100000 + int(seed_episode_index) * 1000
        ),
        "model_noise_seed_first_call": (
            None
            if args.model_noise_seed is None
            else int(args.model_noise_seed) + int(seed_episode_index) * 100000
        ),
    }
    if args.save_determinism_diagnostics:
        determinism_dir.mkdir(parents=True, exist_ok=False)
    next_action_visualization_frame = 0
    # Keep the discrete command across model chunks.  The measured gripper
    # state can lag the fake attach/release operation by a few simulation steps.
    gripper_command_open = float(observation.gripper_open) > 0.5
    # Legacy libero_delta carries this value between chunks. The strict
    # delta_width_initial_sync protocol overwrites it with each chunk's first
    # executed row, after one physical episode-start synchronization.
    previous_predicted_width = observed_gripper_width(observation)
    gripper_initial_sync_applied = False
    gripper_initial_sync_info = None
    gripper_transition_count = 0
    # A gripper transition is not sent until its EEF waypoint is reached. If
    # that waypoint exhausts Mover retries and triggers replanning, retain the
    # transition and apply it to a reached waypoint from the new chunk.
    pending_gripper_open = None
    pending_gripper_origin = None
    pending_gripper_stored_count = 0
    pending_gripper_applied_count = 0
    pending_gripper_cancelled_count = 0
    # This latch records a close command issued during this episode. The reset
    # state being closed is not itself a close event.
    gripper_closed_latched = False
    episode_started_at = time.monotonic()
    close_laptop_contact_seek_active = False

    def append_determinism_state(
        post_observation,
        simulator_action,
        execution_phase_code,
        mover_attempt,
        current_model_call,
        current_chunk_row,
    ):
        nonlocal determinism_state_count
        if not args.save_determinism_diagnostics:
            return
        task = getattr(task_env, "_task", None)
        water_snapshot = _water_plants_snapshot(task_env)
        state_values = finite_json_value({
            "eef_pose7_world": np.asarray(
                post_observation.gripper_pose, dtype=np.float64
            ).tolist(),
            "gripper_open_fraction": float(post_observation.gripper_open),
            "gripper_width_m": float(observed_gripper_width(post_observation)),
            "gripper_joint_positions": np.asarray(
                getattr(post_observation, "gripper_joint_positions", []),
                dtype=np.float64,
            ).tolist(),
            "waterer_pose7_world": object_pose7_or_none(
                getattr(task, "waterer", None)
            ),
            "waterer_head_pose7_world": object_pose7_or_none(
                getattr(task, "head", None)
            ),
            "pour_point_pose7_world": object_pose7_or_none(
                getattr(task, "pour_point", None)
            ),
            "success_sensor_pose7_world": object_pose7_or_none(
                getattr(task, "success_sensor", None)
            ),
            "water_task": water_snapshot,
        })
        state_bytes = json.dumps(
            state_values, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
        record = finite_json_value({
            "simulator_step_index": int(determinism_state_count),
            "model_call": int(current_model_call),
            "chunk_row_index": int(current_chunk_row),
            "execution_phase_code": int(execution_phase_code),
            "mover_attempt": int(mover_attempt),
            "simulator_action_world8": np.asarray(
                simulator_action, dtype=np.float64
            ).tolist(),
            "simulator_action_sha256": array_sha256(simulator_action),
            "state_sha256": hashlib.sha256(state_bytes).hexdigest(),
            **state_values,
        })
        with open(determinism_state_log, "a", encoding="utf-8") as file:
            file.write(json.dumps(record, separators=(",", ":"), allow_nan=False))
            file.write("\n")
        determinism_state_count += 1

    def save_frame_pointcloud(frame_observation, frame_index):
        """Persist exactly the sampled XYZRGB cloud used by the model input adapter."""
        nonlocal frame_pointcloud_count
        if not args.save_frame_pointclouds:
            return
        if int(frame_index) % int(args.frame_pointcloud_every_n_frames) != 0:
            return
        point_seed = (
            int(args.seed) * 100000
            + int(seed_episode_index) * 1000
            + int(frame_index)
        )
        frame_model_observation = live_model_observation(
            frame_observation,
            args,
            seed=point_seed,
        )
        path = frame_pointcloud_dir / (
            "frame_" + str(frame_index).zfill(6) + "_model_input_raw.ply"
        )
        write_model_input_ply_binary(
            path,
            frame_model_observation["point_cloud"],
            comments=(
                "coordinate_frame current_eef",
                "source live_rlbench_model_input",
                "task " + str(task_name),
                "episode_index " + str(episode_index),
                "video_frame_index " + str(frame_index),
                "point_sampling_seed " + str(point_seed),
                "pointseg_probabilities unavailable_no_model_call_at_every_frame",
            ),
        )
        frame_pointcloud_count += 1

    save_frame_pointcloud(observation, 0)

    def print_episode_progress(next_model_call: int, stage: str) -> None:
        """Print a lightweight single-episode progress bar without extra I/O."""
        total = max(int(effective_model_call_budget), 1)
        completed = min(max(int(next_model_call), 0), total)
        width = 20
        filled = int(round(width * completed / total))
        bar = "#" * filled + "." * (width - filled)
        elapsed = time.monotonic() - episode_started_at
        print(
            "[eval-trajectory] task="
            + task_name
            + " episode="
            + str(episode_index + 1)
            + " ["
            + bar
            + "] "
            + str(completed)
            + "/"
            + str(total)
            + " model_calls frame="
            + str(len(executed_actions))
            + " success_rate="
            + str(int(completed_successes))
            + "/"
            + str(int(completed_episodes))
            + " elapsed_s="
            + format(elapsed, ".1f")
            + " stage="
            + stage,
            flush=True,
        )

    try:
        while (
            model_calls < effective_model_call_budget
            and not termination
            and not success
            and (
                args.max_policy_action_steps <= 0
                or policy_action_steps_attempted < args.max_policy_action_steps
            )
        ):
            print_episode_progress(model_calls + 1, "model_start")
            print(
                "[eval-frame] "
                + json.dumps(
                    {
                        "stage": "model_start",
                        "task": task_name,
                        "episode": int(episode_index),
                        "frame": int(len(executed_actions)),
                        "physics_frame": int(physics_frame_count),
                        "model_call": int(model_calls + 1),
                    },
                    separators=(",", ":"),
                ),
                flush=True,
            )
            # The policy's current frame is the virtual TCP. At the calibrated
            # 0.09 m length this is exactly Panda_tip; otherwise it moves with
            # the selected virtual gripper and model outputs are conjugated
            # back to Panda_tip before controller execution.
            chunk_anchor_world = observation_model_tcp_world(observation, args)
            model_observation = live_model_observation(
                observation,
                args,
                seed=args.seed * 100000
                + seed_episode_index * 1000
                + model_calls,
            )
            if args.save_determinism_diagnostics and model_calls == 0:
                reset_front_rgb_raw = np.asarray(observation.front_rgb)
                first_model_rgb = np.asarray(model_observation["front"])
                first_model_point_cloud = np.asarray(
                    model_observation["point_cloud"]
                )
                first_model_state = np.asarray(model_observation["state"])
                task = getattr(task_env, "_task", None)
                np.savez_compressed(
                    determinism_dir / "first_observation_and_model_input.npz",
                    reset_front_rgb_raw=reset_front_rgb_raw,
                    model_front_rgb=first_model_rgb,
                    model_point_cloud=first_model_point_cloud,
                    model_state=first_model_state,
                    reset_eef_pose7_world=np.asarray(
                        observation.gripper_pose, dtype=np.float64
                    ),
                    reset_model_virtual_tcp_pose7_world=np.asarray(
                        matrix_to_pose7(chunk_anchor_world), dtype=np.float64
                    ),
                    physical_eef_to_model_tcp=np.asarray(
                        physical_eef_to_model_tcp(args), dtype=np.float64
                    ),
                    reset_gripper_width_m=np.asarray(
                        observed_gripper_width(observation), dtype=np.float64
                    ),
                    reset_waterer_pose7_world=np.asarray(
                        object_pose7_or_none(getattr(task, "waterer", None))
                        or [],
                        dtype=np.float64,
                    ),
                    reset_waterer_head_pose7_world=np.asarray(
                        object_pose7_or_none(getattr(task, "head", None)) or [],
                        dtype=np.float64,
                    ),
                )
                determinism_manifest.update(
                    {
                        "reset_front_rgb_raw_sha256": array_sha256(
                            reset_front_rgb_raw
                        ),
                        "model_front_rgb_sha256": array_sha256(first_model_rgb),
                        "model_point_cloud_sha256": array_sha256(
                            first_model_point_cloud
                        ),
                        "model_state_sha256": array_sha256(first_model_state),
                        "first_observation_file": (
                            "first_observation_and_model_input.npz"
                        ),
                    }
                )
                with open(
                    determinism_manifest_path, "w", encoding="utf-8"
                ) as file:
                    json.dump(determinism_manifest, file, indent=2)
            print_control_log(
                "[model-state]",
                {
                    "model_call": int(model_calls + 1),
                    "actual_state_world_pose7": np.asarray(
                        observation.gripper_pose, dtype=np.float32
                    ).tolist(),
                    "actual_gripper_open": float(observation.gripper_open),
                    "model_state10": np.asarray(
                        model_observation["state"], dtype=np.float32
                    ).tolist(),
                    "chunk_anchor_world_pose7": matrix_to_pose7(
                        chunk_anchor_world
                    ).tolist(),
                    "chunk_anchor_semantics": "model_virtual_tcp",
                    "physical_eef_to_model_tcp_translation_m": (
                        physical_eef_to_model_tcp(args)[:3, 3].tolist()
                    ),
                },
                args.log_control_details,
            )
            chunk = infer.predict_action_chunk_obs(
                model_observation,
                task=language,
                postprocess=True,
                state_pose_mode="identity",
                noise_seed=(
                    None
                    if args.model_noise_seed is None
                    else int(args.model_noise_seed)
                    + seed_episode_index * 100000
                    + int(model_calls)
                ),
            )
            chunk = np.asarray(chunk[0].detach().cpu(), dtype=np.float32)
            if chunk.ndim != 2 or chunk.shape[1] < 10:
                raise ValueError("Expected policy action chunk with shape (T, 10).")
            if args.save_determinism_diagnostics and model_calls == 0:
                np.save(determinism_dir / "first_action_chunk.npy", chunk)
                determinism_manifest.update(
                    {
                        "first_action_chunk_sha256": array_sha256(chunk),
                        "first_action_chunk_shape": list(chunk.shape),
                        "first_action_chunk_file": "first_action_chunk.npy",
                    }
                )
                with open(
                    determinism_manifest_path, "w", encoding="utf-8"
                ) as file:
                    json.dump(determinism_manifest, file, indent=2)
            predicted_chunks.append(chunk.copy())
            model_calls += 1
            executed_chunk_rows_this_call = 0
            attempted_chunk_rows_this_call = 0
            failed_chunk_rows_this_call = 0
            planning_failure_this_call = False
            planning_failure_reasons_this_call = []

            control_frame_index = len(executed_actions)
            print(
                "[eval-frame] "
                + json.dumps(
                    {
                        "stage": "chunk_ready",
                        "task": task_name,
                        "episode": int(episode_index),
                        "frame": int(control_frame_index),
                        "physics_frame": int(physics_frame_count),
                        "model_call": int(model_calls),
                    },
                    separators=(",", ":"),
                ),
                flush=True,
            )
            if args.save_action_chunks:
                chunk_path = (
                    run_dir
                    / "action_chunks"
                    / ("episode_" + str(episode_index).zfill(3))
                    / (
                        "frame_"
                        + str(control_frame_index).zfill(6)
                        + "_model_call_"
                        + str(model_calls).zfill(4)
                        + ".npy"
                    )
                )
                chunk_path.parent.mkdir(parents=True, exist_ok=True)
                np.save(chunk_path, chunk)
                action_chunk_paths.append(str(chunk_path))
            if (
                args.save_action_visualizations
                and control_frame_index >= next_action_visualization_frame
            ):
                try:
                    visualization = save_action_chunk_visualization(
                        run_dir,
                        episode_index,
                        control_frame_index,
                        model_calls,
                        observation,
                        model_observation,
                        chunk_anchor_world,
                        chunk,
                        getattr(infer.policy.model, "last_pointseg_visualization", None),
                        args,
                    )
                    action_visualizations.append(visualization)
                    print("[action-visualization] " + json.dumps(visualization), flush=True)
                except Exception as visualization_error:
                    print("[action-visualization-warning] " + repr(visualization_error), flush=True)
                while next_action_visualization_frame <= control_frame_index:
                    next_action_visualization_frame += args.action_vis_every_n_frames

            start = min(max(args.action_index, 0), len(chunk) - 1)
            stop = min(len(chunk), start + max(args.exec_action_steps, 1))
            if args.max_policy_action_steps > 0:
                remaining_policy_steps = max(
                    int(args.max_policy_action_steps)
                    - int(policy_action_steps_attempted),
                    0,
                )
                stop = min(stop, start + remaining_policy_steps)
            if args.gripper_mode == DELTA_WIDTH_INITIAL_SYNC:
                if not gripper_initial_sync_applied:
                    gripper_initial_sync_info = (
                        set_gripper_absolute_width_position_target(
                            task_env,
                            float(chunk[0, 9]),
                            RLBENCH_PANDA_MAX_WIDTH,
                        )
                    )
                    gripper_initial_sync_applied = True
                    observation = task_env.get_observation()
                    gripper_command_open = bool(
                        gripper_initial_sync_info["command_gripper_open"]
                    )
                    print(
                        "[gripper-initial-sync] "
                        + json.dumps(
                            {
                                "episode": int(episode_index),
                                "model_call": int(model_calls),
                                **gripper_initial_sync_info,
                            },
                            separators=(",", ":"),
                        ),
                        flush=True,
                    )
                # Every freshly predicted chunk is isolated from the previous
                # chunk: its first actually executed row has delta exactly 0.
                previous_predicted_width = initial_delta_reference_for_chunk(
                    chunk, start
                )
            elif model_calls == 1 and start > 0:
                # When execution starts inside a chunk, the preceding model
                # row is the legacy delta reference for the first selected row.
                previous_predicted_width = float(chunk[start - 1, 9])
            continue_after_controller_error = False
            for chunk_row_index in range(start, stop):
                policy_action_steps_attempted += 1
                attempted_chunk_rows_this_call += 1
                row = chunk[chunk_row_index]
                target_eeft = pose9_to_homo_np(row[:9])
                raw_requested_model_tcp_world = chunk_anchor_world @ target_eeft
                raw_requested_world = model_tcp_world_to_physical_eef_world(
                    raw_requested_model_tcp_world, args
                )
                (
                    requested_world,
                    workspace_clipped,
                    raw_requested_xyz,
                ) = clip_world_target_to_pointact_workspace(
                    raw_requested_world, args.clip_within_workspace
                )
                if workspace_clipped:
                    workspace_clipped_action_count += 1
                    print_control_log(
                        "[workspace-clip]",
                        {
                            "episode_index": int(episode_index),
                            "model_call": int(model_calls),
                            "chunk_row_index": int(chunk_row_index),
                            "requested_xyz": raw_requested_xyz.tolist(),
                            "clipped_xyz": requested_world[:3, 3].tolist(),
                            "workspace_min": POINTACT_WORKSPACE_MIN.tolist(),
                            "workspace_max": POINTACT_WORKSPACE_MAX.tolist(),
                        },
                        args.log_control_details,
                    )
                previous_command_open = bool(gripper_command_open)
                gripper_closed_latched_before_row = bool(gripper_closed_latched)
                if args.gripper_mode == ABSOLUTE_WIDTH:
                    target_gripper_open, gripper_event = absolute_width_gripper_target(
                        row[9], args.gripper_open_threshold
                    )
                    width_change = float(row[9]) - float(previous_predicted_width)
                else:
                    (
                        target_gripper_open,
                        gripper_event,
                        width_change,
                    ) = libero_style_gripper_target(
                        previous_predicted_width,
                        row[9],
                        gripper_command_open,
                        args.gripper_delta_threshold,
                        alignment=args.gripper_delta_alignment,
                        next_width=(
                            chunk[chunk_row_index + 1, 9]
                            if chunk_row_index + 1 < len(chunk)
                            else None
                        ),
                        open_threshold=args.gripper_delta_open_threshold,
                        close_threshold=args.gripper_delta_close_threshold,
                    )
                    pending_before_decode = pending_gripper_open
                    (
                        target_gripper_open,
                        gripper_event,
                        pending_gripper_open,
                        pending_gripper_replayed,
                    ) = resolve_pending_delta_gripper_target(
                        target_gripper_open,
                        gripper_event,
                        pending_gripper_open,
                    )
                    if (
                        pending_before_decode is not None
                        and pending_gripper_open is None
                    ):
                        pending_gripper_cancelled_count += 1
                        print_control_log(
                            "[gripper-pending-cancelled]",
                            {
                                "episode_index": int(episode_index),
                                "model_call": int(model_calls),
                                "chunk_row_index": int(chunk_row_index),
                                "pending_target_gripper_open": bool(
                                    pending_before_decode
                                ),
                                "replacement_event": gripper_event,
                                "replacement_target_gripper_open": bool(
                                    target_gripper_open
                                ),
                            },
                            args.log_control_details,
                        )
                if args.gripper_mode == ABSOLUTE_WIDTH:
                    pending_gripper_replayed = False
                if args.gripper_lock_after_close:
                    if gripper_closed_latched:
                        if target_gripper_open:
                            gripper_event = "lock_after_close"
                        target_gripper_open = False
                    elif previous_command_open and not target_gripper_open:
                        # Do not treat a reset state that is already closed as
                        # a close event. Latch only on an actual open->closed
                        # command transition in this episode.
                        gripper_closed_latched = True
                gripper_state_changed = bool(
                    target_gripper_open != previous_command_open
                )
                gripper_transition_requires_reach = bool(
                    gripper_state_changed
                    and (
                        args.gripper_open_require_reach
                        if target_gripper_open
                        else args.gripper_close_require_reach
                    )
                )
                if (
                    task_name == "close_laptop_lid"
                    and gripper_state_changed
                    and previous_command_open
                    and not target_gripper_open
                ):
                    close_laptop_contact_seek_active = True
                laptop_contact_offset_applied = bool(
                    task_name == "close_laptop_lid"
                    and gripper_state_changed
                    and previous_command_open
                    and not target_gripper_open
                    and float(args.close_laptop_contact_z_offset_m) != 0.0
                )
                if laptop_contact_offset_applied:
                    requested_world = np.asarray(
                        requested_world, dtype=np.float64
                    ).copy()
                    requested_world[2, 3] += float(
                        args.close_laptop_contact_z_offset_m
                    )
                    print_control_log(
                        "[close-laptop-contact-offset]",
                        {
                            "episode_index": int(episode_index),
                            "model_call": int(model_calls),
                            "chunk_row_index": int(chunk_row_index),
                            "world_z_offset_m": float(
                                args.close_laptop_contact_z_offset_m
                            ),
                            "offset_target_state_world_pose7": matrix_to_pose7(
                                requested_world
                            ).tolist(),
                        },
                        args.log_control_details,
                    )
                laptop_contact_seek_applied = False
                if (
                    task_name == "close_laptop_lid"
                    and close_laptop_contact_seek_active
                    and not target_gripper_open
                    and float(args.close_laptop_contact_seek_z_offset_m) != 0.0
                ):
                    laptop_diagnostics = collect_close_laptop_diagnostics(
                        task_env, task_name
                    )
                    lid_displacement = float(
                        laptop_diagnostics.get(
                            "laptop_joint_displacement_rad", 0.0
                        )
                    )
                    if lid_displacement >= float(
                        args.close_laptop_contact_established_rad
                    ):
                        close_laptop_contact_seek_active = False
                    else:
                        requested_world = np.asarray(
                            requested_world, dtype=np.float64
                        ).copy()
                        requested_world[2, 3] += float(
                            args.close_laptop_contact_seek_z_offset_m
                        )
                        laptop_contact_seek_applied = True
                        print_control_log(
                            "[close-laptop-contact-seek]",
                            {
                                "episode_index": int(episode_index),
                                "model_call": int(model_calls),
                                "chunk_row_index": int(chunk_row_index),
                                "world_z_offset_m": float(
                                    args.close_laptop_contact_seek_z_offset_m
                                ),
                                "lid_displacement_rad": lid_displacement,
                                "contact_established_threshold_rad": float(
                                    args.close_laptop_contact_established_rad
                                ),
                            },
                            args.log_control_details,
                        )
                print_control_log(
                    "[model-action]",
                    {
                        "model_call": int(model_calls),
                        "chunk_row_index": int(chunk_row_index),
                        "model_action_relative10": row[:10].tolist(),
                        "raw_target_model_tcp_world_pose7": matrix_to_pose7(
                            raw_requested_model_tcp_world
                        ).tolist(),
                        "raw_target_state_world_pose7": matrix_to_pose7(
                            raw_requested_world
                        ).tolist(),
                        "target_state_world_pose7": matrix_to_pose7(
                            requested_world
                        ).tolist(),
                        "workspace_clipped": bool(workspace_clipped),
                        "target_gripper_width_m": float(row[9]),
                        "previous_predicted_gripper_width_m": float(previous_predicted_width),
                        "predicted_gripper_width_change_m": float(width_change),
                        "gripper_delta_alignment": args.gripper_delta_alignment,
                        "gripper_protocol": args.gripper_mode,
                        "gripper_event": gripper_event,
                        "pending_gripper_replayed": bool(
                            pending_gripper_replayed
                        ),
                        "pending_target_gripper_open": (
                            None
                            if pending_gripper_open is None
                            else bool(pending_gripper_open)
                        ),
                        "target_gripper_open": bool(target_gripper_open),
                        "gripper_state_changed": gripper_state_changed,
                        "gripper_transition_requires_reach": bool(
                            gripper_transition_requires_reach
                        ),
                        "close_laptop_contact_z_offset_applied": (
                            laptop_contact_offset_applied
                        ),
                        "close_laptop_contact_seek_applied": (
                            laptop_contact_seek_applied
                        ),
                        "command_gripper_discrete": int(bool(target_gripper_open)),
                        "gripper_closed_latched": bool(gripper_closed_latched),
                    },
                    args.log_control_details,
                )
                diagnostic_context = {
                    "task_name": task_name,
                    "episode_index": int(episode_index),
                    "model_call": int(model_calls),
                    "chunk_row_index": int(chunk_row_index),
                    "raw_requested_target_model_tcp_world_pose7": matrix_to_pose7(
                        raw_requested_model_tcp_world
                    ).tolist(),
                    "raw_requested_target_state_world_pose7": matrix_to_pose7(
                        raw_requested_world
                    ).tolist(),
                    "requested_target_state_world_pose7": matrix_to_pose7(
                        requested_world
                    ).tolist(),
                    "workspace_clipped": bool(workspace_clipped),
                    "target_gripper_width_m": float(row[9]),
                    "target_gripper_open": bool(target_gripper_open),
                    "gripper_transition_requires_reach": bool(
                        gripper_transition_requires_reach
                    ),
                }

                def record_executed_action(
                    simulator_action,
                    execution_phase_code,
                    mover_attempt,
                    post_observation,
                ):
                    """Capture one model/controller pair only after step() returns."""
                    executed_simulator_actions.append(
                        np.asarray(simulator_action, dtype=np.float32).copy()
                    )
                    executed_model_actions.append(
                        np.asarray(row[:10], dtype=np.float32).copy()
                    )
                    executed_action_indices.append(
                        np.asarray(
                            [
                                int(episode_index),
                                int(model_calls),
                                int(chunk_row_index),
                                int(execution_phase_code),
                                int(mover_attempt),
                                int(len(executed_simulator_actions) - 1),
                            ],
                            dtype=np.int64,
                        )
                    )
                    try:
                        append_determinism_state(
                            post_observation=post_observation,
                            simulator_action=simulator_action,
                            execution_phase_code=execution_phase_code,
                            mover_attempt=mover_attempt,
                            current_model_call=model_calls,
                            current_chunk_row=chunk_row_index,
                        )
                    except Exception as diagnostic_error:
                        # Diagnostics must never change the controller result.
                        if args.save_determinism_diagnostics:
                            with open(
                                determinism_dir / "diagnostic_errors.jsonl",
                                "a",
                                encoding="utf-8",
                            ) as diagnostic_error_file:
                                diagnostic_error_file.write(
                                    json.dumps(
                                        {
                                            "model_call": int(model_calls),
                                            "chunk_row_index": int(chunk_row_index),
                                            "error": repr(diagnostic_error),
                                        },
                                        separators=(",", ":"),
                                    )
                                    + "\n"
                                )

                if gripper_state_changed:
                    gripper_transition_count += 1
                previous_predicted_width = float(row[9])
                gripper_command_open = bool(target_gripper_open)
                if args.execution_mode in {"dataset_step", "bounded_step"}:
                    command_world = requested_world
                    was_limited = False
                    if args.execution_mode == "bounded_step":
                        current_world = pose9_to_homo_np(
                            pose7_to_pose9(observation.gripper_pose)
                        )
                        command_world, was_limited = limit_absolute_eef_target(
                            current_world,
                            requested_world,
                            args.max_eef_position_step,
                            args.max_eef_rotation_step,
                        )
                    try:
                        step_result = execute_dataset_target_with_pointact_mover(
                            task_env=task_env,
                            observation=observation,
                            command_world=command_world,
                            target_gripper_open=target_gripper_open,
                            previous_gripper_open=previous_command_open,
                            was_limited=was_limited,
                            args=args,
                            diagnostic_context=diagnostic_context,
                            execution_recorder=record_executed_action,
                        )
                    except Exception as control_error:
                        planning_failure_this_call = True
                        planning_failure_reasons_this_call.append(
                            "controller_exception"
                        )
                        try:
                            observation = task_env.get_observation()
                        except Exception:
                            # A controller rejection normally happens before a
                            # simulation step. Keep the last valid observation
                            # if RLBench cannot refresh it here.
                            pass
                        print_control_log(
                            "[control-error]",
                            {
                                "episode_index": int(episode_index),
                                "model_call": int(model_calls),
                                "chunk_row_index": int(chunk_row_index),
                                "actual_state_world_pose7": np.asarray(
                                    observation.gripper_pose, dtype=np.float32
                                ).tolist(),
                                "command_target_world_pose7": matrix_to_pose7(
                                    command_world
                                ).tolist(),
                                "error": repr(control_error),
                            },
                            args.log_control_details,
                        )
                        controller_continue_errors += 1
                        skipped_controller_waypoints += 1
                        rejected_chunk_count += 1
                        controller_continue_failures.append(
                            {
                                "model_call": int(model_calls),
                                "chunk_row_index": int(chunk_row_index),
                                "error": repr(control_error),
                            }
                        )
                        if args.song_continue_failed_chunk_rows:
                            failed_chunk_rows_this_call += 1
                            song_failed_chunk_row_count += 1
                            if (
                                args.controller_continue_max_errors > 0
                                and controller_continue_errors
                                > args.controller_continue_max_errors
                            ):
                                error = repr(control_error)
                                end_reason = "controller_error_limit_exceeded"
                                print(
                                    "[song-controller-error-limit] "
                                    + json.dumps(
                                        {
                                            "episode": int(episode_index),
                                            "model_call": int(model_calls),
                                            "chunk_row": int(chunk_row_index),
                                            "errors": int(
                                                controller_continue_errors
                                            ),
                                            "max_errors": int(
                                                args.controller_continue_max_errors
                                            ),
                                            "reason": "controller_exception",
                                            "episode_result": "failure",
                                            "next": "next_episode",
                                            "episode_retry": False,
                                            "error": repr(control_error),
                                        },
                                        separators=(",", ":"),
                                    ),
                                    flush=True,
                                )
                                break
                            fallback_executed = False
                            fallback_error = None
                            if (
                                args.song_gripper_fallback_on_control_failure
                                and gripper_state_changed
                            ):
                                try:
                                    step_result = (
                                        execute_gripper_only_after_control_failure(
                                            task_env=task_env,
                                            observation=observation,
                                            target_gripper_open=target_gripper_open,
                                            diagnostic_context={
                                                **diagnostic_context,
                                                "arm_failure_reason": (
                                                    "controller_exception"
                                                ),
                                                "arm_failure": repr(control_error),
                                            },
                                            execution_recorder=(
                                                record_executed_action
                                            ),
                                            log_control_details=(
                                                args.log_control_details
                                            ),
                                        )
                                    )
                                    fallback_executed = True
                                    song_gripper_fallback_count += 1
                                    if (
                                        pending_gripper_open is not None
                                        and bool(pending_gripper_open)
                                        == bool(target_gripper_open)
                                    ):
                                        pending_gripper_applied_count += 1
                                        pending_gripper_open = None
                                        pending_gripper_origin = None
                                except Exception as gripper_fallback_error:
                                    fallback_error = repr(gripper_fallback_error)
                                    song_gripper_fallback_failure_count += 1
                                    gripper_command_open = (
                                        recover_discrete_gripper_command_after_control_failure(
                                            previous_command_open,
                                            observation.gripper_open,
                                        )
                                    )
                                    gripper_closed_latched = (
                                        gripper_closed_latched_before_row
                                    )
                            else:
                                gripper_command_open = (
                                    recover_discrete_gripper_command_after_control_failure(
                                        previous_command_open,
                                        observation.gripper_open,
                                    )
                                )
                                gripper_closed_latched = (
                                    gripper_closed_latched_before_row
                                )
                            fallback_event = {
                                "episode": int(episode_index),
                                "model_call": int(model_calls),
                                "chunk_row": int(chunk_row_index),
                                "errors": int(controller_continue_errors),
                                "reason": "controller_exception",
                                "continue": "next_chunk_row",
                                "gripper_transition_requested": bool(
                                    gripper_state_changed
                                ),
                                "gripper_fallback_executed": bool(
                                    fallback_executed
                                ),
                                "gripper_fallback_error": fallback_error,
                                "error": repr(control_error),
                            }
                            song_gripper_fallback_events.append(fallback_event)
                            print(
                                "[song-controller-continue-row] "
                                + json.dumps(fallback_event, separators=(",", ":")),
                                flush=True,
                            )
                            if not fallback_executed:
                                # No environment action was produced for this
                                # row. Continue with the following predicted
                                # row instead of discarding the chunk.
                                continue
                            # A gripper-only action was executed successfully.
                            # Fall through to the common observation/video and
                            # task-success bookkeeping below.
                        elif (
                            args.controller_error_mode == "continue_episode"
                            and (
                                args.controller_continue_max_errors <= 0
                                or controller_continue_errors
                                <= args.controller_continue_max_errors
                            )
                        ):
                            continue_after_controller_error = True
                            discarded_rows = max(
                                int(stop) - int(chunk_row_index) - 1, 0
                            )
                            control_error_reinferences += 1
                            discarded_chunk_rows += discarded_rows
                            reinference_event = {
                                "model_call": int(model_calls),
                                "chunk_row_index": int(chunk_row_index),
                                "reason": "controller_exception",
                                "discarded_chunk_rows": int(discarded_rows),
                                "error": repr(control_error),
                            }
                            control_error_reinference_events.append(
                                reinference_event
                            )
                            if args.gripper_mode == LEGACY_LIBERO_DELTA:
                                previous_predicted_width = observed_gripper_width(
                                    observation
                                )
                            if (
                                gripper_transition_requires_reach
                            ):
                                if pending_gripper_open is None:
                                    pending_gripper_stored_count += 1
                                pending_gripper_open = bool(target_gripper_open)
                                if pending_gripper_origin is None:
                                    pending_gripper_origin = {
                                        "model_call": int(model_calls),
                                        "chunk_row_index": int(chunk_row_index),
                                        "reason": "controller_exception",
                                    }
                            gripper_command_open = (
                                recover_discrete_gripper_command_after_control_failure(
                                    previous_command_open,
                                    observation.gripper_open,
                                )
                            )
                            gripper_closed_latched = (
                                gripper_closed_latched_before_row
                            )
                            print(
                                "[controller-reinfer] "
                                + json.dumps(
                                    {
                                        "episode": int(episode_index),
                                        "model_call": int(model_calls),
                                        "chunk_row": int(chunk_row_index),
                                        "errors": int(controller_continue_errors),
                                        "reason": "controller_exception",
                                        "discarded_chunk_rows": int(
                                            discarded_rows
                                        ),
                                        "error": repr(control_error),
                                    },
                                    separators=(",", ":"),
                                ),
                                flush=True,
                            )
                            break
                        elif not args.song_continue_failed_chunk_rows:
                            raise
                else:
                    step_result = execute_absolute_target_with_midpoints(
                        task_env,
                        observation,
                        requested_world,
                        target_gripper_open,
                        args,
                        diagnostic_context=diagnostic_context,
                        execution_recorder=record_executed_action,
                    )
                song_mover_exhausted = bool(
                    args.song_continue_failed_chunk_rows
                    and "mover_reached" in step_result
                    and not step_result["mover_reached"]
                    and not step_result["success"]
                    and not step_result["termination"]
                )
                if (
                    song_mover_exhausted
                    and args.song_gripper_fallback_on_control_failure
                    and gripper_state_changed
                    and gripper_transition_requires_reach
                    and not step_result.get(
                        "gripper_after_reach_executed", False
                    )
                ):
                    try:
                        fallback_result = (
                            execute_gripper_only_after_control_failure(
                                task_env=task_env,
                                observation=step_result["observation"],
                                target_gripper_open=target_gripper_open,
                                diagnostic_context={
                                    **diagnostic_context,
                                    "arm_failure_reason": "mover_unreached",
                                    "mover_attempts": int(
                                        step_result.get("mover_attempts", 0)
                                    ),
                                    "final_position_error_m": float(
                                        step_result.get(
                                            "final_position_error", float("inf")
                                        )
                                    ),
                                    "final_rotation_error_rad": float(
                                        step_result.get(
                                            "final_rotation_error", float("inf")
                                        )
                                    ),
                                },
                                execution_recorder=record_executed_action,
                                log_control_details=args.log_control_details,
                            )
                        )
                        step_result["actions"].extend(
                            fallback_result["actions"]
                        )
                        step_result["observations"].extend(
                            fallback_result["observations"]
                        )
                        step_result["observation"] = fallback_result[
                            "observation"
                        ]
                        step_result["success"] = bool(
                            step_result["success"]
                            or fallback_result["success"]
                        )
                        step_result["termination"] = bool(
                            step_result["termination"]
                            or fallback_result["termination"]
                        )
                        step_result[
                            "gripper_after_control_failure_executed"
                        ] = True
                        song_gripper_fallback_count += 1
                        fallback_event = {
                            "episode": int(episode_index),
                            "model_call": int(model_calls),
                            "chunk_row": int(chunk_row_index),
                            "reason": "mover_unreached",
                            "target_gripper_open": bool(target_gripper_open),
                            "gripper_fallback_executed": True,
                        }
                        song_gripper_fallback_events.append(fallback_event)
                        print(
                            "[song-gripper-fallback] "
                            + json.dumps(fallback_event, separators=(",", ":")),
                            flush=True,
                        )
                        if (
                            pending_gripper_open is not None
                            and bool(pending_gripper_open)
                            == bool(target_gripper_open)
                        ):
                            pending_gripper_applied_count += 1
                            pending_gripper_open = None
                            pending_gripper_origin = None
                    except Exception as gripper_fallback_error:
                        song_gripper_fallback_failure_count += 1
                        fallback_event = {
                            "episode": int(episode_index),
                            "model_call": int(model_calls),
                            "chunk_row": int(chunk_row_index),
                            "reason": "mover_unreached",
                            "target_gripper_open": bool(target_gripper_open),
                            "gripper_fallback_executed": False,
                            "gripper_fallback_error": repr(
                                gripper_fallback_error
                            ),
                        }
                        song_gripper_fallback_events.append(fallback_event)
                        gripper_command_open = (
                            recover_discrete_gripper_command_after_control_failure(
                                previous_command_open,
                                step_result["observation"].gripper_open,
                            )
                        )
                        gripper_closed_latched = (
                            gripper_closed_latched_before_row
                        )
                        print(
                            "[song-gripper-fallback-failed] "
                            + json.dumps(fallback_event, separators=(",", ":")),
                            flush=True,
                        )
                if not step_result["ok"] and args.controller_error_mode == "continue_episode":
                    controller_continue_errors += 1
                    skipped_controller_waypoints += 1
                    controller_continue_failures.append(
                        {
                            "model_call": int(model_calls),
                            "chunk_row_index": int(chunk_row_index),
                            "error": step_result["error"],
                        }
                    )
                    if (
                        args.controller_continue_max_errors <= 0
                        or controller_continue_errors
                        <= args.controller_continue_max_errors
                    ):
                        continue_after_controller_error = True
                if step_result["was_limited"]:
                    limited_action_count += 1
                jacobian_intermediate_action_count += int(
                    step_result.get(
                        "jacobian_intermediate_actions",
                        max(len(step_result["actions"]) - 1, 0),
                    )
                )
                jacobian_recursive_midpoint_count += int(
                    step_result["recursive_midpoints"]
                )
                threshold_midpoint_count += int(step_result["threshold_midpoints"])
                reached_waypoint_count += int(step_result["reached"])
                mover_attempt_count += int(step_result.get("mover_attempts", 0))
                mover_retry_count += int(step_result.get("mover_retries", 0))
                mover_target_count += int("mover_reached" in step_result)
                mover_reached_target_count += int(
                    step_result.get("mover_reached", False)
                )
                mover_unreached_target_count += int(
                    "mover_reached" in step_result
                    and not step_result["mover_reached"]
                )
                gripper_after_reach_action_count += int(
                    step_result.get("gripper_after_reach_executed", False)
                )
                if (
                    pending_gripper_open is not None
                    and step_result.get("gripper_after_reach_executed", False)
                    and bool(target_gripper_open) == bool(pending_gripper_open)
                ):
                    pending_gripper_applied_count += 1
                    print_control_log(
                        "[gripper-pending-applied]",
                        {
                            "episode_index": int(episode_index),
                            "model_call": int(model_calls),
                            "chunk_row_index": int(chunk_row_index),
                            "target_gripper_open": bool(target_gripper_open),
                            "origin": pending_gripper_origin,
                        },
                        args.log_control_details,
                    )
                    pending_gripper_open = None
                    pending_gripper_origin = None
                for intermediate_observation, action in zip(
                    step_result["observations"], step_result["actions"]
                ):
                    executed_actions.append(action)
                    save_frame_pointcloud(
                        intermediate_observation,
                        len(executed_actions),
                    )
                    frame = video_front_rgb(
                        task_env,
                        intermediate_observation.front_rgb,
                        args.save_video and args.video_refresh_rgb,
                    )
                    if draw_pour_point and not draw_task_stages:
                        frame = draw_pour_point_on_front_image(
                            frame, intermediate_observation, task_env
                        )
                    if draw_task_stages:
                        frame = draw_water_plants_stage_overlay(
                            frame,
                            intermediate_observation,
                            task_env,
                            success=bool(success or step_result["success"]),
                            termination=bool(
                                termination or step_result["termination"]
                            ),
                        )
                    if draw_phone_success_sensor:
                        frame = draw_phone_success_sensor_overlay(
                            frame,
                            intermediate_observation,
                            task_env,
                            success=bool(success or step_result["success"]),
                            termination=bool(
                                termination or step_result["termination"]
                            ),
                        )
                    raw_frames.append(frame.copy())
                    frame = annotate_video_frame(
                        frame,
                        task_name,
                        episode_index,
                        frame_index=len(executed_actions),
                        physics_frame_index=physics_frame_count + 1,
                        model_call=model_calls,
                        chunk_row=chunk_row_index,
                    )
                    frames.append(frame)
                    physics_frame_count += 1
                    print(
                        "[eval-frame] "
                        + json.dumps(
                            {
                                "task": task_name,
                                "episode": int(episode_index),
                                "frame": int(len(executed_actions)),
                                "physics_frame": int(physics_frame_count),
                                "model_call": int(model_calls),
                                "chunk_row": int(chunk_row_index),
                            },
                            separators=(",", ":"),
                        ),
                        flush=True,
                    )
                observation = step_result["observation"]
                success = bool(success or step_result["success"])
                termination = bool(termination or step_result["termination"])
                mover_unreached = bool(
                    "mover_reached" in step_result
                    and not step_result["mover_reached"]
                    and not success
                    and not termination
                )
                if mover_unreached and args.continue_chunk_on_mover_unreached:
                    planning_failure_this_call = True
                    planning_failure_reasons_this_call.append(
                        "mover_unreached_continued"
                    )
                    controller_continue_errors += 1
                    skipped_controller_waypoints += 1
                    if args.song_continue_failed_chunk_rows:
                        failed_chunk_rows_this_call += 1
                        song_failed_chunk_row_count += 1
                    mover_error = (
                        "pointact_mover_unreached_after_"
                        + str(int(step_result.get("mover_attempts", 0)))
                        + "_attempts"
                    )
                    failure = {
                        "model_call": int(model_calls),
                        "chunk_row_index": int(chunk_row_index),
                        "error": mover_error,
                        "mover_attempts": int(
                            step_result.get("mover_attempts", 0)
                        ),
                        "final_position_error_m": float(
                            step_result.get("final_position_error", float("inf"))
                        ),
                        "final_rotation_error_rad": float(
                            step_result.get("final_rotation_error", float("inf"))
                        ),
                        "recovery": "continue_current_chunk",
                    }
                    controller_continue_failures.append(failure)
                    can_continue_chunk = bool(
                        args.controller_continue_max_errors <= 0
                        or controller_continue_errors
                        <= args.controller_continue_max_errors
                    )
                    rejected_chunk_count += 1
                    if not can_continue_chunk:
                        error = mover_error
                        end_reason = "controller_error_limit_exceeded"
                        print(
                            "[song-controller-error-limit] "
                            + json.dumps(
                                {
                                    "episode": int(episode_index),
                                    "model_call": int(model_calls),
                                    "chunk_row": int(chunk_row_index),
                                    "errors": int(controller_continue_errors),
                                    "max_errors": int(
                                        args.controller_continue_max_errors
                                    ),
                                    "reason": "mover_unreached",
                                    "episode_result": "failure",
                                    "next": "next_episode",
                                    "episode_retry": False,
                                    "error": mover_error,
                                },
                                separators=(",", ":"),
                            ),
                            flush=True,
                        )
                        break
                    mover_unreached_continued_count += 1
                    mover_unreached_continue_events.append(failure)
                    if (
                        gripper_transition_requires_reach
                        and not step_result.get(
                            "gripper_after_reach_executed", False
                        )
                        and not step_result.get(
                            "gripper_after_control_failure_executed", False
                        )
                    ):
                        if pending_gripper_open is None:
                            pending_gripper_stored_count += 1
                        pending_gripper_open = bool(target_gripper_open)
                        if pending_gripper_origin is None:
                            pending_gripper_origin = {
                                "model_call": int(model_calls),
                                "chunk_row_index": int(chunk_row_index),
                                "reason": "mover_unreached_continued",
                            }
                        # The deferred command was not physically issued. Keep
                        # the logical state aligned with the measured command
                        # so a later reached row can replay the pending event.
                        gripper_command_open = previous_command_open
                        gripper_closed_latched = (
                            gripper_closed_latched_before_row
                        )
                    print(
                        "[controller-continue-chunk] "
                        + json.dumps(
                            {
                                "episode": int(episode_index),
                                "model_call": int(model_calls),
                                "chunk_row": int(chunk_row_index),
                                "errors": int(controller_continue_errors),
                                "reason": "mover_unreached",
                                "mover_attempts": int(
                                    step_result.get("mover_attempts", 0)
                                ),
                                "pending_target_gripper_open": (
                                    None
                                    if pending_gripper_open is None
                                    else bool(pending_gripper_open)
                                ),
                            },
                            separators=(",", ":"),
                        ),
                        flush=True,
                    )
                mover_failed = bool(
                    mover_unreached
                    and not args.continue_chunk_on_mover_unreached
                    and args.reinfer_on_control_error
                )
                if mover_failed:
                    planning_failure_this_call = True
                    planning_failure_reasons_this_call.append("mover_unreached")
                    controller_continue_errors += 1
                    skipped_controller_waypoints += 1
                    mover_error = (
                        "pointact_mover_unreached_after_"
                        + str(int(step_result.get("mover_attempts", 0)))
                        + "_attempts"
                    )
                    failure = {
                        "model_call": int(model_calls),
                        "chunk_row_index": int(chunk_row_index),
                        "error": mover_error,
                        "mover_attempts": int(
                            step_result.get("mover_attempts", 0)
                        ),
                        "final_position_error_m": float(
                            step_result.get("final_position_error", float("inf"))
                        ),
                        "final_rotation_error_rad": float(
                            step_result.get("final_rotation_error", float("inf"))
                        ),
                    }
                    controller_continue_failures.append(failure)
                    can_reinfer = bool(
                        args.controller_continue_max_errors <= 0
                        or controller_continue_errors
                        <= args.controller_continue_max_errors
                    )
                    rejected_chunk_count += 1
                    if can_reinfer:
                        continue_after_controller_error = True
                        discarded_rows = max(
                            int(stop) - int(chunk_row_index) - 1, 0
                        )
                        control_error_reinferences += 1
                        discarded_chunk_rows += discarded_rows
                        control_error_reinference_events.append(
                            {
                                **failure,
                                "reason": "mover_unreached",
                                "discarded_chunk_rows": int(discarded_rows),
                            }
                        )
                        if args.gripper_mode == LEGACY_LIBERO_DELTA:
                            previous_predicted_width = observed_gripper_width(
                                observation
                            )
                        if (
                            gripper_transition_requires_reach
                            and not step_result.get(
                                "gripper_after_reach_executed", False
                            )
                        ):
                            if pending_gripper_open is None:
                                pending_gripper_stored_count += 1
                            pending_gripper_open = bool(target_gripper_open)
                            if pending_gripper_origin is None:
                                pending_gripper_origin = {
                                    "model_call": int(model_calls),
                                    "chunk_row_index": int(chunk_row_index),
                                    "reason": "mover_unreached",
                                }
                            gripper_command_open = previous_command_open
                            gripper_closed_latched = (
                                gripper_closed_latched_before_row
                            )
                        print(
                            "[controller-reinfer] "
                            + json.dumps(
                                {
                                    "episode": int(episode_index),
                                    "model_call": int(model_calls),
                                    "chunk_row": int(chunk_row_index),
                                    "errors": int(controller_continue_errors),
                                    "reason": "mover_unreached",
                                    "mover_attempts": int(
                                        step_result.get("mover_attempts", 0)
                                    ),
                                    "discarded_chunk_rows": int(
                                        discarded_rows
                                    ),
                                    "pending_target_gripper_open": (
                                        None
                                        if pending_gripper_open is None
                                        else bool(pending_gripper_open)
                                    ),
                                },
                                separators=(",", ":"),
                            ),
                            flush=True,
                        )
                    else:
                        error = mover_error
                        end_reason = (
                            "controller_error_limit_exceeded"
                            if (
                                args.controller_continue_max_errors > 0
                                and controller_continue_errors
                                > args.controller_continue_max_errors
                            )
                            else "waypoint_controller_rejected"
                        )
                    break
                if not step_result["ok"]:
                    planning_failure_this_call = True
                    planning_failure_reasons_this_call.append(
                        "waypoint_controller_rejected"
                    )
                    rejected_chunk_count += 1
                    failure = {
                        "model_call": int(model_calls),
                        "requested_position_distance_m": float(
                            step_result["requested_position_distance"]
                        ),
                        "requested_rotation_distance_rad": float(
                            step_result["requested_rotation_distance"]
                        ),
                        "threshold_segments": int(step_result["segments"]),
                        "threshold_midpoints": int(step_result["threshold_midpoints"]),
                        "recursive_midpoints": int(
                            step_result["recursive_midpoints"]
                        ),
                        "final_position_error_m": float(
                            step_result["final_position_error"]
                        ),
                        "final_rotation_error_rad": float(
                            step_result["final_rotation_error"]
                        ),
                        "error": step_result["error"],
                    }
                    jacobian_failures.append(failure)
                    print("[jacobian-replan] " + json.dumps(failure), flush=True)
                    if continue_after_controller_error:
                        discarded_rows = max(
                            int(stop) - int(chunk_row_index) - 1, 0
                        )
                        control_error_reinferences += 1
                        discarded_chunk_rows += discarded_rows
                        control_error_reinference_events.append(
                            {
                                **failure,
                                "reason": "waypoint_controller_rejected",
                                "discarded_chunk_rows": int(discarded_rows),
                            }
                        )
                        if args.gripper_mode == LEGACY_LIBERO_DELTA:
                            previous_predicted_width = observed_gripper_width(
                                observation
                            )
                        gripper_command_open = (
                            recover_discrete_gripper_command_after_control_failure(
                                previous_command_open,
                                observation.gripper_open,
                            )
                        )
                        print(
                            "[controller-reinfer] "
                            + json.dumps(
                                {
                                    "episode": int(episode_index),
                                    "model_call": int(model_calls),
                                    "chunk_row": int(chunk_row_index),
                                    "errors": int(controller_continue_errors),
                                    "reason": "waypoint_controller_rejected",
                                    "discarded_chunk_rows": int(
                                        discarded_rows
                                    ),
                                    "error": step_result["error"],
                                },
                                separators=(",", ":"),
                            ),
                            flush=True,
                        )
                    else:
                        error = step_result["error"]
                        end_reason = "waypoint_controller_rejected"
                    break
                executed_chunk_rows_this_call += 1
                if success or termination:
                    break

            if end_reason == "controller_error_limit_exceeded":
                # SONG deliberately does not reset/retry an episode. Once the
                # cumulative controller-error budget is exhausted, stop before
                # granting model-call compensation and let the outer rollout
                # advance directly to the next episode.
                break

            if (
                args.song_continue_failed_chunk_rows
                and attempted_chunk_rows_this_call > 0
                and failed_chunk_rows_this_call
                >= attempted_chunk_rows_this_call
                and not success
                and not termination
            ):
                song_all_rows_failed_call_count += 1
                planning_failure_this_call = True
                planning_failure_reasons_this_call.append(
                    "all_executed_chunk_rows_failed"
                )
                print(
                    "[song-all-chunk-rows-failed-reinfer] "
                    + json.dumps(
                        {
                            "episode": int(episode_index),
                            "model_call": int(model_calls),
                            "attempted_chunk_rows": int(
                                attempted_chunk_rows_this_call
                            ),
                            "failed_chunk_rows": int(
                                failed_chunk_rows_this_call
                            ),
                            "next": "reobserve_and_predict_new_chunk",
                        },
                        separators=(",", ":"),
                    ),
                    flush=True,
                )

            if planning_failure_this_call:
                consecutive_planning_failure_calls += 1
            else:
                consecutive_planning_failure_calls = 0

            compensation_reasons = []
            if (
                args.model_call_compensation_min_executed_rows > 0
                and executed_chunk_rows_this_call
                < args.model_call_compensation_min_executed_rows
            ):
                compensation_reasons.append("insufficient_executed_chunk_rows")
            if (
                args.model_call_compensation_consecutive_planning_failures > 0
                and consecutive_planning_failure_calls
                >= args.model_call_compensation_consecutive_planning_failures
            ):
                compensation_reasons.append("consecutive_planning_failures")
            if (
                compensation_reasons
                and not success
                and not termination
                and model_call_compensations < max_model_call_compensations
            ):
                model_call_compensations += 1
                effective_model_call_budget += 1
                compensation_event = {
                    "model_call": int(model_calls),
                    "reasons": compensation_reasons,
                    "executed_chunk_rows": int(executed_chunk_rows_this_call),
                    "planning_failure": bool(planning_failure_this_call),
                    "planning_failure_reasons": list(
                        dict.fromkeys(planning_failure_reasons_this_call)
                    ),
                    "consecutive_planning_failure_calls": int(
                        consecutive_planning_failure_calls
                    ),
                    "compensation_index": int(model_call_compensations),
                    "effective_model_call_budget": int(
                        effective_model_call_budget
                    ),
                }
                model_call_compensation_events.append(compensation_event)
                print(
                    "[model-call-budget-compensation] "
                    + json.dumps(compensation_event, separators=(",", ":")),
                    flush=True,
                )
            if continue_after_controller_error and not success and not termination:
                end_reason = None
                continue
            if end_reason in {
                "waypoint_controller_rejected",
                "controller_error_limit_exceeded",
            }:
                break
    except Exception as exc:
        error = repr(exc)
        if end_reason is None:
            end_reason = "controller_error"
    if success:
        end_reason = "success"
    elif termination:
        end_reason = "environment_termination"
    elif (
        args.max_policy_action_steps > 0
        and policy_action_steps_attempted >= args.max_policy_action_steps
    ):
        end_reason = "max_policy_action_steps"
    elif end_reason is None and model_calls >= effective_model_call_budget:
        end_reason = "max_model_calls"
    elif end_reason is None:
        end_reason = "loop_exit"

    if draw_task_stages and frames and not success:
        frames[-1] = draw_water_plants_stage_overlay(
            frames[-1],
            observation,
            task_env,
            success=False,
            termination=termination,
            result_override="RESULT: FAIL (%s)" % end_reason,
        )
    if draw_phone_success_sensor and frames:
        frames[-1] = draw_phone_success_sensor_overlay(
            frames[-1],
            observation,
            task_env,
            success=success,
            termination=termination,
        )

    retain_diagnostic_artifacts = (
        not args.failure_artifacts_only or not success
    )
    retain_video = bool(args.save_video)
    if (
        args.save_action_visualizations
        and not retain_diagnostic_artifacts
    ):
        successful_action_vis_dir = (
            run_dir
            / "action_visualizations"
            / ("episode_" + str(episode_index).zfill(3))
        )
        if successful_action_vis_dir.is_dir():
            shutil.rmtree(successful_action_vis_dir)
        action_visualizations = []

    episode_result = {
        "episode_index": int(episode_index),
        "seed_episode_index": int(seed_episode_index),
        "task": task_name,
        "language": language,
        "execution_mode": args.execution_mode,
        "draw_pour_point": bool(draw_pour_point),
        "draw_task_stages": bool(draw_task_stages),
        "draw_phone_success_sensor": bool(draw_phone_success_sensor),
        "gripper_mode": args.gripper_mode,
        "gripper_protocol": (
            "one_time_physical_width_sync_then_per_chunk_first_executed_row_self_reference"
            if args.gripper_mode == DELTA_WIDTH_INITIAL_SYNC
            else (
                "measured_episode_anchor_then_cross_chunk_prediction_carry"
                if args.gripper_mode == LEGACY_LIBERO_DELTA
                else "per_row_absolute_width_threshold"
            )
        ),
        "gripper_initial_sync_applied": bool(gripper_initial_sync_applied),
        "gripper_initial_sync": gripper_initial_sync_info,
        "gripper_after_reach": bool(args.gripper_after_reach),
        "gripper_close_require_reach": bool(
            args.gripper_close_require_reach
        ),
        "gripper_open_require_reach": bool(
            args.gripper_open_require_reach
        ),
        "gripper_lock_after_close": bool(args.gripper_lock_after_close),
        "gripper_closed_latched": bool(gripper_closed_latched),
        "gripper_transitions": int(gripper_transition_count),
        "pending_gripper_stored": int(pending_gripper_stored_count),
        "pending_gripper_applied": int(pending_gripper_applied_count),
        "pending_gripper_cancelled": int(pending_gripper_cancelled_count),
        "pending_gripper_open_at_end": (
            None
            if pending_gripper_open is None
            else bool(pending_gripper_open)
        ),
        "success": bool(success),
        "failure_artifacts_only": bool(args.failure_artifacts_only),
        "diagnostic_artifacts_retained": bool(retain_diagnostic_artifacts),
        "video_retained": bool(retain_video),
        "terminated": bool(termination),
        "end_reason": end_reason,
        "model_calls": int(model_calls),
        "base_model_call_budget": int(base_model_call_budget),
        "effective_model_call_budget": int(effective_model_call_budget),
        "model_call_compensations": int(model_call_compensations),
        "max_model_call_compensations": int(max_model_call_compensations),
        "model_call_compensation_min_executed_rows": int(
            args.model_call_compensation_min_executed_rows
        ),
        "model_call_compensation_consecutive_planning_failures": int(
            args.model_call_compensation_consecutive_planning_failures
        ),
        "model_call_compensation_events": model_call_compensation_events,
        "policy_action_steps_attempted": int(policy_action_steps_attempted),
        "max_policy_action_steps": int(args.max_policy_action_steps),
        "environment_actions": int(len(executed_simulator_actions)),
        "physics_frames": int(physics_frame_count),
        "video_frames": int(len(frames)),
        "limited_eef_actions": int(limited_action_count),
        "workspace_clipped_actions": int(workspace_clipped_action_count),
        "mover_targets": int(mover_target_count),
        "mover_reached_targets": int(mover_reached_target_count),
        "mover_attempts": int(mover_attempt_count),
        "mover_retries": int(mover_retry_count),
        "mover_unreached_targets": int(mover_unreached_target_count),
        "continue_chunk_on_mover_unreached": bool(
            args.continue_chunk_on_mover_unreached
        ),
        "mover_unreached_continued": int(mover_unreached_continued_count),
        "mover_unreached_continue_events": mover_unreached_continue_events,
        "gripper_after_reach_actions": int(gripper_after_reach_action_count),
        "song_failed_chunk_rows": int(song_failed_chunk_row_count),
        "song_all_rows_failed_model_calls": int(
            song_all_rows_failed_call_count
        ),
        "song_gripper_fallback_actions": int(song_gripper_fallback_count),
        "song_gripper_fallback_failures": int(
            song_gripper_fallback_failure_count
        ),
        "song_gripper_fallback_events": song_gripper_fallback_events,
        "episode_retries_enabled": not bool(args.disable_episode_retries),
        "jacobian_intermediate_actions": int(jacobian_intermediate_action_count),
        "jacobian_recursive_midpoints": int(jacobian_recursive_midpoint_count),
        "threshold_midpoints": int(threshold_midpoint_count),
        "reached_waypoints": int(reached_waypoint_count),
        "rejected_chunks": int(rejected_chunk_count),
        "jacobian_failures": jacobian_failures,
        "controller_error_mode": args.controller_error_mode,
        "reinfer_on_control_error": bool(args.reinfer_on_control_error),
        "controller_continue_max_errors": int(
            args.controller_continue_max_errors
        ),
        "controller_error_limit_exceeded": bool(
            end_reason == "controller_error_limit_exceeded"
        ),
        "controller_continue_errors": int(controller_continue_errors),
        "skipped_controller_waypoints": int(skipped_controller_waypoints),
        "controller_continue_failures": controller_continue_failures,
        "control_error_reinferences": int(control_error_reinferences),
        "discarded_chunk_rows": int(discarded_chunk_rows),
        "control_error_reinference_events": control_error_reinference_events,
        "action_visualizations": action_visualizations,
        "action_chunks": action_chunk_paths,
        "frame_pointclouds_enabled": bool(args.save_frame_pointclouds),
        "frame_pointcloud_every_n_frames": int(
            args.frame_pointcloud_every_n_frames
        ),
        "frame_pointcloud_count": int(frame_pointcloud_count),
        "frame_pointcloud_dir": (
            str(frame_pointcloud_dir.relative_to(run_dir))
            if args.save_frame_pointclouds
            else None
        ),
        "determinism_diagnostics_enabled": bool(
            args.save_determinism_diagnostics
        ),
        "determinism_diagnostics_dir": (
            str(determinism_dir.relative_to(run_dir))
            if args.save_determinism_diagnostics
            else None
        ),
        "determinism_state_count": int(determinism_state_count),
        "error": error,
    }
    if args.save_determinism_diagnostics:
        determinism_manifest.update(
            {
                "success": bool(success),
                "termination": bool(termination),
                "end_reason": end_reason,
                "model_calls": int(model_calls),
                "policy_action_steps_attempted": int(
                    policy_action_steps_attempted
                ),
                "simulator_state_records": int(determinism_state_count),
                "executed_states_file": "executed_states.jsonl",
            }
        )
        with open(determinism_manifest_path, "w", encoding="utf-8") as file:
            json.dump(determinism_manifest, file, indent=2)
    if dataset_reset_spec is not None:
        episode_result.update(
            {
                "dataset_local_episode_index": int(
                    dataset_reset_spec["local_episode_index"]
                ),
                "dataset_global_episode_index": int(
                    dataset_reset_spec["global_episode_index"]
                ),
                "dataset_initial_task_state": str(
                    dataset_reset_spec["state_path"]
                ),
                "dataset_reset_method": dataset_reset_method,
                "dataset_initial_state_validation": dataset_initial_state_validation,
            }
        )
    if retain_video:
        video_path = run_dir / "videos" / ("episode_" + str(episode_index).zfill(3) + ".mp4")
        video_path.parent.mkdir(parents=True, exist_ok=True)
        video_frames = resize_video_frames(
            frames, args.video_width, args.video_height
        )
        video_frames = annotate_final_task_result_frames(video_frames, success)
        raw_video_frames = resize_video_frames(
            raw_frames, args.video_width, args.video_height
        )
        raw_frame_stack = np.asarray(raw_video_frames, dtype=np.uint8)
        episode_result["video_rgb_min"] = int(raw_frame_stack.min())
        episode_result["video_rgb_max"] = int(raw_frame_stack.max())
        episode_result["video_rgb_mean"] = float(raw_frame_stack.mean())
        if episode_result["video_rgb_max"] == 0:
            print(
                "[video-warning] episode="
                + str(episode_index)
                + " all saved RGB frames are zero; check RLBench camera rendering.",
                flush=True,
            )
        imageio.mimsave(video_path, video_frames, fps=args.video_fps, macro_block_size=1)
        episode_result["video"] = str(video_path.relative_to(run_dir))
    if args.save_action_records:
        actions_path = (
            run_dir
            / "actions"
            / ("episode_" + str(episode_index).zfill(3) + "_actions.npy")
        )
        model_chunks_path = (
            run_dir
            / "model_chunks"
            / ("episode_" + str(episode_index).zfill(3) + "_model_chunks.npy")
        )
        actions_path.parent.mkdir(parents=True, exist_ok=True)
        model_chunks_path.parent.mkdir(parents=True, exist_ok=True)
        simulator_action_array = _execution_array(
            executed_simulator_actions,
            8,
            np.float32,
            "simulator actions",
        )
        np.save(actions_path, simulator_action_array)
        np.save(
            model_chunks_path,
            np.asarray(predicted_chunks, dtype=np.float32),
        )
        episode_result["executed_actions"] = str(actions_path.relative_to(run_dir))
        episode_result["model_chunks"] = str(model_chunks_path.relative_to(run_dir))
        episode_result.update(
            save_episode_executed_action_alignment(
                run_dir=run_dir,
                episode_index=episode_index,
                model_actions=executed_model_actions,
                controller_actions=simulator_action_array,
                execution_indices=executed_action_indices,
            )
        )
    return episode_result


def apply_action_replay_parity(args):
    """Remove evaluator-only control heuristics and match dataset action replay."""
    if not args.action_replay_parity:
        return
    if args.dataset_root is None:
        raise ValueError("--action-replay-parity requires --dataset-root.")

    policy_config_path = args.policy_path.expanduser().resolve() / "config.json"
    if not policy_config_path.is_file():
        raise FileNotFoundError(
            "Policy config does not exist for action-replay parity: "
            + str(policy_config_path)
        )
    with open(policy_config_path, "r", encoding="utf-8") as file:
        policy_config = json.load(file)
    policy_action_steps = int(policy_config.get("n_action_steps", 16))
    visual_features = [
        value
        for value in policy_config.get("input_features", {}).values()
        if isinstance(value, dict) and value.get("type") == "VISUAL"
    ]
    policy_image_size = 256
    if visual_features:
        visual_shape = visual_features[0].get("shape", [])
        if len(visual_shape) == 3:
            policy_image_size = int(visual_shape[-1])

    dataset_root = args.dataset_root.expanduser().resolve()
    zarr_attribute_files = sorted(
        (dataset_root / "point_clouds").glob("episode_*.zarr/.zattrs")
    )
    if not zarr_attribute_files:
        raise FileNotFoundError(
            "Cannot infer recorded point count: no point-cloud Zarr metadata in "
            + str(dataset_root)
        )
    with open(zarr_attribute_files[0], "r", encoding="utf-8") as file:
        point_cloud_attributes = json.load(file)
    point_cloud_shape = point_cloud_attributes.get("shape", [])
    if len(point_cloud_shape) != 3:
        raise ValueError(
            "Invalid recorded point-cloud shape: " + str(point_cloud_shape)
        )
    recorded_point_count = int(point_cloud_shape[1])

    parity_settings = {
        "arm_action_mode": "planning",
        "execution_mode": "dataset_step",
        "collision_checking": False,
        "action_index": 0,
        "exec_action_steps": policy_action_steps,
        "gripper_mode": DELTA_WIDTH_INITIAL_SYNC,
        "gripper_lock_after_close": False,
        "max_eef_position_step": 0.0,
        "max_eef_rotation_step": 0.0,
        "mover_rotation_tolerance": 0.0,
        "mover_gripper_rotation_tolerance": 0.0,
        "simulation_timestep": 0.0,
        "num_points": recorded_point_count,
        "gripper_points": 500,
        "add_gripper_cloud": True,
        "gripper_template": "reap",
        "image_size": policy_image_size,
        "water_plant_collision": "enabled",
        "water_drop_collision": "original",
        "video_refresh_rgb": False,
        "draw_pour_point": False,
        "draw_task_stages": False,
        "draw_phone_success_sensor": False,
        "save_action_visualizations": False,
        "visualize_foreground": False,
    }
    for name, value in parity_settings.items():
        setattr(args, name, value)
    args.action_replay_parity_resolved = {
        **parity_settings,
        "policy_config": str(policy_config_path),
        "dataset_point_cloud_metadata": str(zarr_attribute_files[0]),
        "recorded_initial_robot_state_restored": True,
        "model_action_refresh_steps": policy_action_steps,
        "dataset_and_model_input_contract": (
            "recorded reset, controller family, point-cloud input, virtual gripper, "
            "and task physics match dataset replay"
        ),
    }


def apply_simulator_robustness_optimizations(args):
    """Resolve the legacy or SONG simulator execution profile."""
    song_enabled = bool(args.simulator_robustness_optimizations_song)
    legacy_enabled = bool(args.simulator_robustness_optimizations)
    if song_enabled:
        # The new profile replaces (rather than stacks on top of) the legacy
        # PointACT profile, whose parser default remains True for old commands.
        args.simulator_robustness_optimizations = False
        # Dataset collection stores the expert FK target without PointACT's
        # narrower controller-side XYZ clipping. The point-cloud input still
        # uses the shared RLBENCH_SCENE_BOUNDS crop in both collection and eval.
        settings = {
            "clip_within_workspace": False,
            "mover_max_tries": (
                int(args.mover_max_tries)
                if args.mover_max_tries is not None
                else 2
            ),
            "planning_same_target_settle": True,
            "reinfer_on_control_error": True,
            "continue_chunk_on_mover_unreached": True,
            "gripper_after_reach": True,
            "gripper_close_require_reach": True,
            "gripper_open_require_reach": True,
            "pointact_pyrep_compat": True,
            "controller_error_mode": "continue_episode",
            "controller_error_retries": 0,
            "controller_error_retry_timeout_seconds": 0.0,
            # Keep an explicit CLI limit. Older SONG runs used 200 as the
            # profile default, but evaluation callers may intentionally use a
            # smaller bound such as --controller-continue-max-errors 5.
            "controller_continue_max_errors": (
                args.controller_continue_max_errors
                if args.controller_continue_max_errors > 0
                else 200
            ),
        }
        profile = "song"
    elif legacy_enabled:
        settings = {
            "clip_within_workspace": True,
            "mover_max_tries": (
                int(args.mover_max_tries)
                if args.mover_max_tries is not None
                else 10
            ),
            "planning_same_target_settle": False,
            "reinfer_on_control_error": True,
            "gripper_after_reach": True,
            "pointact_pyrep_compat": True,
            "controller_error_mode": "continue_episode",
            "controller_error_retries": 0,
            "controller_error_retry_timeout_seconds": 0.0,
            "controller_continue_max_errors": 0,
        }
        profile = "legacy_pointact"
    else:
        settings = {
            "clip_within_workspace": False,
            "mover_max_tries": (
                int(args.mover_max_tries)
                if args.mover_max_tries is not None
                else 1
            ),
            "planning_same_target_settle": False,
            "reinfer_on_control_error": False,
            "gripper_after_reach": False,
            "pointact_pyrep_compat": False,
            "controller_error_mode": "retry_episode",
            "controller_error_retries": 0,
            "controller_error_retry_timeout_seconds": 0.0,
            "controller_continue_max_errors": 0,
        }
        profile = "raw_controller"
    for name, value in settings.items():
        setattr(args, name, value)
    # The legacy aggregate switch remains backwards compatible with explicit
    # direction-specific flags. The SONG settings above intentionally force
    # both directions to require reach before its control-failure fallback.
    if args.gripper_close_require_reach is None:
        args.gripper_close_require_reach = bool(args.gripper_after_reach)
    if args.gripper_open_require_reach is None:
        args.gripper_open_require_reach = bool(args.gripper_after_reach)
    args.disable_episode_retries = bool(song_enabled)
    args.song_continue_failed_chunk_rows = bool(song_enabled)
    args.song_gripper_fallback_on_control_failure = bool(song_enabled)
    args.simulator_robustness_optimizations_resolved = {
        "enabled": bool(legacy_enabled and not song_enabled),
        "superseded_by_song_profile": bool(song_enabled),
        "profile": profile,
        **settings,
        "gripper_close_require_reach": bool(
            args.gripper_close_require_reach
        ),
        "gripper_open_require_reach": bool(
            args.gripper_open_require_reach
        ),
        "continue_chunk_on_mover_unreached": bool(
            args.continue_chunk_on_mover_unreached
        ),
        "scope": (
            "simulator/controller robustness only; does not alter point-cloud or "
            "gripper input parameters"
        ),
    }
    args.simulator_robustness_optimizations_song_resolved = {
        "enabled": bool(song_enabled),
        "profile": profile,
        **settings,
        "gripper_close_require_reach": bool(
            args.gripper_close_require_reach
        ),
        "gripper_open_require_reach": bool(
            args.gripper_open_require_reach
        ),
        "continue_failed_chunk_rows": bool(
            args.song_continue_failed_chunk_rows
        ),
        "gripper_fallback_on_control_failure": bool(
            args.song_gripper_fallback_on_control_failure
        ),
        "episode_retries_enabled": not bool(args.disable_episode_retries),
        "action_workspace_clip_matches_training": True,
        "point_cloud_crop": "shared_RLBENCH_SCENE_BOUNDS",
    }


def main():
    global CONTROL_LOG_PATH

    args = parse_args()
    if not np.isclose(
        float(RLBENCH_REAP_ALIGNED_GRIPPER_LEN),
        EXPECTED_RLBENCH_REAP_TCP_CALIBRATION_LEN_M,
        rtol=0.0,
        atol=1e-12,
    ):
        raise RuntimeError(
            "The REAP virtual-TCP calibration must remain fixed at 0.09 m; "
            "do not edit RLBENCH_REAP_ALIGNED_GRIPPER_LEN for an ablation. "
            "Restore it to 0.09 and select the test geometry with "
            "--gripper-len instead."
        )
    torch_determinism = configure_torch_determinism(args.deterministic_torch)
    apply_action_replay_parity(args)
    apply_simulator_robustness_optimizations(args)
    if args.planner_max_time_ms is not None:
        if args.planner_max_time_ms <= 0:
            raise ValueError("--planner-max-time-ms must be positive.")
        os.environ["RLBENCH_PLANNER_MAX_TIME_MS"] = str(args.planner_max_time_ms)
    os.environ["RLBENCH_WATER_PLANT_COLLISION"] = args.water_plant_collision
    os.environ["RLBENCH_WATER_DROP_COLLISION"] = args.water_drop_collision
    if args.episodes <= 0:
        raise ValueError("--episodes must be positive.")
    if args.max_model_calls <= 0:
        raise ValueError("--max-model-calls must be positive.")
    if args.model_call_compensation_min_executed_rows < 0:
        raise ValueError(
            "--model-call-compensation-min-executed-rows must be non-negative."
        )
    if args.model_call_compensation_consecutive_planning_failures < 0:
        raise ValueError(
            "--model-call-compensation-consecutive-planning-failures must be "
            "non-negative."
        )
    if args.max_model_call_compensations < -1:
        raise ValueError("--max-model-call-compensations must be -1 or non-negative.")
    if args.max_policy_action_steps < 0:
        raise ValueError("--max-policy-action-steps must be non-negative.")
    if args.controller_error_retries < -1:
        raise ValueError("--controller-error-retries must be -1 or non-negative.")
    if (
        args.controller_error_mode == "retry_episode"
        and args.controller_error_retries != 0
        and args.controller_error_retry_timeout_seconds <= 0.0
    ):
        raise ValueError("--controller-error-retry-timeout-seconds must be positive.")
    if args.controller_continue_max_errors < 0:
        raise ValueError("--controller-continue-max-errors must be non-negative.")
    if args.waypoint_position_tolerance < 0.0:
        raise ValueError("--waypoint-position-tolerance must be non-negative.")
    if args.waypoint_rotation_tolerance < 0.0:
        raise ValueError("--waypoint-rotation-tolerance must be non-negative.")
    if args.mover_max_tries <= 0:
        raise ValueError("--mover-max-tries must be positive.")
    if args.mover_position_tolerance <= 0.0:
        raise ValueError("--mover-position-tolerance must be positive.")
    if args.mover_rotation_tolerance < 0.0:
        raise ValueError("--mover-rotation-tolerance must be non-negative.")
    if args.mover_gripper_position_tolerance <= 0.0:
        raise ValueError("--mover-gripper-position-tolerance must be positive.")
    if args.mover_gripper_rotation_tolerance < 0.0:
        raise ValueError(
            "--mover-gripper-rotation-tolerance must be non-negative."
        )
    if (
        args.execution_mode == "adaptive"
        and (
            args.mover_max_tries > 1
            or args.gripper_close_require_reach
            or args.gripper_open_require_reach
        )
    ):
        raise ValueError(
            "PointACT Mover retries and --gripper-after-reach require "
            "--execution-mode dataset_step or bounded_step."
        )
    if args.gripper_delta_threshold < 0.0:
        raise ValueError("--gripper-delta-threshold must be non-negative.")
    if args.gripper_len < 0.0:
        raise ValueError("--gripper-len must be non-negative.")
    if args.gripper_delta_open_threshold is None:
        args.gripper_delta_open_threshold = args.gripper_delta_threshold
    if args.gripper_delta_close_threshold is None:
        args.gripper_delta_close_threshold = args.gripper_delta_threshold
    if args.gripper_delta_open_threshold < 0.0:
        raise ValueError("--gripper-delta-open-threshold must be non-negative.")
    if args.gripper_delta_close_threshold < 0.0:
        raise ValueError("--gripper-delta-close-threshold must be non-negative.")
    if (
        args.gripper_mode == DELTA_WIDTH_INITIAL_SYNC
        and args.gripper_delta_alignment != "current_minus_previous"
    ):
        raise ValueError(
            "delta_width_initial_sync requires --gripper-delta-alignment "
            "current_minus_previous so every chunk's first executed row has zero delta."
        )
    if args.gripper_open_threshold < 0.0:
        raise ValueError("--gripper-open-threshold must be non-negative.")
    if args.waypoint_max_control_steps <= 0:
        raise ValueError("--waypoint-max-control-steps must be positive.")
    if args.joint_velocity_kp <= 0.0:
        raise ValueError("--joint-velocity-kp must be positive.")
    if args.joint_velocity_max_speed <= 0.0:
        raise ValueError("--joint-velocity-max-speed must be positive.")
    if args.joint_velocity_joint_tolerance < 0.0:
        raise ValueError("--joint-velocity-joint-tolerance must be non-negative.")
    if args.joint_velocity_stall_steps <= 0:
        raise ValueError("--joint-velocity-stall-steps must be positive.")
    if args.linear_planning_max_joint_goal_distance < 0.0:
        raise ValueError("--linear-planning-max-joint-goal-distance must be non-negative.")
    if args.linear_planning_max_joint_path_length < 0.0:
        raise ValueError("--linear-planning-max-joint-path-length must be non-negative.")
    if args.linear_planning_max_joint_path_ratio < 0.0:
        raise ValueError("--linear-planning-max-joint-path-ratio must be non-negative.")
    if args.linear_planning_max_single_joint_travel < 0.0:
        raise ValueError("--linear-planning-max-single-joint-travel must be non-negative.")
    if args.franka_ik_max_iterations <= 0:
        raise ValueError("--franka-ik-max-iterations must be positive.")
    if args.franka_ik_tolerance <= 0.0:
        raise ValueError("--franka-ik-tolerance must be positive.")
    if args.franka_ik_damping <= 0.0:
        raise ValueError("--franka-ik-damping must be positive.")
    if args.video_width <= 0 or args.video_height <= 0:
        raise ValueError("--video-width and --video-height must be positive.")
    if args.action_vis_every_n_frames <= 0:
        raise ValueError("--action-vis-every-n-frames must be positive.")
    if args.frame_pointcloud_every_n_frames <= 0:
        raise ValueError("--frame-pointcloud-every-n-frames must be positive.")
    if args.action_vis_max_points <= 0:
        raise ValueError("--action-vis-max-points must be positive.")
    if args.action_vis_image_width <= 0:
        raise ValueError("--action-vis-image-width must be positive.")
    if args.dataset_episodes is not None and args.dataset_root is None:
        raise ValueError("--dataset-episodes requires --dataset-root.")
    if args.episode_indices is not None and args.dataset_root is not None:
        raise ValueError("--episode-indices cannot be combined with --dataset-root.")
    selected_episode_indices = (
        None
        if args.episode_indices is None
        else parse_dataset_episode_indices(args.episode_indices)
    )
    if selected_episode_indices is not None:
        args.episodes = len(selected_episode_indices)
    if not args.policy_path.expanduser().is_dir():
        raise FileNotFoundError("Policy path does not exist: " + str(args.policy_path))
    if not os.environ.get("DISPLAY"):
        print("[warning] DISPLAY is not set; RLBench normally needs DISPLAY=:99.")

    from rlbench import Environment
    from rlbench.action_modes.action_mode import MoveArmThenGripper
    from rlbench.action_modes.arm_action_modes import (
        ArmActionMode,
        EndEffectorPoseViaIK,
        EndEffectorPoseViaLinearPlanning,
        EndEffectorPoseViaPlanning,
        RelativeFrame,
        assert_action_shape,
        assert_unit_quaternion,
    )
    from rlbench.backend.exceptions import InvalidActionError
    from rlbench.action_modes.gripper_action_modes import Discrete
    from pyrep.errors import IKError

    pointact_pyrep_compatibility = {
        "requested": bool(args.pointact_pyrep_compat),
        "applied": False,
        "implementation": "disabled",
    }
    if args.pointact_pyrep_compat:
        pointact_pyrep_compatibility = apply_pointact_pyrep_compatibility_patch()
        print(
            "[pointact-pyrep-compat] "
            + json.dumps(pointact_pyrep_compatibility, separators=(",", ":")),
            flush=True,
        )

    class EndEffectorPoseViaJointVelocityWaypoint(ArmActionMode):
        """Use an absolute EEF pose as a waypoint and track its IK joints by velocity."""

        def __init__(
            self,
            kp: float,
            max_speed: float,
            joint_tolerance: float,
            max_control_steps: int,
            stall_steps: int,
        ):
            self._kp = float(kp)
            self._max_speed = float(max_speed)
            self._joint_tolerance = float(joint_tolerance)
            self._max_control_steps = int(max_control_steps)
            self._stall_steps = int(stall_steps)

        def action(self, scene, action: np.ndarray):
            assert_action_shape(action, (7,))
            assert_unit_quaternion(action[3:])
            target = np.asarray(action, dtype=np.float64)
            if not scene.check_target_in_workspace(target[:3]):
                raise InvalidActionError(
                    "A joint-velocity waypoint target is outside of workspace."
                )

            arm = scene.robot.arm
            try:
                joint_target = np.asarray(
                    arm.solve_ik_via_jacobian(
                        target[:3], quaternion=target[3:], relative_to=None
                    ),
                    dtype=np.float64,
                )
            except IKError as error:
                raise InvalidActionError(
                    "Could not solve the EEF waypoint with Jacobian IK."
                ) from error

            stalled = 0
            try:
                for _ in range(self._max_control_steps):
                    current = np.asarray(arm.get_joint_positions(), dtype=np.float64)
                    error = joint_target - current
                    if float(np.max(np.abs(error))) <= self._joint_tolerance:
                        break
                    velocity = np.clip(
                        self._kp * error,
                        -self._max_speed,
                        self._max_speed,
                    )
                    arm.set_joint_target_velocities(velocity.tolist())
                    scene.step()
                    success, _ = scene.task.success()
                    if success:
                        break
                    updated = np.asarray(arm.get_joint_positions(), dtype=np.float64)
                    if float(np.max(np.abs(updated - current))) < 1e-5:
                        stalled += 1
                        if stalled >= self._stall_steps:
                            break
                    else:
                        stalled = 0
            finally:
                arm.set_joint_target_velocities(np.zeros_like(joint_target).tolist())

        def action_shape(self, scene) -> tuple:
            return 7,

        def set_control_mode(self, robot):
            robot.arm.set_control_loop_enabled(False)
            robot.arm.set_motor_locked_at_zero_velocity(True)

    class EndEffectorPoseViaFrankaIKServo(ArmActionMode):
        """Pinocchio Panda IK with CoppeliaSim joint-position servo execution."""

        def __init__(self, max_iterations, tolerance, damping, max_control_steps, stall_steps):
            try:
                import pinocchio as pin
            except ImportError as error:
                raise RuntimeError(
                    "franka_ik_servo requires Pinocchio. Install it with "
                    f"'{sys.executable} -m pip install pin'."
                ) from error
            self._pin = pin
            urdf_path = RL_BENCH_ROOT / "urdfs/panda/panda.urdf"
            self._model = pin.buildModelFromUrdf(str(urdf_path))
            self._data = self._model.createData()
            self._tip_frame_id = self._model.getFrameId("Pandatip")
            self._max_iterations = int(max_iterations)
            self._tolerance = float(tolerance)
            self._damping = float(damping)
            self._max_control_steps = int(max_control_steps)
            self._stall_steps = int(stall_steps)
            self._sim_from_urdf = None

        def _joint_configuration(self, arm):
            q = self._pin.neutral(self._model)
            q[:7] = np.asarray(arm.get_joint_positions(), dtype=np.float64)
            return q

        def _tip_pose(self, q):
            self._pin.forwardKinematics(self._model, self._data, q)
            self._pin.updateFramePlacements(self._model, self._data)
            return self._data.oMf[self._tip_frame_id]

        def _solve(self, arm, target_world):
            q_current = self._joint_configuration(arm)
            current_urdf_tip = self._tip_pose(q_current)
            current_sim_tip = np.asarray(arm.get_tip().get_matrix(), dtype=np.float64)
            if self._sim_from_urdf is None:
                self._sim_from_urdf = current_sim_tip @ np.linalg.inv(
                    current_urdf_tip.homogeneous
                )
            target_urdf_matrix = np.linalg.inv(self._sim_from_urdf) @ target_world
            target_urdf = self._pin.SE3(
                target_urdf_matrix[:3, :3], target_urdf_matrix[:3, 3]
            )
            # The Panda is redundant. Start from the measured configuration and
            # deterministic elbow/wrist perturbations, then keep the closest valid solution.
            initial_guesses = [q_current]
            for joint_index, offset in ((2, -0.5), (2, 0.5), (4, -0.5), (4, 0.5), (6, -0.5), (6, 0.5)):
                guess = q_current.copy()
                guess[joint_index] = np.clip(
                    guess[joint_index] + offset,
                    self._model.lowerPositionLimit[joint_index],
                    self._model.upperPositionLimit[joint_index],
                )
                initial_guesses.append(guess)
            best_error = float("inf")
            for initial_q in initial_guesses:
                q = initial_q.copy()
                for _ in range(self._max_iterations):
                    current_tip = self._tip_pose(q)
                    # Pinocchio's LOCAL frame Jacobian uses the target-to-current
                    # error convention. Reversing this SE(3) product makes DLS
                    # step away from the goal instead of towards it.
                    error = self._pin.log6(target_urdf.actInv(current_tip)).vector
                    error_norm = float(np.linalg.norm(error))
                    best_error = min(best_error, error_norm)
                    if error_norm <= self._tolerance:
                        return q[:7]
                    jacobian = self._pin.computeFrameJacobian(
                        self._model,
                        self._data,
                        q,
                        self._tip_frame_id,
                        self._pin.ReferenceFrame.LOCAL,
                    )[:, :7]
                    joint_delta = -jacobian.T @ np.linalg.solve(
                        jacobian @ jacobian.T
                        + self._damping * np.eye(6, dtype=np.float64),
                        error,
                    )
                    joint_delta = np.clip(joint_delta, -0.15, 0.15)
                    q[:7] = np.clip(
                        q[:7] + joint_delta * 0.5,
                        self._model.lowerPositionLimit[:7],
                        self._model.upperPositionLimit[:7],
                    )
            raise InvalidActionError(
                "Pinocchio Franka IK did not converge for the EEF waypoint; "
                + "best_se3_error=" + format(best_error, ".5f")
            )

        def action(self, scene, action: np.ndarray):
            assert_action_shape(action, (7,))
            assert_unit_quaternion(action[3:])
            target = np.asarray(action, dtype=np.float64)
            if not scene.check_target_in_workspace(target[:3]):
                raise InvalidActionError("A Franka IK servo target is outside of workspace.")
            arm = scene.robot.arm
            target_matrix = np.eye(4, dtype=np.float64)
            target_matrix[:3, :3] = self._pin.Quaternion(
                float(target[6]), float(target[3]), float(target[4]), float(target[5])
            ).matrix()
            target_matrix[:3, 3] = target[:3]
            joint_target = self._solve(arm, target_matrix)
            arm.set_joint_target_positions(joint_target.tolist())
            stalled = 0
            for _ in range(self._max_control_steps):
                current = np.asarray(arm.get_joint_positions(), dtype=np.float64)
                if float(np.max(np.abs(joint_target - current))) <= 0.01:
                    break
                scene.step()
                success, _ = scene.task.success()
                if success:
                    break
                updated = np.asarray(arm.get_joint_positions(), dtype=np.float64)
                stalled = stalled + 1 if float(np.max(np.abs(updated - current))) < 1e-5 else 0
                if stalled >= self._stall_steps:
                    break
            arm.set_joint_target_positions(np.asarray(arm.get_joint_positions()).tolist())

        def action_shape(self, scene):
            return 7,

    class EndEffectorPoseViaFrankaCartesianServo(ArmActionMode):
        """MuJoCo-style Cartesian velocity servo backed by Pinocchio Jacobians."""

        def __init__(self, damping, max_speed, max_control_steps):
            try:
                import pinocchio as pin
            except ImportError as error:
                raise RuntimeError(
                    "franka_cartesian_servo requires Pinocchio. Install it with "
                    f"'{sys.executable} -m pip install pin'."
                ) from error
            self._pin = pin
            self._model = pin.buildModelFromUrdf(
                str(RL_BENCH_ROOT / "urdfs/panda/panda.urdf")
            )
            self._data = self._model.createData()
            self._tip_frame_id = self._model.getFrameId("Pandatip")
            self._damping = float(damping)
            self._max_speed = float(max_speed)
            self._max_control_steps = int(max_control_steps)
            self._sim_from_urdf = None

        def _q(self, arm):
            q = self._pin.neutral(self._model)
            q[:7] = np.asarray(arm.get_joint_positions(), dtype=np.float64)
            return q

        def _urdf_tip(self, q):
            self._pin.forwardKinematics(self._model, self._data, q)
            self._pin.updateFramePlacements(self._model, self._data)
            return self._data.oMf[self._tip_frame_id]

        def action(self, scene, action: np.ndarray):
            assert_action_shape(action, (7,))
            assert_unit_quaternion(action[3:])
            arm = scene.robot.arm
            q = self._q(arm)
            current_urdf = self._urdf_tip(q)
            current_sim = np.asarray(arm.get_tip().get_matrix(), dtype=np.float64)
            if self._sim_from_urdf is None:
                self._sim_from_urdf = current_sim @ np.linalg.inv(current_urdf.homogeneous)
            target_sim = np.eye(4, dtype=np.float64)
            target_sim[:3, :3] = self._pin.Quaternion(
                float(action[6]), float(action[3]), float(action[4]), float(action[5])
            ).matrix()
            target_sim[:3, 3] = np.asarray(action[:3], dtype=np.float64)
            target_urdf_matrix = np.linalg.inv(self._sim_from_urdf) @ target_sim
            target_urdf = self._pin.SE3(
                target_urdf_matrix[:3, :3], target_urdf_matrix[:3, 3]
            )
            try:
                for _ in range(self._max_control_steps):
                    q = self._q(arm)
                    current = self._urdf_tip(q)
                    error = self._pin.log6(current.actInv(target_urdf)).vector
                    jacobian = self._pin.computeFrameJacobian(
                        self._model, self._data, q, self._tip_frame_id,
                        self._pin.ReferenceFrame.LOCAL,
                    )[:, :7]
                    qdot = jacobian.T @ np.linalg.solve(
                        jacobian @ jacobian.T
                        + self._damping * np.eye(6, dtype=np.float64),
                        error,
                    )
                    qdot = np.clip(qdot, -self._max_speed, self._max_speed)
                    arm.set_joint_target_velocities(qdot.tolist())
                    scene.step()
                    success, _ = scene.task.success()
                    if success:
                        break
            finally:
                arm.set_joint_target_velocities(np.zeros(7, dtype=np.float64).tolist())

        def action_shape(self, scene):
            return 7,

        def set_control_mode(self, robot):
            robot.arm.set_control_loop_enabled(False)
            robot.arm.set_motor_locked_at_zero_velocity(True)

    dataset_reset_specs = None
    if args.dataset_root is not None:
        dataset_reset_specs = load_dataset_reset_specs(
            args.dataset_root,
            args.task,
            args.dataset_episodes,
            args.episodes,
        )
        args.episodes = len(dataset_reset_specs)

    if args.run_dir is None:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        run_dir = args.output_dir.expanduser().resolve() / (args.task + "_" + stamp)
    else:
        run_dir = args.run_dir.expanduser().resolve()
    run_dir.mkdir(parents=True, exist_ok=False)
    if args.log_control_details or args.save_control_log:
        CONTROL_LOG_PATH = run_dir / "control.log"
    config = vars(args).copy()
    config["torch_determinism"] = torch_determinism
    config["scene_bounds"] = RLBENCH_SCENE_BOUNDS.tolist()
    config["pointact_workspace_min"] = POINTACT_WORKSPACE_MIN.tolist()
    config["pointact_workspace_max"] = POINTACT_WORKSPACE_MAX.tolist()
    config["pointact_pyrep_compatibility"] = pointact_pyrep_compatibility
    config["executed_action_alignment"] = {
        "enabled": bool(args.save_action_records),
        "directory": "executed_action_alignment",
        "model_action_shape": ["executed_environment_steps", 10],
        "one_file_per_episode": True,
        "file_suffix": "_executed_model_actions_relative10.npy",
        "capture": (
            "online_after_successful_controller_action_including_song_"
            "gripper_only_fallback"
        ),
        "simulator_actions_directory": "actions",
    }
    if args.gripper_template == "reap":
        config["virtual_gripper"] = canonical_reap_metadata()
        config["virtual_gripper"].update(
            {
                "selected_gripper_len_m": float(args.gripper_len),
                "selected_virtual_gripper_local_offset_m": [
                    0.0,
                    0.0,
                    -float(args.gripper_len),
                ],
                "virtual_tcp_sync": bool(args.sync_virtual_gripper_tcp),
                "physical_eef_to_model_tcp_translation_m": (
                    physical_eef_to_model_tcp(args)[:3, 3].tolist()
                ),
                "model_frame_gripper_len_m": float(
                    gripper_len_in_model_tcp_frame(args)
                ),
                "model_state_semantics": (
                    "identity_normalized_current_virtual_tcp"
                    if args.sync_virtual_gripper_tcp
                    else "identity_normalized_physical_panda_tip"
                ),
                "controller_target_semantics": "physical_panda_tip",
            }
        )
    config["environment"] = {
        "display": os.environ.get("DISPLAY"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "eval_worker_slot": os.environ.get("EVAL_WORKER_SLOT"),
        "eval_workers": os.environ.get("EVAL_WORKERS"),
        "eval_display_base": os.environ.get("EVAL_DISPLAY_BASE"),
        "eval_gpu_id": os.environ.get("EVAL_GPU_ID"),
        "coppeliasim_root": os.environ.get("COPPELIASIM_ROOT"),
        "rlbench_planner_max_time_ms": os.environ.get(
            "RLBENCH_PLANNER_MAX_TIME_MS"
        ),
        "qt_qpa_platform": os.environ.get("QT_QPA_PLATFORM"),
        "qt_qpa_platform_plugin_path": os.environ.get(
            "QT_QPA_PLATFORM_PLUGIN_PATH"
        ),
        "qt_x11_no_mitshm": os.environ.get("QT_X11_NO_MITSHM"),
        "water_plant_collision": args.water_plant_collision,
        "water_drop_collision": args.water_drop_collision,
        "python": sys.executable,
    }
    launch_command_file = os.environ.get("EVAL_LAUNCH_COMMAND_FILE")
    if launch_command_file:
        command_path = Path(launch_command_file)
        if command_path.is_file():
            config["bash_invocation"] = command_path.read_text(encoding="utf-8").strip()
            config["command_file"] = str(command_path.resolve())
    if dataset_reset_specs is not None:
        config["dataset_reset_global_episodes"] = [
            int(spec["global_episode_index"]) for spec in dataset_reset_specs
        ]
    with open(run_dir / "config.json", "w", encoding="utf-8") as file:
        json.dump(config, file, default=str, indent=2)
    config_log_line = "[eval-config] " + json.dumps(
        config, default=str, separators=(",", ":")
    )
    print(config_log_line, flush=True)
    with open(run_dir / "eval_parameters.log", "w", encoding="utf-8") as file:
        file.write(config_log_line + "\n")
    if CONTROL_LOG_PATH is not None:
        with open(CONTROL_LOG_PATH, "w", encoding="utf-8") as file:
            file.write(config_log_line + "\n")

    infer = SmolVLA_ModelInference(
        policy_path=args.policy_path.expanduser(),
        device=args.device,
        visualize_foreground=args.visualize_foreground,
    )
    # Saving an action PLY also saves the exact LitePT operation probabilities
    # produced by that same model forward pass.  This does not open a viewer.
    if args.save_action_visualizations:
        infer.policy.model.capture_pointseg_visualization = True
    if args.arm_action_mode == "planning":
        arm_action_mode = EndEffectorPoseViaPlanning(
            absolute_mode=True,
            frame=RelativeFrame.WORLD,
            collision_checking=args.collision_checking,
            settle_same_target_without_replanning=(
                args.planning_same_target_settle
            ),
        )
    elif args.arm_action_mode == "linear_planning":
        arm_action_mode = EndEffectorPoseViaLinearPlanning(
            absolute_mode=True,
            frame=RelativeFrame.WORLD,
            collision_checking=args.collision_checking,
            max_joint_goal_distance=(
                args.linear_planning_max_joint_goal_distance
            ),
            max_joint_path_length=args.linear_planning_max_joint_path_length,
            max_joint_path_ratio=args.linear_planning_max_joint_path_ratio,
            max_single_joint_travel=(
                args.linear_planning_max_single_joint_travel
            ),
        )
    elif args.arm_action_mode == "ik":
        arm_action_mode = EndEffectorPoseViaIK(
            absolute_mode=True,
            frame=RelativeFrame.WORLD,
            collision_checking=args.collision_checking,
        )
    elif args.arm_action_mode == "joint_velocity_waypoint":
        arm_action_mode = EndEffectorPoseViaJointVelocityWaypoint(
            kp=args.joint_velocity_kp,
            max_speed=args.joint_velocity_max_speed,
            joint_tolerance=args.joint_velocity_joint_tolerance,
            max_control_steps=args.waypoint_max_control_steps,
            stall_steps=args.joint_velocity_stall_steps,
        )
    elif args.arm_action_mode == "franka_ik_servo":
        arm_action_mode = EndEffectorPoseViaFrankaIKServo(
            max_iterations=args.franka_ik_max_iterations,
            tolerance=args.franka_ik_tolerance,
            damping=args.franka_ik_damping,
            max_control_steps=args.waypoint_max_control_steps,
            stall_steps=args.joint_velocity_stall_steps,
        )
    else:
        arm_action_mode = EndEffectorPoseViaFrankaCartesianServo(
            damping=args.franka_ik_damping,
            max_speed=args.joint_velocity_max_speed,
            max_control_steps=args.waypoint_max_control_steps,
        )

    env = Environment(
        MoveArmThenGripper(arm_action_mode, Discrete()),
        obs_config=make_observation_config(args.image_size),
        headless=True,
        static_positions=False,
    )
    results = []
    try:
        env.launch()
        args.worldflow_t_base_world = None
        if bool(getattr(infer.policy.config, "worldflow_enable", False)):
            reference_frame = str(
                getattr(infer.policy.config, "worldflow_reference_frame", "")
            )
            if reference_frame != "robot_base":
                raise ValueError(
                    "RLBench online WorldFlow evaluation currently requires "
                    "worldflow_reference_frame='robot_base', got "
                    + repr(reference_frame)
                )
            t_world_base = rlbench_panda_link0_to_world_matrix(env._robot.arm)
            args.worldflow_t_base_world = np.linalg.inv(t_world_base)
            print(
                "[worldflow-eval] "
                + json.dumps(
                    {
                        "enabled": True,
                        "reference_frame": reference_frame,
                        "current_ee_pose_key": "worldflow.current_ee_pose",
                        "current_ee_pose_semantics": "T_base_model_virtual_tcp pose9",
                        "policy_output_semantics": "ego_current_virtual_tcp_relative_pose9_plus_gripper",
                        "controller_target_semantics": "T_world_physical_panda_tip",
                        "physical_eef_to_model_tcp_translation_m": (
                            physical_eef_to_model_tcp(args)[:3, 3].tolist()
                        ),
                        "T_world_base": np.asarray(t_world_base).tolist(),
                        "T_base_world": np.asarray(args.worldflow_t_base_world).tolist(),
                    },
                    separators=(",", ":"),
                ),
                flush=True,
            )
        task_env = env.get_task(task_class_from_name(args.task))
        task_env.set_variation(args.variation)
        rollout_episode_indices = (
            list(range(args.episodes))
            if selected_episode_indices is None
            else selected_episode_indices
        )
        for episode_position, episode_index in enumerate(rollout_episode_indices):
            dataset_reset_spec = (
                None
                if dataset_reset_specs is None
                else dataset_reset_specs[episode_position]
            )
            attempt = 0
            retry_started_at = time.monotonic()
            while True:
                original_seed = args.seed
                original_model_noise_seed = args.model_noise_seed
                args.seed = original_seed + attempt
                args.model_noise_seed = original_model_noise_seed + attempt
                try:
                    result = run_episode(
                        task_env,
                        infer,
                        args,
                        episode_index,
                        args.task,
                        run_dir,
                        dataset_reset_spec=dataset_reset_spec,
                        completed_successes=sum(int(item["success"]) for item in results),
                        completed_episodes=len(results),
                    )
                finally:
                    args.seed = original_seed
                    args.model_noise_seed = original_model_noise_seed
                retryable = (
                    not args.disable_episode_retries
                    and not result["success"]
                    and result["end_reason"]
                    in {"controller_error", "waypoint_controller_rejected"}
                )
                retry_elapsed_seconds = time.monotonic() - retry_started_at
                retry_timed_out = retry_elapsed_seconds >= args.controller_error_retry_timeout_seconds
                retry_limit_reached = (
                    args.controller_error_retries >= 0
                    and attempt >= args.controller_error_retries
                )
                if not retryable or retry_timed_out or retry_limit_reached:
                    result["controller_error_attempts"] = int(attempt)
                    result["controller_error_retry_elapsed_seconds"] = float(retry_elapsed_seconds)
                    result["controller_error_retry_timed_out"] = bool(retryable and retry_timed_out)
                    break
                attempt += 1
                print(
                    "[eval-retry] task="
                    + args.task
                    + " episode="
                    + str(episode_index)
                    + " failed_attempt="
                    + str(attempt)
                    + " error="
                    + str(result["error"]),
                    flush=True,
                )
                print(
                    "[eval-retry] resetting same episode before attempt="
                    + str(attempt + 1),
                    flush=True,
                )
            results.append(result)
            completed_successes = sum(int(item["success"]) for item in results)
            completed_episodes = len(results)
            print(
                "[eval-progress] task="
                + args.task
                + " episode="
                + str(episode_position + 1)
                + "/"
                + str(args.episodes)
                + " episode_index="
                + str(episode_index)
                + " current_success="
                + str(result["success"])
                + " task_successes="
                + str(completed_successes)
                + "/"
                + str(completed_episodes)
                + " task_success_rate="
                + format(completed_successes / max(completed_episodes, 1), ".3f")
                + " actions="
                + str(result["environment_actions"])
                + " frames="
                + str(result["video_frames"])
                + " end_reason="
                + str(result["end_reason"])
                + " language="
                + result["language"]
            )
            if result["error"] is not None:
                print("[episode-error] " + result["error"])
    finally:
        env.shutdown()
        infer.close()

    success_count = sum(int(item["success"]) for item in results)
    summary = {
        "task": args.task,
        "episodes": len(results),
        "successes": success_count,
        "success_rate": success_count / max(len(results), 1),
        "executed_action_alignment": {
            "enabled": bool(args.save_action_records),
            "directory": "executed_action_alignment" if args.save_action_records else None,
            "executed_environment_steps": int(
                sum(int(item["environment_actions"]) for item in results)
            ),
            "file_suffix": (
                "_executed_model_actions_relative10.npy"
                if args.save_action_records
                else None
            ),
        },
        "results": results,
    }
    with open(run_dir / "summary.json", "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)
    print(
        "[task-summary] task="
        + args.task
        + " episodes="
        + str(len(results))
        + " successes="
        + str(success_count)
        + " success_rate="
        + format(success_count / max(len(results), 1), ".3f")
        + " run_dir="
        + str(run_dir)
    )
    print(json.dumps(summary, indent=2))
    print("[done] " + str(run_dir))


if __name__ == "__main__":
    main()
