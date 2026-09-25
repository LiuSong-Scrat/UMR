#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Collect RLBench expert demonstrations into the Song LeRobot format.

This file is deliberately independent from RLBench's original
``dataset_generator.py``.  RLBench's generator removes point clouds before
writing a demo, while this collector reads the live observations first and
stores the clouds in LeRobot-compatible episode sidecars.

The main table has the same 10-dimensional convention used by the existing
CALVIN/Song data:

    [x, y, z, first_rotation_column(3), second_rotation_column(3), width]

For ``observation.state``, the first nine values are the achieved EEF pose
expressed relative to the EEF pose in frame zero. By default, ``action`` is the
RLBench expert's commanded Panda joint target converted by FK and expressed
relative to the same EEF0. ``--action-label-mode executed`` instead uses the
achieved next EEF state as the action label. The point cloud sidecar contains
the finite front-camera world cloud transformed to the current EEF frame,
followed by RGB in [0, 255].

RLBench's expert itself uses JointVelocity + Discrete gripper internally.
Those original expert commands are saved separately as
``raw_expert_actions/episode_XXXXXX.npy``.  They are not silently substituted
for the EEF labels in the main LeRobot action column.
"""

import argparse
import importlib
import json
import math
import multiprocessing as mp
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import zarr


# Make the repository's LeRobot package and the existing point-cloud helpers
# importable when this file is run directly by an absolute path. The defaults
# support the current monorepo layout; environment variables support a
# standalone RLBench checkout.
RL_BENCH_ROOT = Path(
    os.environ.get("RLBENCH_ROOT", str(Path(__file__).resolve().parents[1]))
).expanduser().resolve()
REPO_ROOT = Path(
    os.environ.get("LEROBOT_ROOT", str(RL_BENCH_ROOT.parents[1]))
).expanduser().resolve()
SONG_SCRIPT_ROOT = Path(
    os.environ.get(
        "SONG_SCRIPTS",
        str(REPO_ROOT / "benchmarks" / "song_real_libero" / "scripts"),
    )
).expanduser().resolve()
COPPELIASIM_ROOT = Path(
    os.environ.get("COPPELIASIM_ROOT", str(RL_BENCH_ROOT / "../CoppeliaSim"))
).expanduser().resolve()
DEFAULT_OUTPUT_ROOT = (
    RL_BENCH_ROOT
    / "datasets"
    / ("rlbench_lerobot_" + time.strftime("%Y%m%d_%H%M%S"))
)
os.environ.setdefault("DISPLAY", ":99")
os.environ.setdefault("QT_QPA_PLATFORM", "xcb")
os.environ.setdefault("COPPELIASIM_ROOT", str(COPPELIASIM_ROOT))
os.environ.setdefault("QT_QPA_PLATFORM_PLUGIN_PATH", str(COPPELIASIM_ROOT))
# RLBench validates ordinary expert waypoints during reset.  That validation
# uses RRTConnect and needs a little more time after random task placement.
# Keep this separate from action-execution planner timing.
os.environ.setdefault("RLBENCH_WAYPOINT_PLANNER_MAX_TIME_MS", "50")
library_path = os.environ.get("LD_LIBRARY_PATH", "")
if str(COPPELIASIM_ROOT) not in library_path.split(":"):
    os.environ["LD_LIBRARY_PATH"] = str(COPPELIASIM_ROOT) + (":" + library_path if library_path else "")
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(SONG_SCRIPT_ROOT))

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from libero_setting.libero_pointcloud_utils import (
    RLBENCH_PANDA_GRIPPER_TEMPLATE,
    RLBENCH_PANDA_GRIPPER_TEMPLATE_VERSION,
    RLBENCH_PANDA_MAX_WIDTH,
    add_world_gripper_clouds_to_episode,
    fast_inverse_homogeneous,
    pose9_to_homo_np,
    sample_or_repeat_points,
)
from reap_gripper import (
    LIBERO_GRIPPER_TEMPLATE,
    LIBERO_GRIPPER_TEMPLATE_VERSION,
    LIBERO_GRIPPER_WIDTH_NORMALIZATION_MAX,
    LIBERO_REAP_GRIPPER_LEN,
    LIBERO_REAP_OPENING_MAX_WIDTH,
    LIBERO_REAP_TEMPLATE_MAX_WIDTH,
    canonical_reap_metadata,
    libero_reap_width_percent_from_physical,
)
from worldflow_sidecars import (
    RLBENCH_PANDA_LINK0_FRAME_VERSION,
    RLBENCH_PANDA_LINK0_TRANSFORM_SOURCE,
    WORLD_BASE_ACTION_TARGET_EE_POSE_DIR,
    WORLD_BASE_EE_POSE_DIR,
    build_robot_base_episode_sidecars,
    rlbench_panda_link0_to_world_matrix,
    validate_rigid_transform,
    write_sidecar_metadata as write_robot_base_sidecar_metadata,
)
from lerobot.policies.smolvla.song_pointseg import save_point_clouds_zarr


DEFAULT_TASKS = [
    # "close_box",
    # "close_laptop_lid",
    "toilet_seat_down",
    "sweep_to_dustpan",
    "close_fridge",
    # "phone_on_base",
    # "take_umbrella_out_of_umbrella_stand",
    # "take_frame_off_hanger",
    # "stack_wine",
    "water_plants",
]

FEATURE_NAMES = ["x", "y", "z", "x1", "y1", "z1", "x2", "y2", "z2", "gripper"]
ACTION_LABEL_MODES = ("expert_target", "executed")
POINT_DIR = "point_clouds"
POSE_DIR = "world_ee_poses"
RAW_ACTION_DIR = "raw_expert_actions"
RAW_ACTION_FULL_DIR = "raw_expert_actions_full"
TASK_STATE_DIR = "initial_task_states"
OBJECT_STATE_DIR = "initial_object_states"
# Historical Song/RLBench point-cloud crop kept for dataset/evaluation compatibility.
RLBENCH_SCENE_BOUNDS = np.asarray(
    [-0.5, -1, 0.7505, 1.5, 1, 2.0], dtype=np.float32
)
OBJECT_STATE_KEYS = [
    "initial_object_names",
    "initial_object_handles",
    "initial_object_types",
    "initial_object_parent_handles",
    "initial_object_parent_names",
    "initial_object_poses",
    "initial_object_linear_velocities",
    "initial_object_angular_velocities",
    "initial_object_joint_positions",
    "initial_object_joint_velocities",
    "initial_object_joint_target_positions",
    "initial_object_joint_target_velocities",
]


def action_semantics(action_label_mode):
    if action_label_mode == "expert_target":
        return "expert Panda joint target converted by FK relative to episode EEF0"
    if action_label_mode == "executed":
        return "achieved next-frame EEF pose relative to episode EEF0"
    raise ValueError("Unknown action label mode: " + str(action_label_mode))


def action_semantics_version(action_label_mode):
    if action_label_mode == "expert_target":
        return "rlbench_expert_joint_target_fk_eef0_object_state_v3"
    if action_label_mode == "executed":
        return "rlbench_achieved_next_eef_state_eef0_v1"
    raise ValueError("Unknown action label mode: " + str(action_label_mode))


def parse_args():
    parser = argparse.ArgumentParser(description="Collect RLBench expert data as Song LeRobot point clouds.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Dataset output directory. Defaults to a timestamped directory under benchmarks/RLBench/datasets.",
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=None,
        help=(
            "Optional explicit artifact directory. Normally this is derived from "
            "--output-root; it is required when --pack-only rebuilds a dataset at "
            "a different temporary output path."
        ),
    )
    parser.add_argument(
        "--pack-only",
        action="store_true",
        help=(
            "Do not launch RLBench or collect demos. Read the completed records and "
            "collection settings from --artifact-root/manifest.json and rebuild only "
            "the LeRobot dataset."
        ),
    )
    parser.add_argument(
        "--artifacts-only",
        action="store_true",
        help=(
            "Collect and validate raw per-episode artifacts, then stop without "
            "building a LeRobot dataset or PointSeg cache."
        ),
    )
    parser.add_argument("--repo-id", default="rlbench_song_pointcloud")
    parser.add_argument("--tasks", nargs="*", default=None)
    parser.add_argument("--all-tasks", action="store_true")
    parser.add_argument("--episodes-per-task", type=int, default=10)
    parser.add_argument("--episode-start", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument(
        "--episode-indices",
        nargs="+",
        type=int,
        default=None,
        help=(
            "Collect these exact local episode indices instead of a contiguous "
            "range. Currently supported with --collection-workers 1."
        ),
    )
    parser.add_argument("--variation", type=int, default=0)
    parser.add_argument(
        "--collection-seed",
        type=int,
        default=None,
        help="Optional NumPy seed for reproducible live-demo initialization.",
    )
    parser.add_argument("--num-points", type=int, default=10000)
    parser.add_argument(
        "--cache-current-points",
        type=int,
        default=None,
        help="Number of points retained in the current-frame PointSeg cache. Defaults to --num-points.",
    )
    parser.add_argument(
        "--cache-future-points",
        type=int,
        default=None,
        help="Number of points retained in the future-frame PointSeg cache. Defaults to --cache-current-points.",
    )
    parser.add_argument("--gripper-points", type=int, default=500)
    parser.add_argument(
        "--gripper-template",
        choices=[RLBENCH_PANDA_GRIPPER_TEMPLATE, LIBERO_GRIPPER_TEMPLATE],
        default=LIBERO_GRIPPER_TEMPLATE,
        help=(
            "Virtual gripper merged into point clouds before PointSeg caching. "
            "reap is the canonical RLBench-aligned REAP v4 gripper."
        ),
    )
    parser.add_argument("--gripper-max-width", type=float, default=RLBENCH_PANDA_MAX_WIDTH)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument(
        "--front-camera-position-m",
        type=float,
        nargs=3,
        default=None,
        metavar=("X", "Y", "Z"),
        help=(
            "Optional world-frame position override for RLBench cam_front. "
            "The default scene camera is unchanged when omitted."
        ),
    )
    parser.add_argument(
        "--front-camera-look-at-m",
        type=float,
        nargs=3,
        default=None,
        metavar=("X", "Y", "Z"),
        help=(
            "Optional world-frame look-at point for cam_front. The camera's "
            "local +Z axis is aimed at this point while preserving world +Z "
            "as image-up as closely as possible."
        ),
    )
    parser.add_argument("--max-demo-attempts", type=int, default=10)
    parser.add_argument(
        "--post-success-frames",
        type=int,
        default=10,
        help=(
            "Trim live expert demos after the first successful observation, "
            "retaining this many following dataset frames. Transition alignment "
            "automatically retains one extra terminal observation. Use -1 to disable."
        ),
    )
    parser.add_argument(
        "--stop-after-success",
        action="store_true",
        help=(
            "Stop the live expert waypoint program as soon as task success is "
            "observed. Hold the last command only long enough to retain "
            "--post-success-frames (plus one terminal observation for transition "
            "alignment), rather than executing all remaining waypoints."
        ),
    )
    parser.add_argument(
        "--min-first-success-frame",
        type=int,
        default=0,
        help=(
            "Reject a live expert candidate whose task-success condition is "
            "already reached before this frame. This guards matched-scene "
            "repairs against a dynamic object settling into the goal during reset."
        ),
    )
    parser.add_argument(
        "--min-initial-scene-distance-m",
        type=float,
        default=0.0,
        help=(
            "Reject a newly reset scene when its task-object pose signature is "
            "closer than this distance to any already accepted episode of the "
            "same task. Rotation is converted to metres using "
            "--scene-rotation-radius-m. Zero disables scene de-duplication."
        ),
    )
    parser.add_argument(
        "--scene-rotation-radius-m",
        type=float,
        default=0.10,
        help=(
            "Lever arm used to convert initial object quaternion/joint angular "
            "differences into the metre-equivalent scene-distance metric."
        ),
    )
    parser.add_argument(
        "--abort-on-episode-failure",
        action="store_true",
        help=(
            "Abort collection after the configured attempts for one episode "
            "instead of retrying that episode indefinitely. This is intended "
            "for isolated exact-scene planner diagnostics."
        ),
    )
    parser.add_argument(
        "--expert-path-mode",
        choices=[
            "linear_then_rrt",
            "linear_only",
            "segmented_linear",
            "franka_cartesian_servo",
            "cartesian_detour",
            "segmented_linear_then_rrt",
            "segmented_linear_then_best_of_n",
            "best_of_n_all_points",
            "rrt_only",
            "phone_predefined_path_waypoint3",
            "phone_prm_waypoint3",
            "phone_prm_0_3",
            "phone_prm_all",
            "phone_cartesian_0_3",
            "phone_staged_linear_0_3",
            "phone_staged_linear_then_rrt_0_3",
            "phone_rrt_cartesian_shortcut_0_3",
            "phone_best_ik_joint_interp_0_3",
            "phone_baseframe_reference_ik_0_3",
            "phone_best_of_n_0_3",
        ],
        default="linear_then_rrt",
        help=(
            "Expert waypoint planner. linear_then_rrt preserves RLBench's "
            "linear-first/RRTConnect-fallback behavior; linear_only forces "
            "ordinary Point waypoints through Cartesian linear IK and never "
            "calls the nonlinear planner; segmented_linear divides the same "
            "Cartesian line and quaternion SLERP into short IK segments; "
            "franka_cartesian_servo follows that line and shortest-arc SLERP "
            "online with a damped-Jacobian joint-velocity servo and a "
            "start-configuration nullspace target; it never calls OMPL; "
            "cartesian_detour deterministically searches bounded smooth arcs "
            "around the same line, solves every arc by continuous linear IK, "
            "and selects the smallest feasible deviation without OMPL; "
            "segmented_linear_then_rrt uses the same short targets but permits "
            "RRTConnect only for an individually infeasible segment; "
            "segmented_linear_then_best_of_n first tries the same complete "
            "short-arc segmented path, then samples full-waypoint stock paths "
            "and rejects Cartesian, rotational, or joint-space loops; "
            "best_of_n_all_points samples the stock collision-aware planner "
            "multiple times at every ordinary Point waypoint, rejects joint "
            "wraps and Cartesian/orientation loops, and executes the lowest-cost path; "
            "rrt_only forces ordinary Point "
            "waypoints directly through nonlinear RRTConnect; "
            "phone_predefined_path_waypoint3 leaves the pickup side on the "
            "stock planner and replaces only phone_on_base waypoint3 with one "
            "direct CoppeliaSim CartesianPath whose orientation follows the "
            "short quaternion arc; it has no nonlinear fallback; "
            "phone_prm_waypoint3 leaves the pickup side on the stock planner "
            "and uses only OMPL PRM candidates for phone_on_base waypoint3, "
            "selecting the candidate with the smallest measured Cartesian, "
            "orientation, and joint-space motion; "
            "phone_prm_0_3 uses PRM only at phone waypoint0/waypoint3 and "
            "strict Cartesian-linear IK at all other Point waypoints; it "
            "never invokes an RRT-family planner; "
            "phone_prm_all uses scored PRM candidates at every phone Point "
            "waypoint, without invoking an RRT-family planner; "
            "phone_cartesian_0_3 replaces phone_on_base waypoint0 and "
            "waypoint3 with dynamically authored Cartesian paths. A stock "
            "plan supplies reachable EEF positions and a safe fallback; the "
            "candidate selector compares short-arc, long-arc, guide-orientation, "
            "and stock paths, prefers a negative waypoint0 wrist branch, and "
            "executes the lowest-cost feasible choice. Predefined "
            "Cartesian paths are preserved; phone_staged_linear_0_3 instead "
            "uses only deterministic direct or lift/translate/descend linear "
            "IK candidates for phone waypoint0/waypoint3. It never requests "
            "an RRT guide and never silently falls back to the stock planner; "
            "phone_staged_linear_then_rrt_0_3 uses the same deterministic "
            "staged targets but permits a locally scored RRT fallback only "
            "for an individually infeasible segment; "
            "phone_rrt_cartesian_shortcut_0_3 first obtains a collision-aware "
            "RRT guide for phone waypoint0/waypoint3 and greedily replaces "
            "the guide's bends with the longest feasible Cartesian-linear "
            "shortcuts; "
            "phone_best_ik_joint_interp_0_3 samples multiple collision-free "
            "IK goal configurations and executes the collision-free joint "
            "interpolation whose EEF path has the smallest Cartesian arc; "
            "phone_baseframe_reference_ik_0_3 retrieves nearby clean IK "
            "branches by robot-base-frame waypoint pose, locally refines each "
            "branch to the exact live target, and selects a collision-free "
            "low-joint-travel interpolation; "
            "phone_best_of_n_0_3 samples multiple collision-aware stock plans "
            "for those two waypoints and rejects large joint-space loops before "
            "selecting the lowest-travel candidate."
        ),
    )
    parser.add_argument(
        "--allow-rrt",
        action="store_true",
        help=(
            "Explicitly opt back into a path mode that can call RRT/OMPL. "
            "RRT is forbidden by default for new dataset collection because "
            "a successful but highly curved path is not an acceptable expert "
            "demonstration. This switch exists only for deliberate historical "
            "reproduction and planner diagnostics."
        ),
    )
    parser.add_argument(
        "--phone-path-candidates",
        type=int,
        default=12,
        help=(
            "Number of independently sampled collision-aware plans for each "
            "phone waypoint0/waypoint3 call in phone_best_of_n_0_3 mode."
        ),
    )
    parser.add_argument(
        "--phone-reference-artifact",
        type=Path,
        default=None,
        help=(
            "Optional successful phone episode arrays.npz. Its waypoint-end "
            "joint configurations seed the same known IK branch; each seeded "
            "goal is locally refined and revalidated, never replayed blindly."
        ),
    )
    parser.add_argument(
        "--phone-smooth-ik-reference-root",
        type=Path,
        default=None,
        help=(
            "Artifact directory containing clean phone_on_base episodes used "
            "as robot-base-frame IK branch references by "
            "phone_baseframe_reference_ik_0_3."
        ),
    )
    parser.add_argument(
        "--phone-smooth-ik-exclude-episodes",
        type=int,
        nargs="*",
        default=(),
        help=(
            "Episode indices excluded from the smooth-IK reference bank, "
            "normally the trajectories currently being repaired."
        ),
    )
    parser.add_argument(
        "--phone-postgrasp-lift-m",
        type=float,
        default=0.08,
        help=(
            "Vertical retract distance used by phone_prm_all immediately "
            "after grasping the handset. Larger values provide clean "
            "clearance for near-base scenes before horizontal transport."
        ),
    )
    parser.add_argument(
        "--phone-best-of-n-waypoint3-only",
        action="store_true",
        help=(
            "With phone_best_of_n_0_3, phone_rrt_cartesian_shortcut_0_3, or "
            "a phone_staged_linear*_0_3 mode, "
            "leave waypoint0 on the selected base planner and run expensive "
            "best-of-N hard-guarded search only at waypoint3, after the phone "
            "has been grasped. For the shortcut mode, --phone-path-candidates "
            "controls the number of independently sampled RRT guides."
        ),
    )
    parser.add_argument(
        "--path-xyz-loop-floor-m",
        type=float,
        default=0.08,
        help=(
            "Minimum allowed Cartesian path length before best-of-N loop "
            "rejection. The effective limit is max(this value, 3x direct XYZ "
            "distance). A larger value can preserve necessary obstacle detours "
            "for very short waypoint displacements."
        ),
    )
    parser.add_argument(
        "--path-rotation-excess-limit-rad",
        type=float,
        default=-1.0,
        help=(
            "For phone_best_of_n_0_3 waypoint3 only, reject a scored planner "
            "candidate when its cumulative EEF rotation exceeds the direct "
            "start-to-target quaternion geodesic by more than this many radians. "
            "A negative value disables this guard."
        ),
    )
    parser.add_argument(
        "--roll-path-max-lateral-deviation-m",
        type=float,
        default=0.10,
        help=(
            "For roll-symmetry path-cost selection, reject a 0/180-degree "
            "candidate when its EEF leaves the straight start-to-target chord "
            "by more than this distance. A negative value disables the guard."
        ),
    )
    parser.add_argument(
        "--roll-path-max-detour-ratio",
        type=float,
        default=2.5,
        help=(
            "For roll-symmetry path-cost selection, reject an EEF path whose "
            "length/direct-distance ratio exceeds this value. A negative value "
            "disables the guard."
        ),
    )
    parser.add_argument(
        "--roll-path-max-wrist-travel-rad",
        type=float,
        default=4.5,
        help=(
            "Maximum summed absolute travel of Panda joints 5-7 for one "
            "roll-symmetry candidate. A negative value disables the guard."
        ),
    )
    parser.add_argument(
        "--roll-path-max-joint-step-rad",
        type=float,
        default=0.50,
        help=(
            "Maximum change of any Panda joint between adjacent candidate "
            "path configurations. A negative value disables the guard."
        ),
    )
    parser.add_argument(
        "--phone-waypoint3-max-detour-ratio",
        type=float,
        default=3.0,
        help=(
            "Hard Cartesian path-length/direct-distance ratio limit for "
            "phone_best_of_n_0_3 waypoint3. A negative value disables this "
            "guard. The default 3.0 preserves the historical loop guard."
        ),
    )
    parser.add_argument(
        "--phone-waypoint3-max-lateral-deviation-m",
        type=float,
        default=-1.0,
        help=(
            "Hard maximum EEF deviation from the start-to-waypoint3 chord in "
            "phone_best_of_n_0_3. A negative value disables this guard."
        ),
    )
    parser.add_argument(
        "--phone-waypoint3-max-joint-travel-rad",
        type=float,
        default=float(np.pi),
        help=(
            "Hard maximum cumulative travel of any one arm joint for "
            "hard-guarded phone waypoint3 candidates. A negative value "
            "disables this guard."
        ),
    )
    parser.add_argument(
        "--phone-waypoint3-exact-execution",
        action="store_true",
        help=(
            "For the hard-guarded phone waypoint3 shortcut search, densely "
            "resample the selected joint path and execute those exact joint "
            "configurations one per simulator step. This prevents legacy "
            "Reflexxes interpolation from following a substantially different "
            "EEF curve than the one accepted by the geometric guards."
        ),
    )
    parser.add_argument(
        "--phone-waypoint3-exact-step-rad",
        type=float,
        default=0.04,
        help=(
            "Maximum absolute change of any joint between consecutive exact "
            "waypoint3 execution samples. Must be positive."
        ),
    )
    parser.add_argument(
        "--phone-roll-symmetry",
        action="store_true",
        help=(
            "For phone_best_of_n_0_3, treat a 180-degree rotation about the "
            "waypoint's local tool-Z axis as an equivalent parallel-gripper "
            "pose. At every Point waypoint, select whichever of the original "
            "and finger-swapped orientations has the smaller quaternion "
            "geodesic distance from the current real EEF orientation. XYZ and "
            "gripper extension commands are unchanged."
        ),
    )
    parser.add_argument(
        "--phone-waypoint0-roll-branch",
        choices=["authored", "finger_swapped"],
        default=None,
        help=(
            "For a single phone_on_base collection, force only waypoint0 to "
            "use either its authored orientation or the equivalent 180-degree "
            "rotation about its local tool-Z axis. The treatment of later Point "
            "waypoints is controlled by --phone-later-waypoint-roll-policy."
        ),
    )
    parser.add_argument(
        "--phone-later-waypoint-roll-policy",
        choices=["authored", "nearest", "latched", "opposite_latched"],
        default="authored",
        help=(
            "With --phone-waypoint0-roll-branch, either preserve authored "
            "orientations after waypoint0 (historical behavior), at every later "
            "Point waypoint choose between authored and local-tool-Z+pi using "
            "the smaller quaternion geodesic distance from the current real EEF, "
            "latch the exact waypoint0 branch through the remaining chain, "
            "or use opposite_latched to swap the two fingers by rotating "
            "waypoint1 and every later Point waypoint by 180 degrees relative "
            "to the waypoint0 branch."
        ),
    )
    parser.add_argument(
        "--phone-waypoint-local-z-offset-deg",
        type=float,
        default=0.0,
        help=(
            "Apply one additional constant local-tool-Z rotation to the "
            "selected phone waypoint0 branch and every later Point waypoint. "
            "This is a targeted near-base IK escape hatch; zero preserves the "
            "authored/finger-swapped task poses exactly."
        ),
    )
    parser.add_argument(
        "--dual-waypoint0-roll-select-shorter",
        action="store_true",
        help=(
            "For a single phone_on_base or take_frame_off_hanger task, restore "
            "each initial scene and run both authored (0 degree) and local-tool-Z "
            "+pi (180 degree) waypoint0 branches. Later Point waypoints follow "
            "--phone-later-waypoint-roll-policy. Retain the successful branch that "
            "reaches waypoint0 in fewer recorded frames for newly sampled scenes. "
            "For exact matched-scene repair, --matched-scene-demo-candidates can "
            "sample each branch repeatedly and prioritizes the smallest measured "
            "waypoint3 EEF rotation."
        ),
    )
    parser.add_argument(
        "--waypoint-roll-symmetry",
        action="store_true",
        help=(
            "For every ordinary Point waypoint, compare its authored orientation "
            "with the equivalent 180-degree local tool-Z finger-swapped orientation "
            "and plan to whichever is rotationally closer to the current real EEF. "
            "Waypoint XYZ, gripper commands, and PredefinedPath waypoints are unchanged."
        ),
    )
    parser.add_argument(
        "--waypoint-roll-symmetry-selection",
        choices=["endpoint_geodesic", "path_cost", "chain_path_cost"],
        default="endpoint_geodesic",
        help=(
            "How --waypoint-roll-symmetry chooses between authored and "
            "finger-swapped orientations. endpoint_geodesic preserves the "
            "original endpoint-only rule. path_cost plans both alternatives, "
            "rejects paths with >270 degree cumulative EEF rotation, >180 "
            "degree single-joint travel, or >3x Cartesian detours, and selects "
            "the lowest measured FK path cost. chain_path_cost evaluates one "
            "consistent authored or finger-swapped branch across the complete "
            "waypoint0..N chain, selects the lowest-cost fully feasible chain, "
            "and reuses those exact paths during execution. This task-specific "
            "mode is currently limited to take_frame_off_hanger."
        ),
    )
    parser.add_argument(
        "--waypoint-roll-force-branch",
        choices=[
            "authored",
            "finger_swapped",
            "finger_swapped_waypoint3",
            "finger_swapped_until_waypoint2",
            "finger_swapped_waypoint0",
        ],
        default=None,
        help=(
            "With --waypoint-roll-symmetry, force every ordinary Point waypoint "
            "to one consistent authored or local-tool-Z+pi branch. The "
            "finger_swapped_waypoint3 option swaps only the carried transfer "
            "target and restores the authored branch at waypoint4. The "
            "finger_swapped_until_waypoint2 option keeps the corrected grasp "
            "orientation through waypoint0/1/2, then rotates continuously while "
            "moving toward the authored waypoint3 placement pose. The "
            "finger_swapped_waypoint0 option selects the swapped branch at the "
            "initial approach waypoint and latches that same branch through "
            "waypoint1 and every later waypoint. This prevents an artificial "
            "180-degree rotation between the pre-grasp and grasp poses."
        ),
    )
    parser.add_argument(
        "--segmented-linear-segments",
        type=int,
        default=30,
        help=(
            "Number of short Cartesian IK segments per ordinary waypoint when "
            "--expert-path-mode=segmented_linear (default: 30)."
        ),
    )
    parser.add_argument(
        "--segmented-use-stock-guide",
        action="store_true",
        help=(
            "Sample segmented Cartesian positions along one stock RRT path "
            "instead of the direct XYZ chord. Orientation still follows "
            "shortest-arc SLERP, so translation and rotation remain coupled."
        ),
    )
    parser.add_argument(
        "--franka-servo-linear-speed-m-s",
        type=float,
        default=0.10,
        help="Cartesian reference speed for franka_cartesian_servo.",
    )
    parser.add_argument(
        "--franka-servo-angular-speed-rad-s",
        type=float,
        default=0.75,
        help="Shortest-arc orientation reference speed for franka_cartesian_servo.",
    )
    parser.add_argument(
        "--franka-servo-max-joint-speed-rad-s",
        type=float,
        default=0.80,
        help="Per-joint velocity bound for franka_cartesian_servo.",
    )
    parser.add_argument(
        "--franka-servo-max-joint-accel-step-rad-s",
        type=float,
        default=0.12,
        help="Maximum per-simulation-step joint-velocity change.",
    )
    parser.add_argument(
        "--franka-servo-damping",
        type=float,
        default=1e-3,
        help="Damped least-squares regularizer for franka_cartesian_servo.",
    )
    parser.add_argument(
        "--franka-servo-nullspace-gain",
        type=float,
        default=0.35,
        help="Gain that keeps redundant joints near the waypoint start branch.",
    )
    parser.add_argument(
        "--franka-servo-max-steps",
        type=int,
        default=500,
        help="Maximum physics steps for one expert waypoint servo.",
    )
    parser.add_argument(
        "--franka-servo-save-debug",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Save a frame/error-labelled MP4 and telemetry NPZ per servo "
            "waypoint. Disabled by default to avoid large diagnostic output."
        ),
    )
    parser.add_argument(
        "--cartesian-detour-max-offset-m",
        type=float,
        default=0.20,
        help=(
            "Maximum midpoint deviation searched by cartesian_detour. "
            "Candidates are smooth sin(pi*t) arcs, not sampled RRT paths."
        ),
    )
    parser.add_argument(
        "--cartesian-detour-offset-step-m",
        type=float,
        default=0.025,
        help="Offset-magnitude resolution for cartesian_detour.",
    )
    parser.add_argument(
        "--cartesian-carried-min-clearance-m",
        type=float,
        default=0.0,
        help=(
            "Minimum vertical clearance candidate for targeted carried-object "
            "Cartesian repair. Zero keeps the normal 4 cm-first search; a "
            "larger value is useful when an exact scene already proved every "
            "lower clearance kinematically infeasible."
        ),
    )
    parser.add_argument(
        "--cartesian-carried-direct-joint-first",
        action="store_true",
        help=(
            "For targeted carried-object waypoint repairs, sample multiple IK "
            "solutions for the authored endpoint and first try an exactly "
            "tracked joint interpolation selected by measured EEF path length, "
            "rotation, backtracking, lift, and collision checks. No OMPL/RRT is "
            "used. If no bounded candidate exists, continue with Cartesian "
            "staged candidates."
        ),
    )
    parser.add_argument(
        "--cartesian-carried-direct-min-rise-m",
        type=float,
        default=0.0,
        help=(
            "Minimum smooth EEF rise above max(start_z, target_z) required "
            "for a sampled direct-joint carried-object candidate. This is a "
            "geometric clearance filter, not an inserted staged waypoint."
        ),
    )
    parser.add_argument(
        "--cartesian-carried-direct-max-backtrack-m",
        type=float,
        default=0.035,
        help=(
            "Maximum EEF progress reversal admitted for a sampled direct-"
            "joint carried-object candidate. Keep this small; it exists for "
            "near-base scenes whose best branch has a slight local retreat."
        ),
    )
    parser.add_argument(
        "--cartesian-reference-deform-rise-m",
        type=float,
        default=-1.0,
        help=(
            "When non-negative, deform the saved successful joint branch for "
            "a targeted carried waypoint onto a straight XY path with this "
            "sinusoidal vertical rise and shortest-arc quaternion SLERP. The "
            "deformation uses continuous IK homotopy and never calls OMPL/RRT."
        ),
    )
    parser.add_argument(
        "--cartesian-reference-deform-xyz-blend",
        type=float,
        default=0.0,
        help=(
            "Blend the targeted reference path's measured EEF XYZ into the "
            "straight start-target chord during IK homotopy: 0 is straight, "
            "1 preserves the reference XYZ curve. Intermediate values "
            "progressively remove detours while keeping the known IK branch."
        ),
    )
    parser.add_argument(
        "--cartesian-reference-deform-orientation-blend",
        type=float,
        default=1.0,
        help=(
            "Blend each targeted reference FK orientation toward the direct "
            "shortest-arc SLERP orientation: 0 preserves the known path, 1 "
            "uses the full shortest arc. This allows the largest continuous "
            "loop reduction to be found without changing XYZ."
        ),
    )
    parser.add_argument(
        "--cartesian-reference-deform-required",
        action="store_true",
        help=(
            "For a targeted waypoint with a reference segment, reject the "
            "candidate immediately when reference homotopy is infeasible "
            "instead of entering unrelated Cartesian/joint fallbacks."
        ),
    )
    parser.add_argument(
        "--cartesian-reference-deform-preserve-orientation",
        action="store_true",
        help=(
            "During reference-branch homotopy, deform only XYZ and retain "
            "the reference FK orientation. This isolates excessive height "
            "from orientation feasibility before a later shortest-arc pass."
        ),
    )
    parser.add_argument(
        "--cartesian-reference-deform-roll-symmetry",
        action="store_true",
        help=(
            "During reference-branch homotopy, globally select original or "
            "local-tool-Z+pi orientation at every sample to minimize cumulative "
            "rotation, while forcing the last sample back to the authored pose."
        ),
    )
    parser.add_argument(
        "--joint-interp-max-lateral-deviation-m",
        type=float,
        default=0.40,
        help=(
            "Maximum EEF deviation from the start-goal chord for the "
            "collision-free joint-interpolation fallback."
        ),
    )
    parser.add_argument(
        "--segmented-fallback-algorithm",
        choices=["RRTConnect", "BITstar"],
        default="RRTConnect",
        help=(
            "OMPL algorithm used only for an individually infeasible "
            "segmented Cartesian target when --allow-rrt is explicit."
        ),
    )
    parser.add_argument(
        "--segmented-linear-waypoints",
        nargs="*",
        default=None,
        help=(
            "Optional ordinary waypoint names to which segmented linear modes "
            "are applied, for example waypoint0 waypoint2. Other Point "
            "waypoints retain RLBench's stock planner. By default all Point "
            "waypoints are segmented."
        ),
    )
    parser.add_argument(
        "--replay-random-seeds-from-artifacts",
        type=Path,
        default=None,
        help=(
            "Optional artifact root from an earlier collection. Before each live "
            "demo reset, restore that episode's complete saved NumPy MT19937 state. "
            "This provides matched initial scenes for planner A/B comparisons."
        ),
    )
    parser.add_argument(
        "--phone-base-max-initial-distance-m",
        type=float,
        default=None,
        help=(
            "For phone_on_base only, retain demos whose initial phone-center to "
            "success-sensor-center distance is at most this many meters."
        ),
    )
    parser.add_argument(
        "--phone-eef-max-initial-distance-m",
        type=float,
        default=None,
        help=(
            "For phone_on_base only, sample the scene before expert planning and "
            "retain it only when the initial phone-center to physical EEF-tip "
            "distance is at most this many meters."
        ),
    )
    parser.add_argument(
        "--phone-robot-base-min-initial-distance-m",
        type=float,
        default=None,
        help=(
            "For phone_on_base only, reject a random scene before expert planning "
            "unless the initial phone center is at least this far from panda_link0. "
            "This is useful for quickly collecting easy, untwisted phone reaches."
        ),
    )
    parser.add_argument(
        "--replay-scenes-from-artifacts",
        type=Path,
        default=None,
        help=(
            "Artifact root from a matched collection. Restore each episode's "
            "configuration tree and readable per-object snapshot before expert "
            "planning, enabling strict planner A/B collection on the same scenes."
        ),
    )
    parser.add_argument(
        "--matched-scene-demo-candidates",
        type=int,
        default=1,
        help=(
            "When replaying an exact saved scene, independently execute this "
            "many complete successful expert demonstrations and retain the one "
            "with the lowest measured joint/EEF travel. This captures path "
            "dependencies across all waypoints instead of optimizing each "
            "waypoint in isolation."
        ),
    )
    parser.add_argument(
        "--matched-scene-selection-objective",
        choices=[
            "waypoint3_rotation",
            "waypoint_path_geometry",
            "waypoint2_path_geometry",
            "waypoint0_path_geometry",
            "waypoint0_balanced_geometry",
            "phone_weighted_geometry",
            "phone_joint_smoothness",
            "wine_postgrasp_geometry",
        ],
        default="waypoint3_rotation",
        help=(
            "Objective used to retain complete matched-scene dual-roll demos. "
            "waypoint3_rotation preserves the rotation-focused repair behavior; "
            "waypoint_path_geometry minimizes actual executed waypoint0/waypoint3 "
            "Cartesian detour, lateral deviation, and backtracking; "
            "waypoint2_path_geometry prioritizes the actual waypoint2 Cartesian "
            "arc score and lateral deviation for matched-scene frame repairs; "
            "waypoint0_path_geometry lexicographically prioritizes the actual "
            "initial-to-waypoint0 lateral deviation and arc score before the "
            "later waypoint3 path; waypoint0_balanced_geometry groups waypoint0 "
            "lateral deviation into 2.5 cm bands, then minimizes waypoint3 "
            "Cartesian detour and excess rotation within the best band; "
            "phone_weighted_geometry jointly scores both Cartesian segments "
            "and waypoint3 excess rotation; phone_joint_smoothness prioritizes "
            "continuous low-travel Panda joint motion (especially the wrist) "
            "before Cartesian tie-breaks; wine_postgrasp_geometry rejects an "
            "excessive upward hump during waypoint3 and then minimizes the "
            "remaining height, lateral deviation, and Cartesian detour."
        ),
    )
    parser.add_argument(
        "--wine-postgrasp-max-excess-z-m",
        type=float,
        default=0.02,
        help=(
            "With --matched-scene-selection-objective=wine_postgrasp_geometry, "
            "reject a complete candidate when waypoint3 rises this far above "
            "both segment endpoints."
        ),
    )
    parser.add_argument(
        "--collection-workers",
        type=int,
        default=1,
        help=(
            "Number of independent RLBench environments used in parallel. "
            "Each worker needs its own X display; default is 1."
        ),
    )
    parser.add_argument(
        "--collection-display-base",
        type=int,
        default=None,
        help=(
            "Base X display number for parallel collection. Worker i uses "
            ":(base+i). If omitted, the current DISPLAY number is used."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument(
        "--delete-artifacts-after-pack",
        action="store_true",
        help=(
            "After each episode is successfully saved and its sidecars are verified, "
            "delete that episode's raw artifact directory to reduce peak disk use."
        ),
    )
    parser.add_argument("--skip-pointseg-cache", action="store_true")
    parser.add_argument("--cache-output-dir", type=Path, default=None)
    parser.add_argument(
        "--cache-python",
        type=Path,
        default=Path(os.environ.get("CACHE_PYTHON", sys.executable)),
    )
    parser.add_argument("--cache-batch-size", type=int, default=4)
    parser.add_argument("--cache-num-workers", type=int, default=8)
    parser.add_argument("--cache-device", default="cuda")
    parser.add_argument("--cache-vis-count", type=int, default=8)
    parser.add_argument("--cache-vis-one-episode-per-task", action="store_true")
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
    parser.add_argument("--overwrite-cache", action="store_true")
    parser.add_argument(
        "--action-alignment",
        choices=["transition", "observation"],
        default="transition",
        help=(
            "transition stores observation[t] with the expert command that moves "
            "to the next frame; observation preserves RLBench's post-step misc action."
        ),
    )
    parser.add_argument(
        "--action-label-mode",
        choices=ACTION_LABEL_MODES,
        default="expert_target",
        help=(
            "expert_target keeps the nominal expert joint-target FK label; "
            "executed uses the achieved EEF state at the aligned next frame."
        ),
    )
    parser.add_argument(
        "--generate-world-base-worldflow-sidecars",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Generate achieved and commanded EEF pose sidecars in the complete "
            "Panda robot-base frame (default: enabled)."
        ),
    )
    return parser.parse_args()


def pose7_to_pose9(pose7):
    """Convert RLBench [xyz, qx, qy, qz, qw] to the Song 9D pose."""
    from scipy.spatial.transform import Rotation

    pose7 = np.asarray(pose7, dtype=np.float32).reshape(-1)
    rotation = Rotation.from_quat(pose7[3:7]).as_matrix().astype(np.float32)
    return np.concatenate((pose7[:3], rotation[:, 0], rotation[:, 1])).astype(np.float32)


def configuration_tree_to_bytes(configuration_tree):
    """Copy a CoppeliaSim configuration tree into ordinary Python bytes.

    PyRep 4.1 currently returns CoppeliaSim's native ``char *`` pointer even
    though ``get_configuration_tree()`` is annotated as returning ``bytes``.
    The first four bytes of this binary buffer store its complete byte length.
    It must not be copied with ``ffi.string()`` because configuration trees
    contain zero bytes and would be truncated at the first zero.
    """
    if isinstance(configuration_tree, bytes):
        return configuration_tree
    if isinstance(configuration_tree, (bytearray, memoryview)):
        return bytes(configuration_tree)

    from pyrep.backend import sim

    try:
        header = bytes(sim.ffi.buffer(configuration_tree, 4))
        byte_count = int.from_bytes(header, byteorder="little", signed=False)
        if byte_count < 4 or byte_count > 256 * 1024 * 1024:
            raise RuntimeError(
                "Invalid CoppeliaSim configuration-tree size: " + str(byte_count)
            )
        return bytes(sim.ffi.buffer(configuration_tree, byte_count))
    finally:
        # simGetConfigurationTree allocates this native buffer. After making
        # the Python copy, release it exactly once to avoid one leak per demo.
        sim.simReleaseBuffer(configuration_tree)


def capture_initial_object_states(task):
    """Capture readable per-object state at the recorded first frame."""
    objects = task.get_base().get_objects_in_tree(
        exclude_base=False, first_generation_only=False
    )
    count = len(objects)
    names = []
    handles = np.empty(count, dtype=np.int64)
    types = []
    parent_handles = np.full(count, -1, dtype=np.int64)
    parent_names = [""] * count
    poses = np.full((count, 7), np.nan, dtype=np.float32)
    linear_velocities = np.full((count, 3), np.nan, dtype=np.float32)
    angular_velocities = np.full((count, 3), np.nan, dtype=np.float32)
    joint_positions = np.full(count, np.nan, dtype=np.float32)
    joint_velocities = np.full(count, np.nan, dtype=np.float32)
    joint_target_positions = np.full(count, np.nan, dtype=np.float32)
    joint_target_velocities = np.full(count, np.nan, dtype=np.float32)

    for index, obj in enumerate(objects):
        names.append(obj.get_name())
        handles[index] = obj.get_handle()
        object_type = obj.get_type()
        types.append(getattr(object_type, "name", str(object_type)))
        try:
            parent = obj.get_parent()
            if parent is not None:
                parent_handles[index] = parent.get_handle()
                parent_names[index] = parent.get_name()
        except Exception:
            pass
        try:
            poses[index] = np.asarray(obj.get_pose(), dtype=np.float32)
        except Exception:
            pass
        try:
            linear, angular = obj.get_velocity()
            linear_velocities[index] = np.asarray(linear, dtype=np.float32)
            angular_velocities[index] = np.asarray(angular, dtype=np.float32)
        except Exception:
            pass
        if hasattr(obj, "get_joint_position"):
            try:
                joint_positions[index] = float(obj.get_joint_position())
                joint_velocities[index] = float(obj.get_joint_velocity())
                joint_target_positions[index] = float(obj.get_joint_target_position())
                joint_target_velocities[index] = float(obj.get_joint_target_velocity())
            except Exception:
                pass

    return {
        "initial_object_names": np.asarray(names, dtype="U256"),
        "initial_object_handles": handles,
        "initial_object_types": np.asarray(types, dtype="U64"),
        "initial_object_parent_handles": parent_handles,
        "initial_object_parent_names": np.asarray(parent_names, dtype="U256"),
        "initial_object_poses": poses,
        "initial_object_linear_velocities": linear_velocities,
        "initial_object_angular_velocities": angular_velocities,
        "initial_object_joint_positions": joint_positions,
        "initial_object_joint_velocities": joint_velocities,
        "initial_object_joint_target_positions": joint_target_positions,
        "initial_object_joint_target_velocities": joint_target_velocities,
    }


def capture_initial_scene_arrays(task_env, random_state):
    """Capture one reset scene for exact replay and online de-duplication."""
    configuration_tree, object_count = task_env._task.get_state()
    configuration_bytes = configuration_tree_to_bytes(configuration_tree)
    object_states = capture_initial_object_states(task_env._task)
    if len(object_states["initial_object_names"]) != int(object_count):
        raise RuntimeError(
            "Task state and per-object state counts differ: "
            + str(object_count)
            + " versus "
            + str(len(object_states["initial_object_names"]))
        )
    arrays = {
        "initial_task_state_bytes": np.frombuffer(
            configuration_bytes, dtype=np.uint8
        ).copy(),
        "initial_task_state_object_count": np.int64(object_count),
        "demo_random_seed_state": np.asarray(random_state[1], dtype=np.uint32),
        "demo_random_seed_position": np.int64(random_state[2]),
        "demo_random_seed_has_gauss": np.int64(random_state[3]),
        "demo_random_seed_cached_gaussian": np.float64(random_state[4]),
        "demo_num_reset_attempts": np.int64(task_env._scene._attempts + 1),
    }
    arrays.update(object_states)
    return arrays


def initial_scene_signature(arrays):
    """Return named task-object poses/joints used by the novelty metric."""
    names = [str(name) for name in np.asarray(arrays["initial_object_names"])]
    poses = np.asarray(arrays["initial_object_poses"], dtype=np.float64)
    joints = np.asarray(
        arrays.get("initial_object_joint_positions", np.full(len(names), np.nan)),
        dtype=np.float64,
    )
    return {
        name: {
            "pose": poses[index].copy(),
            "joint": float(joints[index]),
        }
        for index, name in enumerate(names)
    }


def initial_scene_distance_m(left, right, rotation_radius_m):
    """Maximum named-object pose change in a metre-equivalent metric."""
    common_names = sorted(set(left) & set(right))
    if not common_names:
        return float("inf")
    distances = []
    for name in common_names:
        left_pose = np.asarray(left[name]["pose"], dtype=np.float64)
        right_pose = np.asarray(right[name]["pose"], dtype=np.float64)
        if np.isfinite(left_pose).all() and np.isfinite(right_pose).all():
            distances.append(float(np.linalg.norm(left_pose[:3] - right_pose[:3])))
            left_quaternion = left_pose[3:7] / np.linalg.norm(left_pose[3:7])
            right_quaternion = right_pose[3:7] / np.linalg.norm(right_pose[3:7])
            angle = 2.0 * np.arccos(
                np.clip(abs(float(np.dot(left_quaternion, right_quaternion))), -1.0, 1.0)
            )
            distances.append(float(rotation_radius_m) * float(angle))
        left_joint = float(left[name]["joint"])
        right_joint = float(right[name]["joint"])
        if np.isfinite(left_joint) and np.isfinite(right_joint):
            distances.append(
                float(rotation_radius_m)
                * abs(float(np.arctan2(
                    np.sin(left_joint - right_joint),
                    np.cos(left_joint - right_joint),
                )))
            )
    return max(distances) if distances else float("inf")


def nearest_initial_scene(candidate, accepted, rotation_radius_m):
    """Return (distance, episode index) for the closest accepted scene."""
    if not accepted:
        return float("inf"), None
    distances = [
        (
            initial_scene_distance_m(candidate, signature, rotation_radius_m),
            int(episode_index),
        )
        for episode_index, signature in accepted
    ]
    return min(distances, key=lambda item: item[0])


def demo_random_state_from_arrays(arrays):
    """Rebuild the complete NumPy RNG state saved with a collected demo."""
    return (
        "MT19937",
        np.asarray(arrays["demo_random_seed_state"], dtype=np.uint32).copy(),
        int(arrays["demo_random_seed_position"]),
        int(arrays["demo_random_seed_has_gauss"]),
        float(arrays["demo_random_seed_cached_gaussian"]),
    )


def restore_initial_object_states_from_arrays(task, arrays):
    """Restore the readable per-object snapshot stored in one collection artifact."""
    objects = task.get_base().get_objects_in_tree(
        exclude_base=False, first_generation_only=False
    )
    current_by_name = {obj.get_name(): obj for obj in objects}
    recorded_names = [str(name) for name in arrays["initial_object_names"]]
    missing = [name for name in recorded_names if name not in current_by_name]
    if missing:
        raise RuntimeError(
            "Cannot restore matched scene; task objects are missing: "
            + ", ".join(missing)
        )

    for index, name in enumerate(recorded_names):
        obj = current_by_name[name]
        pose = np.asarray(arrays["initial_object_poses"][index], dtype=np.float64)
        if np.isfinite(pose).all():
            obj.set_pose(pose.tolist(), reset_dynamics=True)
        joint_position = float(arrays["initial_object_joint_positions"][index])
        if np.isfinite(joint_position) and hasattr(obj, "set_joint_position"):
            obj.set_joint_position(joint_position, disable_dynamics=True)
        joint_target = float(
            arrays["initial_object_joint_target_positions"][index]
        )
        if np.isfinite(joint_target) and hasattr(obj, "set_joint_target_position"):
            obj.set_joint_target_position(joint_target)
        joint_target_velocity = float(
            arrays["initial_object_joint_target_velocities"][index]
        )
        if np.isfinite(joint_target_velocity) and hasattr(
            obj, "set_joint_target_velocity"
        ):
            obj.set_joint_target_velocity(joint_target_velocity)

    # Joint setters can step or move descendants. Reapply absolute poses last.
    for index, name in enumerate(recorded_names):
        pose = np.asarray(arrays["initial_object_poses"][index], dtype=np.float64)
        if np.isfinite(pose).all():
            current_by_name[name].set_pose(pose.tolist(), reset_dynamics=True)


def restore_task_environment_from_artifact_arrays(task_env, arrays):
    """Reset robot/task bookkeeping, then restore one exact recorded task scene."""
    random_state = demo_random_state_from_arrays(arrays)
    reset_attempts = int(arrays["demo_num_reset_attempts"])

    class RecordedDemoPlacement:
        num_reset_attempts = reset_attempts

    np.random.set_state(random_state)
    descriptions, _ = task_env.reset(RecordedDemoPlacement())
    configuration_bytes = np.asarray(
        arrays["initial_task_state_bytes"], dtype=np.uint8
    ).tobytes()
    object_count = int(arrays["initial_task_state_object_count"])
    task_env._task.restore_state((configuration_bytes, object_count))
    restore_initial_object_states_from_arrays(task_env._task, arrays)
    return descriptions, task_env.get_observation(), random_state, reset_attempts


def trim_demo_after_first_success(demo, post_success_frames, action_alignment):
    """Remove expert retreat motion while preserving transition alignment."""
    original_observations = len(demo)
    first_success_frame = getattr(demo, "first_success_frame", None)
    terminal_observations = 1 if action_alignment == "transition" else 0
    kept_observations = original_observations
    if post_success_frames >= 0 and first_success_frame is not None:
        kept_observations = min(
            original_observations,
            int(first_success_frame) + 1 + int(post_success_frames) + terminal_observations,
        )
        if kept_observations < original_observations:
            demo._observations = demo._observations[:kept_observations]

    demo.success_trim_first_success_frame = first_success_frame
    demo.success_trim_original_observations = original_observations
    demo.success_trim_kept_observations = kept_observations
    demo.success_trim_removed_observations = original_observations - kept_observations
    demo.success_trim_post_success_frames = int(post_success_frames)
    demo.success_trim_transition_terminal_observation = bool(terminal_observations)
    print(
        "[success-trim] first_success_frame="
        + str(first_success_frame)
        + " original_observations="
        + str(original_observations)
        + " kept_observations="
        + str(kept_observations)
        + " removed_observations="
        + str(original_observations - kept_observations)
        + " post_success_dataset_frames="
        + str(post_success_frames),
        flush=True,
    )
    return demo


def collect_live_demo_from_current_scene(
    task_env,
    post_success_frames,
    action_alignment,
    stop_after_success=False,
):
    """Run the RLBench expert without allowing TaskEnvironment to reset again."""
    control_loop_enabled = task_env._robot.arm.joints[0].is_control_loop_enabled()
    task_env._robot.arm.set_control_loop_enabled(True)
    try:
        stop_after_success_observations = None
        if stop_after_success:
            if post_success_frames < 0:
                raise ValueError(
                    "--stop-after-success requires non-negative "
                    "--post-success-frames"
                )
            stop_after_success_observations = int(post_success_frames) + (
                1 if action_alignment == "transition" else 0
            )
        demo = task_env._scene.get_demo(
            stop_after_success_observations=stop_after_success_observations
        )
        return trim_demo_after_first_success(
            demo, post_success_frames, action_alignment
        )
    finally:
        task_env._robot.arm.set_control_loop_enabled(control_loop_enabled)


def pose9_to_umi(pose_sequence):
    """Express every world pose in the first EEF frame of this episode."""
    poses = np.asarray(pose_sequence, dtype=np.float32)
    transforms = pose9_to_homo_np(poses)
    first_inverse = fast_inverse_homogeneous(transforms[0])
    relative = first_inverse[None] @ transforms
    return np.concatenate((relative[:, :3, 3], relative[:, :3, 0], relative[:, :3, 1]), axis=1)


def matrix_to_pose9(matrix):
    matrix = np.asarray(matrix, dtype=np.float64)
    return np.concatenate((matrix[:3, 3], matrix[:3, 0], matrix[:3, 1])).astype(np.float32)


def rotation_about_axis(axis, point, angle):
    axis = np.asarray(axis, dtype=np.float64)
    point = np.asarray(point, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    cosine = math.cos(float(angle))
    sine = math.sin(float(angle))
    one_minus_cosine = 1.0 - cosine
    rotation = np.array(
        [
            [cosine + x * x * one_minus_cosine,
             x * y * one_minus_cosine - z * sine,
             x * z * one_minus_cosine + y * sine],
            [y * x * one_minus_cosine + z * sine,
             cosine + y * y * one_minus_cosine,
             y * z * one_minus_cosine - x * sine],
            [z * x * one_minus_cosine - y * sine,
             z * y * one_minus_cosine + x * sine,
             cosine + z * z * one_minus_cosine],
        ],
        dtype=np.float64,
    )
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = point - rotation @ point
    return transform


def read_panda_fk_model(environment):
    """Read the fixed Panda kinematic model once before collecting tasks."""
    arm = environment._robot.arm
    original_positions = arm.get_joint_positions()
    arm.set_joint_positions([0.0] * 7, disable_dynamics=True)
    home_joint_positions = np.asarray(arm.get_joint_positions(), dtype=np.float64)
    axes = []
    points = []
    for joint in arm.joints:
        matrix = np.asarray(joint.get_matrix(), dtype=np.float64)
        points.append(matrix[:3, 3].copy())
        axes.append(matrix[:3, 2].copy())
    home_tip = np.asarray(arm.get_tip().get_matrix(), dtype=np.float64)
    arm.set_joint_positions(original_positions, disable_dynamics=True)
    return axes, points, home_tip, home_joint_positions


def fk_pose(model, joint_positions):
    axes, points, home_tip, home_joint_positions = model
    transform = np.eye(4, dtype=np.float64)
    for axis, point, angle, home_angle in zip(
        axes, points, joint_positions, home_joint_positions
    ):
        transform = transform @ rotation_about_axis(axis, point, angle - home_angle)
    return transform @ home_tip


def expert_actions_to_eef0(raw_actions, world_poses, first_width, gripper_max_width, fk_model):
    """Convert recorded Panda joint targets directly into EEF0 pose targets."""
    raw_actions = np.asarray(raw_actions, dtype=np.float32)
    world_poses = np.asarray(world_poses, dtype=np.float32)
    world_to_eef0 = fast_inverse_homogeneous(pose9_to_homo_np(world_poses[0]))
    actions = np.empty((len(raw_actions), 10), dtype=np.float32)
    actions[0] = 0.0
    actions[0, 3] = 1.0
    actions[0, 7] = 1.0
    actions[0, 9] = float(first_width)
    position_errors = []
    for frame_index in range(1, len(raw_actions)):
        raw_action = raw_actions[frame_index]
        if raw_action.shape != (8,) or not np.isfinite(raw_action).all():
            raise RuntimeError("Missing RLBench expert joint target at frame " + str(frame_index))
        target_world = fk_pose(fk_model, raw_action[:7])
        target_eef0 = world_to_eef0 @ target_world
        actions[frame_index, :9] = matrix_to_pose9(target_eef0)
        actions[frame_index, 9] = float(np.clip(raw_action[7], 0.0, 1.0)) * gripper_max_width
        position_errors.append(float(np.linalg.norm(target_world[:3, 3] - world_poses[frame_index, :3])))
    return actions, np.asarray(position_errors, dtype=np.float32)


def task_class_from_name(task_name):
    class_name = "".join(part[:1].upper() + part[1:] for part in task_name.split("_"))
    module = importlib.import_module("rlbench.tasks." + task_name)
    return getattr(module, class_name)


def resolve_tasks(args):
    if args.all_tasks:
        import rlbench

        names = []
        for name in rlbench.TASKS:
            name = name.replace(".py", "")
            if not name.startswith("place_holder"):
                names.append(name)
        return sorted(names)
    if args.tasks:
        return list(args.tasks)
    return list(DEFAULT_TASKS)


def make_observation_config(image_size):
    from rlbench import CameraConfig, ObservationConfig

    config = ObservationConfig()
    config.set_all(False)
    config.front_camera = CameraConfig(
        rgb=True, depth=False, point_cloud=True, mask=False, image_size=(image_size, image_size)
    )
    config.wrist_camera = CameraConfig(
        rgb=False, depth=False, point_cloud=False, mask=False, image_size=(image_size, image_size)
    )
    config.gripper_open = True
    config.gripper_pose = True
    config.joint_positions = True
    config.joint_velocities = True
    config.record_gripper_closing = True
    return config


def configure_front_camera(position_m=None, look_at_m=None):
    """Apply an optional absolute cam_front override without changing defaults."""
    if position_m is None and look_at_m is None:
        return None

    from pyrep.objects.vision_sensor import VisionSensor
    from scipy.spatial.transform import Rotation

    camera = VisionSensor("cam_front")
    if position_m is not None:
        camera.set_position(np.asarray(position_m, dtype=np.float64).tolist())
    if look_at_m is not None:
        position = np.asarray(camera.get_position(), dtype=np.float64)
        forward = np.asarray(look_at_m, dtype=np.float64) - position
        forward_norm = float(np.linalg.norm(forward))
        if forward_norm <= 1e-9:
            raise ValueError("cam_front position and look-at point must differ")
        forward /= forward_norm
        world_up = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
        local_x = np.cross(world_up, forward)
        local_x_norm = float(np.linalg.norm(local_x))
        if local_x_norm <= 1e-9:
            world_up = np.asarray([0.0, 1.0, 0.0], dtype=np.float64)
            local_x = np.cross(world_up, forward)
            local_x_norm = float(np.linalg.norm(local_x))
        local_x /= local_x_norm
        local_y = np.cross(forward, local_x)
        rotation = np.column_stack((local_x, local_y, forward))
        camera.set_quaternion(Rotation.from_matrix(rotation).as_quat().tolist())

    info = {
        "position_m": list(map(float, camera.get_position())),
        "quaternion_xyzw": list(map(float, camera.get_quaternion())),
        "look_at_m": (
            None if look_at_m is None else list(map(float, look_at_m))
        ),
    }
    print("[front-camera-override] " + json.dumps(info, sort_keys=True), flush=True)
    return info


def color_to_uint8(rgb):
    rgb = np.asarray(rgb)
    if rgb.dtype != np.uint8:
        rgb = rgb.astype(np.float32)
        if rgb.size and float(np.nanmax(rgb)) <= 1.0:
            rgb = rgb * 255.0
        rgb = np.clip(rgb, 0.0, 255.0).astype(np.uint8)
    return rgb


def observation_cloud(observation, fallback_to_all_finite=False):
    """Return front-camera points inside the historical world-space crop."""
    if observation.front_point_cloud is None:
        raise RuntimeError("RLBench did not return a front-camera point cloud.")
    xyz = np.asarray(observation.front_point_cloud, dtype=np.float32).reshape(-1, 3)
    if observation.front_rgb is None:
        rgb = np.zeros((len(xyz), 3), dtype=np.uint8)
    else:
        rgb = color_to_uint8(observation.front_rgb).reshape(-1, 3)
    count = min(len(xyz), len(rgb))
    xyz = xyz[:count]
    rgb = rgb[:count]
    lower = RLBENCH_SCENE_BOUNDS[:3]
    upper = RLBENCH_SCENE_BOUNDS[3:]
    valid = np.isfinite(xyz).all(axis=1)
    valid &= (xyz >= lower).all(axis=1)
    valid &= (xyz <= upper).all(axis=1)
    if not valid.any():
        finite = np.isfinite(xyz).all(axis=1)
        if fallback_to_all_finite and finite.any():
            # A live evaluation camera can see only floor/background points
            # outside the training crop after task randomization. Preserve
            # those finite observations instead of aborting before inference.
            valid = finite
        else:
            raise RuntimeError(
                "RLBench returned no usable front-camera point-cloud points "
                "(finite=%d, in_bounds=%d)." % (int(finite.sum()), int(valid.sum()))
            )
    return np.concatenate((xyz[valid], rgb[valid].astype(np.float32)), axis=1).astype(np.float32)


def make_episode_arrays(
    demo,
    num_points,
    gripper_points,
    gripper_template,
    gripper_max_width,
    fps,
    seed,
    fk_model,
    t_world_base,
    generate_world_base_worldflow_sidecars,
    action_alignment,
    action_label_mode,
):
    """Convert one successful RLBench Demo object into numpy arrays."""
    observations = [demo[index] for index in range(len(demo))]
    if len(observations) < 2:
        raise RuntimeError("The expert demo contains fewer than two observations.")

    world_poses = []
    grippers = []
    world_clouds = []
    raw_actions = []
    images = []

    for frame_index, observation in enumerate(observations):
        world_poses.append(pose7_to_pose9(observation.gripper_pose))
        # Keep the continuous simulator opening amount. RLBench's Panda
        # physical width is represented as [0, gripper_max_width] meters.
        open_value = float(np.clip(observation.gripper_open, 0.0, 1.0))
        grippers.append(open_value * float(gripper_max_width))
        world_clouds.append(
            sample_or_repeat_points(
                # Some task cameras are entirely outside the historical crop;
                # retain their finite camera points instead of rejecting the demo.
                observation_cloud(observation, fallback_to_all_finite=True),
                num_points,
                seed + frame_index,
            )
        )
        images.append(np.asarray(observation.front_rgb, dtype=np.uint8))

        command = observation.misc.get("joint_position_action")
        if command is None:
            raw_actions.append(np.full((8,), np.nan, dtype=np.float32))
        else:
            command = np.asarray(command, dtype=np.float32).reshape(-1)
            if command.size != 8:
                raise RuntimeError("RLBench expert joint_position_action is not 8-dimensional.")
            raw_actions.append(command)

    world_poses = np.asarray(world_poses, dtype=np.float32)
    grippers = np.asarray(grippers, dtype=np.float32)
    world_clouds = np.asarray(world_clouds, dtype=np.float32)

    if gripper_template == LIBERO_GRIPPER_TEMPLATE:
        # Preserve LIBERO's physical width / 0.1 normalization and fixed
        # four-box body dimensions, but recover the two-finger aperture with
        # the same 0.1 m scale so the virtual gap equals RLBench's physical gap.
        # The shape is translated to local Z=-0.09 m so its forward tip aligns
        # with the RLBench Panda tip. RLBench state/action widths remain in
        # their native physical [0, 0.08] m range.
        cloud_gripper_widths = libero_reap_width_percent_from_physical(grippers)
        cloud_widths_are_normalized = True
        cloud_gripper_max_width = None
        cloud_gripper_opening_max_width = LIBERO_REAP_OPENING_MAX_WIDTH
        cloud_gripper_len = LIBERO_REAP_GRIPPER_LEN
    else:
        cloud_gripper_widths = grippers
        cloud_widths_are_normalized = False
        cloud_gripper_max_width = gripper_max_width
        cloud_gripper_opening_max_width = None
        cloud_gripper_len = 0.0

    # Merge the selected template before PointSeg sees the stored point cloud.
    world_clouds = add_world_gripper_clouds_to_episode(
        world_clouds,
        world_poses,
        cloud_gripper_widths,
        total_points=num_points,
        gripper_points=gripper_points,
        gripper_template=gripper_template,
        gripper_len=cloud_gripper_len,
        seed=seed,
        drop_strategy="tail",
        shuffle_points=False,
        widths_are_normalized=cloud_widths_are_normalized,
        gripper_max_width=cloud_gripper_max_width,
        gripper_opening_max_width=cloud_gripper_opening_max_width,
    )
    # add_world_gripper_clouds_to_episode already returns current-EFF clouds.
    point_clouds = world_clouds
    relative_poses = pose9_to_umi(world_poses)
    states = np.concatenate((relative_poses, grippers[:, None]), axis=1).astype(np.float32)
    expert_target_actions, fk_position_errors = expert_actions_to_eef0(
        raw_actions,
        world_poses,
        grippers[0],
        gripper_max_width,
        fk_model,
    )
    actions = expert_target_actions

    if action_label_mode == "executed":
        # The alignment slices below turn this into state[t+1] for transition
        # mode and state[t] for observation mode.
        actions = states.copy()
    elif action_label_mode != "expert_target":
        raise ValueError("Unknown action label mode: " + str(action_label_mode))

    # RLBench attaches the command to the observation recorded after that
    # command was executed. For a LeRobot transition, pair command i+1 with
    # observation i and drop the terminal observation with no next command.
    raw_actions_array = np.asarray(raw_actions, dtype=np.float32)
    if action_alignment == "transition":
        if len(actions) < 2:
            raise RuntimeError("A transition-aligned episode needs at least two observations.")
        output = {
            "actions": actions[1:],
            "states": states[:-1],
            "point_clouds": point_clouds[:-1],
            "world_ee_poses": world_poses[:-1],
            "raw_expert_actions": raw_actions_array[1:],
            "raw_expert_actions_full": raw_actions_array,
            "images": np.asarray(images[:-1], dtype=np.uint8),
            "timestamps": np.arange(len(actions) - 1, dtype=np.float32) / float(fps),
            "fk_position_errors": fk_position_errors,
            "action_alignment": action_alignment,
            "action_label_mode": action_label_mode,
        }
        commanded_targets_eef0 = expert_target_actions[1:, :9]
    elif action_alignment == "observation":
        output = {
            "actions": actions,
            "states": states,
            "point_clouds": point_clouds,
            "world_ee_poses": world_poses,
            "raw_expert_actions": raw_actions_array,
            "raw_expert_actions_full": raw_actions_array,
            "images": np.asarray(images, dtype=np.uint8),
            "timestamps": np.arange(len(actions), dtype=np.float32) / float(fps),
            "fk_position_errors": fk_position_errors,
            "action_alignment": action_alignment,
            "action_label_mode": action_label_mode,
        }
        commanded_targets_eef0 = expert_target_actions[:, :9]
    else:
        raise ValueError("Unknown action alignment: " + str(action_alignment))

    if generate_world_base_worldflow_sidecars:
        sidecar_actions = np.zeros((len(commanded_targets_eef0), 10), dtype=np.float32)
        sidecar_actions[:, :9] = commanded_targets_eef0
        base_ee, base_target, sidecar_validation = build_robot_base_episode_sidecars(
            output["world_ee_poses"],
            sidecar_actions,
            t_world_base,
        )
        output["world_base_ee_poses"] = base_ee
        output["world_base_action_target_ee_poses"] = base_target
        output["worldflow_base_sidecar_validation"] = sidecar_validation
        output["T_world_base"] = np.asarray(t_world_base, dtype=np.float64)
        output["T_base_world"] = np.linalg.inv(np.asarray(t_world_base, dtype=np.float64))
    return output


def validate_captured_episode(arrays, task_name, episode_index, gripper_points):
    """Reject camera/point-cloud failures before an episode reaches the dataset."""
    images = np.asarray(arrays["images"])
    if images.size == 0 or int(images.max()) == 0:
        raise RuntimeError(
            "front RGB is completely black for task="
            + str(task_name)
            + " episode="
            + str(episode_index)
            + "; the worker display/camera did not render"
        )

    point_clouds = np.asarray(arrays["point_clouds"], dtype=np.float32)
    if point_clouds.ndim != 3 or point_clouds.shape[-1] != 6:
        raise RuntimeError("captured point-cloud array has an invalid shape: " + str(point_clouds.shape))
    scene_points = point_clouds[:, :-int(min(gripper_points, point_clouds.shape[1])), :]
    scene_rgb = scene_points[..., 3:6]
    if scene_rgb.size == 0 or float(np.max(scene_rgb)) <= 0.0:
        raise RuntimeError(
            "front point-cloud RGB is empty for task="
            + str(task_name)
            + " episode="
            + str(episode_index)
        )

    # The stored cloud is in the current EEF frame. Reconstruct frame-0 world
    # coordinates and verify that the camera cloud still lies in the RLBench crop.
    world_pose0 = pose9_to_homo_np(np.asarray(arrays["world_ee_poses"])[0, :9])
    world_xyz = (
        scene_points[0, ..., :3] @ world_pose0[:3, :3].T
        + world_pose0[:3, 3]
    )
    lower = RLBENCH_SCENE_BOUNDS[:3] - 0.05
    upper = RLBENCH_SCENE_BOUNDS[3:] + 0.05
    in_bounds = np.isfinite(world_xyz).all(axis=1) & (world_xyz >= lower).all(axis=1) & (world_xyz <= upper).all(axis=1)
    if float(in_bounds.mean()) < 0.98:
        raise RuntimeError(
            "frame-0 point cloud is inconsistent with the RLBench world crop for task="
            + str(task_name)
            + " episode="
            + str(episode_index)
            + f" (in_bounds={float(in_bounds.mean()):.3f})"
        )


def episode_name(task_name, local_index):
    return task_name + "__episode_" + str(local_index).zfill(5)


def artifact_is_complete(path, require_world_base_worldflow_sidecars=True):
    arrays_path = path / "arrays.npz"
    if not (
        path.is_dir()
        and arrays_path.is_file()
        and (path / "point_clouds.zarr").is_dir()
        and (path / "record.json").is_file()
    ):
        return False

    # v1 artifacts did not save the task configuration tree. They cannot
    # restore the first-frame object layout and must not be reused silently.
    try:
        with np.load(arrays_path) as arrays:
            complete = (
                "initial_task_state_bytes" in arrays.files
                and "initial_task_state_object_count" in arrays.files
                and "raw_expert_actions_full" in arrays.files
                and all(key in arrays.files for key in OBJECT_STATE_KEYS)
            )
            if require_world_base_worldflow_sidecars:
                complete = complete and all(
                    key in arrays.files
                    for key in (
                        "world_base_ee_poses",
                        "world_base_action_target_ee_poses",
                        "T_world_base",
                        "T_base_world",
                    )
                )
            return complete
    except Exception:
        return False


def save_artifact(path, task_name, local_index, arrays, description, variation):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    if temp.exists():
        shutil.rmtree(temp)
    temp.mkdir(parents=True)
    artifact_arrays = {
        "actions": arrays["actions"],
        "states": arrays["states"],
        "world_ee_poses": arrays["world_ee_poses"],
        "raw_expert_actions": arrays["raw_expert_actions"],
        "raw_expert_actions_full": arrays["raw_expert_actions_full"],
        "images": arrays["images"],
        "timestamps": arrays["timestamps"],
        "fk_position_errors": arrays["fk_position_errors"],
        "initial_task_state_bytes": arrays["initial_task_state_bytes"],
        "initial_task_state_object_count": arrays["initial_task_state_object_count"],
        "demo_random_seed_state": arrays["demo_random_seed_state"],
        "demo_random_seed_position": arrays["demo_random_seed_position"],
        "demo_random_seed_has_gauss": arrays["demo_random_seed_has_gauss"],
        "demo_random_seed_cached_gaussian": arrays["demo_random_seed_cached_gaussian"],
        "demo_num_reset_attempts": arrays["demo_num_reset_attempts"],
    }
    artifact_arrays.update(
        {key: arrays[key] for key in OBJECT_STATE_KEYS if key in arrays}
    )
    artifact_arrays.update(
        {
            key: arrays[key]
            for key in (
                "world_base_ee_poses",
                "world_base_action_target_ee_poses",
                "T_world_base",
                "T_base_world",
                "expert_path_mode",
                "phone_path_candidates",
                "phone_roll_symmetry",
                "phone_waypoint0_roll_branch",
                "phone_later_waypoint_roll_policy",
                "phone_waypoint0_default_branch",
                "phone_waypoint0_authored_angle_rad",
                "phone_waypoint0_finger_swapped_angle_rad",
                "waypoint_roll_symmetry",
                "waypoint_roll_authored_fallback",
                "waypoint_roll_primary_error",
                "expert_linear_path_calls",
                "expert_rrt_path_calls",
                "expert_cartesian_path_calls",
                "expert_cartesian_stock_fallback_calls",
                "initial_phone_to_base_sensor_distance_m",
                "initial_phone_to_eef_distance_m",
                "initial_phone_to_robot_base_distance_m",
                "matched_scene_source",
                "success_trim_first_success_frame",
                "success_trim_original_observations",
                "success_trim_kept_observations",
                "success_trim_removed_observations",
                "success_trim_post_success_frames",
                "success_trim_transition_terminal_observation",
                "initial_scene_nearest_distance_m",
                "initial_scene_nearest_episode",
                "dual_waypoint0_selected_branch",
                "dual_waypoint0_authored_frames",
                "dual_waypoint0_finger_swapped_frames",
                "waypoint_end_names",
                "waypoint_end_frames",
            )
            if key in arrays
        }
    )
    # Raw artifacts retain every array, but RGB and repeated state fields are
    # highly compressible.  Compression materially reduces the footprint of a
    # 10-task x 100-episode collection without changing the on-disk schema.
    np.savez_compressed(
        temp / "arrays.npz",
        **artifact_arrays,
    )
    save_point_clouds_zarr(temp / "point_clouds.zarr", arrays["point_clouds"], compression_level=3)
    record = {
        "task": task_name,
        "local_episode_index": int(local_index),
        "description": str(description),
        "variation": int(variation),
        "frames": int(len(arrays["actions"])),
        "reset_first_rgb_mae": float(arrays["reset_first_rgb_mae"]),
        "initial_task_state_object_count": int(arrays["initial_task_state_object_count"]),
        "initial_object_state_count": int(len(arrays.get("initial_object_names", []))),
        "fk_target_vs_achieved_position_error_median_m": (
            float(np.median(arrays["fk_position_errors"]))
            if len(arrays["fk_position_errors"])
            else 0.0
        ),
        "fk_target_vs_achieved_position_error_max_m": (
            float(np.max(arrays["fk_position_errors"]))
            if len(arrays["fk_position_errors"])
            else 0.0
        ),
        "action_alignment": str(arrays["action_alignment"]),
        "action_label_mode": str(arrays["action_label_mode"]),
        "first_success_frame": (
            None
            if int(arrays.get("success_trim_first_success_frame", -1)) < 0
            else int(arrays["success_trim_first_success_frame"])
        ),
        "success_trim_original_observations": int(
            arrays.get("success_trim_original_observations", len(arrays["actions"]))
        ),
        "success_trim_kept_observations": int(
            arrays.get("success_trim_kept_observations", len(arrays["actions"]))
        ),
        "success_trim_removed_observations": int(
            arrays.get("success_trim_removed_observations", 0)
        ),
        "post_success_frames": int(
            arrays.get("success_trim_post_success_frames", -1)
        ),
        "world_base_worldflow_sidecars": bool("world_base_ee_poses" in arrays),
        "expert_path_mode": str(arrays.get("expert_path_mode", "linear_then_rrt")),
        "waypoint_roll_symmetry": bool(
            arrays.get("waypoint_roll_symmetry", False)
        ),
        "waypoint_roll_authored_fallback": bool(
            arrays.get("waypoint_roll_authored_fallback", False)
        ),
        "waypoint_roll_primary_error": str(
            arrays.get("waypoint_roll_primary_error", "")
        ),
        "expert_linear_path_calls": int(arrays.get("expert_linear_path_calls", 0)),
        "expert_rrt_path_calls": int(arrays.get("expert_rrt_path_calls", 0)),
        "expert_cartesian_path_calls": int(
            arrays.get("expert_cartesian_path_calls", 0)
        ),
        "expert_cartesian_stock_fallback_calls": int(
            arrays.get("expert_cartesian_stock_fallback_calls", 0)
        ),
        "worldflow_base_sidecar_validation": arrays.get(
            "worldflow_base_sidecar_validation"
        ),
        "created_unix_s": time.time(),
    }
    if "initial_phone_to_base_sensor_distance_m" in arrays:
        record["initial_phone_to_base_sensor_distance_m"] = float(
            arrays["initial_phone_to_base_sensor_distance_m"]
        )
    if "initial_phone_to_eef_distance_m" in arrays:
        record["initial_phone_to_eef_distance_m"] = float(
            arrays["initial_phone_to_eef_distance_m"]
        )
    if "matched_scene_source" in arrays:
        record["matched_scene_source"] = str(arrays["matched_scene_source"])
    if "initial_scene_nearest_distance_m" in arrays:
        distance = float(arrays["initial_scene_nearest_distance_m"])
        record["initial_scene_nearest_distance_m"] = (
            None if distance < 0.0 else distance
        )
        nearest_episode = int(arrays.get("initial_scene_nearest_episode", -1))
        record["initial_scene_nearest_episode"] = (
            None if nearest_episode < 0 else nearest_episode
        )
    if "dual_waypoint0_selected_branch" in arrays:
        record["dual_waypoint0_selected_branch"] = str(
            arrays["dual_waypoint0_selected_branch"]
        )
        record["dual_waypoint0_authored_frames"] = int(
            arrays["dual_waypoint0_authored_frames"]
        )
        record["dual_waypoint0_finger_swapped_frames"] = int(
            arrays["dual_waypoint0_finger_swapped_frames"]
        )
    with open(temp / "record.json", "w", encoding="utf-8") as file:
        json.dump(record, file, indent=2)
    if path.exists():
        shutil.rmtree(path)
    temp.rename(path)


def copy_tree_with_hardlinks(source, destination):
    """Copy a Zarr directory without duplicating data blocks on this disk."""
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    for source_path in source.rglob("*"):
        relative = source_path.relative_to(source)
        destination_path = destination / relative
        if source_path.is_dir():
            destination_path.mkdir()
        else:
            destination_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.link(source_path, destination_path)
            except OSError:
                shutil.copy2(source_path, destination_path)


def write_sidecar_meta(root, t_world_base=None, action_alignment="transition"):
    point_dir = root / POINT_DIR
    pose_dir = root / POSE_DIR
    raw_dir = root / RAW_ACTION_DIR
    raw_full_dir = root / RAW_ACTION_FULL_DIR
    task_state_dir = root / TASK_STATE_DIR
    object_state_dir = root / OBJECT_STATE_DIR
    point_dir.mkdir(parents=True, exist_ok=True)
    pose_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_full_dir.mkdir(parents=True, exist_ok=True)
    task_state_dir.mkdir(parents=True, exist_ok=True)
    object_state_dir.mkdir(parents=True, exist_ok=True)
    if t_world_base is not None:
        write_robot_base_sidecar_metadata(
            root,
            t_world_base,
            RLBENCH_PANDA_LINK0_TRANSFORM_SOURCE,
            action_alignment=action_alignment,
            base_frame_definition=RLBENCH_PANDA_LINK0_FRAME_VERSION,
        )
    with open(point_dir / "meta.json", "w", encoding="utf-8") as file:
        json.dump(
            {
                "key": "observation.point_cloud",
                "dtype": "float32",
                "shape": [None, 6],
                "layout": "episode_array",
                "storage_format": "zarr",
                "zarr_encoding": "packed_xyz_float16_rgb_uint8",
                "path_format": "point_clouds/episode_{episode_index:06d}.zarr",
                "coordinate_frame": "current_eff",
                "source_frame": "RLBench_world",
            },
            file,
            indent=2,
        )
    with open(pose_dir / "meta.json", "w", encoding="utf-8") as file:
        json.dump(
            {
                "key": "worldflow.ee_poses",
                "shape": [9],
                "dtype": "float32",
                "coordinate_frame": "RLBench_world",
                "path_format": "world_ee_poses/episode_{episode_index:06d}.npy",
            },
            file,
            indent=2,
        )
    with open(raw_dir / "meta.json", "w", encoding="utf-8") as file:
        json.dump(
            {
                "key": "rlbench.raw_expert_action",
                "shape": [8],
                "dtype": "float32",
                "layout": "episode_npy",
                "values": "7 joint positions followed by 0=closed/1=open",
                "alignment": "same transition index as action; first command moves observation[t] to observation[t+1]",
                "path_format": "raw_expert_actions/episode_{episode_index:06d}.npy",
            },
            file,
            indent=2,
        )
    with open(raw_full_dir / "meta.json", "w", encoding="utf-8") as file:
        json.dump(
            {
                "key": "rlbench.raw_expert_action_full_demo",
                "shape": [None, 8],
                "dtype": "float32",
                "values": "RLBench demo observations including row 0 NaN and post-step commands",
                "path_format": "raw_expert_actions_full/episode_{episode_index:06d}.npy",
            },
            file,
            indent=2,
        )
    with open(task_state_dir / "meta.json", "w", encoding="utf-8") as file:
        json.dump(
            {
                "key": "rlbench.initial_task_state",
                "layout": "episode_npz",
                "path_format": "initial_task_states/episode_{episode_index:06d}.npz",
                "configuration_bytes": "RLBench Task.get_state() configuration tree as uint8",
                "object_count": "Object count checked by RLBench Task.restore_state()",
                "purpose": "Restore the recorded first-frame task-object layout before action replay",
            },
            file,
            indent=2,
        )
    with open(object_state_dir / "meta.json", "w", encoding="utf-8") as file:
        json.dump(
            {
                "key": "rlbench.initial_object_states",
                "layout": "episode_npz",
                "path_format": "initial_object_states/episode_{episode_index:06d}.npz",
                "coordinate_frame": "RLBench_world",
                "fields": {
                    "initial_object_names": "object names",
                    "initial_object_handles": "CoppeliaSim handles",
                    "initial_object_types": "PyRep ObjectType names",
                    "initial_object_parent_handles": "parent handles, -1 means none",
                    "initial_object_parent_names": "parent names, empty means none",
                    "initial_object_poses": "[x,y,z,qx,qy,qz,qw]",
                    "initial_object_linear_velocities": "world linear velocity m/s",
                    "initial_object_angular_velocities": "world angular velocity rad/s",
                    "initial_object_joint_positions": "NaN for non-joints",
                    "initial_object_joint_velocities": "NaN for non-joints",
                    "initial_object_joint_target_positions": "NaN for non-joints",
                    "initial_object_joint_target_velocities": "NaN for non-joints",
                },
                "purpose": "Readable per-object reset snapshot; exact replay uses initial_task_states.",
            },
            file,
            indent=2,
        )


def write_frame_to_dataset(dataset, task, arrays, frame_index):
    image_key = "observation.images.front"
    dataset.add_frame(
        {
            "task": task,
            "action": arrays["actions"][frame_index],
            "observation.state": arrays["states"][frame_index],
            image_key: arrays["images"][frame_index],
        }
    )


def create_dataset(
    output_root,
    repo_id,
    fps,
    image_size,
    t_world_base=None,
    action_alignment="transition",
):
    features = {
        "action": {"dtype": "float32", "shape": (10,), "names": FEATURE_NAMES},
        "observation.state": {"dtype": "float32", "shape": (10,), "names": FEATURE_NAMES},
        "observation.images.front": {
            "dtype": "image",
            "shape": (image_size, image_size, 3),
            "names": ["height", "width", "channels"],
        },
    }
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        root=output_root,
        fps=fps,
        features=features,
        robot_type="rlbench_panda",
        use_videos=False,
    )
    write_sidecar_meta(
        dataset.root,
        t_world_base=t_world_base,
        action_alignment=action_alignment,
    )
    return dataset


def verify_packed_episode(
    output_root,
    episode_index,
    expected_frames,
    expected_points,
    require_world_base_worldflow_sidecars=True,
):
    """Verify durable episode outputs before a source artifact may be deleted."""

    stem = "episode_" + str(episode_index).zfill(6)
    point_path = output_root / POINT_DIR / (stem + ".zarr")
    point_group = zarr.open(str(point_path), mode="r")
    expected_shape = (int(expected_frames), int(expected_points), 3)
    xyz_shape = tuple(point_group["xyz"].shape)
    rgb_shape = tuple(point_group["rgb"].shape)
    if xyz_shape != expected_shape or rgb_shape != expected_shape:
        raise RuntimeError(
            "Packed point-cloud shape mismatch for "
            + stem
            + ": xyz="
            + str(xyz_shape)
            + " rgb="
            + str(rgb_shape)
            + " expected="
            + str(expected_shape)
        )
    required = [
        output_root / POSE_DIR / (stem + ".npy"),
        output_root / RAW_ACTION_DIR / (stem + ".npy"),
        output_root / RAW_ACTION_FULL_DIR / (stem + ".npy"),
        output_root / TASK_STATE_DIR / (stem + ".npz"),
    ]
    if require_world_base_worldflow_sidecars:
        required.extend(
            [
                output_root / WORLD_BASE_EE_POSE_DIR / (stem + ".npy"),
                output_root / WORLD_BASE_ACTION_TARGET_EE_POSE_DIR / (stem + ".npy"),
            ]
        )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(
            "Packed episode sidecars are missing before artifact cleanup: "
            + ", ".join(missing)
        )
    if require_world_base_worldflow_sidecars:
        for directory_name in (
            WORLD_BASE_EE_POSE_DIR,
            WORLD_BASE_ACTION_TARGET_EE_POSE_DIR,
        ):
            pose_path = output_root / directory_name / (stem + ".npy")
            poses = np.load(pose_path, mmap_mode="r")
            if poses.shape != (int(expected_frames), 9) or poses.dtype != np.float32:
                raise RuntimeError(
                    "Packed WorldFlow sidecar shape/dtype mismatch: "
                    + str(pose_path)
                    + " shape="
                    + str(poses.shape)
                    + " dtype="
                    + str(poses.dtype)
                )
            if not np.isfinite(poses).all():
                raise RuntimeError("Packed WorldFlow sidecar contains NaN/Inf: " + str(pose_path))


def pack_artifacts(args, artifact_root, records, expected_episode_count):
    output_root = args.output_root.expanduser().resolve()
    if output_root.exists():
        complete_marker = output_root / "meta" / "rlbench_conversion_complete.json"
        if complete_marker.is_file() and not args.overwrite:
            try:
                with open(complete_marker, "r", encoding="utf-8") as file:
                    complete_meta = json.load(file)
            except (OSError, ValueError, TypeError) as error:
                raise RuntimeError(
                    "Could not read the existing RLBench conversion marker: "
                    + str(complete_marker)
                ) from error
            existing_mode = str(complete_meta.get("action_label_mode", "expert_target"))
            if existing_mode != str(args.action_label_mode):
                raise RuntimeError(
                    "Existing dataset uses action_label_mode="
                    + existing_mode
                    + ", requested "
                    + str(args.action_label_mode)
                    + ". Use a different --output-root or pass --overwrite."
                )
            if args.generate_world_base_worldflow_sidecars:
                missing_worldflow = [
                    str(output_root / directory_name)
                    for directory_name in (
                        WORLD_BASE_EE_POSE_DIR,
                        WORLD_BASE_ACTION_TARGET_EE_POSE_DIR,
                    )
                    if not (output_root / directory_name / "meta.json").is_file()
                ]
                if missing_worldflow:
                    raise RuntimeError(
                        "Existing complete dataset predates robot-base WorldFlow sidecars: "
                        + ", ".join(missing_worldflow)
                        + ". Run the standalone sidecar backfill tool; the collector will not "
                        "delete a complete dataset implicitly."
                    )
                existing_base_frame = complete_meta.get("world_base_frame_definition")
                if existing_base_frame != RLBENCH_PANDA_LINK0_FRAME_VERSION:
                    raise RuntimeError(
                        "Existing complete dataset uses an incompatible or unversioned "
                        "WorldFlow base frame: "
                        + repr(existing_base_frame)
                        + "; required "
                        + repr(RLBENCH_PANDA_LINK0_FRAME_VERSION)
                        + ". Run the standalone sidecar backfill tool with the Panda-link0 "
                        "transform; the collector will not silently reuse old-base sidecars."
                    )
            print("[skip pack] complete output already exists: " + str(output_root))
            return
        print("[pack] removing an incomplete output before rebuilding: " + str(output_root))
        shutil.rmtree(output_root)
    output_root.parent.mkdir(parents=True, exist_ok=True)
    t_world_base = None
    if args.generate_world_base_worldflow_sidecars:
        if not records:
            raise RuntimeError("Cannot pack robot-base WorldFlow sidecars without episodes")
        first_record = records[0]
        first_artifact = artifact_root / episode_name(
            first_record["task"], first_record["local_episode_index"]
        )
        with np.load(first_artifact / "arrays.npz") as first_arrays:
            t_world_base = validate_rigid_transform(
                first_arrays["T_world_base"],
                "packed T_world_base",
            )
    dataset = create_dataset(
        output_root,
        args.repo_id,
        args.fps,
        args.image_size,
        t_world_base=t_world_base,
        action_alignment=args.action_alignment,
    )
    worldflow_validation = None
    if args.generate_world_base_worldflow_sidecars:
        worldflow_validation = {
            "episode_count": 0,
            "frame_count": 0,
            "achieved_roundtrip_max_abs": 0.0,
            "action_target_roundtrip_max_abs": 0.0,
            "achieved_rotation_orthogonality_max_abs": 0.0,
            "target_rotation_orthogonality_max_abs": 0.0,
            "achieved_rotation_determinant_min": float("inf"),
            "achieved_rotation_determinant_max": float("-inf"),
            "target_rotation_determinant_min": float("inf"),
            "target_rotation_determinant_max": float("-inf"),
        }
    try:
        for episode_index, record in enumerate(records):
            artifact = artifact_root / episode_name(record["task"], record["local_episode_index"])
            with np.load(artifact / "arrays.npz") as arrays_file:
                arrays = {
                    "actions": arrays_file["actions"],
                    "states": arrays_file["states"],
                    "world_ee_poses": arrays_file["world_ee_poses"],
                    "raw_expert_actions": arrays_file["raw_expert_actions"],
                    "raw_expert_actions_full": arrays_file["raw_expert_actions_full"],
                    "images": arrays_file["images"],
                    "timestamps": arrays_file["timestamps"],
                    "initial_task_state_bytes": arrays_file["initial_task_state_bytes"],
                    "initial_task_state_object_count": arrays_file["initial_task_state_object_count"],
                    "demo_random_seed_state": arrays_file["demo_random_seed_state"],
                    "demo_random_seed_position": arrays_file["demo_random_seed_position"],
                    "demo_random_seed_has_gauss": arrays_file["demo_random_seed_has_gauss"],
                    "demo_random_seed_cached_gaussian": arrays_file["demo_random_seed_cached_gaussian"],
                    "demo_num_reset_attempts": arrays_file["demo_num_reset_attempts"],
                }
                arrays.update(
                    {
                        key: arrays_file[key]
                        for key in OBJECT_STATE_KEYS
                        if key in arrays_file.files
                    }
                )
                arrays.update(
                    {
                        key: arrays_file[key]
                        for key in (
                            "world_base_ee_poses",
                            "world_base_action_target_ee_poses",
                            "T_world_base",
                            "T_base_world",
                        )
                        if key in arrays_file.files
                    }
                )
            if args.generate_world_base_worldflow_sidecars:
                episode_t_world_base = validate_rigid_transform(
                    arrays["T_world_base"],
                    "episode T_world_base",
                )
                if not np.allclose(
                    episode_t_world_base,
                    t_world_base,
                    atol=2e-6,
                    rtol=0.0,
                ):
                    raise RuntimeError(
                        "RLBench Panda base transform changed across collection workers/episodes"
                    )
                with open(artifact / "record.json", "r", encoding="utf-8") as file:
                    artifact_record = json.load(file)
                metrics = artifact_record.get("worldflow_base_sidecar_validation")
                if not isinstance(metrics, dict):
                    raise RuntimeError(
                        "Artifact is missing WorldFlow sidecar validation: " + str(artifact)
                    )
                worldflow_validation["episode_count"] += 1
                worldflow_validation["frame_count"] += int(metrics["frames"])
                for key in (
                    "achieved_roundtrip_max_abs",
                    "action_target_roundtrip_max_abs",
                    "achieved_rotation_orthogonality_max_abs",
                    "target_rotation_orthogonality_max_abs",
                    "achieved_rotation_determinant_max",
                    "target_rotation_determinant_max",
                ):
                    worldflow_validation[key] = max(
                        float(worldflow_validation[key]), float(metrics[key])
                    )
                for key in (
                    "achieved_rotation_determinant_min",
                    "target_rotation_determinant_min",
                ):
                    worldflow_validation[key] = min(
                        float(worldflow_validation[key]), float(metrics[key])
                    )
            copy_tree_with_hardlinks(
                artifact / "point_clouds.zarr",
                output_root / POINT_DIR / ("episode_" + str(episode_index).zfill(6) + ".zarr"),
            )
            np.save(output_root / POSE_DIR / ("episode_" + str(episode_index).zfill(6) + ".npy"), arrays["world_ee_poses"])
            np.save(output_root / RAW_ACTION_DIR / ("episode_" + str(episode_index).zfill(6) + ".npy"), arrays["raw_expert_actions"])
            np.save(
                output_root / RAW_ACTION_FULL_DIR / ("episode_" + str(episode_index).zfill(6) + ".npy"),
                arrays["raw_expert_actions_full"],
            )
            if args.generate_world_base_worldflow_sidecars:
                stem = "episode_" + str(episode_index).zfill(6) + ".npy"
                np.save(
                    output_root / WORLD_BASE_EE_POSE_DIR / stem,
                    np.ascontiguousarray(arrays["world_base_ee_poses"], dtype=np.float32),
                )
                np.save(
                    output_root / WORLD_BASE_ACTION_TARGET_EE_POSE_DIR / stem,
                    np.ascontiguousarray(
                        arrays["world_base_action_target_ee_poses"], dtype=np.float32
                    ),
                )
            np.savez(
                output_root / TASK_STATE_DIR / ("episode_" + str(episode_index).zfill(6) + ".npz"),
                configuration_bytes=arrays["initial_task_state_bytes"],
                object_count=arrays["initial_task_state_object_count"],
                demo_random_seed_state=arrays["demo_random_seed_state"],
                demo_random_seed_position=arrays["demo_random_seed_position"],
                demo_random_seed_has_gauss=arrays["demo_random_seed_has_gauss"],
                demo_random_seed_cached_gaussian=arrays["demo_random_seed_cached_gaussian"],
                demo_num_reset_attempts=arrays["demo_num_reset_attempts"],
            )
            object_state_arrays = {
                key: arrays[key] for key in OBJECT_STATE_KEYS if key in arrays
            }
            if object_state_arrays:
                np.savez_compressed(
                    output_root
                    / OBJECT_STATE_DIR
                    / ("episode_" + str(episode_index).zfill(6) + ".npz"),
                    **object_state_arrays,
                )
            for frame_index in range(len(arrays["actions"])):
                write_frame_to_dataset(dataset, record["description"], arrays, frame_index)
            dataset.save_episode()
            print("[pack] episode=" + str(episode_index) + " task=" + record["task"] + " frames=" + str(record["frames"]))
            if args.delete_artifacts_after_pack:
                verify_packed_episode(
                    output_root,
                    episode_index,
                    expected_frames=len(arrays["actions"]),
                    expected_points=args.num_points,
                    require_world_base_worldflow_sidecars=(
                        args.generate_world_base_worldflow_sidecars
                    ),
                )
                shutil.rmtree(artifact)
                print(
                    "[artifact-cleanup] verified packed episode="
                    + str(episode_index)
                    + "; deleted="
                    + str(artifact),
                    flush=True,
                )
    finally:
        dataset.finalize()
    if args.generate_world_base_worldflow_sidecars:
        expected_frames = sum(int(record["frames"]) for record in records)
        if int(worldflow_validation["episode_count"]) != len(records):
            raise RuntimeError("WorldFlow validation episode count does not match packed records")
        if int(worldflow_validation["frame_count"]) != expected_frames:
            raise RuntimeError("WorldFlow validation frame count does not match packed records")
        achieved_files = list((output_root / WORLD_BASE_EE_POSE_DIR).glob("episode_*.npy"))
        target_files = list(
            (output_root / WORLD_BASE_ACTION_TARGET_EE_POSE_DIR).glob("episode_*.npy")
        )
        if len(achieved_files) != len(records) or len(target_files) != len(records):
            raise RuntimeError(
                "Packed WorldFlow sidecar file count does not match dataset episodes"
            )
        with open(
            output_root / "meta" / "rlbench_worldflow_robot_base_sidecars.json",
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                {
                    "complete": True,
                    "dataset_root": str(output_root),
                    "action_alignment": str(args.action_alignment),
                    "target_semantics": "commanded action target; never achieved next pose",
                    "transform_source": (
                        RLBENCH_PANDA_LINK0_TRANSFORM_SOURCE
                    ),
                    "base_frame_definition": RLBENCH_PANDA_LINK0_FRAME_VERSION,
                    "T_world_base": np.asarray(t_world_base, dtype=np.float64).tolist(),
                    "T_base_world": np.linalg.inv(
                        np.asarray(t_world_base, dtype=np.float64)
                    ).tolist(),
                    "validation": worldflow_validation,
                },
                file,
                indent=2,
            )
    packed_task_order = list(dict.fromkeys(item["task"] for item in records))
    post_success_frames_by_task = {
        task: sorted(
            {
                int(item.get("post_success_frames", args.post_success_frames))
                for item in records
                if item["task"] == task
            }
        )
        for task in packed_task_order
    }
    packed_post_success_values = sorted(
        {value for values in post_success_frames_by_task.values() for value in values}
    )
    packed_post_success_frames = (
        packed_post_success_values[0]
        if len(packed_post_success_values) == 1
        else None
    )
    with open(output_root / "meta" / "rlbench_conversion.json", "w", encoding="utf-8") as file:
        json.dump(
            {
                "source": "RLBench live expert policy",
                "expert_action_mode": "MoveArmThenGripper(JointVelocity, Discrete)",
                "main_action_semantics": action_semantics(args.action_label_mode),
                "action_alignment": str(args.action_alignment),
                "action_label_mode": str(args.action_label_mode),
                "post_success_frames": packed_post_success_frames,
                "post_success_frames_by_task": post_success_frames_by_task,
                "min_first_success_frame": int(args.min_first_success_frame),
                "post_success_trim_semantics": (
                    "keep the first successful dataset frame plus N following frames; "
                    "transition alignment retains one additional terminal observation"
                ),
                "observation_state_semantics": "achieved EEF pose relative to episode EEF0",
                "world_base_worldflow_sidecars": bool(
                    args.generate_world_base_worldflow_sidecars
                ),
                "world_base_frame_definition": (
                    RLBENCH_PANDA_LINK0_FRAME_VERSION
                    if args.generate_world_base_worldflow_sidecars
                    else None
                ),
                "world_base_ee_pose_semantics": (
                    "T_base_ee = inverse(T_world_base) @ T_world_ee"
                    if args.generate_world_base_worldflow_sidecars
                    else None
                ),
                "world_base_action_target_semantics": (
                    "commanded expert target: inverse(T_world_base) @ "
                    "T_world_eef0 @ T_eef0_target"
                    if args.generate_world_base_worldflow_sidecars
                    else None
                ),
                "T_world_base": (
                    np.asarray(t_world_base, dtype=np.float64).tolist()
                    if t_world_base is not None
                    else None
                ),
                "T_base_world": (
                    np.linalg.inv(np.asarray(t_world_base, dtype=np.float64)).tolist()
                    if t_world_base is not None
                    else None
                ),
                "point_cloud_semantics": "finite front-camera world cloud plus selected virtual gripper template transformed to current EEF",
                "scene_bounds": RLBENCH_SCENE_BOUNDS.tolist(),
                "gripper_template": (
                    LIBERO_GRIPPER_TEMPLATE_VERSION
                    if args.gripper_template == LIBERO_GRIPPER_TEMPLATE
                    else RLBENCH_PANDA_GRIPPER_TEMPLATE_VERSION
                ),
                "gripper_template_name": str(args.gripper_template),
                "gripper_template_version": (
                    LIBERO_GRIPPER_TEMPLATE_VERSION
                    if args.gripper_template == LIBERO_GRIPPER_TEMPLATE
                    else RLBENCH_PANDA_GRIPPER_TEMPLATE_VERSION
                ),
                "virtual_gripper_width_normalization_max_m": (
                    LIBERO_GRIPPER_WIDTH_NORMALIZATION_MAX
                    if args.gripper_template == LIBERO_GRIPPER_TEMPLATE
                    else float(args.gripper_max_width)
                ),
                "virtual_gripper_geometry_max_width_m": (
                    LIBERO_REAP_TEMPLATE_MAX_WIDTH
                    if args.gripper_template == LIBERO_GRIPPER_TEMPLATE
                    else RLBENCH_PANDA_MAX_WIDTH
                ),
                "virtual_gripper_opening_max_width_m": (
                    LIBERO_REAP_OPENING_MAX_WIDTH
                    if args.gripper_template == LIBERO_GRIPPER_TEMPLATE
                    else float(args.gripper_max_width)
                ),
                "virtual_gripper_local_offset_m": (
                    [0.0, 0.0, -LIBERO_REAP_GRIPPER_LEN]
                    if args.gripper_template == LIBERO_GRIPPER_TEMPLATE
                    else [0.0, 0.0, 0.0]
                ),
                "initial_task_state_semantics": "RLBench task configuration tree at reset_to_demo first frame",
                "initial_object_state_semantics": "Readable per-object world state at reset_to_demo first frame",
                "collection_workers": int(args.collection_workers),
                "expert_path_mode": str(args.expert_path_mode),
                "phone_path_candidates": int(args.phone_path_candidates),
                "phone_roll_symmetry": bool(args.phone_roll_symmetry),
                "waypoint_roll_symmetry": bool(args.waypoint_roll_symmetry),
                "replay_random_seeds_from_artifacts": (
                    None
                    if args.replay_random_seeds_from_artifacts is None
                    else str(
                        Path(args.replay_random_seeds_from_artifacts)
                        .expanduser()
                        .resolve()
                    )
                ),
                "collection_seed": (
                    None if args.collection_seed is None else int(args.collection_seed)
                ),
                "phone_base_max_initial_distance_m": (
                    None
                    if args.phone_base_max_initial_distance_m is None
                    else float(args.phone_base_max_initial_distance_m)
                ),
                "phone_eef_max_initial_distance_m": (
                    None
                    if args.phone_eef_max_initial_distance_m is None
                    else float(args.phone_eef_max_initial_distance_m)
                ),
                "phone_robot_base_min_initial_distance_m": (
                    None
                    if args.phone_robot_base_min_initial_distance_m is None
                    else float(args.phone_robot_base_min_initial_distance_m)
                ),
                "replay_scenes_from_artifacts": (
                    None
                    if args.replay_scenes_from_artifacts is None
                    else str(Path(args.replay_scenes_from_artifacts).expanduser().resolve())
                ),
                "source_artifacts_retained": not bool(args.delete_artifacts_after_pack),
                "artifact_cleanup_mode": (
                    "delete_each_episode_after_verified_pack"
                    if args.delete_artifacts_after_pack
                    else "retain"
                ),
                "episode_count": len(records),
                # Preserve first occurrence: this is the order used by
                # LeRobotDataset to assign task_index in tasks.parquet.
                "tasks": packed_task_order,
            },
            file,
            indent=2,
        )
    if len(records) == expected_episode_count:
        with open(output_root / "meta" / "rlbench_conversion_complete.json", "w", encoding="utf-8") as file:
            json.dump(
                {
                    "complete": True,
                    "episode_count": len(records),
                    "action_label_mode": str(args.action_label_mode),
                    "world_base_worldflow_sidecars": bool(
                        args.generate_world_base_worldflow_sidecars
                    ),
                    "world_base_frame_definition": (
                        RLBENCH_PANDA_LINK0_FRAME_VERSION
                        if args.generate_world_base_worldflow_sidecars
                        else None
                    ),
                },
                file,
                indent=2,
            )


def collection_config_signature(args, tasks):
    template_version = (
        LIBERO_GRIPPER_TEMPLATE_VERSION
        if args.gripper_template == LIBERO_GRIPPER_TEMPLATE
        else RLBENCH_PANDA_GRIPPER_TEMPLATE_VERSION
    )
    return {
        "tasks": list(tasks),
        "episodes_per_task": int(args.episodes_per_task),
        "episode_start": int(getattr(args, "episode_start", 0)),
        "episode_indices": (
            None
            if args.episode_indices is None
            else list(map(int, args.episode_indices))
        ),
        "variation": int(args.variation),
        "collection_seed": (
            None if args.collection_seed is None else int(args.collection_seed)
        ),
        "num_points": int(args.num_points),
        "gripper_points": int(args.gripper_points),
        "gripper_max_width": float(args.gripper_max_width),
        "gripper_template": str(args.gripper_template),
        "gripper_template_version": template_version,
        "virtual_gripper_width_normalization_max_m": (
            LIBERO_GRIPPER_WIDTH_NORMALIZATION_MAX
            if args.gripper_template == LIBERO_GRIPPER_TEMPLATE
            else float(args.gripper_max_width)
        ),
        "virtual_gripper_geometry_max_width_m": (
            LIBERO_REAP_TEMPLATE_MAX_WIDTH
            if args.gripper_template == LIBERO_GRIPPER_TEMPLATE
            else RLBENCH_PANDA_MAX_WIDTH
        ),
        "virtual_gripper_opening_max_width_m": (
            LIBERO_REAP_OPENING_MAX_WIDTH
            if args.gripper_template == LIBERO_GRIPPER_TEMPLATE
            else float(args.gripper_max_width)
        ),
        "virtual_gripper_len_m": (
            LIBERO_REAP_GRIPPER_LEN
            if args.gripper_template == LIBERO_GRIPPER_TEMPLATE
            else 0.0
        ),
        "image_size": int(args.image_size),
        "fps": int(args.fps),
        "collection_workers": int(args.collection_workers),
        "expert_path_mode": str(args.expert_path_mode),
        "allow_rrt": bool(args.allow_rrt),
        "phone_path_candidates": int(args.phone_path_candidates),
        "phone_smooth_ik_reference_root": (
            None
            if args.phone_smooth_ik_reference_root is None
            else str(
                Path(args.phone_smooth_ik_reference_root)
                .expanduser()
                .resolve()
            )
        ),
        "phone_smooth_ik_exclude_episodes": list(
            map(int, args.phone_smooth_ik_exclude_episodes)
        ),
        "phone_best_of_n_waypoint3_only": bool(
            args.phone_best_of_n_waypoint3_only
        ),
        "path_xyz_loop_floor_m": float(args.path_xyz_loop_floor_m),
        "path_rotation_excess_limit_rad": float(
            args.path_rotation_excess_limit_rad
        ),
        "roll_path_max_lateral_deviation_m": float(
            args.roll_path_max_lateral_deviation_m
        ),
        "roll_path_max_detour_ratio": float(
            args.roll_path_max_detour_ratio
        ),
        "roll_path_max_wrist_travel_rad": float(
            args.roll_path_max_wrist_travel_rad
        ),
        "roll_path_max_joint_step_rad": float(
            args.roll_path_max_joint_step_rad
        ),
        "phone_waypoint3_max_detour_ratio": float(
            args.phone_waypoint3_max_detour_ratio
        ),
        "phone_waypoint3_max_lateral_deviation_m": float(
            args.phone_waypoint3_max_lateral_deviation_m
        ),
        "phone_waypoint3_max_joint_travel_rad": float(
            args.phone_waypoint3_max_joint_travel_rad
        ),
        "phone_waypoint3_exact_execution": bool(
            args.phone_waypoint3_exact_execution
        ),
        "phone_waypoint3_exact_step_rad": float(
            args.phone_waypoint3_exact_step_rad
        ),
        "phone_roll_symmetry": bool(args.phone_roll_symmetry),
        "phone_waypoint0_roll_branch": args.phone_waypoint0_roll_branch,
        "phone_later_waypoint_roll_policy": str(
            args.phone_later_waypoint_roll_policy
        ),
        "phone_waypoint_local_z_offset_deg": float(
            args.phone_waypoint_local_z_offset_deg
        ),
        "dual_waypoint0_roll_select_shorter": bool(
            args.dual_waypoint0_roll_select_shorter
        ),
        "waypoint_roll_symmetry": bool(args.waypoint_roll_symmetry),
        "waypoint_roll_force_branch": args.waypoint_roll_force_branch,
        "waypoint_roll_symmetry_selection": str(
            args.waypoint_roll_symmetry_selection
        ),
        "segmented_linear_segments": int(args.segmented_linear_segments),
        "franka_cartesian_servo": {
            "linear_speed_m_s": float(args.franka_servo_linear_speed_m_s),
            "angular_speed_rad_s": float(args.franka_servo_angular_speed_rad_s),
            "max_joint_speed_rad_s": float(
                args.franka_servo_max_joint_speed_rad_s
            ),
            "max_joint_accel_step_rad_s": float(
                args.franka_servo_max_joint_accel_step_rad_s
            ),
            "damping": float(args.franka_servo_damping),
            "nullspace_gain": float(args.franka_servo_nullspace_gain),
            "max_steps": int(args.franka_servo_max_steps),
            "save_debug": bool(args.franka_servo_save_debug),
            "position_tolerance_m": 0.003,
            "rotation_tolerance_rad": 0.04,
        },
        "cartesian_detour": {
            "max_offset_m": float(args.cartesian_detour_max_offset_m),
            "offset_step_m": float(args.cartesian_detour_offset_step_m),
            "segments": int(args.segmented_linear_segments),
            "curve": "linear_xyz_plus_sin_pi_t_times_fixed_offset",
            "orientation": "shortest_arc_quaternion_slerp",
            "planner": "none_sequential_linear_ik_only",
            "joint_interp_max_lateral_deviation_m": float(
                args.joint_interp_max_lateral_deviation_m
            ),
        },
        "segmented_fallback_algorithm": str(
            args.segmented_fallback_algorithm
        ),
        "segmented_linear_waypoints": (
            None
            if args.segmented_linear_waypoints is None
            else list(args.segmented_linear_waypoints)
        ),
        "replay_random_seeds_from_artifacts": (
            None
            if args.replay_random_seeds_from_artifacts is None
            else str(Path(args.replay_random_seeds_from_artifacts).expanduser().resolve())
        ),
        "phone_base_max_initial_distance_m": (
            None
            if args.phone_base_max_initial_distance_m is None
            else float(args.phone_base_max_initial_distance_m)
        ),
        "phone_eef_max_initial_distance_m": (
            None
            if args.phone_eef_max_initial_distance_m is None
            else float(args.phone_eef_max_initial_distance_m)
        ),
        "phone_robot_base_min_initial_distance_m": (
            None
            if args.phone_robot_base_min_initial_distance_m is None
            else float(args.phone_robot_base_min_initial_distance_m)
        ),
        "replay_scenes_from_artifacts": (
            None
            if args.replay_scenes_from_artifacts is None
            else str(Path(args.replay_scenes_from_artifacts).expanduser().resolve())
        ),
        "matched_scene_demo_candidates": int(args.matched_scene_demo_candidates),
        "action_semantics_version": action_semantics_version(args.action_label_mode),
        "action_label_mode": str(args.action_label_mode),
        "action_alignment": str(args.action_alignment),
        "post_success_frames": int(args.post_success_frames),
        "min_first_success_frame": int(args.min_first_success_frame),
        "stop_after_success": bool(args.stop_after_success),
        "min_initial_scene_distance_m": float(
            args.min_initial_scene_distance_m
        ),
        "scene_rotation_radius_m": float(args.scene_rotation_radius_m),
        "front_camera_position_m": (
            None
            if args.front_camera_position_m is None
            else list(map(float, args.front_camera_position_m))
        ),
        "front_camera_look_at_m": (
            None
            if args.front_camera_look_at_m is None
            else list(map(float, args.front_camera_look_at_m))
        ),
        "artifacts_only": bool(args.artifacts_only),
        "generate_world_base_worldflow_sidecars": bool(
            args.generate_world_base_worldflow_sidecars
        ),
        "world_base_frame_definition": (
            RLBENCH_PANDA_LINK0_FRAME_VERSION
            if args.generate_world_base_worldflow_sidecars
            else None
        ),
        "scene_bounds": RLBENCH_SCENE_BOUNDS.tolist(),
    }


def collect(args, tasks, artifact_root):
    from rlbench import Environment
    from rlbench.action_modes.action_mode import MoveArmThenGripper
    from rlbench.action_modes.arm_action_modes import JointVelocity
    from rlbench.action_modes.gripper_action_modes import Discrete
    from rlbench.backend.scene import Scene as BackendScene
    from rlbench.backend.task import Task as BackendTask
    from rlbench.backend.waypoints import Point, PredefinedPath
    from pyrep.objects.cartesian_path import CartesianPath
    from pyrep.const import ConfigurationPathAlgorithms as Algos
    from pyrep.errors import ConfigurationError, ConfigurationPathError, IKError
    from pyrep.robots.configuration_paths.arm_configuration_path import (
        ArmConfigurationPath,
    )

    class DenseJointServoPath(ArmConfigurationPath):
        """Track a checked dense joint polyline with physical motor control.

        Unlike ``ArmConfigurationPath``, this does not ask Reflexxes/RML to
        reparameterize across the whole polyline. Unlike the exact diagnostic
        trackers, it never teleports joints or disables dynamics. Each dense
        target must be reached by the simulator's joint servos before the next
        one is issued, preserving both path topology and physical grasps.
        """

        def __init__(self, arm, path_points, waypoint_name=""):
            configurations = np.asarray(
                path_points, dtype=np.float64
            ).reshape(-1, int(arm.get_joint_count()))
            if len(configurations) == 0:
                raise ConfigurationPathError("empty dense servo path")
            super().__init__(arm, configurations.reshape(-1))
            self._dense_configurations = configurations
            self._dense_index = 0
            self._dense_stall_steps = 0
            self._waypoint_name = str(waypoint_name)
            # A carried rigid body can leave a persistent 0.02--0.055 rad
            # motor error at one dense sample because of contact forces.  Do
            # not deadlock on that intermediate sample: the next nearby target
            # supplies the force needed to continue along the same topology.
            self._position_tolerance = 0.060
            self._final_tolerance = 0.010

        def step(self):
            if self._path_done:
                raise RuntimeError("Dense servo path has already completed")
            target = self._dense_configurations[self._dense_index]
            self._joint_position_action = target.copy()
            self._arm.set_joint_target_positions(target.tolist())
            current = np.asarray(
                self._arm.get_joint_positions(), dtype=np.float64
            )
            error = float(np.max(np.abs(current - target)))
            tolerance = (
                self._final_tolerance
                if self._dense_index == len(self._dense_configurations) - 1
                else self._position_tolerance
            )
            if error <= tolerance:
                self._dense_stall_steps = 0
                if self._dense_index == len(self._dense_configurations) - 1:
                    self._path_done = True
                    return True
                self._dense_index += 1
            else:
                self._dense_stall_steps += 1
                if (
                    self._dense_index < len(self._dense_configurations) - 1
                    and self._dense_stall_steps >= 10
                    and error <= 0.15
                ):
                    # Contact may hold a joint a few degrees behind a dense
                    # sample. Advance one nearby setpoint after ten physics
                    # steps so the motor can keep pulling along the same
                    # branch; never skip when lag exceeds the 0.15-rad guard.
                    print(
                        "[dense-servo-bounded-advance] waypoint="
                        + self._waypoint_name
                        + " index="
                        + str(self._dense_index)
                        + " lag_rad="
                        + format(error, ".6f"),
                        flush=True,
                    )
                    self._dense_index += 1
                    self._dense_stall_steps = 0
                    return False
                if self._dense_stall_steps > 500:
                    raise ConfigurationPathError(
                        "Dense servo stalled at waypoint "
                        + self._waypoint_name
                        + " index="
                        + str(self._dense_index)
                        + " error_rad="
                        + format(error, ".6f")
                    )
            return False

        def set_to_start(self, disable_dynamics=False):
            self._arm.set_joint_positions(
                self._dense_configurations[0].tolist(),
                disable_dynamics=disable_dynamics,
            )
            self._dense_index = 0
            self._dense_stall_steps = 0
            self._path_done = False
            self._joint_position_action = None

    if args.collection_seed is not None:
        effective_collection_seed = int(args.collection_seed) + int(
            getattr(args, "episode_start", 0)
        )
        np.random.seed(effective_collection_seed)
        print(
            "[collection-seed] base="
            + str(args.collection_seed)
            + " effective="
            + str(effective_collection_seed),
            flush=True,
        )

    original_point_get_path = Point.get_path
    original_predefined_get_path = PredefinedPath.get_path
    linear_path_calls = 0
    rrt_path_calls = 0
    cartesian_path_calls = 0
    cartesian_stock_fallback_calls = 0
    phone_waypoint0_default_selection_info = None
    forced_waypoint0_roll_branch = args.phone_waypoint0_roll_branch
    forced_waypoint_roll_branch = args.waypoint_roll_force_branch
    roll_symmetry_execution_active = False
    phone_waypoint_feasibility_active = False
    original_task_feasible = None
    path_fk_model = None
    # Keep the default collector behavior identical to upstream RLBench:
    # ordinary Point waypoints use Point.get_path(), which tries the linear
    # path and falls back to RRTConnect. Other legacy modes remain opt-in so a
    # historical command cannot silently change the expert trajectory.
    rrt_free_expert_path_modes = {
        "linear_only", "segmented_linear", "franka_cartesian_servo",
        "cartesian_detour", "phone_prm_0_3", "phone_prm_all",
        "phone_baseframe_reference_ik_0_3",
    }
    if not args.allow_rrt:
        if (
            args.expert_path_mode not in rrt_free_expert_path_modes
            and args.expert_path_mode != "linear_then_rrt"
        ):
            raise ValueError(
                "RRT/OMPL path fallback is disabled for dataset collection. "
                "Use --expert-path-mode linear_only, segmented_linear, "
                "franka_cartesian_servo, cartesian_detour, or the upstream "
                "linear_then_rrt mode. "
                "Only deliberate historical reproduction may add --allow-rrt."
            )
        if (
            args.expert_path_mode == "segmented_linear"
            and args.segmented_linear_waypoints is not None
        ):
            raise ValueError(
                "RRT-free segmented collection must cover every ordinary Point "
                "waypoint; omit --segmented-linear-waypoints."
            )
    if args.phone_best_of_n_waypoint3_only:
        original_task_feasible = BackendTask._feasible

        def phone_waypoint_feasibility_context(task, waypoints):
            nonlocal phone_waypoint_feasibility_active
            phone_waypoint_feasibility_active = True
            try:
                return original_task_feasible(task, waypoints)
            finally:
                phone_waypoint_feasibility_active = False

        BackendTask._feasible = phone_waypoint_feasibility_context
    elif (
        args.expert_path_mode in (
            "phone_prm_0_3",
            "phone_prm_all",
            "phone_baseframe_reference_ik_0_3",
        )
        and args.replay_scenes_from_artifacts is not None
    ):
        original_task_feasible = BackendTask._feasible

        def matched_phone_prm_feasibility(task, waypoints):
            # The stored scene already comes from a successful expert demo.
            # PRM is re-planned from the restored state during the real run;
            # replaying its intermediate endpoint by set_to_end() here can put
            # the IK chain on a different numerical branch at waypoint1.
            if task.get_name() == "phone_on_base":
                return True, -1
            return original_task_feasible(task, waypoints)

        BackendTask._feasible = matched_phone_prm_feasibility
    if args.expert_path_mode in (
        "phone_cartesian_0_3",
        "phone_predefined_path_waypoint3",
        "phone_prm_waypoint3",
        "phone_prm_0_3",
        "phone_prm_all",
        "phone_staged_linear_0_3",
        "phone_staged_linear_then_rrt_0_3",
        "phone_rrt_cartesian_shortcut_0_3",
        "phone_best_ik_joint_interp_0_3",
        "phone_baseframe_reference_ik_0_3",
        "phone_best_of_n_0_3",
    ) and tasks != ["phone_on_base"]:
        raise ValueError(
            "The selected phone waypoint path mode is intentionally limited "
            "to a single --tasks phone_on_base collection."
        )
    if (
        args.expert_path_mode == "phone_baseframe_reference_ik_0_3"
        and args.phone_smooth_ik_reference_root is None
    ):
        raise ValueError(
            "phone_baseframe_reference_ik_0_3 requires "
            "--phone-smooth-ik-reference-root"
        )
    if args.phone_waypoint3_exact_execution and not (
        args.expert_path_mode == "phone_rrt_cartesian_shortcut_0_3"
        and args.phone_best_of_n_waypoint3_only
    ):
        raise ValueError(
            "--phone-waypoint3-exact-execution requires "
            "--expert-path-mode phone_rrt_cartesian_shortcut_0_3 and "
            "--phone-best-of-n-waypoint3-only"
        )
    if (
        args.phone_waypoint0_roll_branch is not None
        and tasks != ["phone_on_base"]
    ):
        raise ValueError(
            "--phone-waypoint0-roll-branch is limited to a single "
            "--tasks phone_on_base collection."
        )
    if (
        args.dual_waypoint0_roll_select_shorter
        and tasks not in (["phone_on_base"], ["take_frame_off_hanger"])
    ):
        raise ValueError(
            "--dual-waypoint0-roll-select-shorter requires a single "
            "phone_on_base or take_frame_off_hanger task."
        )
    if (
        args.waypoint_roll_symmetry_selection == "chain_path_cost"
        and tasks not in (["take_frame_off_hanger"], ["phone_on_base"])
    ):
        raise ValueError(
            "--waypoint-roll-symmetry-selection=chain_path_cost is currently "
            "limited to a single take_frame_off_hanger or phone_on_base task."
        )
    if args.waypoint_roll_force_branch is not None and not args.waypoint_roll_symmetry:
        raise ValueError(
            "--waypoint-roll-force-branch requires --waypoint-roll-symmetry"
        )
    if args.expert_path_mode == "cartesian_detour":
        from pyrep.robots.configuration_paths.arm_configuration_path import (
            ArmConfigurationPath,
        )
        from scipy.optimize import least_squares
        from scipy.spatial.transform import Rotation, Slerp

        cartesian_detour_post_step_target = None
        cartesian_detour_post_step_final = None
        original_cartesian_detour_scene_step = BackendScene.step

        def cartesian_detour_scene_step(scene):
            """Correct a dense target after physics and before observation."""
            nonlocal cartesian_detour_post_step_target
            nonlocal cartesian_detour_post_step_final
            original_cartesian_detour_scene_step(scene)
            if cartesian_detour_post_step_target is None:
                return
            target = np.asarray(
                cartesian_detour_post_step_target, dtype=np.float64
            )
            scene.robot.arm.set_joint_positions(
                target.tolist(), disable_dynamics=True
            )
            scene.robot.arm.set_joint_target_positions(target.tolist())
            if cartesian_detour_post_step_final is not None:
                actual = np.asarray(
                    scene.robot.arm.get_joint_positions(), dtype=np.float64
                )
                print(
                    "[cartesian-detour-post-step-final] waypoint="
                    + str(cartesian_detour_post_step_final)
                    + " max_joint_error_rad="
                    + format(float(np.max(np.abs(actual - target))), ".9f"),
                    flush=True,
                )
            cartesian_detour_post_step_target = None
            cartesian_detour_post_step_final = None

        BackendScene.step = cartesian_detour_scene_step

        class TrackingArmConfigurationPath(ArmConfigurationPath):
            """Apply one checked dense joint configuration per physics step.

            Each target is also re-applied immediately after the physics step,
            before RLBench records its observation. No OMPL/RRT is involved.
            """

            def __init__(self, arm, path_points, robot=None, waypoint_name=""):
                super().__init__(arm, np.asarray(path_points).reshape(-1))
                self._tracked_configs = np.asarray(
                    path_points, dtype=np.float64
                ).reshape(-1, int(arm.get_joint_count()))
                self._robot = robot
                self._waypoint_name = str(waypoint_name)
                self._tracked_index = 0

            def step(self):
                nonlocal cartesian_detour_post_step_target
                nonlocal cartesian_detour_post_step_final
                if self._path_done:
                    raise RuntimeError("Dense path has already completed")
                target = self._tracked_configs[self._tracked_index]
                self._joint_position_action = target.copy()
                self._arm.set_joint_positions(
                    target.tolist(), disable_dynamics=True
                )
                self._arm.set_joint_target_positions(target.tolist())
                cartesian_detour_post_step_target = target.copy()
                is_final = self._tracked_index >= len(self._tracked_configs) - 1
                cartesian_detour_post_step_final = (
                    self._waypoint_name if is_final else None
                )
                if is_final:
                    self._path_done = True
                    return True
                self._tracked_index += 1
                return False

            def set_to_start(self, disable_dynamics=False):
                nonlocal cartesian_detour_post_step_target
                nonlocal cartesian_detour_post_step_final
                self._arm.set_joint_positions(
                    self._tracked_configs[0].tolist(),
                    disable_dynamics=disable_dynamics,
                )
                self._tracked_index = 0
                self._path_done = False
                self._joint_position_action = None
                cartesian_detour_post_step_target = None
                cartesian_detour_post_step_final = None

        cartesian_reference_waypoint_paths = {}
        cartesian_reference_waypoint_poses = {}
        if args.phone_reference_artifact is not None:
            reference_arrays = np.load(
                args.phone_reference_artifact, allow_pickle=False
            )
            reference_names = [
                str(name) for name in reference_arrays["waypoint_end_names"]
            ]
            reference_frames = np.asarray(
                reference_arrays["waypoint_end_frames"], dtype=np.int64
            )
            reference_actions = np.asarray(
                reference_arrays["raw_expert_actions"], dtype=np.float64
            )
            reference_poses = np.asarray(
                reference_arrays["world_ee_poses"], dtype=np.float64
            )
            reference_start = 0
            for reference_name, reference_frame in zip(
                reference_names, reference_frames
            ):
                reference_stop = min(
                    int(reference_frame) + 1, len(reference_actions)
                )
                reference_segment = reference_actions[
                    reference_start:reference_stop, :7
                ].copy()
                if len(reference_segment) > 1:
                    keep = np.ones(len(reference_segment), dtype=bool)
                    keep[1:] = (
                        np.max(
                            np.abs(np.diff(reference_segment, axis=0)),
                            axis=1,
                        )
                        > 1e-7
                    )
                    reference_segment = reference_segment[keep]
                cartesian_reference_waypoint_paths[reference_name] = (
                    reference_segment
                )
                cartesian_reference_waypoint_poses[reference_name] = (
                    reference_poses[
                        reference_start:reference_stop, :7
                    ].copy()
                )
                reference_start = reference_stop
            print(
                "[cartesian-reference-paths] source="
                + str(args.phone_reference_artifact)
                + " waypoints="
                + ",".join(sorted(cartesian_reference_waypoint_paths)),
                flush=True,
            )

        def cartesian_detour_point_get_path(point, ignore_collisions=False):
            """Find the smallest deterministic smooth Cartesian arc that IK can follow."""
            nonlocal linear_path_calls
            arm = point._robot.arm
            joint_count = int(arm.get_joint_count())
            waypoint_name = point._waypoint.get_name()
            if (
                args.segmented_linear_waypoints is not None
                and waypoint_name not in args.segmented_linear_waypoints
            ):
                reference_path = cartesian_reference_waypoint_paths.get(
                    waypoint_name
                )
                if reference_path is not None and len(reference_path):
                    print(
                        "[cartesian-reference-segment] waypoint="
                        + waypoint_name
                        + " points="
                        + str(len(reference_path)),
                        flush=True,
                    )
                    # Replay unaffected portions with the same PyRep path
                    # stepping semantics that produced the source demo.  Only
                    # the repaired transport segment uses exact tracking;
                    # changing grasp-approach physics can otherwise prevent
                    # the bottle from attaching despite identical endpoints.
                    return ArmConfigurationPath(
                        arm, reference_path.reshape(-1)
                    )
                # Task-specific repair runs can target only the defective
                # semantic segment (e.g. stack_wine waypoint3 transport).
                # Keep every unaffected waypoint on ordinary straight-line
                # IK and never reintroduce an OMPL/RRT fallback.
                linear_path_calls += 1
                return arm.get_linear_path(
                    point._waypoint.get_position(),
                    quaternion=point._waypoint.get_quaternion(),
                    ignore_collisions=(
                        point._ignore_collisions or ignore_collisions
                    ),
                )
            start_joints = np.asarray(
                arm.get_joint_positions(), dtype=np.float64
            )
            target_position = np.asarray(
                point._waypoint.get_position(), dtype=np.float64
            )
            target_quaternion = np.asarray(
                point._waypoint.get_quaternion(), dtype=np.float64
            )

            # RLBench calls get_path(ignore_collisions=True) only to verify
            # that the waypoint chain has reachable endpoints before running
            # the demo. Do not perform the expensive curve search twice.
            if ignore_collisions:
                try:
                    endpoint_configs = arm.solve_ik_via_sampling(
                        target_position,
                        quaternion=target_quaternion,
                        ignore_collisions=True,
                        trials=400,
                        max_configs=30,
                    )
                except ConfigurationError as error:
                    raise ConfigurationPathError(
                        "Cartesian-detour endpoint has no IK solution"
                    ) from error
                endpoint_q = min(
                    np.asarray(endpoint_configs, dtype=np.float64),
                    key=lambda candidate: float(
                        np.linalg.norm(candidate - start_joints)
                        + 0.5
                        * np.linalg.norm(
                            candidate[-3:] - start_joints[-3:]
                        )
                    ),
                )
                return ArmConfigurationPath(arm, endpoint_q.reshape(-1))

            start_position = np.asarray(
                arm.get_tip().get_position(), dtype=np.float64
            )
            start_quaternion = np.asarray(
                arm.get_tip().get_quaternion(), dtype=np.float64
            )
            start_quaternion /= np.linalg.norm(start_quaternion)
            target_quaternion /= np.linalg.norm(target_quaternion)
            if float(np.dot(start_quaternion, target_quaternion)) < 0.0:
                target_quaternion *= -1.0
            slerp = Slerp(
                [0.0, 1.0],
                Rotation.from_quat(
                    np.stack((start_quaternion, target_quaternion))
                ),
            )
            segment_count = int(args.segmented_linear_segments)
            fractions = np.linspace(
                0.0, 1.0, segment_count + 1, dtype=np.float64
            )[1:]
            segment_quaternions = slerp(fractions).as_quat()
            direct_vector = target_position - start_position
            direct_distance = float(np.linalg.norm(direct_vector))
            if direct_distance > 1e-9:
                direct_unit = direct_vector / direct_distance
            else:
                direct_unit = np.asarray([1.0, 0.0, 0.0])
            world_z = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
            lateral = np.cross(world_z, direct_unit)
            if float(np.linalg.norm(lateral)) < 1e-8:
                lateral = np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
            lateral /= np.linalg.norm(lateral)
            normal = np.cross(direct_unit, lateral)
            normal /= np.linalg.norm(normal)
            targeted_carried_transport = bool(
                args.segmented_linear_waypoints is not None
                and waypoint_name in args.segmented_linear_waypoints
                and (
                    point._robot.gripper.get_grasped_objects()
                    or (
                        waypoint_name == "waypoint0"
                        and args.phone_reference_artifact is not None
                    )
                )
            )
            if (
                targeted_carried_transport
                and args.cartesian_carried_direct_joint_first
                and waypoint_name == "waypoint3"
            ):
                held_objects = point._robot.gripper.get_grasped_objects()
                held_states = []
                direct_joint_candidates = []
                direct_rotation = float(
                    2.0
                    * np.arccos(
                        np.clip(
                            abs(
                                float(
                                    np.dot(
                                        start_quaternion,
                                        target_quaternion,
                                    )
                                )
                            ),
                            -1.0,
                            1.0,
                        )
                    )
                )
                try:
                    for held_object in held_objects:
                        was_collidable = bool(held_object.is_collidable())
                        held_states.append((held_object, was_collidable))
                        if was_collidable:
                            held_object.set_collidable(False)
                    try:
                        endpoint_configs = arm.solve_ik_via_sampling(
                            target_position,
                            quaternion=target_quaternion,
                            ignore_collisions=False,
                            trials=1600,
                            max_configs=120,
                            distance_threshold=0.8,
                            max_time_ms=100,
                        )
                    except ConfigurationError:
                        endpoint_configs = []
                    for candidate_index, endpoint_config in enumerate(
                        np.asarray(endpoint_configs, dtype=np.float64).reshape(
                            -1, joint_count
                        )
                    ):
                        joint_delta = endpoint_config - start_joints
                        max_joint_delta = float(
                            np.max(np.abs(joint_delta), initial=0.0)
                        )
                        if max_joint_delta > float(np.pi):
                            continue
                        step_count = max(
                            2,
                            int(math.ceil(max_joint_delta / 0.015)),
                        )
                        fractions = np.linspace(
                            0.0, 1.0, step_count + 1, dtype=np.float64
                        )[1:]
                        configurations = (
                            start_joints[None]
                            + fractions[:, None] * joint_delta[None]
                        )
                        positions = [start_position.copy()]
                        quaternions = [start_quaternion.copy()]
                        collision = False
                        for configuration in configurations:
                            arm.set_joint_positions(
                                configuration.tolist(), disable_dynamics=True
                            )
                            if arm.check_arm_collision():
                                collision = True
                                break
                            positions.append(
                                np.asarray(
                                    arm.get_tip().get_position(),
                                    dtype=np.float64,
                                )
                            )
                            quaternion = np.asarray(
                                arm.get_tip().get_quaternion(),
                                dtype=np.float64,
                            )
                            quaternion /= np.linalg.norm(quaternion)
                            quaternions.append(quaternion)
                        arm.set_joint_positions(
                            start_joints.tolist(), disable_dynamics=True
                        )
                        if collision or len(positions) != len(configurations) + 1:
                            continue
                        positions = np.asarray(positions, dtype=np.float64)
                        rotation_length = 0.0
                        for previous_quaternion, quaternion in zip(
                            quaternions[:-1], quaternions[1:], strict=True
                        ):
                            rotation_length += float(
                                2.0
                                * np.arccos(
                                    np.clip(
                                        abs(
                                            float(
                                                np.dot(
                                                    previous_quaternion,
                                                    quaternion,
                                                )
                                            )
                                        ),
                                        -1.0,
                                        1.0,
                                    )
                                )
                            )
                        path_length = float(
                            np.linalg.norm(
                                np.diff(positions, axis=0), axis=1
                            ).sum()
                        )
                        chord = target_position - start_position
                        direct_distance = float(np.linalg.norm(chord))
                        if direct_distance > 1e-9:
                            chord_unit = chord / direct_distance
                            progress = (
                                positions - start_position
                            ) @ chord_unit
                            closest = (
                                start_position
                                + progress[:, None] * chord_unit
                            )
                            lateral_deviation = float(
                                np.linalg.norm(
                                    positions - closest, axis=1
                                ).max(initial=0.0)
                            )
                            progress_backtrack = float(
                                np.maximum(0.0, -np.diff(progress)).sum()
                            )
                        else:
                            lateral_deviation = path_length
                            progress_backtrack = path_length
                        rise = float(
                            positions[:, 2].max(initial=start_position[2])
                            - max(start_position[2], target_position[2])
                        )
                        detour_ratio = float(
                            path_length / max(direct_distance, 1e-9)
                        )
                        rotation_excess = float(
                            max(0.0, rotation_length - direct_rotation)
                        )
                        rejected = bool(
                            # A redundant Panda joint chord can exceed the EEF
                            # endpoint geodesic while still remaining monotonic
                            # and well below a full turn. Reject actual loops,
                            # but do not force a high Cartesian lift merely to
                            # remove a kinematically unavoidable wrist sweep.
                            rotation_excess > 2.5
                            or rotation_length > 1.5 * np.pi
                            or detour_ratio > 2.0
                            or lateral_deviation > 0.12
                            or progress_backtrack
                            > float(
                                args.cartesian_carried_direct_max_backtrack_m
                            )
                            or rise + 1e-9
                            < float(
                                args.cartesian_carried_direct_min_rise_m
                            )
                            or rise > 0.065
                        )
                        metrics = {
                            "candidate": int(candidate_index),
                            "points": int(len(configurations)),
                            "path_length_m": path_length,
                            "direct_m": direct_distance,
                            "detour_ratio": detour_ratio,
                            "lateral_deviation_m": lateral_deviation,
                            "progress_backtrack_m": progress_backtrack,
                            "rise_m": rise,
                            "direct_rotation_rad": direct_rotation,
                            "rotation_rad": rotation_length,
                            "rotation_excess_rad": rotation_excess,
                            "joint_travel_rad": float(
                                np.abs(joint_delta).sum()
                            ),
                            "max_joint_delta_rad": max_joint_delta,
                            "rejected": rejected,
                        }
                        print(
                            "[cartesian-carried-direct-joint-candidate] waypoint="
                            + waypoint_name
                            + " metrics="
                            + json.dumps(metrics, sort_keys=True),
                            flush=True,
                        )
                        if not rejected:
                            score = float(
                                5.0 * rotation_length
                                + 3.0 * path_length
                                + 8.0 * lateral_deviation
                                + 10.0 * progress_backtrack
                                + 4.0 * rise
                                + 0.05 * np.abs(joint_delta).sum()
                            )
                            direct_joint_candidates.append(
                                (score, configurations.copy(), metrics)
                            )
                finally:
                    for held_object, was_collidable in held_states:
                        held_object.set_collidable(was_collidable)
                    arm.set_joint_positions(
                        start_joints.tolist(), disable_dynamics=True
                    )
                if direct_joint_candidates:
                    selected_direct_joint = min(
                        direct_joint_candidates, key=lambda item: item[0]
                    )
                    print(
                        "[cartesian-carried-direct-joint-selected] waypoint="
                        + waypoint_name
                        + " metrics="
                        + json.dumps(
                            selected_direct_joint[2], sort_keys=True
                        ),
                        flush=True,
                    )
                    return TrackingArmConfigurationPath(
                        arm,
                        selected_direct_joint[1],
                        robot=point._robot,
                        waypoint_name=waypoint_name,
                    )
                print(
                    "[cartesian-carried-direct-joint-unavailable] waypoint="
                    + waypoint_name,
                    flush=True,
                )
            carried_clearance_candidates = tuple(
                clearance
                for clearance in (
                    0.04, 0.06, 0.08, 0.10, 0.12, 0.15, 0.20
                )
                if clearance + 1e-9
                >= float(args.cartesian_carried_min_clearance_m)
            )
            if not carried_clearance_candidates:
                raise ConfigurationPathError(
                    "No carried-object clearance candidate satisfies "
                    "--cartesian-carried-min-clearance-m"
                )
            if targeted_carried_transport:
                reference_deform_rise_m = float(
                    args.cartesian_reference_deform_rise_m
                )
                reference_deform_joints = (
                    cartesian_reference_waypoint_paths.get(waypoint_name)
                )
                reference_deform_poses = (
                    cartesian_reference_waypoint_poses.get(waypoint_name)
                )
                if (
                    reference_deform_rise_m >= 0.0
                    and reference_deform_joints is not None
                    and reference_deform_poses is not None
                    and len(reference_deform_joints) >= 2
                ):
                    sample_count = min(100, len(reference_deform_joints))
                    reference_indices = np.rint(
                        np.linspace(
                            0,
                            len(reference_deform_joints) - 1,
                            sample_count,
                        )
                    ).astype(np.int64)
                    progress_values = np.linspace(
                        0.0, 1.0, sample_count + 1
                    )[1:]
                    desired_quaternions = slerp(
                        progress_values
                    ).as_quat()
                    if args.cartesian_reference_deform_roll_symmetry:
                        reference_sample_quaternions = []
                        for reference_index in reference_indices:
                            arm.set_joint_positions(
                                np.asarray(
                                    reference_deform_joints[
                                        reference_index
                                    ],
                                    dtype=np.float64,
                                ).tolist(),
                                disable_dynamics=True,
                            )
                            reference_sample_quaternions.append(
                                np.asarray(
                                    arm.get_tip().get_quaternion(),
                                    dtype=np.float64,
                                )
                            )
                        arm.set_joint_positions(
                            start_joints.tolist(), disable_dynamics=True
                        )
                        reference_sample_quaternions = np.asarray(
                            reference_sample_quaternions,
                            dtype=np.float64,
                        )
                        reference_sample_quaternions /= np.linalg.norm(
                            reference_sample_quaternions,
                            axis=1,
                            keepdims=True,
                        )
                        swapped_sample_quaternions = (
                            Rotation.from_quat(reference_sample_quaternions)
                            * Rotation.from_rotvec(
                                np.tile(
                                    np.asarray(
                                        [0.0, 0.0, math.pi],
                                        dtype=np.float64,
                                    ),
                                    (sample_count, 1),
                                )
                            )
                        ).as_quat()
                        orientation_options = np.stack(
                            (
                                reference_sample_quaternions,
                                swapped_sample_quaternions,
                            ),
                            axis=1,
                        )
                        costs = np.full(
                            (sample_count, 2), np.inf, dtype=np.float64
                        )
                        parents = np.full(
                            (sample_count, 2), -1, dtype=np.int64
                        )
                        costs[0, 0] = float(
                                2.0
                                * np.arccos(
                                    np.clip(
                                        abs(
                                            float(
                                                np.dot(
                                                    start_quaternion,
                                                    orientation_options[
                                                        0, 0
                                                    ],
                                                )
                                            )
                                        ),
                                        -1.0,
                                        1.0,
                                    )
                                )
                            )
                        costs[0, 1] = np.inf
                        for index in range(1, sample_count):
                            for state in range(2):
                                transitions = []
                                for previous_state in range(2):
                                    angle = float(
                                        2.0
                                        * np.arccos(
                                            np.clip(
                                                abs(
                                                    float(
                                                        np.dot(
                                                            orientation_options[
                                                                index - 1,
                                                                previous_state,
                                                            ],
                                                            orientation_options[
                                                                index, state
                                                            ],
                                                        )
                                                    )
                                                ),
                                                -1.0,
                                                1.0,
                                            )
                                        )
                                    )
                                    transitions.append(
                                        costs[index - 1, previous_state]
                                        + angle
                                    )
                                parents[index, state] = int(
                                    np.argmin(transitions)
                                )
                                costs[index, state] = float(
                                    min(transitions)
                                )
                        selected_states = np.zeros(
                            sample_count, dtype=np.int64
                        )
                        selected_states[-1] = 0
                        for index in range(sample_count - 1, 0, -1):
                            selected_states[index - 1] = parents[
                                index, selected_states[index]
                            ]
                        desired_quaternions = orientation_options[
                            np.arange(sample_count), selected_states
                        ]
                    deformed_configurations = []
                    previous_deformed_joints = start_joints.copy()
                    deform_valid = True
                    deform_held_states = []
                    try:
                        for held_object in point._robot.gripper.get_grasped_objects():
                            was_collidable = bool(held_object.is_collidable())
                            deform_held_states.append(
                                (held_object, was_collidable)
                            )
                            if was_collidable:
                                held_object.set_collidable(False)
                        for output_index, (
                            reference_index,
                            progress,
                            desired_quaternion,
                        ) in enumerate(
                            zip(
                                reference_indices,
                                progress_values,
                                desired_quaternions,
                            )
                        ):
                            reference_joints = np.asarray(
                                reference_deform_joints[reference_index],
                                dtype=np.float64,
                            )
                            arm.set_joint_positions(
                                reference_joints.tolist(),
                                disable_dynamics=True,
                            )
                            reference_position = np.asarray(
                                arm.get_tip().get_position(),
                                dtype=np.float64,
                            )
                            reference_quaternion = np.asarray(
                                arm.get_tip().get_quaternion(),
                                dtype=np.float64,
                            )
                            reference_quaternion /= np.linalg.norm(
                                reference_quaternion
                            )
                            if (
                                args.cartesian_reference_deform_roll_symmetry
                                and output_index > 0
                            ):
                                arm.set_joint_positions(
                                    previous_deformed_joints.tolist(),
                                    disable_dynamics=True,
                                )
                                reference_position = np.asarray(
                                    arm.get_tip().get_position(),
                                    dtype=np.float64,
                                )
                                reference_quaternion = np.asarray(
                                    arm.get_tip().get_quaternion(),
                                    dtype=np.float64,
                                )
                                reference_quaternion /= np.linalg.norm(
                                    reference_quaternion
                                )
                                reference_joints = (
                                    previous_deformed_joints.copy()
                                )
                            if args.cartesian_reference_deform_preserve_orientation:
                                desired_quaternion = reference_quaternion.copy()
                            else:
                                orientation_blend = float(
                                    args.cartesian_reference_deform_orientation_blend
                                )
                                if not 0.0 <= orientation_blend <= 1.0:
                                    raise ValueError(
                                        "--cartesian-reference-deform-orientation-blend "
                                        "must be in [0, 1]"
                                    )
                                orientation_blend_slerp = Slerp(
                                    [0.0, 1.0],
                                    Rotation.from_quat(
                                        np.stack(
                                            (
                                                reference_quaternion,
                                                desired_quaternion,
                                            )
                                        )
                                    ),
                                )
                                desired_quaternion = orientation_blend_slerp(
                                    [orientation_blend]
                                ).as_quat()[0]
                            straight_position = (
                                (1.0 - progress) * start_position
                                + progress * target_position
                            )
                            xyz_blend = float(
                                args.cartesian_reference_deform_xyz_blend
                            )
                            if not 0.0 <= xyz_blend <= 1.0:
                                raise ValueError(
                                    "--cartesian-reference-deform-xyz-blend "
                                    "must be in [0, 1]"
                                )
                            desired_position = (
                                (1.0 - xyz_blend) * straight_position
                                + xyz_blend * reference_position
                            )
                            desired_position = desired_position.copy()
                            desired_position[2] += (
                                reference_deform_rise_m
                                * math.sin(math.pi * progress)
                            )
                            homotopy_slerp = Slerp(
                                [0.0, 1.0],
                                Rotation.from_quat(
                                    np.stack(
                                        (
                                            reference_quaternion,
                                            desired_quaternion,
                                        )
                                    )
                                ),
                            )
                            homotopy_joints = reference_joints.copy()
                            for alpha in np.linspace(0.0, 1.0, 7)[1:]:
                                arm.set_joint_positions(
                                    homotopy_joints.tolist(),
                                    disable_dynamics=True,
                                )
                                homotopy_position = (
                                    (1.0 - alpha) * reference_position
                                    + alpha * desired_position
                                )
                                homotopy_quaternion = homotopy_slerp(
                                    [alpha]
                                ).as_quat()[0]
                                try:
                                    homotopy_joints = np.asarray(
                                        arm.solve_ik_via_jacobian(
                                            homotopy_position,
                                            quaternion=homotopy_quaternion,
                                        ),
                                        dtype=np.float64,
                                    )
                                except (IKError, ConfigurationError):
                                    deform_valid = False
                                    break
                            if not deform_valid:
                                break
                            if float(
                                np.max(
                                    np.abs(
                                        homotopy_joints
                                        - previous_deformed_joints
                                    )
                                )
                            ) > 0.45:
                                deform_valid = False
                                break
                            arm.set_joint_positions(
                                homotopy_joints.tolist(),
                                disable_dynamics=True,
                            )
                            if arm.check_arm_collision():
                                deform_valid = False
                                break
                            deformed_configurations.append(
                                homotopy_joints.copy()
                            )
                            previous_deformed_joints = homotopy_joints.copy()
                    finally:
                        for held_object, was_collidable in deform_held_states:
                            held_object.set_collidable(was_collidable)
                        arm.set_joint_positions(
                            start_joints.tolist(), disable_dynamics=True
                        )
                    if (
                        deform_valid
                        and len(deformed_configurations) == sample_count
                    ):
                        print(
                            "[cartesian-reference-homotopy-selected] waypoint="
                            + waypoint_name
                            + " points="
                            + str(sample_count)
                            + " rise_m="
                            + format(reference_deform_rise_m, ".6f")
                            + " xyz_blend="
                            + format(
                                float(
                                    args.cartesian_reference_deform_xyz_blend
                                ),
                                ".6f",
                            )
                            + " orientation_blend="
                            + format(
                                float(
                                    args.cartesian_reference_deform_orientation_blend
                                ),
                                ".6f",
                            )
                            + " preserve_orientation="
                            + str(
                                bool(
                                    args.cartesian_reference_deform_preserve_orientation
                                )
                            )
                            + " roll_symmetry="
                            + str(
                                bool(
                                    args.cartesian_reference_deform_roll_symmetry
                                )
                            ),
                            flush=True,
                        )
                        return TrackingArmConfigurationPath(
                            arm,
                            np.asarray(
                                deformed_configurations,
                                dtype=np.float64,
                            ),
                            robot=point._robot,
                            waypoint_name=waypoint_name,
                        )
                    print(
                        "[cartesian-reference-homotopy-unavailable] waypoint="
                        + waypoint_name
                        + " solved="
                        + str(len(deformed_configurations))
                        + "/"
                        + str(sample_count),
                        flush=True,
                    )
                    if args.cartesian_reference_deform_required:
                        raise ConfigurationPathError(
                            "Required Cartesian reference homotopy is infeasible "
                            "for " + waypoint_name
                        )
                # A rack insertion is not well represented by one joint-space
                # chord. Search a small family of explicit Cartesian paths:
                # short vertical clearance, straight horizontal transfer, then
                # straight descent. Quaternion progress remains shortest-arc
                # and monotonic across the three legs.
                staged_candidates = []
                held_states = []
                try:
                    for held_object in point._robot.gripper.get_grasped_objects():
                        was_collidable = bool(held_object.is_collidable())
                        held_states.append((held_object, was_collidable))
                        if was_collidable:
                            held_object.set_collidable(False)
                    for clearance_m in carried_clearance_candidates:
                        clearance_z = max(
                            start_position[2], target_position[2]
                        ) + clearance_m
                        lift_position = start_position.copy()
                        lift_position[2] = clearance_z
                        overhead_position = target_position.copy()
                        overhead_position[2] = clearance_z
                        for lift_rotation_progress, overhead_rotation_progress in (
                            (0.0, 0.5),
                            (0.0, 0.75),
                            (0.0, 1.0),
                            (0.25, 0.75),
                            (0.5, 1.0),
                        ):
                            arm.set_joint_positions(
                                start_joints.tolist(), disable_dynamics=True
                            )
                            try:
                                leg1 = arm.get_linear_path(
                                    lift_position,
                                    quaternion=slerp(
                                        [lift_rotation_progress]
                                    ).as_quat()[0],
                                    steps=80,
                                    ignore_collisions=True,
                                )
                                q1 = np.asarray(
                                    leg1._path_points, dtype=np.float64
                                ).reshape(-1, joint_count)
                                leg1.set_to_end(disable_dynamics=True)
                                leg2 = arm.get_linear_path(
                                    overhead_position,
                                    quaternion=slerp(
                                        [overhead_rotation_progress]
                                    ).as_quat()[0],
                                    steps=140,
                                    ignore_collisions=True,
                                )
                                q2 = np.asarray(
                                    leg2._path_points, dtype=np.float64
                                ).reshape(-1, joint_count)
                                leg2.set_to_end(disable_dynamics=True)
                                leg3 = arm.get_linear_path(
                                    target_position,
                                    quaternion=target_quaternion,
                                    steps=100,
                                    ignore_collisions=True,
                                )
                                q3 = np.asarray(
                                    leg3._path_points, dtype=np.float64
                                ).reshape(-1, joint_count)
                                configurations = np.vstack(
                                    (q1, q2[1:], q3[1:])
                                )
                                sequence = np.vstack(
                                    (start_joints[None], configurations)
                                )
                                joint_steps = np.abs(
                                    np.diff(sequence, axis=0)
                                )
                                joint_travel = joint_steps.sum(axis=0)
                                if (
                                    float(joint_steps.max(initial=0.0)) > 0.25
                                    or float(joint_travel.max(initial=0.0))
                                    > 1.5 * math.pi
                                    or float(joint_travel[-3:].sum()) > 7.0
                                ):
                                    raise ConfigurationPathError(
                                        "carried staged Cartesian joint guards"
                                    )
                                xyz_length = float(
                                    np.linalg.norm(
                                        lift_position - start_position
                                    )
                                    + np.linalg.norm(
                                        overhead_position - lift_position
                                    )
                                    + np.linalg.norm(
                                        target_position - overhead_position
                                    )
                                )
                                metrics = {
                                    "clearance_m": float(clearance_m),
                                    "lift_rotation_progress": float(
                                        lift_rotation_progress
                                    ),
                                    "overhead_rotation_progress": float(
                                        overhead_rotation_progress
                                    ),
                                    "path_length_m": xyz_length,
                                    "max_joint_travel_rad": float(
                                        joint_travel.max(initial=0.0)
                                    ),
                                    "wrist_travel_rad": float(
                                        joint_travel[-3:].sum()
                                    ),
                                    "points": int(len(configurations)),
                                }
                                score = float(
                                    xyz_length
                                    + 2.0 * clearance_m
                                    + 0.02 * joint_travel.sum()
                                    + 0.05 * joint_travel[-3:].sum()
                                )
                                staged_candidates.append(
                                    (score, configurations.copy(), metrics)
                                )
                            except (
                                ConfigurationError,
                                ConfigurationPathError,
                                IKError,
                            ):
                                pass
                            finally:
                                arm.set_joint_positions(
                                    start_joints.tolist(),
                                    disable_dynamics=True,
                                )
                        if staged_candidates:
                            break
                finally:
                    for held_object, was_collidable in held_states:
                        held_object.set_collidable(was_collidable)
                    arm.set_joint_positions(
                        start_joints.tolist(), disable_dynamics=True
                    )
                if staged_candidates:
                    selected_staged = min(
                        staged_candidates, key=lambda item: item[0]
                    )
                    print(
                        "[cartesian-carried-staged-selected] waypoint="
                        + waypoint_name
                        + " metrics="
                        + json.dumps(selected_staged[2], sort_keys=True),
                        flush=True,
                    )
                    if tasks == ["stack_wine"]:
                        # StackWine uses RLBench's registered-object grasp
                        # attachment. Execute the already checked Cartesian IK
                        # samples exactly; Reflexxes re-interpolation can choose
                        # a different redundant-arm branch and turn a bounded
                        # planned motion into a 300-500 degree wrist loop.
                        return TrackingArmConfigurationPath(
                            arm,
                            selected_staged[1],
                            robot=point._robot,
                            waypoint_name=waypoint_name,
                        )
                    return DenseJointServoPath(
                        arm,
                        selected_staged[1],
                        waypoint_name=waypoint_name,
                    )
                # The legacy whole-leg IK generator can reject a valid path
                # near a singularity. Fall back to local Jacobian continuation
                # on the same lift/across/down geometry, always seeding each
                # tiny solve from the preceding configuration.
                jacobian_staged_candidates = []
                for clearance_m in carried_clearance_candidates:
                    clearance_z = max(
                        start_position[2], target_position[2]
                    ) + clearance_m
                    lift_position = start_position.copy()
                    lift_position[2] = clearance_z
                    overhead_position = target_position.copy()
                    overhead_position[2] = clearance_z
                    for lift_progress, overhead_progress in (
                        (0.0, 0.5),
                        (0.0, 0.75),
                        (0.0, 1.0),
                        (0.25, 0.75),
                        (0.5, 1.0),
                    ):
                        node_positions = (
                            start_position,
                            lift_position,
                            overhead_position,
                            target_position,
                        )
                        node_rotation_progress = (
                            0.0,
                            float(lift_progress),
                            float(overhead_progress),
                            1.0,
                        )
                        configurations = []
                        arm.set_joint_positions(
                            start_joints.tolist(), disable_dynamics=True
                        )
                        try:
                            for leg_index in range(3):
                                for fraction in np.linspace(
                                    0.0, 1.0, 41, dtype=np.float64
                                )[1:]:
                                    position = (
                                        (1.0 - float(fraction))
                                        * node_positions[leg_index]
                                        + float(fraction)
                                        * node_positions[leg_index + 1]
                                    )
                                    rotation_progress = (
                                        (1.0 - float(fraction))
                                        * node_rotation_progress[leg_index]
                                        + float(fraction)
                                        * node_rotation_progress[leg_index + 1]
                                    )
                                    quaternion = slerp(
                                        [rotation_progress]
                                    ).as_quat()[0]
                                    previous = np.asarray(
                                        arm.get_joint_positions(),
                                        dtype=np.float64,
                                    )
                                    joints = np.asarray(
                                        arm.solve_ik_via_jacobian(
                                            position,
                                            quaternion=quaternion,
                                        ),
                                        dtype=np.float64,
                                    )
                                    if float(
                                        np.max(np.abs(joints - previous))
                                    ) > 0.20:
                                        raise ConfigurationPathError(
                                            "Jacobian staged IK branch jump"
                                        )
                                    arm.set_joint_positions(
                                        joints.tolist(), disable_dynamics=True
                                    )
                                    configurations.append(joints.copy())
                            configurations = np.asarray(
                                configurations, dtype=np.float64
                            )
                            sequence = np.vstack(
                                (start_joints[None], configurations)
                            )
                            steps = np.abs(np.diff(sequence, axis=0))
                            travel = steps.sum(axis=0)
                            if (
                                float(travel.max(initial=0.0))
                                > 1.5 * math.pi
                                or float(travel[-3:].sum()) > 7.0
                            ):
                                raise ConfigurationPathError(
                                    "Jacobian staged IK travel guards"
                                )
                            xyz_length = float(
                                np.linalg.norm(
                                    lift_position - start_position
                                )
                                + np.linalg.norm(
                                    overhead_position - lift_position
                                )
                                + np.linalg.norm(
                                    target_position - overhead_position
                                )
                            )
                            metrics = {
                                "solver": "sequential_jacobian",
                                "clearance_m": float(clearance_m),
                                "lift_rotation_progress": float(lift_progress),
                                "overhead_rotation_progress": float(
                                    overhead_progress
                                ),
                                "path_length_m": xyz_length,
                                "max_joint_travel_rad": float(
                                    travel.max(initial=0.0)
                                ),
                                "wrist_travel_rad": float(
                                    travel[-3:].sum()
                                ),
                                "points": int(len(configurations)),
                            }
                            score = float(
                                xyz_length
                                + 2.0 * clearance_m
                                + 0.02 * travel.sum()
                                + 0.05 * travel[-3:].sum()
                            )
                            jacobian_staged_candidates.append(
                                (score, configurations.copy(), metrics)
                            )
                        except (
                            ConfigurationError,
                            ConfigurationPathError,
                            IKError,
                        ):
                            pass
                        finally:
                            arm.set_joint_positions(
                                start_joints.tolist(), disable_dynamics=True
                            )
                    if jacobian_staged_candidates:
                        break
                if jacobian_staged_candidates:
                    selected_staged = min(
                        jacobian_staged_candidates, key=lambda item: item[0]
                    )
                    print(
                        "[cartesian-carried-jacobian-staged-selected] waypoint="
                        + waypoint_name
                        + " metrics="
                        + json.dumps(selected_staged[2], sort_keys=True),
                        flush=True,
                    )
                    return DenseJointServoPath(
                        arm,
                        selected_staged[1],
                        waypoint_name=waypoint_name,
                    )
                sampled_cartesian_candidates = []
                sampled_cartesian_held_states = []
                try:
                    for held_object in point._robot.gripper.get_grasped_objects():
                        was_collidable = bool(held_object.is_collidable())
                        sampled_cartesian_held_states.append(
                            (held_object, was_collidable)
                        )
                        if was_collidable:
                            held_object.set_collidable(False)
                    for clearance_m in carried_clearance_candidates:
                        clearance_z = max(
                            start_position[2], target_position[2]
                        ) + clearance_m
                        lift_position = start_position.copy()
                        lift_position[2] = clearance_z
                        overhead_position = target_position.copy()
                        overhead_position[2] = clearance_z
                        for schedule_name, rotation_progress_fn in (
                            ("coupled", lambda fraction: fraction),
                            (
                                "rotation_late",
                                lambda fraction: max(
                                    0.0, 1.5 * fraction - 0.5
                                ),
                            ),
                            (
                                "rotation_early",
                                lambda fraction: min(
                                    1.0, 1.5 * fraction
                                ),
                            ),
                        ):
                            targets = []
                            for fraction in np.linspace(0.0, 1.0, 9)[1:]:
                                targets.append(
                                    (
                                        (1.0 - fraction) * start_position
                                        + fraction * lift_position,
                                        start_quaternion,
                                    )
                                )
                            for fraction in np.linspace(0.0, 1.0, 25)[1:]:
                                targets.append(
                                    (
                                        (1.0 - fraction) * lift_position
                                        + fraction * overhead_position,
                                        slerp(
                                            [rotation_progress_fn(fraction)]
                                        ).as_quat()[0],
                                    )
                                )
                            for fraction in np.linspace(0.0, 1.0, 9)[1:]:
                                targets.append(
                                    (
                                        (1.0 - fraction) * overhead_position
                                        + fraction * target_position,
                                        target_quaternion,
                                    )
                                )
                            arm.set_joint_positions(
                                start_joints.tolist(), disable_dynamics=True
                            )
                            previous_joints = start_joints.copy()
                            configurations = []
                            valid = True
                            for position, quaternion in targets:
                                try:
                                    goals = arm.solve_ik_via_sampling(
                                        np.asarray(position, dtype=np.float64),
                                        quaternion=np.asarray(
                                            quaternion, dtype=np.float64
                                        ),
                                        ignore_collisions=False,
                                        trials=900,
                                        max_configs=24,
                                        distance_threshold=0.45,
                                        max_time_ms=15,
                                    )
                                except ConfigurationError:
                                    valid = False
                                    break
                                goals = np.asarray(goals, dtype=np.float64)
                                if len(goals) == 0:
                                    valid = False
                                    break
                                goal = min(
                                    goals,
                                    key=lambda candidate: float(
                                        np.linalg.norm(
                                            candidate - previous_joints
                                        )
                                        + 0.5
                                        * np.linalg.norm(
                                            candidate[-3:]
                                            - previous_joints[-3:]
                                        )
                                    ),
                                )
                                if float(
                                    np.max(np.abs(goal - previous_joints))
                                ) > 0.35:
                                    valid = False
                                    break
                                arm.set_joint_positions(
                                    goal.tolist(), disable_dynamics=True
                                )
                                if arm.check_arm_collision():
                                    valid = False
                                    break
                                configurations.append(goal.copy())
                                previous_joints = goal.copy()
                            arm.set_joint_positions(
                                start_joints.tolist(), disable_dynamics=True
                            )
                            if not valid or len(configurations) != len(targets):
                                continue
                            configurations = np.asarray(
                                configurations, dtype=np.float64
                            )
                            sequence = np.vstack(
                                (start_joints[None], configurations)
                            )
                            joint_travel = np.abs(
                                np.diff(sequence, axis=0)
                            ).sum(axis=0)
                            if (
                                float(joint_travel.max(initial=0.0))
                                > 1.5 * math.pi
                                or float(joint_travel[-3:].sum()) > 7.0
                            ):
                                continue
                            xyz_length = float(
                                np.linalg.norm(
                                    lift_position - start_position
                                )
                                + np.linalg.norm(
                                    overhead_position - lift_position
                                )
                                + np.linalg.norm(
                                    target_position - overhead_position
                                )
                            )
                            metrics = {
                                "solver": "sequential_sampled_cartesian_ik",
                                "clearance_m": float(clearance_m),
                                "rotation_schedule": schedule_name,
                                "path_length_m": xyz_length,
                                "rotation_rad": float(
                                    (
                                        Rotation.from_quat(
                                            start_quaternion
                                        ).inv()
                                        * Rotation.from_quat(
                                            target_quaternion
                                        )
                                    ).magnitude()
                                ),
                                "max_joint_travel_rad": float(
                                    joint_travel.max(initial=0.0)
                                ),
                                "wrist_travel_rad": float(
                                    joint_travel[-3:].sum()
                                ),
                                "points": int(len(configurations)),
                            }
                            score = float(
                                xyz_length
                                + 2.0 * clearance_m
                                + 0.02 * joint_travel.sum()
                                + 0.05 * joint_travel[-3:].sum()
                            )
                            sampled_cartesian_candidates.append(
                                (score, configurations.copy(), metrics)
                            )
                        if sampled_cartesian_candidates:
                            break
                finally:
                    for held_object, was_collidable in sampled_cartesian_held_states:
                        held_object.set_collidable(was_collidable)
                    arm.set_joint_positions(
                        start_joints.tolist(), disable_dynamics=True
                    )
                if sampled_cartesian_candidates:
                    selected_staged = min(
                        sampled_cartesian_candidates,
                        key=lambda item: item[0],
                    )
                    print(
                        "[cartesian-carried-sampled-cartesian-selected] waypoint="
                        + waypoint_name
                        + " metrics="
                        + json.dumps(selected_staged[2], sort_keys=True),
                        flush=True,
                    )
                    return TrackingArmConfigurationPath(
                        arm,
                        selected_staged[1],
                        robot=point._robot,
                        waypoint_name=waypoint_name,
                    )
                sampled_staged_candidates = []

                def sampled_bounded_leg(
                    leg_start_joints,
                    leg_start_position,
                    leg_start_quaternion,
                    leg_target_position,
                    leg_target_quaternion,
                ):
                    arm.set_joint_positions(
                        leg_start_joints.tolist(), disable_dynamics=True
                    )
                    try:
                        goals = arm.solve_ik_via_sampling(
                            leg_target_position,
                            quaternion=leg_target_quaternion,
                            ignore_collisions=False,
                            trials=1800,
                            max_configs=24,
                            distance_threshold=0.65,
                            max_time_ms=20,
                        )
                    except ConfigurationError:
                        return None
                    leg_direct = float(
                        np.linalg.norm(
                            leg_target_position - leg_start_position
                        )
                    )
                    if leg_direct > 1e-9:
                        leg_unit = (
                            leg_target_position - leg_start_position
                        ) / leg_direct
                    else:
                        leg_unit = np.asarray([1.0, 0.0, 0.0])
                    choices = []
                    for goal in np.asarray(goals, dtype=np.float64):
                        delta = goal - leg_start_joints
                        if float(np.max(np.abs(delta))) > math.pi:
                            continue
                        count = max(
                            12,
                            int(
                                math.ceil(
                                    float(np.max(np.abs(delta))) / 0.020
                                )
                            ),
                        )
                        configs = (
                            leg_start_joints[None]
                            + np.linspace(0.0, 1.0, count + 1)[1:, None]
                            * delta[None]
                        )
                        xyz = [leg_start_position.copy()]
                        quats = [leg_start_quaternion.copy()]
                        collision = False
                        for config in configs:
                            arm.set_joint_positions(
                                config.tolist(), disable_dynamics=True
                            )
                            if arm.check_arm_collision():
                                collision = True
                                break
                            xyz.append(
                                np.asarray(
                                    arm.get_tip().get_position(),
                                    dtype=np.float64,
                                )
                            )
                            quats.append(
                                np.asarray(
                                    arm.get_tip().get_quaternion(),
                                    dtype=np.float64,
                                )
                            )
                        arm.set_joint_positions(
                            leg_start_joints.tolist(), disable_dynamics=True
                        )
                        if collision or len(xyz) != len(configs) + 1:
                            continue
                        xyz = np.asarray(xyz, dtype=np.float64)
                        quats = np.asarray(quats, dtype=np.float64)
                        quats /= np.linalg.norm(
                            quats, axis=1, keepdims=True
                        )
                        length = float(
                            np.linalg.norm(
                                np.diff(xyz, axis=0), axis=1
                            ).sum()
                        )
                        progress = (xyz - leg_start_position) @ leg_unit
                        closest = (
                            leg_start_position
                            + progress[:, None] * leg_unit
                        )
                        lateral_error = float(
                            np.linalg.norm(
                                xyz - closest, axis=1
                            ).max(initial=0.0)
                        )
                        backtrack = float(
                            np.maximum(0.0, -np.diff(progress)).sum()
                        )
                        rotation = float(
                            np.sum(
                                2.0
                                * np.arccos(
                                    np.clip(
                                        np.abs(
                                            np.sum(
                                                quats[:-1]
                                                * quats[1:],
                                                axis=1,
                                            )
                                        ),
                                        -1.0,
                                        1.0,
                                    )
                                )
                            )
                        )
                        endpoint_error = float(
                            np.linalg.norm(xyz[-1] - leg_target_position)
                        )
                        if (
                            endpoint_error > 0.004
                            or length > max(0.03, 2.0 * leg_direct)
                            or lateral_error > 0.10
                            or backtrack > 0.05
                            or rotation > math.pi
                        ):
                            continue
                        score = float(
                            length
                            + 2.0 * lateral_error
                            + backtrack
                            + 0.1 * rotation
                            + 0.02 * np.abs(delta).sum()
                        )
                        choices.append(
                            (
                                score,
                                configs.copy(),
                                goal.copy(),
                                xyz[-1].copy(),
                                quats[-1].copy(),
                            )
                        )
                    if not choices:
                        return None
                    return min(choices, key=lambda item: item[0])

                for clearance_m in carried_clearance_candidates:
                    clearance_z = max(
                        start_position[2], target_position[2]
                    ) + clearance_m
                    lift_position = start_position.copy()
                    lift_position[2] = clearance_z
                    overhead_position = target_position.copy()
                    overhead_position[2] = clearance_z
                    for lift_progress, overhead_progress in (
                        (0.0, 0.5),
                        (0.0, 1.0),
                        (0.5, 1.0),
                    ):
                        node_positions = (
                            lift_position,
                            overhead_position,
                            target_position,
                        )
                        node_quaternions = (
                            slerp([lift_progress]).as_quat()[0],
                            slerp([overhead_progress]).as_quat()[0],
                            target_quaternion,
                        )
                        leg_start_joints = start_joints.copy()
                        leg_start_position = start_position.copy()
                        leg_start_quaternion = start_quaternion.copy()
                        parts = []
                        leg_scores = 0.0
                        valid = True
                        for node_position, node_quaternion in zip(
                            node_positions, node_quaternions
                        ):
                            selected_leg = sampled_bounded_leg(
                                leg_start_joints,
                                leg_start_position,
                                leg_start_quaternion,
                                node_position,
                                node_quaternion,
                            )
                            if selected_leg is None:
                                valid = False
                                break
                            leg_scores += selected_leg[0]
                            parts.append(selected_leg[1])
                            leg_start_joints = selected_leg[2]
                            leg_start_position = selected_leg[3]
                            leg_start_quaternion = selected_leg[4]
                        arm.set_joint_positions(
                            start_joints.tolist(), disable_dynamics=True
                        )
                        if valid:
                            configurations = np.vstack(parts)
                            metrics = {
                                "solver": "sampled_ik_bounded_legs",
                                "clearance_m": float(clearance_m),
                                "lift_rotation_progress": float(lift_progress),
                                "overhead_rotation_progress": float(
                                    overhead_progress
                                ),
                                "points": int(len(configurations)),
                                "leg_score": float(leg_scores),
                            }
                            score = float(
                                leg_scores + 2.0 * clearance_m
                            )
                            sampled_staged_candidates.append(
                                (score, configurations, metrics)
                            )
                    if sampled_staged_candidates:
                        break
                arm.set_joint_positions(
                    start_joints.tolist(), disable_dynamics=True
                )
                if sampled_staged_candidates:
                    selected_staged = min(
                        sampled_staged_candidates, key=lambda item: item[0]
                    )
                    print(
                        "[cartesian-carried-sampled-staged-selected] waypoint="
                        + waypoint_name
                        + " metrics="
                        + json.dumps(selected_staged[2], sort_keys=True),
                        flush=True,
                    )
                    return DenseJointServoPath(
                        arm,
                        selected_staged[1],
                        waypoint_name=waypoint_name,
                    )
                # Some close-to-base wine scenes have valid endpoints but no
                # collision-free *single* joint chord across the entire rack.
                # Keep the same low lift/translate/descend Cartesian route,
                # but anchor the horizontal leg at four intermediate poses.
                # Every small leg is independently bounded and its IK solve is
                # seeded from the preceding solution, preventing a wrist/elbow
                # branch jump without invoking OMPL/RRT.
                sampled_subdivided_candidates = []
                for clearance_m in carried_clearance_candidates:
                    clearance_z = max(
                        start_position[2], target_position[2]
                    ) + clearance_m
                    lift_position = start_position.copy()
                    lift_position[2] = clearance_z
                    overhead_position = target_position.copy()
                    overhead_position[2] = clearance_z
                    for schedule_name, rotation_progress_fn in (
                        ("coupled", lambda fraction: fraction),
                        (
                            "rotation_late",
                            lambda fraction: max(
                                0.0, 1.5 * fraction - 0.5
                            ),
                        ),
                        (
                            "rotation_early",
                            lambda fraction: min(
                                1.0, 1.5 * fraction
                            ),
                        ),
                    ):
                        node_positions = [lift_position]
                        node_quaternions = [start_quaternion]
                        for horizontal_fraction in (0.25, 0.5, 0.75, 1.0):
                            node_positions.append(
                                (1.0 - horizontal_fraction) * lift_position
                                + horizontal_fraction * overhead_position
                            )
                            node_quaternions.append(
                                slerp(
                                    [
                                        rotation_progress_fn(
                                            horizontal_fraction
                                        )
                                    ]
                                ).as_quat()[0]
                            )
                        node_positions.append(target_position)
                        node_quaternions.append(target_quaternion)
                        leg_start_joints = start_joints.copy()
                        leg_start_position = start_position.copy()
                        leg_start_quaternion = start_quaternion.copy()
                        parts = []
                        leg_scores = 0.0
                        valid = True
                        for node_position, node_quaternion in zip(
                            node_positions, node_quaternions
                        ):
                            selected_leg = sampled_bounded_leg(
                                leg_start_joints,
                                leg_start_position,
                                leg_start_quaternion,
                                np.asarray(node_position, dtype=np.float64),
                                np.asarray(node_quaternion, dtype=np.float64),
                            )
                            if selected_leg is None:
                                valid = False
                                break
                            leg_scores += selected_leg[0]
                            parts.append(selected_leg[1])
                            leg_start_joints = selected_leg[2]
                            leg_start_position = selected_leg[3]
                            leg_start_quaternion = selected_leg[4]
                        arm.set_joint_positions(
                            start_joints.tolist(), disable_dynamics=True
                        )
                        if valid:
                            configurations = np.vstack(parts)
                            metrics = {
                                "solver": "sampled_ik_six_bounded_legs",
                                "clearance_m": float(clearance_m),
                                "rotation_schedule": schedule_name,
                                "points": int(len(configurations)),
                                "leg_score": float(leg_scores),
                            }
                            sampled_subdivided_candidates.append(
                                (
                                    float(leg_scores + 2.0 * clearance_m),
                                    configurations,
                                    metrics,
                                )
                            )
                    if sampled_subdivided_candidates:
                        break
                arm.set_joint_positions(
                    start_joints.tolist(), disable_dynamics=True
                )
                if sampled_subdivided_candidates:
                    selected_staged = min(
                        sampled_subdivided_candidates,
                        key=lambda item: item[0],
                    )
                    print(
                        "[cartesian-carried-sampled-subdivided-selected] waypoint="
                        + waypoint_name
                        + " metrics="
                        + json.dumps(selected_staged[2], sort_keys=True),
                        flush=True,
                    )
                    return DenseJointServoPath(
                        arm,
                        selected_staged[1],
                        waypoint_name=waypoint_name,
                    )
                print(
                    "[cartesian-carried-staged-unavailable] waypoint="
                    + waypoint_name,
                    flush=True,
                )

            raw_directions = [
                world_z,
                -world_z,
                lateral,
                -lateral,
                normal,
                -normal,
                world_z + lateral,
                world_z - lateral,
                -world_z + lateral,
                -world_z - lateral,
                world_z + normal,
                world_z - normal,
                -world_z + normal,
                -world_z - normal,
            ]
            directions = []
            for direction in raw_directions:
                direction = np.asarray(direction, dtype=np.float64)
                direction -= direct_unit * float(
                    np.dot(direction, direct_unit)
                )
                norm = float(np.linalg.norm(direction))
                if norm <= 1e-8:
                    continue
                direction /= norm
                if not any(
                    float(np.linalg.norm(direction - existing)) < 1e-6
                    for existing in directions
                ):
                    directions.append(direction)
            max_offset = float(args.cartesian_detour_max_offset_m)
            offset_step = float(args.cartesian_detour_offset_step_m)
            if targeted_carried_transport:
                upward = world_z - direct_unit * float(
                    np.dot(world_z, direct_unit)
                )
                upward /= max(float(np.linalg.norm(upward)), 1e-9)
                if upward[2] < 0.0:
                    upward *= -1.0
                controlled_max_offset = min(0.08, max_offset)
                controlled_step = max(0.02, offset_step)
                controlled_magnitudes = np.arange(
                    controlled_step,
                    controlled_max_offset + 0.5 * controlled_step,
                    controlled_step,
                ).tolist()
                # A small horizontal bow often moves the Panda away from the
                # wrist singularity without lifting the carried object. Search
                # it before any vertical clearance. The zero offset is the
                # true straight-line/shortest-SLERP candidate.
                offsets = [np.zeros(3, dtype=np.float64)]
                offsets.extend(
                    float(magnitude) * sign * lateral
                    for magnitude in controlled_magnitudes
                    for sign in (1.0, -1.0)
                )
                offsets.extend(
                    float(magnitude) * upward
                    for magnitude in controlled_magnitudes
                )
                print(
                    "[cartesian-carried-bounded-arc-search] waypoint="
                    + waypoint_name
                    + " offsets="
                    + str(len(offsets))
                    + " max_offset_m="
                    + format(controlled_max_offset, ".6f"),
                    flush=True,
                )
            else:
                magnitudes = [0.0]
                if max_offset > 0.0 and offset_step > 0.0:
                    magnitudes.extend(
                        np.arange(
                            offset_step,
                            max_offset + 0.5 * offset_step,
                            offset_step,
                        ).tolist()
                    )
                offsets = [np.zeros(3, dtype=np.float64)]
                for magnitude in magnitudes[1:]:
                    offsets.extend(
                        float(magnitude) * direction for direction in directions
                    )
            schedules = [
                "coupled",
                "rotation_first",
                "translation_first",
                "rotation_lead",
                "rotation_lag",
            ]
            if targeted_carried_transport:
                # Coupled progress is the only schedule whose desired
                # orientation is exactly the shortest monotonic SLERP. Other
                # schedules intentionally rotate while nearly stationary and
                # are inappropriate for loop-repair data.
                schedules = ["coupled"]
            candidates_to_try = [
                (offset, schedule)
                for offset in offsets
                for schedule in schedules
            ]

            accepted = []
            failures = []
            for candidate_index, (offset, schedule) in enumerate(
                candidates_to_try
            ):
                combined = []
                try:
                    arm.set_joint_positions(
                        start_joints.tolist(), disable_dynamics=True
                    )
                    for segment_index, fraction in enumerate(fractions):
                        if schedule == "coupled":
                            position_progress = float(fraction)
                            rotation_progress = float(fraction)
                        elif schedule == "rotation_first":
                            position_progress = max(
                                0.0, 2.0 * float(fraction) - 1.0
                            )
                            rotation_progress = min(
                                1.0, 2.0 * float(fraction)
                            )
                        elif schedule == "translation_first":
                            position_progress = min(
                                1.0, 2.0 * float(fraction)
                            )
                            rotation_progress = max(
                                0.0, 2.0 * float(fraction) - 1.0
                            )
                        elif schedule == "rotation_lead":
                            position_progress = float(fraction)
                            rotation_progress = min(
                                1.0, 1.5 * float(fraction)
                            )
                        else:
                            position_progress = float(fraction)
                            rotation_progress = max(
                                0.0, 1.5 * float(fraction) - 0.5
                            )
                        quaternion = slerp(
                            [rotation_progress]
                        ).as_quat()[0]
                        position = (
                            start_position
                            + position_progress * direct_vector
                            + math.sin(math.pi * position_progress) * offset
                        )
                        linear_path_calls += 1
                        segment_path = arm.get_linear_path(
                            position,
                            quaternion=quaternion,
                            steps=3,
                            ignore_collisions=targeted_carried_transport,
                        )
                        points = np.asarray(
                            segment_path._path_points, dtype=np.float64
                        ).reshape(-1, joint_count)
                        if segment_index > 0 and len(points):
                            points = points[1:]
                        combined.append(points)
                        segment_path.set_to_end(disable_dynamics=True)
                    configurations = np.concatenate(combined, axis=0)
                    joint_sequence = np.vstack(
                        (start_joints[None], configurations)
                    )
                    joint_steps = np.abs(np.diff(joint_sequence, axis=0))
                    joint_travel = joint_steps.sum(axis=0)
                    max_joint_step = float(joint_steps.max(initial=0.0))
                    max_joint_travel = float(joint_travel.max(initial=0.0))
                    wrist_travel = float(joint_travel[-3:].sum())
                    total_joint_travel = float(joint_travel.sum())
                    if max_joint_step > 0.35:
                        raise ConfigurationPathError(
                            "adjacent IK jump exceeds 0.35 rad"
                        )
                    if max_joint_travel > math.pi:
                        raise ConfigurationPathError(
                            "single-joint cumulative travel exceeds pi"
                        )
                    if wrist_travel > 4.5:
                        raise ConfigurationPathError(
                            "wrist cumulative travel exceeds 4.5 rad"
                        )
                    metrics = {
                        "candidate": int(candidate_index),
                        "schedule": schedule,
                        "offset_m": np.asarray(offset).tolist(),
                        "max_lateral_deviation_m": float(
                            np.linalg.norm(offset)
                        ),
                        "direct_distance_m": direct_distance,
                        "points": int(len(configurations)),
                        "total_joint_travel_rad": total_joint_travel,
                        "wrist_travel_rad": wrist_travel,
                        "max_joint_travel_rad": max_joint_travel,
                        "max_joint_step_rad": max_joint_step,
                    }
                    score = float(
                        1000.0 * np.linalg.norm(offset)
                        + total_joint_travel
                        + 2.0 * wrist_travel
                    )
                    accepted.append((score, configurations.copy(), metrics))
                    print(
                        "[cartesian-detour-candidate] waypoint="
                        + point._waypoint.get_name()
                        + " metrics="
                        + json.dumps(metrics, sort_keys=True),
                        flush=True,
                    )
                    # Offsets are ordered by magnitude. Once one magnitude is
                    # feasible, larger deviations cannot win the primary cost.
                    selected_magnitude = float(np.linalg.norm(offset))
                    next_index = candidate_index + 1
                    if (
                        next_index >= len(candidates_to_try)
                        or float(
                            np.linalg.norm(candidates_to_try[next_index][0])
                        )
                        > selected_magnitude + 1e-9
                    ):
                        break
                except (ConfigurationPathError, ConfigurationError) as error:
                    failures.append(
                        {
                            "candidate": int(candidate_index),
                            "schedule": schedule,
                            "offset_m": np.asarray(offset).tolist(),
                            "error": str(error),
                        }
                    )
                finally:
                    arm.set_joint_positions(
                        start_joints.tolist(), disable_dynamics=True
                    )
            if not accepted:
                joint_candidates = []
                joint_rejections = []

                def arm_collides_excluding_grasped_objects():
                    """Check arm collisions without treating a held object as an obstacle.

                    RLBench's phone task rigidly attaches the handset to the
                    gripper after the grasp waypoint.  The generic arm collision
                    collection can then report the intentional gripper/handset
                    contact while evaluating waypoint3.  Temporarily making only
                    the grasped objects non-collidable preserves robot self-
                    collision and collisions against the rest of the scene.
                    """
                    grasped_objects = list(
                        point._robot.gripper.get_grasped_objects()
                    )
                    collidable_states = []
                    try:
                        for grasped_object in grasped_objects:
                            was_collidable = bool(
                                grasped_object.is_collidable()
                            )
                            collidable_states.append(
                                (grasped_object, was_collidable)
                            )
                            if was_collidable:
                                grasped_object.set_collidable(False)
                        return bool(arm.check_arm_collision())
                    finally:
                        for grasped_object, was_collidable in collidable_states:
                            grasped_object.set_collidable(was_collidable)

                try:
                    goal_configurations = arm.solve_ik_via_sampling(
                        target_position,
                        quaternion=target_quaternion,
                        ignore_collisions=False,
                        trials=4000,
                        max_configs=100,
                        distance_threshold=0.65,
                        max_time_ms=20,
                    )
                except ConfigurationError:
                    goal_configurations = []
                try:
                    for goal_index, goal_joints in enumerate(
                        np.asarray(goal_configurations, dtype=np.float64)
                    ):
                        # simGetConfigForTipPose may return a configuration
                        # inside its coarse acceptance region.  Refine every
                        # sampled branch with local Jacobian IK before scoring
                        # it as an executable endpoint.
                        arm.set_joint_positions(
                            goal_joints.tolist(), disable_dynamics=True
                        )
                        try:
                            goal_joints = np.asarray(
                                arm.solve_ik_via_jacobian(
                                    target_position,
                                    quaternion=target_quaternion,
                                ),
                                dtype=np.float64,
                            )
                        except Exception:
                            arm.set_joint_positions(
                                start_joints.tolist(), disable_dynamics=True
                            )
                            joint_rejections.append(
                                {
                                    "goal": int(goal_index),
                                    "reason": "endpoint_jacobian_refinement_failed",
                                }
                            )
                            continue
                        arm.set_joint_positions(
                            start_joints.tolist(), disable_dynamics=True
                        )
                        joint_delta = goal_joints - start_joints
                        interpolation_steps = max(
                            2,
                            int(
                                math.ceil(
                                    float(np.max(np.abs(joint_delta))) / 0.015
                                )
                            ),
                        )
                        interpolation = (
                            start_joints[None]
                            + np.linspace(
                                0.0, 1.0, interpolation_steps + 1
                            )[1:, None]
                            * joint_delta[None]
                        )
                        xyz = [start_position.copy()]
                        quaternion_previous = start_quaternion.copy()
                        rotation_length = 0.0
                        collision = False
                        for configuration in interpolation:
                            arm.set_joint_positions(
                                configuration.tolist(),
                                disable_dynamics=True,
                            )
                            if arm_collides_excluding_grasped_objects():
                                collision = True
                                break
                            xyz.append(
                                np.asarray(
                                    arm.get_tip().get_position(),
                                    dtype=np.float64,
                                )
                            )
                            quaternion_current = np.asarray(
                                arm.get_tip().get_quaternion(),
                                dtype=np.float64,
                            )
                            quaternion_current /= np.linalg.norm(
                                quaternion_current
                            )
                            rotation_length += float(
                                2.0
                                * np.arccos(
                                    np.clip(
                                        abs(
                                            float(
                                                np.dot(
                                                    quaternion_previous,
                                                    quaternion_current,
                                                )
                                            )
                                        ),
                                        -1.0,
                                        1.0,
                                    )
                                )
                            )
                            quaternion_previous = quaternion_current
                        arm.set_joint_positions(
                            start_joints.tolist(), disable_dynamics=True
                        )
                        if collision:
                            joint_rejections.append(
                                {"goal": int(goal_index), "reason": "collision"}
                            )
                            continue
                        xyz = np.asarray(xyz, dtype=np.float64)
                        path_length = float(
                            np.linalg.norm(
                                np.diff(xyz, axis=0), axis=1
                            ).sum()
                        )
                        if direct_distance > 1e-9:
                            chord_progress = (
                                xyz - start_position
                            ) @ direct_unit
                            chord_closest = (
                                start_position
                                + chord_progress[:, None] * direct_unit
                            )
                            lateral_deviation = float(
                                np.linalg.norm(
                                    xyz - chord_closest, axis=1
                                ).max(initial=0.0)
                            )
                            progress_backtrack = float(
                                np.maximum(
                                    0.0, -np.diff(chord_progress)
                                ).sum()
                            )
                        else:
                            lateral_deviation = path_length
                            progress_backtrack = path_length
                        joint_travel = np.abs(joint_delta)
                        endpoint_position_error = float(
                            np.linalg.norm(xyz[-1] - target_position)
                        )
                        endpoint_rotation_error = float(
                            2.0
                            * np.arccos(
                                np.clip(
                                    abs(
                                        float(
                                            np.dot(
                                                quaternion_current,
                                                target_quaternion,
                                            )
                                        )
                                    ),
                                    -1.0,
                                    1.0,
                                )
                            )
                        )
                        max_joint_delta = float(
                            joint_travel.max(initial=0.0)
                        )
                        wrist_travel = float(joint_travel[-3:].sum())
                        detour_ratio = float(
                            path_length / max(direct_distance, 1e-9)
                        )
                        rejection_reasons = []
                        if endpoint_position_error > 0.003:
                            rejection_reasons.append("endpoint_position_error")
                        if endpoint_rotation_error > 0.05:
                            rejection_reasons.append("endpoint_rotation_error")
                        if max_joint_delta > 5.0:
                            rejection_reasons.append("single_joint_delta")
                        if wrist_travel > 8.0:
                            rejection_reasons.append("wrist_travel")
                        if rotation_length > 1.5 * math.pi:
                            rejection_reasons.append("eef_rotation")
                        if lateral_deviation > float(
                            args.joint_interp_max_lateral_deviation_m
                        ):
                            rejection_reasons.append("eef_lateral_deviation")
                        if detour_ratio > 3.0:
                            rejection_reasons.append("eef_detour_ratio")
                        if progress_backtrack > 0.10:
                            rejection_reasons.append("eef_progress_backtrack")
                        if rejection_reasons:
                            joint_rejections.append(
                                {
                                    "goal": int(goal_index),
                                    "reasons": rejection_reasons,
                                    "max_joint_delta_rad": max_joint_delta,
                                    "wrist_travel_rad": wrist_travel,
                                    "rotation_rad": rotation_length,
                                    "max_lateral_deviation_m": lateral_deviation,
                                    "detour_ratio": detour_ratio,
                                    "progress_backtrack_m": progress_backtrack,
                                    "endpoint_position_error_m": endpoint_position_error,
                                    "endpoint_rotation_error_rad": endpoint_rotation_error,
                                }
                            )
                            continue
                        metrics = {
                            "fallback": "nearest_collision_free_joint_interpolation",
                            "goal": int(goal_index),
                            "points": int(len(interpolation)),
                            "direct_distance_m": direct_distance,
                            "path_length_m": path_length,
                            "detour_ratio": detour_ratio,
                            "max_lateral_deviation_m": lateral_deviation,
                            "progress_backtrack_m": progress_backtrack,
                            "rotation_rad": rotation_length,
                            "total_joint_travel_rad": float(joint_travel.sum()),
                            "wrist_travel_rad": wrist_travel,
                            "max_joint_delta_rad": max_joint_delta,
                            "endpoint_position_error_m": endpoint_position_error,
                            "endpoint_rotation_error_rad": endpoint_rotation_error,
                        }
                        score = float(
                            20.0 * lateral_deviation
                            + 5.0 * progress_backtrack
                            + max(0.0, path_length - direct_distance)
                            + float(joint_travel.sum())
                            + 2.0 * wrist_travel
                            + rotation_length
                        )
                        joint_candidates.append(
                            (score, interpolation.copy(), metrics)
                        )
                finally:
                    arm.set_joint_positions(
                        start_joints.tolist(), disable_dynamics=True
                    )
                if joint_candidates:
                    _, selected_joint_path, selected_joint_metrics = min(
                        joint_candidates, key=lambda item: item[0]
                    )
                    print(
                        "[cartesian-detour-joint-fallback-selected] waypoint="
                        + point._waypoint.get_name()
                        + " accepted="
                        + str(len(joint_candidates))
                        + "/"
                        + str(len(goal_configurations))
                        + " metrics="
                        + json.dumps(selected_joint_metrics, sort_keys=True),
                        flush=True,
                    )
                    exact_fake_grasp_transport = bool(
                        args.phone_reference_artifact is not None
                        and any(
                            task_token in str(args.phone_reference_artifact)
                            for task_token in ("phone_on_base", "stack_wine")
                        )
                    )
                    if exact_fake_grasp_transport:
                        # The selected joint interpolation already supplies a
                        # low, collision-free XYZ route, but its unconstrained
                        # FK orientation can wind the wrist through 300--500
                        # degrees. Keep those Cartesian positions and solve a
                        # continuous local IK sequence whose quaternion is the
                        # explicit shortest arc from the carried start pose to
                        # the authored target. Both phone and wine are RLBench
                        # fake-grasp objects (parented to the attach point), so
                        # exact path tracking preserves the grasp while avoiding
                        # Reflexxes/RML re-interpolation.
                        selected_xyz = []
                        for configuration in selected_joint_path:
                            arm.set_joint_positions(
                                configuration.tolist(),
                                disable_dynamics=True,
                            )
                            selected_xyz.append(
                                np.asarray(
                                    arm.get_tip().get_position(),
                                    dtype=np.float64,
                                )
                            )
                        arm.set_joint_positions(
                            start_joints.tolist(), disable_dynamics=True
                        )
                        corrected_path = []
                        previous_joints = start_joints.copy()
                        correction_valid = True
                        corrected_quaternions = slerp(
                            np.linspace(
                                0.0,
                                1.0,
                                len(selected_joint_path) + 1,
                                dtype=np.float64,
                            )[1:]
                        ).as_quat()
                        for position, quaternion in zip(
                            selected_xyz, corrected_quaternions
                        ):
                            arm.set_joint_positions(
                                previous_joints.tolist(),
                                disable_dynamics=True,
                            )
                            corrected_joints = None
                            try:
                                jacobian_joints = np.asarray(
                                    arm.solve_ik_via_jacobian(
                                        position,
                                        quaternion=quaternion,
                                    ),
                                    dtype=np.float64,
                                )
                                if float(
                                    np.max(
                                        np.abs(
                                            jacobian_joints
                                            - previous_joints
                                        )
                                    )
                                ) <= 0.35:
                                    corrected_joints = jacobian_joints
                            except (IKError, ConfigurationError):
                                pass
                            if corrected_joints is None:
                                arm.set_joint_positions(
                                    previous_joints.tolist(),
                                    disable_dynamics=True,
                                )
                                try:
                                    sampled_joints = np.asarray(
                                        arm.solve_ik_via_sampling(
                                            position,
                                            quaternion=quaternion,
                                            ignore_collisions=False,
                                            trials=1200,
                                            max_configs=36,
                                            distance_threshold=0.55,
                                            max_time_ms=20,
                                        ),
                                        dtype=np.float64,
                                    )
                                except ConfigurationError:
                                    sampled_joints = np.empty(
                                        (0, len(previous_joints)),
                                        dtype=np.float64,
                                    )
                                nearby_sampled = [
                                    candidate
                                    for candidate in sampled_joints
                                    if float(
                                        np.max(
                                            np.abs(
                                                candidate - previous_joints
                                            )
                                        )
                                    )
                                    <= 0.55
                                ]
                                if nearby_sampled:
                                    corrected_joints = min(
                                        nearby_sampled,
                                        key=lambda candidate: float(
                                            np.linalg.norm(
                                                candidate - previous_joints
                                            )
                                            + 0.5
                                            * np.linalg.norm(
                                                candidate[-3:]
                                                - previous_joints[-3:]
                                            )
                                        ),
                                    )
                            if corrected_joints is None:
                                correction_valid = False
                                break
                            arm.set_joint_positions(
                                corrected_joints.tolist(),
                                disable_dynamics=True,
                            )
                            if arm.check_arm_collision():
                                correction_valid = False
                                break
                            corrected_path.append(corrected_joints.copy())
                            previous_joints = corrected_joints.copy()
                        arm.set_joint_positions(
                            start_joints.tolist(), disable_dynamics=True
                        )
                        if (
                            correction_valid
                            and len(corrected_path) == len(selected_joint_path)
                        ):
                            corrected_path = np.asarray(
                                corrected_path, dtype=np.float64
                            )
                            print(
                                "[cartesian-fake-grasp-shortest-arc-corrected] waypoint="
                                + point._waypoint.get_name()
                                + " points="
                                + str(len(corrected_path))
                                + " target_rotation_rad="
                                + format(
                                    float(
                                        (
                                            Rotation.from_quat(
                                                start_quaternion
                                            ).inv()
                                            * Rotation.from_quat(
                                                target_quaternion
                                            )
                                        ).magnitude()
                                    ),
                                    ".6f",
                                ),
                                flush=True,
                            )
                            return TrackingArmConfigurationPath(
                                arm,
                                corrected_path,
                                robot=point._robot,
                                waypoint_name=point._waypoint.get_name(),
                            )
                        print(
                            "[cartesian-fake-grasp-shortest-arc-unavailable] waypoint="
                            + point._waypoint.get_name()
                            + " solved="
                            + str(len(corrected_path))
                            + "/"
                            + str(len(selected_joint_path)),
                            flush=True,
                        )
                    if (
                        args.segmented_linear_waypoints is not None
                        and not exact_fake_grasp_transport
                    ):
                        return DenseJointServoPath(
                            arm,
                            selected_joint_path,
                            waypoint_name=point._waypoint.get_name(),
                        )
                    if exact_fake_grasp_transport:
                        print(
                            "[cartesian-phone-fake-grasp-exact-tracking] waypoint="
                            + point._waypoint.get_name(),
                            flush=True,
                        )
                    return TrackingArmConfigurationPath(
                        arm,
                        selected_joint_path,
                        robot=point._robot,
                        waypoint_name=point._waypoint.get_name(),
                    )
                print(
                    "[cartesian-detour-failed] waypoint="
                    + point._waypoint.get_name()
                    + " candidates="
                    + str(len(candidates_to_try))
                    + " arc_failures="
                    + str(len(failures))
                    + " joint_goals="
                    + str(len(goal_configurations))
                    + " joint_rejections="
                    + str(len(joint_rejections))
                    + " first_arc_failures="
                    + json.dumps(failures[:5], sort_keys=True)
                    + " first_joint_rejections="
                    + json.dumps(joint_rejections[:5], sort_keys=True),
                    flush=True,
                )
                raise ConfigurationPathError(
                    "No bounded Cartesian detour passed continuous IK and "
                    "joint-path guards"
                )
            _, selected_configurations, selected_metrics = min(
                accepted, key=lambda item: item[0]
            )
            print(
                "[cartesian-detour-selected] waypoint="
                + point._waypoint.get_name()
                + " accepted="
                + str(len(accepted))
                + " metrics="
                + json.dumps(selected_metrics, sort_keys=True),
                flush=True,
            )
            if (
                point._waypoint.get_name() in ("waypoint0", "waypoint3")
                and args.segmented_linear_waypoints is None
            ):
                return TrackingArmConfigurationPath(
                    arm,
                    selected_configurations,
                    robot=point._robot,
                    waypoint_name=point._waypoint.get_name(),
                )
            exact_fake_grasp_detour = bool(
                args.phone_reference_artifact is not None
                and any(
                    task_token in str(args.phone_reference_artifact)
                    for task_token in ("phone_on_base", "stack_wine")
                )
            )
            if exact_fake_grasp_detour:
                return TrackingArmConfigurationPath(
                    arm,
                    selected_configurations,
                    robot=point._robot,
                    waypoint_name=point._waypoint.get_name(),
                )
            return DenseJointServoPath(
                    arm,
                    selected_configurations,
                    waypoint_name=point._waypoint.get_name(),
                )

        Point.get_path = cartesian_detour_point_get_path
    elif args.expert_path_mode == "franka_cartesian_servo":
        from scipy.spatial.transform import Rotation, Slerp
        import pinocchio as pin

        servo_model = pin.buildModelFromUrdf(
            str(RL_BENCH_ROOT / "urdfs/panda/panda.urdf")
        )
        servo_data = servo_model.createData()
        servo_tip_frame_id = servo_model.getFrameId("Pandatip")

        class FrankaCartesianServoPath:
            """Online resolved-rate Cartesian path with branch-preserving nullspace control."""

            def __init__(
                self, arm, target_position, target_quaternion, waypoint_name
            ):
                self._arm = arm
                self._waypoint_name = str(waypoint_name)
                self._target_position = np.asarray(
                    target_position, dtype=np.float64
                )
                self._target_quaternion = np.asarray(
                    target_quaternion, dtype=np.float64
                )
                self._target_quaternion /= np.linalg.norm(
                    self._target_quaternion
                )
                self._initialized = False
                self._done = False
                self._step_index = 0
                self._previous_qdot = np.zeros(7, dtype=np.float64)
                self._debug_frames = []
                self._debug_rows = []
                self._construction_start_q = np.asarray(
                    arm.get_joint_positions(), dtype=np.float64
                )
                self._joint_position_action = self._construction_start_q.copy()

            @staticmethod
            def _rotation_angle(rotation_matrix):
                return float(
                    np.arccos(
                        np.clip(
                            (np.trace(rotation_matrix) - 1.0) / 2.0,
                            -1.0,
                            1.0,
                        )
                    )
                )

            def _pin_q(self):
                q = pin.neutral(servo_model)
                q[:7] = np.asarray(
                    self._arm.get_joint_positions(), dtype=np.float64
                )
                return q

            def _urdf_tip(self, q):
                pin.forwardKinematics(servo_model, servo_data, q)
                pin.updateFramePlacements(servo_model, servo_data)
                return servo_data.oMf[servo_tip_frame_id]

            def _initialize(self):
                self._start_q = np.asarray(
                    self._arm.get_joint_positions(), dtype=np.float64
                )
                self._start_position = np.asarray(
                    self._arm.get_tip().get_position(), dtype=np.float64
                )
                self._start_quaternion = np.asarray(
                    self._arm.get_tip().get_quaternion(), dtype=np.float64
                )
                self._start_quaternion /= np.linalg.norm(
                    self._start_quaternion
                )
                # scipy's Slerp follows the shortest rotation once quaternion
                # signs are put in the same hemisphere explicitly.
                if float(
                    np.dot(self._start_quaternion, self._target_quaternion)
                ) < 0.0:
                    self._target_quaternion *= -1.0
                self._slerp = Slerp(
                    [0.0, 1.0],
                    Rotation.from_quat(
                        np.stack(
                            (self._start_quaternion, self._target_quaternion)
                        )
                    ),
                )
                q = self._pin_q()
                current_urdf = self._urdf_tip(q)
                current_sim = np.asarray(
                    self._arm.get_tip().get_matrix(), dtype=np.float64
                )
                self._sim_from_urdf = (
                    current_sim @ np.linalg.inv(current_urdf.homogeneous)
                )
                try:
                    endpoint_configs = self._arm.solve_ik_via_sampling(
                        self._target_position,
                        quaternion=self._target_quaternion,
                        ignore_collisions=False,
                        trials=600,
                        max_configs=40,
                    )
                except ConfigurationError as error:
                    raise ConfigurationPathError(
                        "Franka Cartesian servo has no collision-free endpoint IK"
                    ) from error
                endpoint_configs = np.asarray(
                    endpoint_configs, dtype=np.float64
                )
                # The endpoint is not used as a precomputed path.  It is only
                # a redundancy target projected through (I - J^+J), which
                # prevents the elbow/wrist from drifting into a singular branch
                # during a long Cartesian translation.
                self._nullspace_target_q = min(
                    endpoint_configs,
                    key=lambda candidate: float(
                        np.linalg.norm(candidate - self._start_q)
                        + 0.5
                        * np.linalg.norm(
                            candidate[-3:] - self._start_q[-3:]
                        )
                    ),
                ).copy()
                distance = float(
                    np.linalg.norm(
                        self._target_position - self._start_position
                    )
                )
                start_rotation = Rotation.from_quat(
                    self._start_quaternion
                ).as_matrix()
                target_rotation = Rotation.from_quat(
                    self._target_quaternion
                ).as_matrix()
                angle = self._rotation_angle(
                    start_rotation.T @ target_rotation
                )
                # RLBench's legacy CoppeliaSim scene uses a 50 ms physics step.
                # Read it from the simulator so this controller remains stable
                # if the scene timestep changes.
                from pyrep.backend import sim
                self._dt = float(sim.simGetSimulationTimeStep())
                duration = max(
                    distance / float(args.franka_servo_linear_speed_m_s),
                    angle / float(args.franka_servo_angular_speed_rad_s),
                    self._dt,
                )
                self._reference_steps = max(
                    1, int(math.ceil(duration / self._dt))
                )
                self._arm.set_joint_target_velocities([0.0] * 7)
                self._arm.set_control_loop_enabled(False)
                self._arm.set_motor_locked_at_zero_velocity(True)
                if args.franka_servo_save_debug:
                    from pyrep.objects.vision_sensor import VisionSensor
                    self._debug_camera = VisionSensor("cam_front")
                self._initialized = True
                print(
                    "[franka-cartesian-servo-start] distance_m="
                    + format(distance, ".6f")
                    + " shortest_rotation_rad="
                    + format(angle, ".6f")
                    + " reference_steps="
                    + str(self._reference_steps),
                    flush=True,
                )

            def _stop(self):
                self._arm.set_joint_target_velocities([0.0] * 7)
                current_q = np.asarray(
                    self._arm.get_joint_positions(), dtype=np.float64
                )
                self._joint_position_action = current_q.copy()
                self._arm.set_joint_target_positions(current_q.tolist())
                self._arm.set_control_loop_enabled(True)

            def _save_debug(self, status):
                debug_root = artifact_root.parent / "servo_debug"
                debug_root.mkdir(parents=True, exist_ok=True)
                stem = self._waypoint_name + "__" + str(status)
                telemetry_path = debug_root / (stem + "__telemetry.npz")
                if self._debug_rows:
                    np.savez_compressed(
                        telemetry_path,
                        telemetry=np.asarray(
                            self._debug_rows, dtype=np.float64
                        ),
                        columns=np.asarray(
                            [
                                "step",
                                "reference_progress",
                                "position_error_m",
                                "rotation_error_rad",
                                "max_abs_qdot_rad_s",
                                "min_joint_limit_margin_rad",
                            ]
                        ),
                    )
                if self._debug_frames:
                    import imageio.v2 as imageio
                    video_path = debug_root / (stem + ".mp4")
                    imageio.mimsave(
                        video_path,
                        self._debug_frames,
                        fps=int(args.fps),
                        codec="libx264",
                        quality=8,
                        macro_block_size=None,
                    )
                    print(
                        "[franka-cartesian-servo-debug] video="
                        + str(video_path)
                        + " telemetry="
                        + str(telemetry_path),
                        flush=True,
                    )

            def step(self):
                if self._done:
                    raise RuntimeError("Franka Cartesian servo path is complete")
                if not self._initialized:
                    self._initialize()
                self._step_index += 1
                progress = min(
                    1.0,
                    float(self._step_index) / float(self._reference_steps),
                )
                desired_position = (
                    self._start_position
                    + progress
                    * (self._target_position - self._start_position)
                )
                desired_quaternion = self._slerp([progress]).as_quat()[0]
                desired_sim = np.eye(4, dtype=np.float64)
                desired_sim[:3, :3] = Rotation.from_quat(
                    desired_quaternion
                ).as_matrix()
                desired_sim[:3, 3] = desired_position
                desired_urdf_matrix = (
                    np.linalg.inv(self._sim_from_urdf) @ desired_sim
                )
                desired_urdf = pin.SE3(
                    desired_urdf_matrix[:3, :3],
                    desired_urdf_matrix[:3, 3],
                )

                q = self._pin_q()
                current_urdf = self._urdf_tip(q)
                error = pin.log6(current_urdf.actInv(desired_urdf)).vector
                jacobian = pin.computeFrameJacobian(
                    servo_model,
                    servo_data,
                    q,
                    servo_tip_frame_id,
                    pin.ReferenceFrame.LOCAL,
                )[:, :7]
                damping = float(args.franka_servo_damping)
                jacobian_pinv = jacobian.T @ np.linalg.solve(
                    jacobian @ jacobian.T
                    + damping * np.eye(6, dtype=np.float64),
                    np.eye(6, dtype=np.float64),
                )
                # Track the moving Cartesian reference and preserve the same
                # redundant elbow/wrist branch in the Jacobian nullspace.
                qdot = jacobian_pinv @ (4.0 * error)
                nullspace = np.eye(7, dtype=np.float64) - jacobian_pinv @ jacobian
                qdot += nullspace @ (
                    float(args.franka_servo_nullspace_gain)
                    * (self._nullspace_target_q - q[:7])
                )
                max_speed = float(args.franka_servo_max_joint_speed_rad_s)
                qdot = np.clip(qdot, -max_speed, max_speed)
                accel_step = float(
                    args.franka_servo_max_joint_accel_step_rad_s
                )
                qdot = np.clip(
                    qdot,
                    self._previous_qdot - accel_step,
                    self._previous_qdot + accel_step,
                )
                self._previous_qdot = qdot.copy()
                self._arm.set_joint_target_velocities(qdot.tolist())
                self._joint_position_action = np.asarray(
                    self._arm.get_joint_positions(), dtype=np.float64
                ) + qdot * self._dt

                actual_position = np.asarray(
                    self._arm.get_tip().get_position(), dtype=np.float64
                )
                actual_rotation = np.asarray(
                    self._arm.get_tip().get_matrix(), dtype=np.float64
                )[:3, :3]
                target_rotation = Rotation.from_quat(
                    self._target_quaternion
                ).as_matrix()
                position_error = float(
                    np.linalg.norm(actual_position - self._target_position)
                )
                rotation_error = self._rotation_angle(
                    actual_rotation.T @ target_rotation
                )
                joint_margin = np.minimum(
                    q[:7] - servo_model.lowerPositionLimit[:7],
                    servo_model.upperPositionLimit[:7] - q[:7],
                )
                if args.franka_servo_save_debug:
                    self._debug_rows.append(
                        [
                            float(self._step_index),
                            float(progress),
                            float(position_error),
                            float(rotation_error),
                            float(np.max(np.abs(qdot))),
                            float(np.min(joint_margin)),
                        ]
                    )
                    debug_rgb = np.asarray(
                        self._debug_camera.capture_rgb(), dtype=np.float64
                    )
                    if (
                        debug_rgb.size
                        and float(np.max(debug_rgb)) <= 1.0 + 1e-6
                    ):
                        debug_rgb *= 255.0
                    debug_rgb = np.clip(
                        debug_rgb, 0.0, 255.0
                    ).astype(np.uint8)
                    import cv2
                    cv2.putText(
                        debug_rgb,
                        "frame=%04d progress=%.3f" % (
                            self._step_index, progress
                        ),
                        (8, 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        (255, 255, 0),
                        1,
                        cv2.LINE_AA,
                    )
                    cv2.putText(
                        debug_rgb,
                        "pos_err=%.4fm rot_err=%.3frad joint_margin=%.3f" % (
                            position_error,
                            rotation_error,
                            float(np.min(joint_margin)),
                        ),
                        (8, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.40,
                        (255, 255, 0),
                        1,
                        cv2.LINE_AA,
                    )
                    self._debug_frames.append(debug_rgb)
                if self._step_index == 1 or self._step_index % 50 == 0:
                    print(
                        "[franka-cartesian-servo-progress] step="
                        + str(self._step_index)
                        + " reference_progress="
                        + format(progress, ".4f")
                        + " final_position_error_m="
                        + format(position_error, ".6f")
                        + " final_rotation_error_rad="
                        + format(rotation_error, ".6f")
                        + " max_abs_qdot_rad_s="
                        + format(float(np.max(np.abs(qdot))), ".6f")
                        + " min_joint_limit_margin_rad="
                        + format(float(np.min(joint_margin)), ".6f"),
                        flush=True,
                    )
                reached = bool(
                    progress >= 1.0
                    and position_error <= 0.003
                    and rotation_error <= 0.04
                )
                if reached:
                    self._done = True
                    self._stop()
                    self._save_debug("success")
                    print(
                        "[franka-cartesian-servo-done] steps="
                        + str(self._step_index)
                        + " position_error_m="
                        + format(position_error, ".6f")
                        + " rotation_error_rad="
                        + format(rotation_error, ".6f"),
                        flush=True,
                    )
                    return True
                if self._step_index >= int(args.franka_servo_max_steps):
                    self._stop()
                    self._save_debug("failed")
                    raise ConfigurationPathError(
                        "Franka Cartesian servo did not reach waypoint: "
                        + "position_error_m="
                        + format(position_error, ".6f")
                        + " rotation_error_rad="
                        + format(rotation_error, ".6f")
                    )
                return False

            def visualize(self):
                return None

            def clear_visualization(self):
                return None

            def get_executed_joint_position_action(self):
                return self._joint_position_action.copy()

            def set_to_end(self, disable_dynamics=False):
                """Compatibility with RLBench's pre-demo feasibility pass.

                The feasibility pass must not run the online controller or
                advance physics.  A collision-ignored Cartesian IK path is
                used only to verify the endpoint and place the arm at that
                endpoint while RLBench checks the remaining waypoint chain.
                The actual demonstration still uses the velocity servo.
                """
                try:
                    endpoint_configs = self._arm.solve_ik_via_sampling(
                        self._target_position,
                        quaternion=self._target_quaternion,
                        ignore_collisions=True,
                        trials=300,
                        max_configs=20,
                    )
                except ConfigurationError as error:
                    raise ConfigurationPathError(
                        "Franka Cartesian servo endpoint has no IK solution"
                    ) from error
                current_q = np.asarray(
                    self._arm.get_joint_positions(), dtype=np.float64
                )
                endpoint_q = min(
                    np.asarray(endpoint_configs, dtype=np.float64),
                    key=lambda candidate: float(
                        np.linalg.norm(candidate - current_q)
                    ),
                )
                self._arm.set_joint_positions(
                    endpoint_q.tolist(), disable_dynamics=disable_dynamics
                )
                self._joint_position_action = np.asarray(
                    self._arm.get_joint_positions(), dtype=np.float64
                )

            def set_to_start(self, disable_dynamics=False):
                self._arm.set_joint_positions(
                    self._construction_start_q.tolist(),
                    disable_dynamics=disable_dynamics,
                )

        def franka_cartesian_servo_point_get_path(
            point, ignore_collisions=False
        ):
            del ignore_collisions
            nonlocal linear_path_calls
            linear_path_calls += 1
            return FrankaCartesianServoPath(
                point._robot.arm,
                point._waypoint.get_position(),
                point._waypoint.get_quaternion(),
                point._waypoint.get_name(),
            )

        Point.get_path = franka_cartesian_servo_point_get_path
    elif args.expert_path_mode == "linear_only":
        def linear_only_point_get_path(point, ignore_collisions=False):
            nonlocal linear_path_calls
            linear_path_calls += 1
            arm = point._robot.arm
            return arm.get_linear_path(
                point._waypoint.get_position(),
                euler=point._waypoint.get_orientation(),
                ignore_collisions=(point._ignore_collisions or ignore_collisions),
            )

        Point.get_path = linear_only_point_get_path
    elif args.expert_path_mode in (
        "segmented_linear",
        "segmented_linear_then_rrt",
        "segmented_linear_then_best_of_n",
    ):
        from pyrep.robots.configuration_paths.arm_configuration_path import (
            ArmConfigurationPath,
        )
        from scipy.spatial.transform import Rotation, Slerp

        def segmented_linear_point_get_path(point, ignore_collisions=False):
            nonlocal linear_path_calls, rrt_path_calls
            waypoint_name = point._waypoint.get_name()
            if (
                args.segmented_linear_waypoints is not None
                and waypoint_name not in args.segmented_linear_waypoints
            ):
                print(
                    "[segmented-linear-stock-waypoint] waypoint="
                    + waypoint_name,
                    flush=True,
                )
                return original_point_get_path(point, ignore_collisions)
            arm = point._robot.arm
            segment_count = int(args.segmented_linear_segments)
            if segment_count <= 0:
                raise ValueError("--segmented-linear-segments must be positive")
            start_joints = np.asarray(arm.get_joint_positions(), dtype=np.float64)
            start_position = np.asarray(arm.get_tip().get_position(), dtype=np.float64)
            target_position = np.asarray(
                point._waypoint.get_position(), dtype=np.float64
            )
            start_quaternion = np.asarray(
                arm.get_tip().get_quaternion(), dtype=np.float64
            )
            target_quaternion = np.asarray(
                point._waypoint.get_quaternion(), dtype=np.float64
            )
            rotations = Rotation.from_quat(
                np.stack((start_quaternion, target_quaternion), axis=0)
            )
            slerp = Slerp([0.0, 1.0], rotations)
            fractions = np.linspace(0.0, 1.0, segment_count + 1)[1:]
            segment_quaternions = slerp(fractions).as_quat()
            segment_positions = None
            if (
                args.expert_path_mode == "segmented_linear_then_best_of_n"
                or args.segmented_use_stock_guide
            ):
                if path_fk_model is None:
                    raise RuntimeError("Panda FK model is unavailable for guided SLERP")
                guide_path = original_point_get_path(point, ignore_collisions)
                guide_configurations = np.asarray(
                    guide_path._path_points, dtype=np.float64
                ).reshape(-1, int(arm.get_joint_count()))
                if len(guide_configurations) == 0:
                    raise ConfigurationPathError("empty stock XYZ guide")
                guide_positions = np.vstack(
                    (
                        start_position[None],
                        np.stack(
                            [
                                fk_pose(path_fk_model, configuration)[:3, 3]
                                for configuration in guide_configurations
                            ],
                            axis=0,
                        ),
                    )
                )
                guide_lengths = np.linalg.norm(
                    np.diff(guide_positions, axis=0), axis=1
                )
                guide_cumulative = np.concatenate(
                    (np.asarray([0.0]), np.cumsum(guide_lengths))
                )
                if guide_cumulative[-1] <= 1e-8:
                    segment_positions = np.repeat(
                        target_position[None], segment_count, axis=0
                    )
                else:
                    sample_distances = fractions * guide_cumulative[-1]
                    segment_positions = np.stack(
                        [
                            np.interp(
                                sample_distances,
                                guide_cumulative,
                                guide_positions[:, axis],
                            )
                            for axis in range(3)
                        ],
                        axis=1,
                    )
                    segment_positions[-1] = target_position
                print(
                    "[segmented-guided-slerp] waypoint="
                    + waypoint_name
                    + " guide_points="
                    + str(len(guide_configurations))
                    + " guide_xyz_m="
                    + format(float(guide_cumulative[-1]), ".6f"),
                    flush=True,
                )
            segment_steps = max(2, int(math.ceil(50.0 / segment_count)))
            combined = []
            segmented_error = None
            try:
                for segment_index, (fraction, quaternion) in enumerate(
                    zip(fractions, segment_quaternions)
                ):
                    position = (
                        segment_positions[segment_index]
                        if segment_positions is not None
                        else start_position
                        + float(fraction) * (target_position - start_position)
                    )
                    linear_path_calls += 1
                    try:
                        path = arm.get_linear_path(
                            position,
                            quaternion=quaternion,
                            steps=segment_steps,
                            ignore_collisions=(
                                point._ignore_collisions or ignore_collisions
                            ),
                        )
                    except ConfigurationPathError:
                        if args.expert_path_mode == "segmented_linear":
                            raise
                        if args.expert_path_mode == "segmented_linear_then_best_of_n":
                            raise
                        rrt_path_calls += 1
                        path = arm.get_nonlinear_path(
                            position,
                            quaternion=quaternion,
                            ignore_collisions=(
                                point._ignore_collisions or ignore_collisions
                            ),
                            trials=100,
                            max_configs=10,
                            trials_per_goal=10,
                            algorithm=getattr(
                                Algos, args.segmented_fallback_algorithm
                            ),
                        )
                    points = np.asarray(path._path_points, dtype=np.float64).reshape(
                        -1, int(arm.get_joint_count())
                    )
                    if segment_index > 0 and len(points):
                        points = points[1:]
                    combined.append(points)
                    path.set_to_end(disable_dynamics=True)
            except ConfigurationPathError as error:
                segmented_error = error
            finally:
                arm.set_joint_positions(start_joints.tolist(), disable_dynamics=True)
            if segmented_error is not None:
                if args.expert_path_mode != "segmented_linear_then_best_of_n":
                    raise segmented_error

                if path_fk_model is None:
                    raise RuntimeError("Panda FK model is unavailable for path scoring")
                candidate_count = int(args.phone_path_candidates)
                candidates = []
                failures = []
                start_transform = fk_pose(path_fk_model, start_joints)
                direct_distance = float(
                    np.linalg.norm(target_position - start_position)
                )
                for candidate_index in range(candidate_count):
                    try:
                        candidate_path = original_point_get_path(
                            point, ignore_collisions
                        )
                        configurations = np.asarray(
                            candidate_path._path_points, dtype=np.float64
                        ).reshape(-1, int(arm.get_joint_count()))
                        if len(configurations) == 0:
                            raise ConfigurationPathError("empty stock path")
                        joint_sequence = np.vstack(
                            (start_joints[None], configurations)
                        )
                        joint_travel_per_axis = np.abs(
                            np.diff(joint_sequence, axis=0)
                        ).sum(axis=0)
                        max_joint_travel = float(joint_travel_per_axis.max())
                        total_joint_travel = float(joint_travel_per_axis.sum())

                        xyz_length = 0.0
                        rotation_length = 0.0
                        previous_transform = start_transform
                        path_positions = [start_transform[:3, 3].copy()]
                        for configuration in configurations:
                            transform = fk_pose(path_fk_model, configuration)
                            path_positions.append(transform[:3, 3].copy())
                            xyz_length += float(
                                np.linalg.norm(
                                    transform[:3, 3]
                                    - previous_transform[:3, 3]
                                )
                            )
                            relative_rotation = (
                                previous_transform[:3, :3].T
                                @ transform[:3, :3]
                            )
                            rotation_length += float(
                                np.arccos(
                                    np.clip(
                                        (np.trace(relative_rotation) - 1.0) / 2.0,
                                        -1.0,
                                        1.0,
                                    )
                                )
                            )
                            previous_transform = transform
                        detour_ratio = float(
                            xyz_length / max(direct_distance, 1e-6)
                        )
                        path_positions = np.asarray(
                            path_positions, dtype=np.float64
                        )
                        if direct_distance > 1e-9:
                            chord_unit = (
                                target_position - start_position
                            ) / direct_distance
                            chord_progress = (
                                path_positions - start_position
                            ) @ chord_unit
                            chord_closest = (
                                start_position
                                + chord_progress[:, None] * chord_unit
                            )
                            max_lateral_deviation = float(
                                np.linalg.norm(
                                    path_positions - chord_closest, axis=1
                                ).max(initial=0.0)
                            )
                            progress_backtrack = float(
                                np.maximum(
                                    0.0, -np.diff(chord_progress)
                                ).sum()
                            )
                        else:
                            max_lateral_deviation = float(
                                np.linalg.norm(
                                    path_positions - start_position, axis=1
                                ).max(initial=0.0)
                            )
                            progress_backtrack = xyz_length
                        distance_to_target = np.linalg.norm(
                            path_positions - target_position, axis=1
                        )
                        target_backtrack = float(
                            np.maximum(
                                0.0, np.diff(distance_to_target)
                            ).sum()
                        )
                        path_excess = float(
                            max(0.0, xyz_length - direct_distance)
                        )
                        geometry_arc_score = float(
                            path_excess
                            + 2.0 * max_lateral_deviation
                            + 2.0 * target_backtrack
                            + progress_backtrack
                        )
                        loop_rejected = bool(
                            max_joint_travel > float(np.pi)
                            or rotation_length > float(1.5 * np.pi)
                            or detour_ratio > 3.0
                            or (
                                args.matched_scene_selection_objective
                                in {
                                    "waypoint_path_geometry",
                                    "waypoint0_path_geometry",
                                    "waypoint0_balanced_geometry",
                                    "phone_weighted_geometry",
                                }
                                and (
                                    detour_ratio > 2.0
                                    or max_lateral_deviation > 0.20
                                )
                            )
                        )
                        score = float(
                            total_joint_travel
                            + 2.0 * rotation_length
                            + 2.0 * xyz_length
                        )
                        metrics = {
                            "candidate": int(candidate_index),
                            "score": score,
                            "xyz_m": xyz_length,
                            "direct_m": direct_distance,
                            "detour_ratio": detour_ratio,
                            "path_excess_m": path_excess,
                            "max_lateral_deviation_m": max_lateral_deviation,
                            "target_distance_backtrack_m": target_backtrack,
                            "progress_backtrack_m": progress_backtrack,
                            "geometry_arc_score_m": geometry_arc_score,
                            "rotation_rad": rotation_length,
                            "total_joint_travel_rad": total_joint_travel,
                            "max_joint_travel_rad": max_joint_travel,
                            "points": int(len(configurations)),
                            "loop_rejected": loop_rejected,
                        }
                        print(
                            "[segmented-best-of-n-candidate] waypoint="
                            + point._waypoint.get_name()
                            + " metrics="
                            + json.dumps(metrics, sort_keys=True),
                            flush=True,
                        )
                        if not loop_rejected:
                            selection_score = (
                                geometry_arc_score
                                if args.matched_scene_selection_objective
                                in {
                                    "waypoint_path_geometry",
                                    "waypoint0_path_geometry",
                                    "waypoint0_balanced_geometry",
                                    "phone_weighted_geometry",
                                }
                                else score
                            )
                            candidates.append(
                                (selection_score, candidate_path, metrics)
                            )
                    except ConfigurationPathError as error:
                        failures.append(
                            {
                                "candidate": int(candidate_index),
                                "error": str(error),
                            }
                        )
                rrt_path_calls += candidate_count
                if not candidates:
                    print(
                        "[segmented-best-of-n-no-acceptable-path] waypoint="
                        + point._waypoint.get_name()
                        + " segmented_error="
                        + repr(segmented_error)
                        + " failures="
                        + json.dumps(failures, sort_keys=True),
                        flush=True,
                    )
                    raise ConfigurationPathError(
                        "Segmented path failed and all full-waypoint fallback "
                        "paths failed or were loop-rejected"
                    )
                _, selected_path, selected_metrics = min(
                    candidates, key=lambda item: item[0]
                )
                print(
                    "[segmented-best-of-n-selected] waypoint="
                    + point._waypoint.get_name()
                    + " accepted="
                    + str(len(candidates))
                    + "/"
                    + str(candidate_count)
                    + " segmented_error="
                    + repr(segmented_error)
                    + " metrics="
                    + json.dumps(selected_metrics, sort_keys=True),
                    flush=True,
                )
                return selected_path
            if not combined or not any(len(points) for points in combined):
                raise RuntimeError("Segmented linear IK returned an empty path")
            return ArmConfigurationPath(
                arm, np.concatenate(combined, axis=0).reshape(-1)
            )

        Point.get_path = segmented_linear_point_get_path
    elif args.expert_path_mode == "best_of_n_all_points":
        def best_of_n_all_points_get_path(point, ignore_collisions=False):
            """Choose the smoothest collision-aware stock path at every Point."""
            candidate_count = int(args.phone_path_candidates)
            if candidate_count <= 0:
                raise ValueError("--phone-path-candidates must be positive")
            if path_fk_model is None:
                raise RuntimeError("Panda FK model is unavailable for path scoring")

            arm = point._robot.arm
            waypoint_name = point._waypoint.get_name()
            joint_count = int(arm.get_joint_count())
            start_joints = np.asarray(arm.get_joint_positions(), dtype=np.float64)
            start_transform = fk_pose(path_fk_model, start_joints)
            target_position = np.asarray(
                point._waypoint.get_position(), dtype=np.float64
            )
            direct_distance = float(
                np.linalg.norm(target_position - start_transform[:3, 3])
            )
            xyz_loop_limit = float(
                max(float(args.path_xyz_loop_floor_m), 3.0 * direct_distance)
            )
            candidates = []
            failures = []
            for candidate_index in range(candidate_count):
                try:
                    path = original_point_get_path(point, ignore_collisions)
                    configurations = np.asarray(
                        path._path_points, dtype=np.float64
                    ).reshape(-1, joint_count)
                    if len(configurations) == 0:
                        raise ConfigurationPathError("empty configuration path")

                    joint_sequence = np.vstack((start_joints[None], configurations))
                    joint_travel_per_axis = np.abs(
                        np.diff(joint_sequence, axis=0)
                    ).sum(axis=0)
                    max_joint_travel = float(joint_travel_per_axis.max())
                    total_joint_travel = float(joint_travel_per_axis.sum())
                    xyz_length = 0.0
                    rotation_length = 0.0
                    previous_transform = start_transform
                    for configuration in configurations:
                        transform = fk_pose(path_fk_model, configuration)
                        xyz_length += float(
                            np.linalg.norm(
                                transform[:3, 3] - previous_transform[:3, 3]
                            )
                        )
                        relative_rotation = (
                            previous_transform[:3, :3].T @ transform[:3, :3]
                        )
                        rotation_length += float(
                            np.arccos(
                                np.clip(
                                    (np.trace(relative_rotation) - 1.0) / 2.0,
                                    -1.0,
                                    1.0,
                                )
                            )
                        )
                        previous_transform = transform
                    loop_rejected = bool(
                        max_joint_travel > float(np.pi)
                        or rotation_length > float(1.5 * np.pi)
                        or xyz_length > xyz_loop_limit
                    )
                    score = float(
                        total_joint_travel
                        + 2.0 * rotation_length
                        + 2.0 * xyz_length
                    )
                    metrics = {
                        "candidate": int(candidate_index),
                        "score": score,
                        "xyz_m": xyz_length,
                        "direct_m": direct_distance,
                        "xyz_loop_limit_m": xyz_loop_limit,
                        "rotation_rad": rotation_length,
                        "total_joint_travel_rad": total_joint_travel,
                        "max_joint_travel_rad": max_joint_travel,
                        "points": int(len(configurations)),
                        "loop_rejected": loop_rejected,
                    }
                    print(
                        "[best-of-n-all-points-candidate] waypoint="
                        + waypoint_name
                        + " metrics="
                        + json.dumps(metrics, sort_keys=True),
                        flush=True,
                    )
                    if not loop_rejected:
                        candidates.append((score, path, metrics))
                except ConfigurationPathError as exc:
                    failures.append(
                        {"candidate": int(candidate_index), "error": str(exc)}
                    )

            if not candidates:
                print(
                    "[best-of-n-all-points-no-acceptable-path] waypoint="
                    + waypoint_name
                    + " failures="
                    + json.dumps(failures, sort_keys=True),
                    flush=True,
                )
                raise ConfigurationPathError(
                    "All collision-aware candidates failed or were loop-rejected for "
                    + waypoint_name
                )
            _, selected_path, selected_metrics = min(
                candidates, key=lambda item: item[0]
            )
            print(
                "[best-of-n-all-points-selected] waypoint="
                + waypoint_name
                + " accepted="
                + str(len(candidates))
                + "/"
                + str(candidate_count)
                + " metrics="
                + json.dumps(selected_metrics, sort_keys=True),
                flush=True,
            )
            return selected_path

        Point.get_path = best_of_n_all_points_get_path
    elif args.expert_path_mode == "rrt_only":
        def rrt_only_point_get_path(point, ignore_collisions=False):
            nonlocal rrt_path_calls
            rrt_path_calls += 1
            arm = point._robot.arm
            return arm.get_nonlinear_path(
                point._waypoint.get_position(),
                euler=point._waypoint.get_orientation(),
                ignore_collisions=(point._ignore_collisions or ignore_collisions),
                trials=100,
                max_configs=10,
                trials_per_goal=10,
                algorithm=Algos.RRTConnect,
            )

        Point.get_path = rrt_only_point_get_path
    elif args.expert_path_mode == "phone_best_of_n_0_3":
        def phone_best_of_n_selected_pose_get_path(point, ignore_collisions=False):
            """Select the smoothest of several collision-aware phone plans."""
            waypoint_name = point._waypoint.get_name()
            if phone_waypoint_feasibility_active:
                return original_point_get_path(point, ignore_collisions)
            if (
                waypoint_name not in ("waypoint0", "waypoint3")
                or (
                    args.phone_best_of_n_waypoint3_only
                    and waypoint_name != "waypoint3"
                )
            ):
                return original_point_get_path(point, ignore_collisions)
            candidate_count = int(args.phone_path_candidates)
            if candidate_count <= 0:
                raise ValueError("--phone-path-candidates must be positive")

            arm = point._robot.arm
            joint_count = int(arm.get_joint_count())
            start_joints = np.asarray(arm.get_joint_positions(), dtype=np.float64)
            start_position = np.asarray(
                arm.get_tip().get_position(), dtype=np.float64
            )
            target_position = np.asarray(
                point._waypoint.get_position(), dtype=np.float64
            )
            direct_distance = float(
                np.linalg.norm(target_position - start_position)
            )
            start_quaternion = np.asarray(
                arm.get_tip().get_quaternion(), dtype=np.float64
            )
            start_quaternion /= np.linalg.norm(start_quaternion)
            target_quaternion = np.asarray(
                point._waypoint.get_quaternion(), dtype=np.float64
            )
            target_quaternion /= np.linalg.norm(target_quaternion)
            direct_rotation = float(
                2.0
                * np.arccos(
                    np.clip(
                        abs(float(np.dot(start_quaternion, target_quaternion))),
                        -1.0,
                        1.0,
                    )
                )
            )
            candidates = []
            failures = []
            for candidate_index in range(candidate_count):
                try:
                    path = original_point_get_path(point, ignore_collisions)
                    configurations = np.asarray(
                        path._path_points, dtype=np.float64
                    ).reshape(-1, joint_count)
                    if len(configurations) == 0:
                        raise ConfigurationPathError("empty configuration path")
                    joint_sequence = np.vstack((start_joints[None], configurations))
                    joint_travel_per_axis = np.abs(
                        np.diff(joint_sequence, axis=0)
                    ).sum(axis=0)
                    max_joint_travel = float(joint_travel_per_axis.max())
                    total_joint_travel = float(joint_travel_per_axis.sum())
                    xyz_length = 0.0
                    rotation_length = 0.0
                    previous_position = start_position.copy()
                    previous_quaternion = start_quaternion.copy()
                    try:
                        for configuration in configurations:
                            arm.set_joint_positions(
                                configuration.tolist(), disable_dynamics=True
                            )
                            position = np.asarray(
                                arm.get_tip().get_position(), dtype=np.float64
                            )
                            quaternion = np.asarray(
                                arm.get_tip().get_quaternion(), dtype=np.float64
                            )
                            xyz_length += float(
                                np.linalg.norm(position - previous_position)
                            )
                            quaternion_dot = float(
                                abs(np.dot(quaternion, previous_quaternion))
                            )
                            rotation_length += float(
                                2.0
                                * np.arccos(
                                    np.clip(quaternion_dot, -1.0, 1.0)
                                )
                            )
                            previous_position = position
                            previous_quaternion = quaternion
                    finally:
                        arm.set_joint_positions(
                            start_joints.tolist(), disable_dynamics=True
                        )
                    detour_ratio = float(
                        xyz_length / max(direct_distance, 1e-6)
                    )
                    path_positions = [start_position.copy()]
                    try:
                        for configuration in configurations:
                            arm.set_joint_positions(
                                configuration.tolist(), disable_dynamics=True
                            )
                            path_positions.append(
                                np.asarray(
                                    arm.get_tip().get_position(), dtype=np.float64
                                )
                            )
                    finally:
                        arm.set_joint_positions(
                            start_joints.tolist(), disable_dynamics=True
                        )
                    path_positions = np.asarray(path_positions, dtype=np.float64)
                    if direct_distance > 1e-9:
                        chord_unit = (
                            target_position - start_position
                        ) / direct_distance
                        chord_progress = (
                            path_positions - start_position
                        ) @ chord_unit
                        chord_closest = (
                            start_position
                            + chord_progress[:, None] * chord_unit
                        )
                        max_lateral_deviation = float(
                            np.linalg.norm(
                                path_positions - chord_closest, axis=1
                            ).max(initial=0.0)
                        )
                    else:
                        max_lateral_deviation = float(
                            np.linalg.norm(
                                path_positions - start_position, axis=1
                            ).max(initial=0.0)
                        )
                    final_joint7 = float(configurations[-1, 6])
                    rotation_excess = float(
                        max(0.0, rotation_length - direct_rotation)
                    )
                    rotation_excess_rejected = bool(
                        waypoint_name == "waypoint3"
                        and float(args.path_rotation_excess_limit_rad) >= 0.0
                        and rotation_excess
                        > float(args.path_rotation_excess_limit_rad)
                    )
                    detour_rejected = bool(
                        waypoint_name == "waypoint3"
                        and float(args.phone_waypoint3_max_detour_ratio) >= 0.0
                        and detour_ratio
                        > float(args.phone_waypoint3_max_detour_ratio)
                    )
                    lateral_rejected = bool(
                        waypoint_name == "waypoint3"
                        and float(
                            args.phone_waypoint3_max_lateral_deviation_m
                        )
                        >= 0.0
                        and max_lateral_deviation
                        > float(
                            args.phone_waypoint3_max_lateral_deviation_m
                        )
                    )
                    loop_rejected = bool(
                        (
                            waypoint_name == "waypoint3"
                            and float(args.phone_waypoint3_max_joint_travel_rad)
                            >= 0.0
                            and max_joint_travel
                            > float(args.phone_waypoint3_max_joint_travel_rad)
                        )
                        or (
                            waypoint_name != "waypoint3"
                            and max_joint_travel > float(np.pi)
                        )
                        or (
                            waypoint_name != "waypoint3"
                            and detour_ratio > 3.0
                        )
                        or detour_rejected
                        or lateral_rejected
                        or rotation_excess_rejected
                    )
                    # The historical non-symmetry mode uses joint-7 sign as a
                    # proxy for the less troublesome wrist branch. Once the
                    # explicit 180-degree gripper symmetry is enabled, that
                    # proxy is invalid: choose purely by measured path cost.
                    branch_penalty = (
                        2.0
                        if (
                            waypoint_name == "waypoint0"
                            and not args.phone_roll_symmetry
                            and not args.dual_waypoint0_roll_select_shorter
                            and final_joint7 >= 0.0
                        )
                        else 0.0
                    )
                    score = float(
                        total_joint_travel
                        + 2.0 * rotation_length
                        + 2.0 * xyz_length
                        + branch_penalty
                    )
                    metrics = {
                        "candidate": int(candidate_index),
                        "score": score,
                        "xyz_m": xyz_length,
                        "direct_m": direct_distance,
                        "detour_ratio": detour_ratio,
                        "max_lateral_deviation_m": max_lateral_deviation,
                        "detour_rejected": detour_rejected,
                        "lateral_rejected": lateral_rejected,
                        "rotation_rad": rotation_length,
                        "direct_rotation_rad": direct_rotation,
                        "rotation_excess_rad": rotation_excess,
                        "rotation_excess_limit_rad": float(
                            args.path_rotation_excess_limit_rad
                        ),
                        "rotation_excess_rejected": rotation_excess_rejected,
                        "total_joint_travel_rad": total_joint_travel,
                        "max_joint_travel_rad": max_joint_travel,
                        "final_joint7": final_joint7,
                        "branch_penalty": branch_penalty,
                        "points": int(len(configurations)),
                        "loop_rejected": loop_rejected,
                    }
                    print(
                        "[phone-best-of-n-candidate] waypoint="
                        + waypoint_name
                        + " metrics="
                        + json.dumps(metrics, sort_keys=True),
                        flush=True,
                    )
                    if not loop_rejected:
                        candidates.append((score, path, metrics))
                except ConfigurationPathError as exc:
                    failures.append(
                        {"candidate": int(candidate_index), "error": str(exc)}
                    )
                finally:
                    arm.set_joint_positions(
                        start_joints.tolist(), disable_dynamics=True
                    )

            if not candidates:
                print(
                    "[phone-best-of-n-no-acceptable-path] waypoint="
                    + waypoint_name
                    + " failures="
                    + json.dumps(failures, sort_keys=True),
                    flush=True,
                )
                raise ConfigurationPathError(
                    "All collision-aware candidates failed or were loop-rejected "
                    "for " + waypoint_name
                )
            selection_pool = candidates
            if waypoint_name == "waypoint0" and not args.phone_roll_symmetry:
                negative_branch_candidates = [
                    item
                    for item in candidates
                    if float(item[2]["final_joint7"]) < 0.0
                ]
                if negative_branch_candidates:
                    selection_pool = negative_branch_candidates
            _, selected_path, selected_metrics = min(
                selection_pool,
                key=lambda item: float(item[0]),
            )
            print(
                "[phone-best-of-n-selected] waypoint="
                + waypoint_name
                + " accepted="
                + str(len(candidates))
                + "/"
                + str(candidate_count)
                + " selection_pool="
                + str(len(selection_pool))
                + " metrics="
                + json.dumps(selected_metrics, sort_keys=True),
                flush=True,
            )
            return selected_path

        def quaternion_multiply_xyzw(left, right):
            """Hamilton product for PyRep's [x, y, z, w] quaternions."""
            x1, y1, z1, w1 = map(float, left)
            x2, y2, z2, w2 = map(float, right)
            return np.asarray(
                [
                    w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                    w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                    w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
                    w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                ],
                dtype=np.float64,
            )

        def quaternion_geodesic_angle(left, right):
            left = np.asarray(left, dtype=np.float64)
            right = np.asarray(right, dtype=np.float64)
            left /= np.linalg.norm(left)
            right /= np.linalg.norm(right)
            return float(
                2.0
                * np.arccos(np.clip(abs(float(np.dot(left, right))), -1.0, 1.0))
            )

        def phone_best_of_n_point_get_path(point, ignore_collisions=False):
            if not args.phone_roll_symmetry:
                return phone_best_of_n_selected_pose_get_path(
                    point, ignore_collisions
                )

            arm = point._robot.arm
            waypoint_name = point._waypoint.get_name()
            current_quaternion = np.asarray(
                arm.get_tip().get_quaternion(), dtype=np.float64
            )
            original_quaternion = np.asarray(
                point._waypoint.get_quaternion(), dtype=np.float64
            )
            # Right multiplication applies the half-turn in the target's local
            # tool frame. Local Z is the REAP/Panda approach axis, so this swaps
            # the two parallel fingers without changing the target TCP or its
            # approach direction.
            finger_swapped_quaternion = quaternion_multiply_xyzw(
                original_quaternion, [0.0, 0.0, 1.0, 0.0]
            )
            finger_swapped_quaternion /= np.linalg.norm(
                finger_swapped_quaternion
            )
            original_angle = quaternion_geodesic_angle(
                current_quaternion, original_quaternion
            )
            finger_swapped_angle = quaternion_geodesic_angle(
                current_quaternion, finger_swapped_quaternion
            )
            use_finger_swapped = finger_swapped_angle + 1e-9 < original_angle
            selected_quaternion = (
                finger_swapped_quaternion
                if use_finger_swapped
                else original_quaternion
            )
            print(
                "[phone-roll-symmetry-selected] waypoint="
                + waypoint_name
                + " route="
                + ("finger_swapped" if use_finger_swapped else "original")
                + " original_angle_rad="
                + format(original_angle, ".6f")
                + " finger_swapped_angle_rad="
                + format(finger_swapped_angle, ".6f")
                + " selected_angle_rad="
                + format(min(original_angle, finger_swapped_angle), ".6f"),
                flush=True,
            )
            point._waypoint.set_quaternion(selected_quaternion.tolist())
            try:
                return phone_best_of_n_selected_pose_get_path(
                    point, ignore_collisions
                )
            finally:
                # Waypoint objects are reused by RLBench. Restore the authored
                # scene orientation so each call compares the same two poses.
                point._waypoint.set_quaternion(original_quaternion.tolist())

        Point.get_path = phone_best_of_n_point_get_path
    elif args.expert_path_mode == "phone_baseframe_reference_ik_0_3":
        from scipy.optimize import least_squares
        from scipy.spatial.transform import Rotation, Slerp

        reference_root = Path(
            args.phone_smooth_ik_reference_root
        ).expanduser().resolve()
        excluded_reference_episodes = set(
            map(int, args.phone_smooth_ik_exclude_episodes)
        )
        reference_bank = {"waypoint0": [], "waypoint3": []}
        reference_paths = sorted(
            reference_root.glob("phone_on_base__episode_*/arrays.npz")
        )
        if not reference_paths:
            raise FileNotFoundError(
                "No phone episode arrays found under smooth-IK reference root: "
                + str(reference_root)
            )
        for reference_path in reference_paths:
            try:
                reference_episode = int(reference_path.parent.name.rsplit("_", 1)[-1])
            except ValueError:
                continue
            if reference_episode in excluded_reference_episodes:
                continue
            with np.load(reference_path, allow_pickle=False) as reference_arrays:
                names = [str(value) for value in reference_arrays["waypoint_end_names"]]
                ends = [int(value) for value in reference_arrays["waypoint_end_frames"]]
                base_targets = np.asarray(
                    reference_arrays["world_base_action_target_ee_poses"],
                    dtype=np.float64,
                )
                raw_actions = np.asarray(
                    reference_arrays["raw_expert_actions_full"],
                    dtype=np.float64,
                )
                for reference_waypoint in reference_bank:
                    if reference_waypoint not in names:
                        continue
                    endpoint = min(
                        ends[names.index(reference_waypoint)],
                        len(base_targets) - 1,
                        len(raw_actions) - 1,
                    )
                    joints = raw_actions[endpoint, :7]
                    pose9 = base_targets[endpoint, :9]
                    if np.isfinite(joints).all() and np.isfinite(pose9).all():
                        reference_bank[reference_waypoint].append(
                            (reference_episode, pose9.copy(), joints.copy())
                        )
        if any(not rows for rows in reference_bank.values()):
            raise RuntimeError(
                "Smooth-IK reference bank is incomplete: "
                + json.dumps(
                    {key: len(value) for key, value in reference_bank.items()},
                    sort_keys=True,
                )
            )
        print(
            "[phone-baseframe-reference-bank] root="
            + str(reference_root)
            + " excluded="
            + json.dumps(sorted(excluded_reference_episodes))
            + " counts="
            + json.dumps(
                {key: len(value) for key, value in reference_bank.items()},
                sort_keys=True,
            ),
            flush=True,
        )

        def pose9_rotation(pose9):
            first = np.asarray(pose9[3:6], dtype=np.float64)
            second = np.asarray(pose9[6:9], dtype=np.float64)
            first /= np.linalg.norm(first)
            second -= first * float(np.dot(first, second))
            second /= np.linalg.norm(second)
            return np.stack((first, second, np.cross(first, second)), axis=1)

        def baseframe_reference_pose_distance(first, second):
            xyz = float(np.linalg.norm(first[:3] - second[:3]))
            relative = pose9_rotation(first).T @ pose9_rotation(second)
            angle = float(
                np.arccos(
                    np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0)
                )
            )
            return xyz + 0.10 * angle, xyz, angle

        def phone_baseframe_reference_ik_get_path(point, ignore_collisions=False):
            """Refine a known normal Panda branch to the exact base-frame pose."""
            nonlocal linear_path_calls
            waypoint_name = point._waypoint.get_name()
            arm = point._robot.arm
            flipped_branch_intermediate = (
                waypoint_name in {"waypoint1", "waypoint2"}
                and args.phone_later_waypoint_roll_policy
                == "opposite_latched"
                and forced_waypoint0_roll_branch is not None
            )
            if (
                waypoint_name not in {"waypoint0", "waypoint3", "waypoint4"}
                and not flipped_branch_intermediate
            ):
                linear_path_calls += 1
                return arm.get_linear_path(
                    point._waypoint.get_position(),
                    quaternion=point._waypoint.get_quaternion(),
                    ignore_collisions=(point._ignore_collisions or ignore_collisions),
                )

            start_joints = np.asarray(arm.get_joint_positions(), dtype=np.float64)
            start_position = np.asarray(
                arm.get_tip().get_position(), dtype=np.float64
            )
            start_quaternion = np.asarray(
                arm.get_tip().get_quaternion(), dtype=np.float64
            )
            start_quaternion /= np.linalg.norm(start_quaternion)
            target_position = np.asarray(
                point._waypoint.get_position(), dtype=np.float64
            )
            target_quaternion = np.asarray(
                point._waypoint.get_quaternion(), dtype=np.float64
            )
            target_quaternion /= np.linalg.norm(target_quaternion)
            target_world = np.eye(4, dtype=np.float64)
            target_world[:3, :3] = Rotation.from_quat(
                target_quaternion
            ).as_matrix()
            target_world[:3, 3] = target_position
            t_world_base = rlbench_panda_link0_to_world_matrix(arm)
            target_base = np.linalg.inv(t_world_base) @ target_world
            target_base_pose9 = matrix_to_pose9(target_base).astype(np.float64)

            ranked_references = sorted(
                (
                    baseframe_reference_pose_distance(target_base_pose9, pose9)
                    + (episode, joints)
                    for episode, pose9, joints in reference_bank.get(
                        waypoint_name, []
                    )
                ),
                key=lambda item: item[:3],
            )
            goal_candidates = []
            refinement_failures = []
            cyclic_joints, joint_intervals = arm.get_joint_intervals()
            joint_lower = np.asarray(
                [
                    -math.pi if cyclic else interval[0]
                    for cyclic, interval in zip(
                        cyclic_joints, joint_intervals, strict=True
                    )
                ],
                dtype=np.float64,
            )
            joint_upper = np.asarray(
                [
                    math.pi if cyclic else interval[0] + interval[1]
                    for cyclic, interval in zip(
                        cyclic_joints, joint_intervals, strict=True
                    )
                ],
                dtype=np.float64,
            )
            joint_span = np.maximum(joint_upper - joint_lower, 1e-6)
            try:
                # Nearby successful episodes provide the intended elbow/wrist
                # branch. Jacobian IK changes that seed only enough to satisfy
                # this scene's exact waypoint pose.
                for pose_score, xyz_error, angle_error, episode, seed_joints in (
                    ranked_references[:0]
                ):
                    seed_joints = np.clip(
                        np.asarray(seed_joints, dtype=np.float64),
                        joint_lower + 1e-5,
                        joint_upper - 1e-5,
                    )
                    # Coppelia's local Jacobian solver frequently declares a
                    # near-base target singular even when the clean reference
                    # elbow branch has a nearby exact solution.  A bounded
                    # least-squares IK keeps the redundant seventh DOF close
                    # to that known-normal posture while solving the exact
                    # live Cartesian target.
                    def reference_regularized_residual(joints):
                        arm.set_joint_positions(
                            np.asarray(joints, dtype=np.float64).tolist(),
                            disable_dynamics=True,
                        )
                        position = np.asarray(
                            arm.get_tip().get_position(), dtype=np.float64
                        )
                        quaternion = np.asarray(
                            arm.get_tip().get_quaternion(), dtype=np.float64
                        )
                        quaternion /= np.linalg.norm(quaternion)
                        rotation_error = (
                            Rotation.from_quat(quaternion).inv()
                            * Rotation.from_quat(target_quaternion)
                        ).as_rotvec()
                        normalized_seed_delta = (
                            np.asarray(joints, dtype=np.float64) - seed_joints
                        ) / joint_span
                        return np.concatenate(
                            (
                                (position - target_position) / 0.002,
                                rotation_error / 0.02,
                                0.015 * normalized_seed_delta,
                            )
                        )

                    try:
                        regularized_result = least_squares(
                            reference_regularized_residual,
                            seed_joints,
                            bounds=(joint_lower + 1e-6, joint_upper - 1e-6),
                            method="trf",
                            x_scale="jac",
                            diff_step=1e-4,
                            ftol=1e-11,
                            xtol=1e-11,
                            gtol=1e-11,
                            max_nfev=1,
                        )
                    except Exception as optimization_error:
                        refinement_failures.append(
                            {
                                "episode": int(episode),
                                "regularized_ik_error": repr(optimization_error),
                            }
                        )
                    else:
                        regularized_goal = np.asarray(
                            regularized_result.x, dtype=np.float64
                        )
                        goal_candidates.append(
                            (
                                "reference_regularized",
                                int(episode),
                                float(pose_score),
                                float(xyz_error),
                                float(angle_error),
                                regularized_goal,
                            )
                        )
                    arm.set_joint_positions(
                        seed_joints.tolist(),
                        disable_dynamics=True,
                    )
                    try:
                        seed_position = np.asarray(
                            arm.get_tip().get_position(), dtype=np.float64
                        )
                        seed_quaternion = np.asarray(
                            arm.get_tip().get_quaternion(), dtype=np.float64
                        )
                        seed_quaternion /= np.linalg.norm(seed_quaternion)
                        homotopy_target_quaternion = target_quaternion.copy()
                        if float(
                            np.dot(seed_quaternion, homotopy_target_quaternion)
                        ) < 0.0:
                            homotopy_target_quaternion *= -1.0
                        homotopy_slerp = Slerp(
                            [0.0, 1.0],
                            Rotation.from_quat(
                                np.stack(
                                    (
                                        seed_quaternion,
                                        homotopy_target_quaternion,
                                    )
                                )
                            ),
                        )
                        refined = np.asarray(seed_joints, dtype=np.float64)
                        for fraction, quaternion in zip(
                            np.linspace(0.0, 1.0, 13)[1:],
                            homotopy_slerp(
                                np.linspace(0.0, 1.0, 13)[1:]
                            ).as_quat(),
                            strict=True,
                        ):
                            arm.set_joint_positions(
                                refined.tolist(), disable_dynamics=True
                            )
                            refined = np.asarray(
                                arm.solve_ik_via_jacobian(
                                    (1.0 - fraction) * seed_position
                                    + fraction * target_position,
                                    quaternion=quaternion,
                                ),
                                dtype=np.float64,
                            )
                    except IKError as error:
                        # Near the base the Panda often sits close to a
                        # singularity, so local Jacobian continuation can fail
                        # even when the desired normal branch exists. Re-run
                        # Coppelia's endpoint sampler while the arm is placed
                        # at the clean reference posture; returned solutions
                        # are then ranked relative to that posture, not home.
                        arm.set_joint_positions(
                            np.asarray(seed_joints, dtype=np.float64).tolist(),
                            disable_dynamics=True,
                        )
                        try:
                            refined_samples = arm.solve_ik_via_sampling(
                                target_position,
                                quaternion=target_quaternion,
                                ignore_collisions=False,
                                trials=800,
                                max_configs=20,
                                distance_threshold=0.8,
                                max_time_ms=50,
                            )
                        except ConfigurationError as sampling_error:
                            refinement_failures.append(
                                {
                                    "episode": int(episode),
                                    "jacobian_error": repr(error),
                                    "sampling_error": repr(sampling_error),
                                }
                            )
                            continue
                        for local_index, refined_sample in enumerate(
                            np.asarray(refined_samples, dtype=np.float64)[:3]
                        ):
                            goal_candidates.append(
                                (
                                    "reference_sampling",
                                    int(episode * 10 + local_index),
                                    float(pose_score),
                                    float(xyz_error),
                                    float(angle_error),
                                    refined_sample,
                                )
                            )
                    else:
                        goal_candidates.append(
                            (
                                "reference_homotopy",
                                int(episode),
                                float(pose_score),
                                float(xyz_error),
                                float(angle_error),
                                refined,
                            )
                        )
                arm.set_joint_positions(
                    start_joints.tolist(), disable_dynamics=True
                )
                # Once waypoint1 closes the gripper, RLBench rigidly attaches
                # the handset to it.  Treating that intentionally grasped
                # object as an obstacle makes Coppelia's endpoint sampler
                # return zero IK configurations for waypoint2 onward.  Hide
                # only the grasped object while sampling: robot self-collision
                # and collisions with every other scene object remain active.
                sampling_held_states = []
                try:
                    for held_object in point._robot.gripper.get_grasped_objects():
                        was_collidable = bool(held_object.is_collidable())
                        sampling_held_states.append((held_object, was_collidable))
                        if was_collidable:
                            held_object.set_collidable(False)
                    try:
                        sampled = arm.solve_ik_via_sampling(
                            target_position,
                            quaternion=target_quaternion,
                            ignore_collisions=False,
                            trials=1600,
                            max_configs=100,
                            distance_threshold=0.8,
                            max_time_ms=100,
                        )
                    except ConfigurationError:
                        sampled = []
                    if len(sampled) == 0:
                        # Near the grasp pose Coppelia's sampler can reject
                        # every endpoint because its coarse collision query
                        # still includes contact geometry belonging to the
                        # attached handset.  Broaden endpoint enumeration only;
                        # every interpolated configuration is collision-checked
                        # explicitly below before it can be selected.
                        print(
                            "[phone-baseframe-reference-ik-endpoint-fallback] "
                            "waypoint=" + waypoint_name
                            + " mode=sample_without_endpoint_collision_filter"
                            + " grasped_objects="
                            + str(len(sampling_held_states)),
                            flush=True,
                        )
                        try:
                            sampled = arm.solve_ik_via_sampling(
                                target_position,
                                quaternion=target_quaternion,
                                ignore_collisions=True,
                                trials=2400,
                                max_configs=120,
                                distance_threshold=0.8,
                                max_time_ms=100,
                            )
                        except ConfigurationError:
                            sampled = []
                finally:
                    for held_object, was_collidable in sampling_held_states:
                        held_object.set_collidable(was_collidable)
                for sampled_index, sampled_joints in enumerate(
                    np.asarray(sampled, dtype=np.float64).reshape(-1, 7)
                ):
                    goal_candidates.append(
                        (
                            "sampling",
                            int(sampled_index),
                            float("inf"),
                            float("inf"),
                            float("inf"),
                            sampled_joints,
                        )
                    )
            finally:
                arm.set_joint_positions(
                    start_joints.tolist(), disable_dynamics=True
                )

            unique_goals = []
            for goal in goal_candidates:
                if any(
                    np.max(np.abs(goal[5] - previous[5])) < 1e-4
                    for previous in unique_goals
                ):
                    continue
                unique_goals.append(goal)

            # Solve several short, explicit Cartesian homotopies from the
            # *actual* current Panda configuration.  This is the important
            # difference from choosing an endpoint IK and linearly blending
            # joints: the redundant elbow/wrist branch is propagated at every
            # pose, so it cannot silently jump to a folded solution.  Small
            # lifted curves cover near-base targets for which the exact chord
            # crosses a singular or colliding posture.
            precomputed_paths = {}
            base_origin_world = np.asarray(t_world_base[:3, 3], dtype=np.float64)
            radial_xy = target_position[:2] - base_origin_world[:2]
            radial_norm = float(np.linalg.norm(radial_xy))
            if radial_norm > 1e-9:
                radial_xy /= radial_norm
            else:
                radial_xy = np.asarray([1.0, 0.0], dtype=np.float64)
            # waypoint3 is the short loaded-handset transfer where selecting
            # an endpoint IK and blending joints can make the EEF draw a large
            # arc even though the two semantic poses are only centimetres
            # apart.  Propagate the current IK branch along the exact Cartesian
            # chord instead.  Other waypoints keep the faster sampled-endpoint
            # solver, for which the chord has already proved well behaved.
            if waypoint_name == "waypoint0":
                # For the long initial approach, propagate one continuous IK
                # branch along the Cartesian chord while the finger-swapped
                # orientation is reached via shortest-arc SLERP.  Selecting
                # only an endpoint IK and interpolating joints can otherwise
                # make the TCP accumulate almost a full extra revolution.
                curve_profiles = (
                    ("direct", 0.0, 0.0),
                    ("lift20", 0.020, 0.0),
                    ("radial20", 0.0, 0.020),
                    ("lift20_radial20", 0.020, 0.020),
                )
            elif (
                waypoint_name == "waypoint1"
                and args.phone_later_waypoint_roll_policy
                == "opposite_latched"
            ):
                # Perform the deliberate finger swap continuously while the
                # gripper travels from the pre-grasp waypoint to the grasp
                # waypoint.  This avoids an in-place 180-degree turn and also
                # handles scenes where RLBench's stock linear path cannot
                # keep a continuous Panda wrist/elbow branch.
                curve_profiles = (("direct", 0.0, 0.0),)
            elif waypoint_name == "waypoint3":
                # A latched 180-degree carry branch can meet the Panda wrist
                # limit at the far end of the exact Cartesian chord.
                # Try small, explicit task-space arcs that move away from the
                # base/singularity while retaining shortest-arc quaternion
                # SLERP.  These are still continuous IK paths and are subject
                # to the same collision, detour and accumulated-rotation
                # checks below; no sampling planner/RRT is involved.
                curve_profiles = (
                    ("direct", 0.0, 0.0),
                    ("lift20", 0.020, 0.0),
                    ("radial20", 0.0, 0.020),
                    ("lift20_radial20", 0.020, 0.020),
                    ("lift40_radial30", 0.040, 0.030),
                )
            elif waypoint_name in {"waypoint3", "waypoint4"}:
                curve_profiles = (("direct", 0.0, 0.0),)
            else:
                curve_profiles = ()
            curve_times = np.linspace(0.0, 1.0, 61)
            curve_slerp = Slerp(
                [0.0, 1.0],
                Rotation.from_quat(
                    np.stack((start_quaternion, target_quaternion))
                ),
            )
            for curve_index, (curve_name, lift_m, radial_m) in enumerate(
                curve_profiles
            ):
                previous_joints = start_joints.copy()
                curve_configurations = []
                curve_failure = None
                for fraction, desired_quaternion in zip(
                    curve_times[1:],
                    curve_slerp(curve_times[1:]).as_quat(),
                    strict=True,
                ):
                    arc_scale = 4.0 * fraction * (1.0 - fraction)
                    if waypoint_name == "waypoint0":
                        # Finish the finger swap during the spacious first
                        # part of the long approach.  Holding the final tool
                        # orientation over the last part lets the redundant
                        # elbow/wrist branch settle before reaching the
                        # near-base pre-grasp pose, instead of concentrating
                        # the turn in its singular neighbourhood.
                        turn_end = 0.65
                        desired_quaternion = curve_slerp(
                            [min(1.0, fraction / turn_end)]
                        ).as_quat()[0]
                        desired_position = (
                            (1.0 - fraction) * start_position
                            + fraction * target_position
                        )
                    elif (
                        waypoint_name == "waypoint1"
                        and args.phone_later_waypoint_roll_policy
                        == "opposite_latched"
                    ):
                        # Complete the deliberate 180-degree finger swap while
                        # still safely above the handset, then descend with a
                        # fixed orientation.  A small lift means this is a
                        # genuine moving turn rather than an in-place spin.
                        turn_end = 0.55
                        if fraction <= turn_end:
                            local = fraction / turn_end
                            move_fraction = 0.10 * local
                            extra_lift = 0.020 * math.sin(math.pi * local)
                            desired_quaternion = curve_slerp(
                                [min(1.0, local)]
                            ).as_quat()[0]
                        else:
                            local = (fraction - turn_end) / (1.0 - turn_end)
                            move_fraction = 0.10 + 0.90 * local
                            extra_lift = 0.0
                            desired_quaternion = target_quaternion
                        desired_position = (
                            (1.0 - move_fraction) * start_position
                            + move_fraction * target_position
                        )
                        desired_position = desired_position.copy()
                        desired_position[2] += extra_lift
                    else:
                        desired_position = (
                            (1.0 - fraction) * start_position
                            + fraction * target_position
                        )
                    desired_position = desired_position.copy()
                    desired_position[2] += lift_m * arc_scale
                    desired_position[:2] += radial_m * arc_scale * radial_xy

                    def continuation_residual(joints):
                        arm.set_joint_positions(
                            np.asarray(joints, dtype=np.float64).tolist(),
                            disable_dynamics=True,
                        )
                        position = np.asarray(
                            arm.get_tip().get_position(), dtype=np.float64
                        )
                        quaternion = np.asarray(
                            arm.get_tip().get_quaternion(), dtype=np.float64
                        )
                        quaternion /= np.linalg.norm(quaternion)
                        rotation_error = (
                            Rotation.from_quat(quaternion).inv()
                            * Rotation.from_quat(desired_quaternion)
                        ).as_rotvec()
                        return np.concatenate(
                            (
                                (position - desired_position) / 0.001,
                                rotation_error / 0.01,
                                0.0002
                                * (np.asarray(joints) - previous_joints)
                                / joint_span,
                            )
                        )

                    try:
                        result = least_squares(
                            continuation_residual,
                            np.clip(
                                previous_joints,
                                joint_lower + 2e-6,
                                joint_upper - 2e-6,
                            ),
                            bounds=(joint_lower + 1e-6, joint_upper - 1e-6),
                            method="trf",
                            x_scale="jac",
                            diff_step=1e-4,
                            ftol=1e-10,
                            xtol=1e-10,
                            gtol=1e-10,
                            max_nfev=100,
                        )
                    except Exception as error:
                        curve_failure = repr(error)
                        break
                    solved_joints = np.asarray(result.x, dtype=np.float64)
                    arm.set_joint_positions(
                        solved_joints.tolist(), disable_dynamics=True
                    )
                    try:
                        solved_joints = np.asarray(
                            arm.solve_ik_via_jacobian(
                                desired_position,
                                quaternion=desired_quaternion,
                            ),
                            dtype=np.float64,
                        )
                    except IKError:
                        # The bounded optimizer remains the valid fallback;
                        # the explicit residual checks below decide whether
                        # its approximation is accurate enough to retain.
                        pass
                    solved_joints = np.clip(
                        solved_joints,
                        joint_lower + 1e-6,
                        joint_upper - 1e-6,
                    )
                    arm.set_joint_positions(
                        solved_joints.tolist(), disable_dynamics=True
                    )
                    solved_position = np.asarray(
                        arm.get_tip().get_position(), dtype=np.float64
                    )
                    solved_quaternion = np.asarray(
                        arm.get_tip().get_quaternion(), dtype=np.float64
                    )
                    solved_quaternion /= np.linalg.norm(solved_quaternion)
                    position_error = float(
                        np.linalg.norm(solved_position - desired_position)
                    )
                    rotation_error = float(
                        2.0
                        * np.arccos(
                            np.clip(
                                abs(
                                    float(
                                        np.dot(
                                            solved_quaternion,
                                            desired_quaternion,
                                        )
                                    )
                                ),
                                -1.0,
                                1.0,
                            )
                        )
                    )
                    joint_step = float(
                        np.max(np.abs(solved_joints - previous_joints))
                    )
                    continuation_rotation_tolerance = (
                        0.15
                        if waypoint_name in {"waypoint3", "waypoint4"}
                        and args.phone_later_waypoint_roll_policy
                        == "opposite_latched"
                        else (
                            0.12
                            if waypoint_name == "waypoint3"
                            else 0.08
                            if waypoint_name == "waypoint0"
                            else 0.04
                        )
                    )
                    continuation_joint_step_tolerance = (
                        0.30
                        if waypoint_name == "waypoint1"
                        and args.phone_later_waypoint_roll_policy
                        == "opposite_latched"
                        else 0.20
                    )
                    if (
                        position_error > 0.003
                        or rotation_error > continuation_rotation_tolerance
                        or joint_step > continuation_joint_step_tolerance
                    ):
                        curve_failure = (
                            f"fraction={fraction:.4f} pos={position_error:.6f} "
                            f"rot={rotation_error:.6f} joint_step={joint_step:.6f}"
                        )
                        break
                    curve_configurations.append(solved_joints.copy())
                    previous_joints = solved_joints
                # A Cartesian continuation can reach the singular target
                # neighbourhood yet miss the exact final orientation by a few
                # hundredths of a radian.  Finish only that short tail with
                # the closest exact sampled IK configuration.  The resulting
                # bridge is still checked below for collisions, joint motion,
                # Cartesian detour and endpoint accuracy.
                if (
                    curve_failure is not None
                    and len(curve_configurations) >= 48
                    and waypoint_name != "waypoint0"
                ):
                    arm.set_joint_positions(
                        previous_joints.tolist(), disable_dynamics=True
                    )
                    try:
                        exact_goals = arm.solve_ik_via_sampling(
                            target_position,
                            quaternion=target_quaternion,
                            ignore_collisions=False,
                            trials=1200,
                            max_configs=40,
                            distance_threshold=0.5,
                            max_time_ms=100,
                        )
                    except ConfigurationError:
                        exact_goals = []
                    exact_goals = np.asarray(
                        exact_goals, dtype=np.float64
                    ).reshape(-1, 7)
                    if len(exact_goals):
                        exact_goal = min(
                            exact_goals,
                            key=lambda candidate: float(
                                np.linalg.norm(candidate - previous_joints)
                            ),
                        )
                        tail_delta = exact_goal - previous_joints
                        tail_steps = max(
                            2,
                            int(
                                math.ceil(
                                    float(np.max(np.abs(tail_delta))) / 0.015
                                )
                            ),
                        )
                        tail = (
                            previous_joints[None]
                            + np.linspace(0.0, 1.0, tail_steps + 1)[1:, None]
                            * tail_delta[None]
                        )
                        curve_configurations.extend(tail)
                        previous_joints = exact_goal
                        curve_failure = None
                arm.set_joint_positions(
                    start_joints.tolist(), disable_dynamics=True
                )
                if curve_failure is not None or not curve_configurations:
                    refinement_failures.append(
                        {
                            "curve": curve_name,
                            "continuation_error": curve_failure,
                            "points": len(curve_configurations),
                        }
                    )
                    continue
                source = "cartesian_continuation_" + curve_name
                source_index = int(curve_index)
                configurations = np.asarray(
                    curve_configurations, dtype=np.float64
                )
                unique_goals.append(
                    (
                        source,
                        source_index,
                        0.0,
                        0.0,
                        0.0,
                        configurations[-1].copy(),
                    )
                )
                precomputed_paths[(source, source_index)] = configurations

            held_states = []
            accepted = []
            rejected = []
            try:
                for held_object in point._robot.gripper.get_grasped_objects():
                    was_collidable = bool(held_object.is_collidable())
                    held_states.append((held_object, was_collidable))
                    if was_collidable:
                        held_object.set_collidable(False)
                for source, source_index, pose_score, ref_xyz, ref_angle, goal in unique_goals:
                    precomputed = precomputed_paths.get((source, source_index))
                    if precomputed is None:
                        delta = goal - start_joints
                        endpoint_max_delta = float(
                            np.max(np.abs(delta), initial=0.0)
                        )
                        step_count = max(
                            2, int(math.ceil(endpoint_max_delta / 0.020))
                        )
                        configurations = (
                            start_joints[None]
                            + np.linspace(0.0, 1.0, step_count + 1)[1:, None]
                            * delta[None]
                        )
                    else:
                        configurations = precomputed
                    joint_path = np.concatenate(
                        (start_joints[None], configurations), axis=0
                    )
                    joint_deltas = np.abs(np.diff(joint_path, axis=0))
                    max_delta = float(joint_deltas.max(initial=0.0))
                    positions = [start_position.copy()]
                    quaternions = [start_quaternion.copy()]
                    collision = False
                    for configuration in configurations:
                        arm.set_joint_positions(
                            configuration.tolist(), disable_dynamics=True
                        )
                        if arm.check_arm_collision():
                            collision = True
                            break
                        positions.append(
                            np.asarray(
                                arm.get_tip().get_position(), dtype=np.float64
                            )
                        )
                        quaternion = np.asarray(
                            arm.get_tip().get_quaternion(), dtype=np.float64
                        )
                        quaternion /= np.linalg.norm(quaternion)
                        quaternions.append(quaternion)
                    arm.set_joint_positions(
                        start_joints.tolist(), disable_dynamics=True
                    )
                    if collision or len(positions) != len(configurations) + 1:
                        rejected.append(
                            {"source": source, "index": source_index, "reason": "collision"}
                        )
                        continue
                    positions = np.asarray(positions, dtype=np.float64)
                    quaternions = np.asarray(quaternions, dtype=np.float64)
                    endpoint_position_error = float(
                        np.linalg.norm(positions[-1] - target_position)
                    )
                    endpoint_rotation_error = float(
                        2.0
                        * np.arccos(
                            np.clip(
                                abs(float(np.dot(quaternions[-1], target_quaternion))),
                                -1.0,
                                1.0,
                            )
                        )
                    )
                    chord = target_position - start_position
                    direct_distance = float(np.linalg.norm(chord))
                    path_length = float(
                        np.linalg.norm(np.diff(positions, axis=0), axis=1).sum()
                    )
                    if direct_distance > 1e-9:
                        unit = chord / direct_distance
                        progress = (positions - start_position) @ unit
                        closest = start_position + progress[:, None] * unit
                        lateral = float(
                            np.linalg.norm(positions - closest, axis=1).max(initial=0.0)
                        )
                        backtrack = float(
                            np.maximum(0.0, -np.diff(progress)).sum()
                        )
                    else:
                        lateral = path_length
                        backtrack = path_length
                    rotation = float(
                        np.sum(
                            2.0
                            * np.arccos(
                                np.clip(
                                    np.abs(
                                        np.sum(
                                            quaternions[:-1] * quaternions[1:], axis=1
                                        )
                                    ),
                                    -1.0,
                                    1.0,
                                )
                            )
                        )
                    )
                    joint_total = float(joint_deltas.sum())
                    wrist_total = float(joint_deltas[:, -3:].sum())
                    detour_ratio = float(path_length / max(direct_distance, 1e-9))
                    goal_joint_limit_margin = float(
                        np.minimum(
                            goal - joint_lower, joint_upper - goal
                        ).min()
                    )
                    if waypoint_name == "waypoint0":
                        taskspace_continuation = source.startswith(
                            "cartesian_continuation_"
                        )
                        limits = {
                            # A close-to-base pickup distributes motion over
                            # all seven Panda joints.  Across the flagged
                            # scenes, clean EEF paths legitimately total up to
                            # about 12 rad even though no individual joint
                            # jumps and the TCP detour stays below 1.6x.
                            # For an explicitly propagated task-space curve,
                            # cumulative multi-joint travel is not evidence of
                            # a TCP loop by itself.  Keep strict geometric and
                            # per-step limits, while allowing the redundant
                            # Panda joints to redistribute along the curve.
                            "joint_total": (
                                20.0 if taskspace_continuation else 12.0
                            ),
                            "wrist_total": (
                                9.0 if taskspace_continuation else 7.0
                            ),
                            "max_delta": 2.80,
                            "detour_ratio": (
                                1.50 if taskspace_continuation else 1.85
                            ),
                            "lateral": (
                                0.18 if taskspace_continuation else 0.35
                            ),
                            "backtrack": 0.10,
                            "rotation": (
                                2.50 if taskspace_continuation else 3.80
                            ),
                        }
                    elif waypoint_name == "waypoint1" and (
                        args.phone_later_waypoint_roll_policy
                        == "opposite_latched"
                    ):
                        limits = {
                            "joint_total": 15.0,
                            "wrist_total": 7.5,
                            "max_delta": 2.2,
                            "detour_ratio": 3.0,
                            "lateral": 0.10,
                            "backtrack": 0.06,
                            "rotation": 3.70,
                        }
                    elif waypoint_name == "waypoint2" and (
                        args.phone_later_waypoint_roll_policy
                        == "opposite_latched"
                    ):
                        limits = {
                            "joint_total": 9.0,
                            "wrist_total": 6.0,
                            "max_delta": 2.2,
                            "detour_ratio": 2.0,
                            "lateral": 0.10,
                            "backtrack": 0.06,
                            "rotation": 4.20,
                        }
                    elif waypoint_name in {"waypoint3", "waypoint4"}:
                        # The loaded-handset transfer has to leave the very
                        # compact near-base grasp branch.  Its clean solutions
                        # are slightly longer and bow out by about 12 cm, but
                        # still have monotonic progress and sub-0.02-rad joint
                        # samples.  Keep those valid exits instead of forcing
                        # a planner loop merely to satisfy the generic bound.
                        latched_waypoint3 = (
                            waypoint_name == "waypoint3"
                            and args.phone_later_waypoint_roll_policy
                            == "latched"
                        )
                        limits = {
                            "joint_total": (
                                7.8
                                if latched_waypoint3
                                else (
                                    8.0
                                    if waypoint_name == "waypoint3"
                                    else 7.5
                                )
                            ),
                            "wrist_total": (
                                5.8 if latched_waypoint3 else 5.2
                            ),
                            "max_delta": 2.2,
                            "detour_ratio": (
                                2.5 if latched_waypoint3 else 2.0
                            ),
                            "lateral": 0.13,
                            "backtrack": (
                                0.07 if latched_waypoint3 else 0.06
                            ),
                            # waypoint4 descends and releases the handset. Its
                            # authored target differs from the compact carry
                            # posture by about two radians even on a clean
                            # path; this is required orientation change, not a
                            # redundant full-arm loop.
                            "rotation": (
                                1.80
                                if latched_waypoint3
                                else (
                                    2.40
                                    if waypoint_name == "waypoint4"
                                    else 1.50
                                )
                            ),
                        }
                    else:
                        limits = {
                            "joint_total": 6.5,
                            "wrist_total": 5.2,
                            "max_delta": 2.2,
                            "detour_ratio": 2.0,
                            "lateral": 0.10,
                            "backtrack": 0.06,
                            "rotation": 1.50,
                        }
                    reasons = []
                    if (
                        waypoint_name == "waypoint1"
                        and args.phone_later_waypoint_roll_policy
                        == "opposite_latched"
                        and not source.startswith(
                            "cartesian_continuation_"
                        )
                    ):
                        reasons.append("requires_staged_cartesian")
                    for name, value in (
                        ("joint_total", joint_total),
                        ("wrist_total", wrist_total),
                        ("max_delta", max_delta),
                        ("detour_ratio", detour_ratio),
                        ("lateral", lateral),
                        ("backtrack", backtrack),
                        ("rotation", rotation),
                    ):
                        if value > limits[name] + 1e-9:
                            reasons.append(name)
                    if endpoint_position_error > 0.003:
                        reasons.append("endpoint_position")
                    endpoint_rotation_limit = (
                        0.15
                        if waypoint_name in {"waypoint3", "waypoint4"}
                        and args.phone_later_waypoint_roll_policy
                        == "opposite_latched"
                        else 0.04
                    )
                    if endpoint_rotation_error > endpoint_rotation_limit:
                        reasons.append("endpoint_rotation")
                    metrics = {
                        "source": source,
                        "source_index": int(source_index),
                        "reference_pose_score": float(pose_score),
                        "reference_xyz_error_m": float(ref_xyz),
                        "reference_rotation_error_rad": float(ref_angle),
                        "joint_total_rad": joint_total,
                        "wrist_total_rad": wrist_total,
                        "max_joint_delta_rad": max_delta,
                        "path_length_m": path_length,
                        "direct_distance_m": direct_distance,
                        "detour_ratio": detour_ratio,
                        "max_lateral_deviation_m": lateral,
                        "progress_backtrack_m": backtrack,
                        "rotation_rad": rotation,
                        "endpoint_position_error_m": endpoint_position_error,
                        "endpoint_rotation_error_rad": endpoint_rotation_error,
                        "goal_joint_positions_rad": goal.tolist(),
                        "goal_min_joint_limit_margin_rad": goal_joint_limit_margin,
                        "points": int(len(configurations)),
                        "rejection_reasons": reasons,
                    }
                    if reasons:
                        rejected.append(metrics)
                        continue
                    joint_limit_penalty = 20.0 * max(
                        0.0, 0.15 - goal_joint_limit_margin
                    )
                    score = float(
                        joint_total
                        + 1.5 * wrist_total
                        + 4.0 * lateral
                        + 2.0 * max(0.0, path_length - direct_distance)
                        + 2.0 * backtrack
                        + 0.25 * rotation
                        + joint_limit_penalty
                    )
                    metrics["joint_limit_penalty"] = joint_limit_penalty
                    metrics["score"] = score
                    accepted.append((score, configurations.copy(), metrics))
            finally:
                for held_object, was_collidable in held_states:
                    held_object.set_collidable(was_collidable)
                arm.set_joint_positions(
                    start_joints.tolist(), disable_dynamics=True
                )
            linear_path_calls += int(len(unique_goals))
            if not accepted:
                print(
                    "[phone-baseframe-reference-ik-unavailable] waypoint="
                    + waypoint_name
                    + " candidates="
                    + str(len(unique_goals))
                    + " refinement_failures="
                    + json.dumps(refinement_failures[:3], sort_keys=True)
                    + " rejected_count="
                    + str(len(rejected))
                    + " rejected_examples="
                    + json.dumps(rejected[:2], sort_keys=True)
                    + " continuation_rejected="
                    + json.dumps(
                        [
                            item
                            for item in rejected
                            if str(item.get("source", "")).startswith(
                                "cartesian_continuation_"
                            )
                        ],
                        sort_keys=True,
                    ),
                    flush=True,
                )
                raise ConfigurationPathError(
                    "No smooth base-frame reference IK path for "
                    + waypoint_name
                    + "; candidates="
                    + str(len(unique_goals))
                    + " refinement_failures="
                    + json.dumps(refinement_failures[:3], sort_keys=True)
                    + " rejected="
                    + json.dumps(rejected[:5], sort_keys=True)
                )
            ranked_accepted = sorted(accepted, key=lambda item: item[0])
            selected = ranked_accepted[0]
            if (
                waypoint_name == "waypoint0"
                and args.phone_later_waypoint_roll_policy
                != "opposite_latched"
            ):
                # Endpoint-only ranking can choose a perfectly smooth pickup
                # IK configuration from which the required 7.5-cm descent to
                # waypoint1 has no continuous linear IK solution.  Test that
                # next semantic segment before committing to waypoint0.  This
                # is a one-step receding-horizon check: waypoint1 is still
                # static here, unlike future waypoints that move with the
                # grasped handset.
                from pyrep.objects.dummy import Dummy

                waypoint1 = Point(Dummy("waypoint1"), point._robot)
                waypoint1_position = waypoint1._waypoint.get_position()
                waypoint1_quaternion = np.asarray(
                    waypoint1._waypoint.get_quaternion(), dtype=np.float64
                )
                waypoint1_lookahead_branch = forced_waypoint0_roll_branch
                if (
                    args.phone_later_waypoint_roll_policy
                    == "opposite_latched"
                    and waypoint1_lookahead_branch is not None
                ):
                    waypoint1_lookahead_branch = (
                        "authored"
                        if waypoint1_lookahead_branch == "finger_swapped"
                        else "finger_swapped"
                    )
                if waypoint1_lookahead_branch == "finger_swapped":
                    x1, y1, z1, w1 = map(float, waypoint1_quaternion)
                    waypoint1_quaternion = np.asarray(
                        [y1, -x1, w1, -z1], dtype=np.float64
                    )
                    waypoint1_quaternion /= np.linalg.norm(
                        waypoint1_quaternion
                    )
                waypoint1_failures = []
                selected = None
                for ranked_index, candidate in enumerate(ranked_accepted[:32]):
                    arm.set_joint_positions(
                        candidate[1][-1].tolist(), disable_dynamics=True
                    )
                    try:
                        arm.get_linear_path(
                            waypoint1_position,
                            quaternion=waypoint1_quaternion.tolist(),
                            ignore_collisions=(
                                waypoint1._ignore_collisions
                                or ignore_collisions
                            ),
                        )
                    except (ConfigurationPathError, ConfigurationError) as error:
                        waypoint1_failures.append(
                            {
                                "rank": int(ranked_index),
                                "source_index": int(candidate[2]["source_index"]),
                                "error": str(error),
                            }
                        )
                        continue
                    selected = candidate
                    selected[2]["waypoint1_linear_lookahead_rank"] = int(
                        ranked_index
                    )
                    break
                arm.set_joint_positions(
                    start_joints.tolist(), disable_dynamics=True
                )
                if selected is None:
                    print(
                        "[phone-baseframe-waypoint1-lookahead-unavailable] "
                        + "tested=" + str(min(32, len(ranked_accepted)))
                        + " failures="
                        + json.dumps(waypoint1_failures[:8], sort_keys=True),
                        flush=True,
                    )
                    raise ConfigurationPathError(
                        "No smooth waypoint0 IK branch can continue linearly "
                        "to waypoint1"
                    )
            continuation_metrics = [
                item[2]
                for item in accepted
                if str(item[2]["source"]).startswith(
                    "cartesian_continuation_"
                )
            ]
            continuation_failures = [
                item
                for item in refinement_failures
                if "curve" in item
            ]
            continuation_rejections = [
                item
                for item in rejected
                if str(item.get("source", "")).startswith(
                    "cartesian_continuation_"
                )
            ]
            if continuation_failures:
                print(
                    "[phone-baseframe-continuation-failures] waypoint="
                    + waypoint_name
                    + " failures="
                    + json.dumps(continuation_failures, sort_keys=True),
                    flush=True,
                )
            if continuation_metrics:
                print(
                    "[phone-baseframe-continuation-candidates] waypoint="
                    + waypoint_name
                    + " metrics="
                    + json.dumps(continuation_metrics, sort_keys=True),
                    flush=True,
                )
            if continuation_rejections:
                print(
                    "[phone-baseframe-continuation-rejections] waypoint="
                    + waypoint_name
                    + " metrics="
                    + json.dumps(continuation_rejections, sort_keys=True),
                    flush=True,
                )
            print(
                "[phone-baseframe-reference-ik-selected] waypoint="
                + waypoint_name
                + " accepted="
                + str(len(accepted))
                + "/"
                + str(len(unique_goals))
                + " target_base_xyz="
                + json.dumps(target_base_pose9[:3].tolist())
                + " metrics="
                + json.dumps(selected[2], sort_keys=True),
                flush=True,
            )
            return ArmConfigurationPath(arm, selected[1].reshape(-1))

        Point.get_path = phone_baseframe_reference_ik_get_path
    elif args.expert_path_mode == "phone_best_ik_joint_interp_0_3":
        from pyrep.robots.configuration_paths.arm_configuration_path import (
            ArmConfigurationPath,
        )

        def phone_best_ik_joint_interp_get_path(
            point, ignore_collisions=False
        ):
            """Choose a collision-free IK branch with the smallest EEF arc."""
            nonlocal linear_path_calls
            waypoint_name = point._waypoint.get_name()
            if phone_waypoint_feasibility_active:
                return original_point_get_path(point, ignore_collisions)
            if (
                waypoint_name not in ("waypoint0", "waypoint3")
                or (
                    args.phone_best_of_n_waypoint3_only
                    and waypoint_name != "waypoint3"
                )
            ):
                return original_point_get_path(point, ignore_collisions)
            arm = point._robot.arm
            start_joints = np.asarray(
                arm.get_joint_positions(), dtype=np.float64
            )
            start_position = np.asarray(
                arm.get_tip().get_position(), dtype=np.float64
            )
            start_quaternion = np.asarray(
                arm.get_tip().get_quaternion(), dtype=np.float64
            )
            start_quaternion /= np.linalg.norm(start_quaternion)
            target_position = np.asarray(
                point._waypoint.get_position(), dtype=np.float64
            )
            target_quaternion = np.asarray(
                point._waypoint.get_quaternion(), dtype=np.float64
            )
            # Do not inherit the task waypoint's broad ignore_collisions flag:
            # that flag allows the close phone scene to be nudged during the
            # approach.  PRM must remain collision-aware.  The held handset is
            # disabled separately below because gripper/handset contact is the
            # intended grasp, not an obstacle.
            effective_ignore_collisions = False
            candidate_count = max(1, int(args.phone_path_candidates))
            try:
                goal_configurations = arm.solve_ik_via_sampling(
                    target_position,
                    quaternion=target_quaternion,
                    ignore_collisions=effective_ignore_collisions,
                    trials=max(600, 40 * candidate_count),
                    max_configs=candidate_count,
                    distance_threshold=0.65,
                    max_time_ms=20,
                )
            except ConfigurationError as exc:
                raise ConfigurationPathError(
                    "No sampled IK goals for joint interpolation"
                ) from exc

            candidates = []
            rejected = []
            try:
                for candidate_index, goal_joints in enumerate(
                    np.asarray(goal_configurations, dtype=np.float64)
                ):
                    joint_delta = goal_joints - start_joints
                    steps = max(
                        2,
                        int(
                            math.ceil(
                                float(np.max(np.abs(joint_delta))) / 0.02
                            )
                        ),
                    )
                    fractions = np.linspace(0.0, 1.0, steps + 1)[1:]
                    configurations = (
                        start_joints[None]
                        + fractions[:, None] * joint_delta[None]
                    )
                    positions = [start_position.copy()]
                    previous_quaternion = start_quaternion.copy()
                    rotation_length = 0.0
                    collision = False
                    for configuration in configurations:
                        arm.set_joint_positions(
                            configuration.tolist(), disable_dynamics=True
                        )
                        if (
                            not effective_ignore_collisions
                            and arm.check_arm_collision()
                        ):
                            collision = True
                            break
                        positions.append(
                            np.asarray(
                                arm.get_tip().get_position(), dtype=np.float64
                            )
                        )
                        quaternion = np.asarray(
                            arm.get_tip().get_quaternion(), dtype=np.float64
                        )
                        quaternion /= np.linalg.norm(quaternion)
                        rotation_length += float(
                            2.0
                            * np.arccos(
                                np.clip(
                                    abs(float(np.dot(quaternion, previous_quaternion))),
                                    -1.0,
                                    1.0,
                                )
                            )
                        )
                        previous_quaternion = quaternion
                    arm.set_joint_positions(
                        start_joints.tolist(), disable_dynamics=True
                    )
                    if collision:
                        rejected.append(
                            {
                                "candidate": int(candidate_index),
                                "reason": "collision",
                            }
                        )
                        continue
                    positions = np.asarray(positions, dtype=np.float64)
                    chord = positions[-1] - positions[0]
                    direct_distance = float(np.linalg.norm(chord))
                    path_length = float(
                        np.linalg.norm(
                            np.diff(positions, axis=0), axis=1
                        ).sum()
                    )
                    if direct_distance > 1e-9:
                        unit = chord / direct_distance
                        progress = (positions - positions[0]) @ unit
                        closest = positions[0] + progress[:, None] * unit
                        lateral = float(
                            np.linalg.norm(
                                positions - closest, axis=1
                            ).max(initial=0.0)
                        )
                        progress_backtrack = float(
                            np.maximum(0.0, -np.diff(progress)).sum()
                        )
                    else:
                        lateral = path_length
                        progress_backtrack = path_length
                    target_distance = np.linalg.norm(
                        positions - positions[-1], axis=1
                    )
                    target_backtrack = float(
                        np.maximum(0.0, np.diff(target_distance)).sum()
                    )
                    arc_score = float(
                        max(0.0, path_length - direct_distance)
                        + 2.0 * lateral
                        + 2.0 * target_backtrack
                        + progress_backtrack
                    )
                    direct_rotation = float(
                        2.0
                        * np.arccos(
                            np.clip(
                                abs(
                                    float(
                                        np.dot(
                                            start_quaternion,
                                            target_quaternion
                                            / np.linalg.norm(target_quaternion),
                                        )
                                    )
                                ),
                                -1.0,
                                1.0,
                            )
                        )
                    )
                    rotation_excess = float(
                        max(0.0, rotation_length - direct_rotation)
                    )
                    max_joint_delta = float(
                        np.abs(joint_delta).max(initial=0.0)
                    )
                    hard_rejected = bool(
                        (
                            waypoint_name == "waypoint3"
                            and float(args.phone_waypoint3_max_joint_travel_rad)
                            >= 0.0
                            and max_joint_delta
                            > float(args.phone_waypoint3_max_joint_travel_rad)
                        )
                        or (
                            waypoint_name != "waypoint3"
                            and max_joint_delta > float(np.pi)
                        )
                        or (
                            waypoint_name == "waypoint3"
                            and float(args.phone_waypoint3_max_detour_ratio) >= 0.0
                            and path_length / max(direct_distance, 1e-9)
                            > float(args.phone_waypoint3_max_detour_ratio)
                        )
                        or (
                            waypoint_name == "waypoint3"
                            and float(
                                args.phone_waypoint3_max_lateral_deviation_m
                            )
                            >= 0.0
                            and lateral
                            > float(
                                args.phone_waypoint3_max_lateral_deviation_m
                            )
                        )
                        or (
                            waypoint_name == "waypoint3"
                            and float(args.path_rotation_excess_limit_rad) >= 0.0
                            and rotation_excess
                            > float(args.path_rotation_excess_limit_rad)
                        )
                    )
                    metrics = {
                        "candidate": int(candidate_index),
                        "steps": int(len(configurations)),
                        "direct_m": direct_distance,
                        "path_length_m": path_length,
                        "detour_ratio": float(
                            path_length / max(direct_distance, 1e-9)
                        ),
                        "max_lateral_deviation_m": lateral,
                        "target_distance_backtrack_m": target_backtrack,
                        "progress_backtrack_m": progress_backtrack,
                        "arc_score_m": arc_score,
                        "joint_travel_rad": float(
                            np.abs(joint_delta).sum()
                        ),
                        "max_joint_delta_rad": float(
                            max_joint_delta
                        ),
                        "direct_rotation_rad": direct_rotation,
                        "rotation_rad": rotation_length,
                        "rotation_excess_rad": rotation_excess,
                        "hard_rejected": hard_rejected,
                    }
                    print(
                        "[phone-best-ik-joint-interp-candidate] waypoint="
                        + waypoint_name
                        + " metrics="
                        + json.dumps(metrics, sort_keys=True),
                        flush=True,
                    )
                    if not hard_rejected:
                        candidates.append(
                            (
                                arc_score,
                                lateral,
                                metrics["joint_travel_rad"],
                                configurations,
                                metrics,
                            )
                        )
            finally:
                arm.set_joint_positions(
                    start_joints.tolist(), disable_dynamics=True
                )
            linear_path_calls += int(len(goal_configurations))
            if not candidates:
                raise ConfigurationPathError(
                    "All sampled IK joint interpolations collided or violated "
                    "waypoint3 geometry guards; rejected="
                    + json.dumps(rejected, sort_keys=True)
                )
            selected = min(candidates, key=lambda item: item[:3])
            print(
                "[phone-best-ik-joint-interp-selected] waypoint="
                + waypoint_name
                + " accepted="
                + str(len(candidates))
                + "/"
                + str(len(goal_configurations))
                + " metrics="
                + json.dumps(selected[4], sort_keys=True),
                flush=True,
            )
            return ArmConfigurationPath(arm, selected[3].reshape(-1))

        Point.get_path = phone_best_ik_joint_interp_get_path
    elif args.expert_path_mode == "phone_rrt_cartesian_shortcut_0_3":
        from pyrep.robots.configuration_paths.arm_configuration_path import (
            ArmConfigurationPath,
        )

        class PhoneExactArmConfigurationPath(ArmConfigurationPath):
            """Execute an already-checked dense joint path without RML reshaping."""

            def __init__(self, arm, path_points):
                super().__init__(arm, path_points)
                self._phone_path_configs = np.asarray(
                    self._path_points, dtype=np.float64
                ).reshape(-1, self._num_joints)
                self._phone_path_index = 0

            def step(self):
                if self._path_done:
                    raise RuntimeError(
                        "This path has already been completed. "
                        "If you want to re-run, then call set_to_start."
                    )
                target = self._phone_path_configs[self._phone_path_index]
                self._joint_position_action = target.copy()
                self._arm.set_joint_positions(target.tolist())
                self._path_done = bool(
                    self._phone_path_index
                    >= len(self._phone_path_configs) - 1
                )
                if not self._path_done:
                    self._phone_path_index += 1
                return self._path_done

            def set_to_start(self, disable_dynamics=False):
                self._arm.set_joint_positions(
                    self._phone_path_configs[0].tolist(),
                    disable_dynamics=disable_dynamics,
                )
                self._path_done = False
                self._phone_path_index = 0
                self._joint_position_action = None

        def densify_phone_joint_path(start_joints, configurations):
            max_step = float(args.phone_waypoint3_exact_step_rad)
            if max_step <= 0.0:
                raise ValueError(
                    "--phone-waypoint3-exact-step-rad must be positive"
                )
            dense = [np.asarray(start_joints, dtype=np.float64).copy()]
            previous = dense[0]
            for target in np.asarray(configurations, dtype=np.float64):
                delta = target - previous
                steps = max(
                    1,
                    int(math.ceil(float(np.max(np.abs(delta))) / max_step)),
                )
                for step_index in range(1, steps + 1):
                    dense.append(
                        previous + delta * (float(step_index) / float(steps))
                    )
                previous = target.copy()
            return np.asarray(dense, dtype=np.float64)

        def phone_rrt_cartesian_shortcut_single_get_path(
            point, ignore_collisions=False
        ):
            """Shortcut a feasible RRT guide with collision-checked XYZ lines."""
            nonlocal linear_path_calls, rrt_path_calls
            waypoint_name = point._waypoint.get_name()
            if (
                waypoint_name not in ("waypoint0", "waypoint3")
                or (
                    args.phone_best_of_n_waypoint3_only
                    and waypoint_name != "waypoint3"
                )
            ):
                return original_point_get_path(point, ignore_collisions)

            arm = point._robot.arm
            joint_count = int(arm.get_joint_count())
            start_joints = np.asarray(
                arm.get_joint_positions(), dtype=np.float64
            )
            start_position = np.asarray(
                arm.get_tip().get_position(), dtype=np.float64
            )
            effective_ignore_collisions = bool(
                point._ignore_collisions or ignore_collisions
            )
            try:
                linear_path_calls += 1
                guide_path = arm.get_linear_path(
                    point._waypoint.get_position(),
                    euler=point._waypoint.get_orientation(),
                    ignore_collisions=effective_ignore_collisions,
                )
            except ConfigurationPathError:
                rrt_path_calls += 1
                guide_path = arm.get_nonlinear_path(
                    point._waypoint.get_position(),
                    euler=point._waypoint.get_orientation(),
                    ignore_collisions=effective_ignore_collisions,
                    trials=100,
                    max_configs=10,
                    trials_per_goal=1,
                    algorithm=Algos.RRTConnect,
                )
            guide_configurations = np.asarray(
                guide_path._path_points, dtype=np.float64
            ).reshape(-1, joint_count)
            if not len(guide_configurations):
                raise ConfigurationPathError("empty RRT shortcut guide")

            # Record the simulator's exact EEF pose on the feasible guide. This
            # avoids FK/model-frame discrepancies when asking linear IK to
            # connect two guide states.
            guide_positions = []
            guide_quaternions = []
            try:
                for configuration in guide_configurations:
                    arm.set_joint_positions(
                        configuration.tolist(), disable_dynamics=True
                    )
                    guide_positions.append(
                        np.asarray(
                            arm.get_tip().get_position(), dtype=np.float64
                        )
                    )
                    guide_quaternions.append(
                        np.asarray(
                            arm.get_tip().get_quaternion(), dtype=np.float64
                        )
                    )
            finally:
                arm.set_joint_positions(
                    start_joints.tolist(), disable_dynamics=True
                )
            guide_positions = np.asarray(guide_positions, dtype=np.float64)
            guide_quaternions = np.asarray(
                guide_quaternions, dtype=np.float64
            )

            combined = []
            current_guide_index = -1
            shortcut_segments = []
            raw_guide_steps = 0
            try:
                while current_guide_index < len(guide_configurations) - 1:
                    first_index = current_guide_index + 1
                    last_index = len(guide_configurations) - 1
                    remaining = last_index - current_guide_index
                    # Probe farthest-first at a bounded set of guide poses.
                    # The immediate next guide state is always retained as a
                    # collision-checked progress fallback.
                    probe_count = min(32, remaining)
                    probe_indices = np.unique(
                        np.linspace(
                            first_index,
                            last_index,
                            probe_count,
                            dtype=np.int64,
                        )
                    )[::-1]
                    chosen_index = None
                    chosen_path = None
                    current_position = np.asarray(
                        arm.get_tip().get_position(), dtype=np.float64
                    )
                    current_quaternion = np.asarray(
                        arm.get_tip().get_quaternion(), dtype=np.float64
                    )
                    for probe_index in probe_indices:
                        target_position = guide_positions[int(probe_index)]
                        target_quaternion = guide_quaternions[int(probe_index)]
                        xyz_distance = float(
                            np.linalg.norm(target_position - current_position)
                        )
                        quaternion_dot = float(
                            abs(
                                np.dot(
                                    current_quaternion
                                    / np.linalg.norm(current_quaternion),
                                    target_quaternion
                                    / np.linalg.norm(target_quaternion),
                                )
                            )
                        )
                        rotation_distance = float(
                            2.0
                            * np.arccos(
                                np.clip(quaternion_dot, -1.0, 1.0)
                            )
                        )
                        steps = max(
                            2,
                            int(math.ceil(xyz_distance / 0.01)),
                            int(math.ceil(rotation_distance / 0.05)),
                        )
                        linear_path_calls += 1
                        try:
                            candidate_path = arm.get_linear_path(
                                target_position,
                                quaternion=target_quaternion,
                                steps=steps,
                                ignore_collisions=effective_ignore_collisions,
                            )
                        except ConfigurationPathError:
                            continue
                        chosen_index = int(probe_index)
                        chosen_path = candidate_path
                        break

                    if chosen_path is None:
                        # The RRT guide itself has already been checked for
                        # collisions. Retaining exactly one next configuration
                        # guarantees progress without introducing a new arc.
                        next_configuration = guide_configurations[first_index]
                        combined.append(next_configuration[None])
                        arm.set_joint_positions(
                            next_configuration.tolist(),
                            disable_dynamics=True,
                        )
                        shortcut_segments.append(
                            {
                                "from": int(current_guide_index),
                                "to": int(first_index),
                                "kind": "guide_step",
                            }
                        )
                        current_guide_index = int(first_index)
                        raw_guide_steps += 1
                        continue

                    segment_points = np.asarray(
                        chosen_path._path_points, dtype=np.float64
                    ).reshape(-1, joint_count)
                    if combined and len(segment_points):
                        segment_points = segment_points[1:]
                    if len(segment_points):
                        combined.append(segment_points)
                    chosen_path.set_to_end(disable_dynamics=True)
                    shortcut_segments.append(
                        {
                            "from": int(current_guide_index),
                            "to": int(chosen_index),
                            "kind": "linear",
                        }
                    )
                    current_guide_index = int(chosen_index)
            finally:
                arm.set_joint_positions(
                    start_joints.tolist(), disable_dynamics=True
                )

            if not combined:
                raise ConfigurationPathError("empty shortcut path")
            configurations = np.concatenate(combined, axis=0)
            positions = [start_position.copy()]
            try:
                for configuration in configurations:
                    arm.set_joint_positions(
                        configuration.tolist(), disable_dynamics=True
                    )
                    positions.append(
                        np.asarray(
                            arm.get_tip().get_position(), dtype=np.float64
                        )
                    )
            finally:
                arm.set_joint_positions(
                    start_joints.tolist(), disable_dynamics=True
                )
            positions = np.asarray(positions, dtype=np.float64)
            chord = positions[-1] - positions[0]
            direct_distance = float(np.linalg.norm(chord))
            path_length = float(
                np.linalg.norm(np.diff(positions, axis=0), axis=1).sum()
            )
            if direct_distance > 1e-9:
                unit = chord / direct_distance
                progress = (positions - positions[0]) @ unit
                closest = positions[0] + progress[:, None] * unit
                max_lateral = float(
                    np.linalg.norm(positions - closest, axis=1).max(
                        initial=0.0
                    )
                )
                progress_backtrack = float(
                    np.maximum(0.0, -np.diff(progress)).sum()
                )
            else:
                max_lateral = path_length
                progress_backtrack = path_length
            target_distance = np.linalg.norm(
                positions - positions[-1], axis=1
            )
            target_backtrack = float(
                np.maximum(0.0, np.diff(target_distance)).sum()
            )
            metrics = {
                "guide_points": int(len(guide_configurations)),
                "output_points": int(len(configurations)),
                "shortcut_segments": int(len(shortcut_segments)),
                "raw_guide_steps": int(raw_guide_steps),
                "direct_m": direct_distance,
                "path_length_m": path_length,
                "detour_ratio": float(
                    path_length / max(direct_distance, 1e-9)
                ),
                "max_lateral_deviation_m": max_lateral,
                "target_distance_backtrack_m": target_backtrack,
                "progress_backtrack_m": progress_backtrack,
            }
            print(
                "[phone-rrt-cartesian-shortcut-selected] waypoint="
                + waypoint_name
                + " metrics="
                + json.dumps(metrics, sort_keys=True)
                + " segments="
                + json.dumps(shortcut_segments),
                flush=True,
            )
            return ArmConfigurationPath(arm, configurations.reshape(-1))

        def phone_rrt_cartesian_shortcut_get_path(
            point, ignore_collisions=False
        ):
            waypoint_name = point._waypoint.get_name()
            if phone_waypoint_feasibility_active:
                return original_point_get_path(point, ignore_collisions)
            candidate_count = (
                int(args.phone_path_candidates)
                if (
                    args.phone_best_of_n_waypoint3_only
                    and waypoint_name == "waypoint3"
                )
                else 1
            )
            if candidate_count <= 1:
                return phone_rrt_cartesian_shortcut_single_get_path(
                    point, ignore_collisions
                )

            arm = point._robot.arm
            joint_count = int(arm.get_joint_count())
            start_joints = np.asarray(arm.get_joint_positions(), dtype=np.float64)
            start_position = np.asarray(
                arm.get_tip().get_position(), dtype=np.float64
            )
            start_quaternion = np.asarray(
                arm.get_tip().get_quaternion(), dtype=np.float64
            )
            start_quaternion /= np.linalg.norm(start_quaternion)
            target_position = np.asarray(
                point._waypoint.get_position(), dtype=np.float64
            )
            target_quaternion = np.asarray(
                point._waypoint.get_quaternion(), dtype=np.float64
            )
            target_quaternion /= np.linalg.norm(target_quaternion)
            direct_distance = float(
                np.linalg.norm(target_position - start_position)
            )
            direct_rotation = float(
                2.0
                * np.arccos(
                    np.clip(
                        abs(float(np.dot(start_quaternion, target_quaternion))),
                        -1.0,
                        1.0,
                    )
                )
            )
            accepted = []
            failures = []
            for candidate_index in range(candidate_count):
                try:
                    path = phone_rrt_cartesian_shortcut_single_get_path(
                        point, ignore_collisions
                    )
                    configurations = np.asarray(
                        path._path_points, dtype=np.float64
                    ).reshape(-1, joint_count)
                    if len(configurations) == 0:
                        raise ConfigurationPathError("empty shortcut path")
                    raw_configuration_count = int(len(configurations))
                    raw_joint_sequence = np.vstack(
                        (start_joints[None], configurations)
                    )
                    raw_joint_steps = np.abs(
                        np.diff(raw_joint_sequence, axis=0)
                    )
                    first_configuration_delta = np.abs(
                        configurations[0] - start_joints
                    )
                    max_raw_joint_step = float(raw_joint_steps.max())
                    max_raw_joint_step_location = np.unravel_index(
                        int(np.argmax(raw_joint_steps)), raw_joint_steps.shape
                    )
                    scored_configurations = densify_phone_joint_path(
                        start_joints, configurations
                    )
                    positions = [start_position.copy()]
                    previous_quaternion = start_quaternion.copy()
                    rotation_length = 0.0
                    try:
                        for configuration in scored_configurations:
                            arm.set_joint_positions(
                                configuration.tolist(), disable_dynamics=True
                            )
                            positions.append(
                                np.asarray(
                                    arm.get_tip().get_position(), dtype=np.float64
                                )
                            )
                            quaternion = np.asarray(
                                arm.get_tip().get_quaternion(), dtype=np.float64
                            )
                            quaternion /= np.linalg.norm(quaternion)
                            rotation_length += float(
                                2.0
                                * np.arccos(
                                    np.clip(
                                        abs(
                                            float(
                                                np.dot(
                                                    quaternion,
                                                    previous_quaternion,
                                                )
                                            )
                                        ),
                                        -1.0,
                                        1.0,
                                    )
                                )
                            )
                            previous_quaternion = quaternion
                    finally:
                        arm.set_joint_positions(
                            start_joints.tolist(), disable_dynamics=True
                        )
                    positions = np.asarray(positions, dtype=np.float64)
                    path_length = float(
                        np.linalg.norm(np.diff(positions, axis=0), axis=1).sum()
                    )
                    detour_ratio = float(
                        path_length / max(direct_distance, 1e-9)
                    )
                    if direct_distance > 1e-9:
                        chord_unit = (
                            target_position - start_position
                        ) / direct_distance
                        progress = (positions - start_position) @ chord_unit
                        closest = start_position + progress[:, None] * chord_unit
                        max_lateral = float(
                            np.linalg.norm(positions - closest, axis=1).max(
                                initial=0.0
                            )
                        )
                    else:
                        max_lateral = path_length
                    joint_sequence = np.vstack(
                        (start_joints[None], scored_configurations)
                    )
                    joint_travel = np.abs(np.diff(joint_sequence, axis=0)).sum(
                        axis=0
                    )
                    max_joint_travel = float(joint_travel.max())
                    total_joint_travel = float(joint_travel.sum())
                    rotation_excess = float(
                        max(0.0, rotation_length - direct_rotation)
                    )
                    rejected = bool(
                        (
                            float(args.phone_waypoint3_max_joint_travel_rad)
                            >= 0.0
                            and max_joint_travel
                            > float(args.phone_waypoint3_max_joint_travel_rad)
                        )
                        or (
                            float(args.phone_waypoint3_max_detour_ratio) >= 0.0
                            and detour_ratio
                            > float(args.phone_waypoint3_max_detour_ratio)
                        )
                        or (
                            float(
                                args.phone_waypoint3_max_lateral_deviation_m
                            )
                            >= 0.0
                            and max_lateral
                            > float(
                                args.phone_waypoint3_max_lateral_deviation_m
                            )
                        )
                        or (
                            float(args.path_rotation_excess_limit_rad) >= 0.0
                            and rotation_excess
                            > float(args.path_rotation_excess_limit_rad)
                        )
                    )
                    score = float(
                        total_joint_travel
                        + 2.0 * rotation_length
                        + 2.0 * path_length
                        + 5.0 * max_lateral
                    )
                    metrics = {
                        "candidate": int(candidate_index),
                        "score": score,
                        "direct_m": direct_distance,
                        "path_length_m": path_length,
                        "detour_ratio": detour_ratio,
                        "max_lateral_deviation_m": max_lateral,
                        "direct_rotation_rad": direct_rotation,
                        "rotation_rad": rotation_length,
                        "rotation_excess_rad": rotation_excess,
                        "total_joint_travel_rad": total_joint_travel,
                        "max_joint_travel_rad": max_joint_travel,
                        "raw_points": raw_configuration_count,
                        "first_configuration_max_delta_rad": float(
                            first_configuration_delta.max()
                        ),
                        "max_raw_joint_step_rad": max_raw_joint_step,
                        "max_raw_joint_step_segment": int(
                            max_raw_joint_step_location[0]
                        ),
                        "max_raw_joint_step_axis": int(
                            max_raw_joint_step_location[1]
                        ),
                        "scored_points": int(len(scored_configurations)),
                        "rejected": rejected,
                    }
                    print(
                        "[phone-shortcut-best-of-n-candidate] waypoint="
                        + waypoint_name
                        + " metrics="
                        + json.dumps(metrics, sort_keys=True),
                        flush=True,
                    )
                    if not rejected:
                        accepted.append(
                            (score, path, scored_configurations, metrics)
                        )
                except ConfigurationPathError as exc:
                    failures.append(
                        {"candidate": int(candidate_index), "error": str(exc)}
                    )
                finally:
                    arm.set_joint_positions(
                        start_joints.tolist(), disable_dynamics=True
                    )
            if not accepted:
                print(
                    "[phone-shortcut-best-of-n-no-acceptable-path] waypoint="
                    + waypoint_name
                    + " failures="
                    + json.dumps(failures, sort_keys=True),
                    flush=True,
                )
                raise ConfigurationPathError(
                    "All shortcut candidates failed or violated waypoint3 "
                    "geometry guards"
                )
            _, selected_path, selected_configurations, selected_metrics = min(
                accepted, key=lambda item: item[0]
            )
            if args.phone_waypoint3_exact_execution:
                selected_path = PhoneExactArmConfigurationPath(
                    arm, selected_configurations.reshape(-1)
                )
            print(
                "[phone-shortcut-best-of-n-selected] waypoint="
                + waypoint_name
                + " accepted="
                + str(len(accepted))
                + "/"
                + str(candidate_count)
                + " metrics="
                + json.dumps(selected_metrics, sort_keys=True),
                flush=True,
            )
            return selected_path

        Point.get_path = phone_rrt_cartesian_shortcut_get_path
    elif args.expert_path_mode in (
        "phone_staged_linear_0_3",
        "phone_staged_linear_then_rrt_0_3",
    ):
        from pyrep.robots.configuration_paths.arm_configuration_path import (
            ArmConfigurationPath,
        )

        def phone_staged_linear_point_get_path(point, ignore_collisions=False):
            """Plan phone travel waypoints without querying a nonlinear guide."""
            nonlocal linear_path_calls, rrt_path_calls
            waypoint_name = point._waypoint.get_name()
            if (
                waypoint_name not in ("waypoint0", "waypoint3")
                or (
                    args.phone_best_of_n_waypoint3_only
                    and waypoint_name != "waypoint3"
                )
            ):
                return original_point_get_path(point, ignore_collisions)

            arm = point._robot.arm
            joint_count = int(arm.get_joint_count())
            start_joints = np.asarray(arm.get_joint_positions(), dtype=np.float64)
            start_position = np.asarray(
                arm.get_tip().get_position(), dtype=np.float64
            )
            start_quaternion = np.asarray(
                arm.get_tip().get_quaternion(), dtype=np.float64
            )
            target_position = np.asarray(
                point._waypoint.get_position(), dtype=np.float64
            )
            target_quaternion = np.asarray(
                point._waypoint.get_quaternion(), dtype=np.float64
            )
            effective_ignore_collisions = bool(
                point._ignore_collisions or ignore_collisions
            )

            def locally_scored_rrt(position, quaternion):
                """Find the smallest Cartesian detour for one staged segment."""
                nonlocal rrt_path_calls
                segment_start_joints = np.asarray(
                    arm.get_joint_positions(), dtype=np.float64
                )
                segment_start_position = np.asarray(
                    arm.get_tip().get_position(), dtype=np.float64
                )
                segment_target_position = np.asarray(position, dtype=np.float64)
                direct = float(
                    np.linalg.norm(
                        segment_target_position - segment_start_position
                    )
                )
                candidates = []
                failures = []
                for candidate_index in range(int(args.phone_path_candidates)):
                    try:
                        candidate_path = arm.get_nonlinear_path(
                            segment_target_position,
                            quaternion=np.asarray(quaternion, dtype=np.float64),
                            ignore_collisions=effective_ignore_collisions,
                            trials=100,
                            max_configs=10,
                            trials_per_goal=10,
                            algorithm=Algos.RRTConnect,
                        )
                        configurations = np.asarray(
                            candidate_path._path_points, dtype=np.float64
                        ).reshape(-1, joint_count)
                        if not len(configurations):
                            raise ConfigurationPathError("empty local RRT path")
                        joint_sequence = np.vstack(
                            (segment_start_joints[None], configurations)
                        )
                        joint_travel = np.abs(
                            np.diff(joint_sequence, axis=0)
                        ).sum(axis=0)
                        positions = [segment_start_position.copy()]
                        try:
                            for configuration in configurations:
                                arm.set_joint_positions(
                                    configuration.tolist(),
                                    disable_dynamics=True,
                                )
                                positions.append(
                                    np.asarray(
                                        arm.get_tip().get_position(),
                                        dtype=np.float64,
                                    )
                                )
                        finally:
                            arm.set_joint_positions(
                                segment_start_joints.tolist(),
                                disable_dynamics=True,
                            )
                        positions = np.asarray(positions, dtype=np.float64)
                        steps = np.linalg.norm(
                            np.diff(positions, axis=0), axis=1
                        )
                        path_length = float(steps.sum())
                        if direct > 1e-9:
                            unit = (
                                segment_target_position
                                - segment_start_position
                            ) / direct
                            progress = (
                                positions - segment_start_position
                            ) @ unit
                            closest = (
                                segment_start_position
                                + progress[:, None] * unit
                            )
                            lateral = float(
                                np.linalg.norm(
                                    positions - closest, axis=1
                                ).max(initial=0.0)
                            )
                            progress_backtrack = float(
                                np.maximum(0.0, -np.diff(progress)).sum()
                            )
                        else:
                            lateral = float(
                                np.linalg.norm(
                                    positions - segment_start_position, axis=1
                                ).max(initial=0.0)
                            )
                            progress_backtrack = path_length
                        target_distance = np.linalg.norm(
                            positions - segment_target_position, axis=1
                        )
                        target_backtrack = float(
                            np.maximum(0.0, np.diff(target_distance)).sum()
                        )
                        detour_ratio = float(
                            path_length / max(direct, 1e-6)
                        )
                        arc_score = float(
                            max(0.0, path_length - direct)
                            + 2.0 * lateral
                            + 2.0 * target_backtrack
                            + progress_backtrack
                        )
                        loop_rejected = bool(
                            float(joint_travel.max(initial=0.0)) > float(np.pi)
                            or detour_ratio > 3.0
                            or lateral > 0.20
                        )
                        metrics = {
                            "candidate": int(candidate_index),
                            "direct_m": direct,
                            "path_length_m": path_length,
                            "detour_ratio": detour_ratio,
                            "max_lateral_deviation_m": lateral,
                            "target_distance_backtrack_m": target_backtrack,
                            "progress_backtrack_m": progress_backtrack,
                            "arc_score_m": arc_score,
                            "max_joint_travel_rad": float(
                                joint_travel.max(initial=0.0)
                            ),
                            "loop_rejected": loop_rejected,
                        }
                        if not loop_rejected:
                            candidates.append(
                                (
                                    arc_score,
                                    float(joint_travel.sum()),
                                    candidate_path,
                                    metrics,
                                )
                            )
                    except ConfigurationPathError as error:
                        failures.append(
                            {
                                "candidate": int(candidate_index),
                                "error": str(error),
                            }
                        )
                rrt_path_calls += int(args.phone_path_candidates)
                if not candidates:
                    raise ConfigurationPathError(
                        "No acceptable local staged RRT path; failures="
                        + json.dumps(failures, sort_keys=True)
                    )
                selected = min(candidates, key=lambda item: item[:2])
                print(
                    "[phone-staged-local-rrt-selected] waypoint="
                    + waypoint_name
                    + " accepted="
                    + str(len(candidates))
                    + "/"
                    + str(int(args.phone_path_candidates))
                    + " metrics="
                    + json.dumps(selected[3], sort_keys=True),
                    flush=True,
                )
                return selected[2]

            def shortest_slerp(fraction):
                q0 = start_quaternion / np.linalg.norm(start_quaternion)
                q1 = target_quaternion / np.linalg.norm(target_quaternion)
                if float(np.dot(q0, q1)) < 0.0:
                    q1 = -q1
                dot = float(np.clip(np.dot(q0, q1), -1.0, 1.0))
                theta = float(np.arccos(dot))
                if abs(theta) < 1e-8:
                    return q0.copy()
                sin_theta = float(np.sin(theta))
                quaternion = (
                    np.sin((1.0 - float(fraction)) * theta) / sin_theta * q0
                    + np.sin(float(fraction) * theta) / sin_theta * q1
                )
                return quaternion / np.linalg.norm(quaternion)

            # Direct is the shortest deterministic candidate. The staged
            # candidates first clear the table/object region vertically, then
            # translate above the target, and only then descend. Unlike the
            # earlier phone_cartesian_0_3 mode, none of these control positions
            # are sampled from an RRT/stock path.
            candidate_targets = [("direct", [(target_position, target_quaternion)])]
            candidate_targets.extend(
                [
                    (
                        "rotate_first",
                        [
                            (start_position, target_quaternion),
                            (target_position, target_quaternion),
                        ],
                    ),
                    (
                        "translate_first",
                        [
                            (target_position, start_quaternion),
                            (target_position, target_quaternion),
                        ],
                    ),
                ]
            )
            for rotation_fraction in (0.25, 0.50, 0.75):
                middle_quaternion = shortest_slerp(rotation_fraction)
                candidate_targets.append(
                    (
                        "rotate_"
                        + format(int(round(rotation_fraction * 100.0)), "02d")
                        + "pct_then_translate",
                        [
                            (start_position, middle_quaternion),
                            (target_position, middle_quaternion),
                            (target_position, target_quaternion),
                        ],
                    )
                )
            for clearance_m in (0.06, 0.10, 0.14, 0.18):
                safe_z = float(max(start_position[2], target_position[2]) + clearance_m)
                lift_position = start_position.copy()
                lift_position[2] = safe_z
                above_target = target_position.copy()
                above_target[2] = safe_z
                candidate_targets.append(
                    (
                        "staged_" + format(int(round(clearance_m * 1000.0)), "03d") + "mm",
                        [
                            (lift_position, shortest_slerp(0.20)),
                            (above_target, shortest_slerp(0.85)),
                            (target_position, target_quaternion),
                        ],
                    )
                )
                candidate_targets.append(
                    (
                        "staged_"
                        + format(int(round(clearance_m * 1000.0)), "03d")
                        + "mm_late_rotation",
                        [
                            (lift_position, start_quaternion),
                            (above_target, target_quaternion),
                            (target_position, target_quaternion),
                        ],
                    )
                )

            feasible_candidates = []
            failed_candidates = []
            for candidate_name, targets in candidate_targets:
                combined = []
                previous_position = start_position.copy()
                previous_quaternion = start_quaternion.copy()
                try:
                    arm.set_joint_positions(
                        start_joints.tolist(), disable_dynamics=True
                    )
                    for segment_index, (position, quaternion) in enumerate(targets):
                        xyz_distance = float(
                            np.linalg.norm(np.asarray(position) - previous_position)
                        )
                        quaternion_dot = float(
                            abs(np.dot(
                                np.asarray(quaternion) / np.linalg.norm(quaternion),
                                previous_quaternion / np.linalg.norm(previous_quaternion),
                            ))
                        )
                        rotation_distance = float(
                            2.0 * np.arccos(np.clip(quaternion_dot, -1.0, 1.0))
                        )
                        steps = max(
                            2,
                            int(math.ceil(xyz_distance / 0.01)),
                            int(math.ceil(rotation_distance / 0.05)),
                        )
                        linear_path_calls += 1
                        try:
                            segment_path = arm.get_linear_path(
                                np.asarray(position, dtype=np.float64),
                                quaternion=np.asarray(quaternion, dtype=np.float64),
                                steps=steps,
                                ignore_collisions=effective_ignore_collisions,
                            )
                        except ConfigurationPathError:
                            if (
                                args.expert_path_mode
                                != "phone_staged_linear_then_rrt_0_3"
                            ):
                                raise
                            segment_path = locally_scored_rrt(
                                position, quaternion
                            )
                        segment_points = np.asarray(
                            segment_path._path_points, dtype=np.float64
                        ).reshape(-1, joint_count)
                        if segment_index > 0 and len(segment_points):
                            segment_points = segment_points[1:]
                        combined.append(segment_points)
                        segment_path.set_to_end(disable_dynamics=True)
                        previous_position = np.asarray(position, dtype=np.float64)
                        previous_quaternion = np.asarray(quaternion, dtype=np.float64)
                    configurations = np.concatenate(combined, axis=0)
                    if len(configurations) == 0:
                        raise ConfigurationPathError("empty staged linear path")
                    joint_sequence = np.vstack((start_joints[None], configurations))
                    joint_travel_per_axis = np.abs(np.diff(joint_sequence, axis=0)).sum(axis=0)
                    max_joint_travel = float(joint_travel_per_axis.max())
                    total_joint_travel = float(joint_travel_per_axis.sum())
                    # A continuous local IK path should not need a full turn in
                    # any single joint. Reject it rather than writing a loop to
                    # the training set.
                    if max_joint_travel > float(np.pi):
                        raise ConfigurationPathError(
                            "loop guard rejected max per-joint travel "
                            + format(max_joint_travel, ".4f")
                            + " rad"
                        )
                    xyz_length = 0.0
                    last_position = start_position.copy()
                    try:
                        for configuration in configurations:
                            arm.set_joint_positions(
                                configuration.tolist(), disable_dynamics=True
                            )
                            position = np.asarray(
                                arm.get_tip().get_position(), dtype=np.float64
                            )
                            xyz_length += float(np.linalg.norm(position - last_position))
                            last_position = position
                    finally:
                        arm.set_joint_positions(
                            start_joints.tolist(), disable_dynamics=True
                        )
                    direct_distance = float(
                        np.linalg.norm(target_position - start_position)
                    )
                    detour_ratio = float(
                        xyz_length / max(direct_distance, 1e-6)
                    )
                    score = float(total_joint_travel + xyz_length)
                    feasible_candidates.append(
                        (
                            score,
                            candidate_name,
                            configurations,
                            {
                                "xyz_m": xyz_length,
                                "direct_m": direct_distance,
                                "detour_ratio": detour_ratio,
                                "total_joint_travel_rad": total_joint_travel,
                                "max_joint_travel_rad": max_joint_travel,
                                "points": int(len(configurations)),
                            },
                        )
                    )
                except ConfigurationPathError as exc:
                    failed_candidates.append(candidate_name + ":" + str(exc))
                finally:
                    arm.set_joint_positions(
                        start_joints.tolist(), disable_dynamics=True
                    )

            if not feasible_candidates:
                print(
                    "[phone-staged-linear-no-feasible-path] waypoint="
                    + waypoint_name
                    + " failures="
                    + json.dumps(failed_candidates),
                    flush=True,
                )
                raise ConfigurationPathError(
                    "No deterministic staged linear path for " + waypoint_name
                )
            selected = min(feasible_candidates, key=lambda item: item[0])
            _, selected_name, selected_configurations, selected_metrics = selected
            print(
                "[phone-staged-linear-selected] waypoint="
                + waypoint_name
                + " route="
                + selected_name
                + " metrics="
                + json.dumps(selected_metrics, sort_keys=True)
                + " rejected="
                + json.dumps(failed_candidates),
                flush=True,
            )
            return ArmConfigurationPath(
                arm, selected_configurations.reshape(-1)
            )

        Point.get_path = phone_staged_linear_point_get_path
    elif args.expert_path_mode == "phone_predefined_path_waypoint3":
        from scipy.spatial.transform import Rotation, Slerp

        def phone_predefined_path_waypoint3_get_path(
            point, ignore_collisions=False
        ):
            """Follow one direct CartesianPath at the post-grasp waypoint3."""
            nonlocal cartesian_path_calls
            waypoint_name = point._waypoint.get_name()
            if waypoint_name != "waypoint3":
                return original_point_get_path(point, ignore_collisions)

            arm = point._robot.arm
            start_joints = np.asarray(
                arm.get_joint_positions(), dtype=np.float64
            )
            start_position = np.asarray(
                arm.get_tip().get_position(), dtype=np.float64
            )
            start_quaternion = np.asarray(
                arm.get_tip().get_quaternion(), dtype=np.float64
            )
            target_position = np.asarray(
                point._waypoint.get_position(), dtype=np.float64
            )
            target_quaternion = np.asarray(
                point._waypoint.get_quaternion(), dtype=np.float64
            )

            # Multiple collinear controls preserve one Cartesian curve while
            # making the intended shortest quaternion arc explicit.  This
            # avoids Euler-angle wrapping inside the legacy path object.
            control_count = 21
            fractions = np.linspace(0.0, 1.0, control_count)
            rotations = Rotation.from_quat(
                np.stack((start_quaternion, target_quaternion), axis=0)
            )
            quaternions = Slerp([0.0, 1.0], rotations)(fractions).as_quat()
            eulers = np.unwrap(
                Rotation.from_quat(quaternions).as_euler("xyz"), axis=0
            )
            positions = (
                start_position[None]
                + fractions[:, None]
                * (target_position - start_position)[None]
            )
            control_points = np.concatenate((positions, eulers), axis=1)
            cartesian_path = CartesianPath.create(
                show_line=False,
                show_orientation=False,
                show_position=False,
                closed_path=False,
                automatic_orientation=False,
                flat_path=False,
            )
            try:
                cartesian_path.insert_control_points(control_points.tolist())
                cartesian_path_calls += 1
                path = PredefinedPath(cartesian_path, point._robot).get_path(
                    ignore_collisions=(
                        point._ignore_collisions or ignore_collisions
                    )
                )
            finally:
                cartesian_path.remove()

            configurations = np.asarray(
                path._path_points, dtype=np.float64
            ).reshape(-1, int(arm.get_joint_count()))
            if len(configurations) == 0:
                raise ConfigurationPathError(
                    "Predefined waypoint3 path returned no configurations"
                )
            joint_sequence = np.vstack((start_joints[None], configurations))
            joint_travel_per_axis = np.abs(
                np.diff(joint_sequence, axis=0)
            ).sum(axis=0)
            xyz = [start_position.copy()]
            quats = [start_quaternion.copy()]
            try:
                for configuration in configurations:
                    arm.set_joint_positions(
                        configuration.tolist(), disable_dynamics=True
                    )
                    xyz.append(
                        np.asarray(arm.get_tip().get_position(), dtype=np.float64)
                    )
                    quats.append(
                        np.asarray(
                            arm.get_tip().get_quaternion(), dtype=np.float64
                        )
                    )
            finally:
                arm.set_joint_positions(
                    start_joints.tolist(), disable_dynamics=True
                )
            xyz = np.asarray(xyz, dtype=np.float64)
            quats = np.asarray(quats, dtype=np.float64)
            path_length = float(
                np.linalg.norm(np.diff(xyz, axis=0), axis=1).sum()
            )
            direct = float(np.linalg.norm(target_position - start_position))
            rotation_length = float(
                np.sum(
                    2.0
                    * np.arccos(
                        np.clip(
                            np.abs(np.sum(quats[:-1] * quats[1:], axis=1)),
                            -1.0,
                            1.0,
                        )
                    )
                )
            )
            metrics = {
                "points": int(len(configurations)),
                "direct_m": direct,
                "path_length_m": path_length,
                "detour_ratio": float(path_length / max(direct, 1e-9)),
                "rotation_rad": rotation_length,
                "max_joint_travel_rad": float(
                    joint_travel_per_axis.max(initial=0.0)
                ),
                "total_joint_travel_rad": float(joint_travel_per_axis.sum()),
            }
            print(
                "[phone-predefined-path-waypoint3] metrics="
                + json.dumps(metrics, sort_keys=True),
                flush=True,
            )
            return path

        Point.get_path = phone_predefined_path_waypoint3_get_path
    elif args.expert_path_mode in (
        "phone_prm_waypoint3", "phone_prm_0_3", "phone_prm_all"
    ):
        from pyrep.robots.configuration_paths.arm_configuration_path import (
            ArmConfigurationPath,
        )
        from pyrep.objects.shape import Shape

        prm_reference_waypoint_joints = {}
        prm_reference_waypoint_paths = {}
        if args.phone_reference_artifact is not None:
            reference_arrays = np.load(
                args.phone_reference_artifact, allow_pickle=False
            )
            reference_names = [
                str(name) for name in reference_arrays["waypoint_end_names"]
            ]
            reference_frames = np.asarray(
                reference_arrays["waypoint_end_frames"], dtype=np.int64
            )
            # ``raw_expert_actions_full`` starts with a synthetic NaN row for
            # transition alignment.  Path reconstruction needs the actual
            # per-transition joint targets, so use the aligned non-NaN array.
            reference_actions = np.asarray(
                reference_arrays["raw_expert_actions"], dtype=np.float64
            )
            for reference_name, reference_frame in zip(
                reference_names, reference_frames
            ):
                prm_reference_waypoint_joints[reference_name] = (
                    reference_actions[int(reference_frame), :7].copy()
                )
            reference_start = 0
            for reference_name, reference_frame in zip(
                reference_names, reference_frames
            ):
                reference_stop = int(reference_frame) + 1
                reference_segment = reference_actions[
                    reference_start:reference_stop, :7
                ].copy()
                if len(reference_segment) > 1:
                    keep = np.ones(len(reference_segment), dtype=bool)
                    keep[1:] = (
                        np.max(
                            np.abs(np.diff(reference_segment, axis=0)),
                            axis=1,
                        )
                        > 1e-7
                    )
                    reference_segment = reference_segment[keep]
                prm_reference_waypoint_paths[reference_name] = (
                    reference_segment
                )
                reference_start = reference_stop
            print(
                "[phone-reference-ik-branches] source="
                + str(args.phone_reference_artifact)
                + " waypoints="
                + ",".join(sorted(prm_reference_waypoint_joints)),
                flush=True,
            )

        prm_held_joint_target = None
        prm_phone_anchor = None
        prm_phone_object = None
        original_prm_scene_step = BackendScene.step

        def prm_scene_step(scene):
            """Keep the arm on the selected PRM polyline through grasp steps."""
            nonlocal prm_phone_anchor, prm_phone_object
            original_prm_scene_step(scene)
            if prm_held_joint_target is None:
                return
            target = np.asarray(prm_held_joint_target, dtype=np.float64)
            scene.robot.arm.set_joint_positions(
                target.tolist(), disable_dynamics=True
            )
            scene.robot.arm.set_joint_target_positions(target.tolist())
            if prm_phone_anchor is not None and prm_phone_object is not None:
                if scene.robot.gripper.get_grasped_objects():
                    prm_phone_anchor = None
                    prm_phone_object = None
                else:
                    prm_phone_object.set_pose(prm_phone_anchor)
                    prm_phone_object.reset_dynamic_object()

        BackendScene.step = prm_scene_step

        class PrmTrackedConfigurationPath(ArmConfigurationPath):
            """Execute every collision-checked PRM configuration exactly."""

            def __init__(self, arm, path_points):
                super().__init__(arm, np.asarray(path_points).reshape(-1))
                self._configs = np.asarray(
                    path_points, dtype=np.float64
                ).reshape(-1, int(arm.get_joint_count()))
                self._index = 0

            def step(self):
                nonlocal prm_held_joint_target
                nonlocal prm_phone_anchor, prm_phone_object
                if self._path_done:
                    raise RuntimeError("PRM tracked path has already completed")
                if (
                    prm_phone_anchor is None
                    and not self._robot_grasped_objects()
                ):
                    prm_phone_object = Shape("phone")
                    prm_phone_anchor = prm_phone_object.get_pose()
                target = self._configs[self._index]
                self._joint_position_action = target.copy()
                self._arm.set_joint_positions(
                    target.tolist(), disable_dynamics=True
                )
                self._arm.set_joint_target_positions(target.tolist())
                prm_held_joint_target = target.copy()
                if self._index >= len(self._configs) - 1:
                    self._path_done = True
                    return True
                self._index += 1
                return False

            def _robot_grasped_objects(self):
                # ArmConfigurationPath does not retain Robot, but the Panda
                # gripper has a stable scene name in this single-task mode.
                from pyrep.robots.end_effectors.panda_gripper import PandaGripper

                return PandaGripper().get_grasped_objects()

            def set_to_start(self, disable_dynamics=False):
                nonlocal prm_held_joint_target
                self._arm.set_joint_positions(
                    self._configs[0].tolist(),
                    disable_dynamics=disable_dynamics,
                )
                self._index = 0
                self._path_done = False
                self._joint_position_action = None
                prm_held_joint_target = self._configs[0].copy()

        def phone_prm_waypoint3_get_path(point, ignore_collisions=False):
            """Select the least circuitous OMPL PRM path at hard phone waypoints."""
            waypoint_name = point._waypoint.get_name()
            use_prm = waypoint_name == "waypoint3" or (
                args.expert_path_mode == "phone_prm_0_3"
                and waypoint_name == "waypoint0"
            ) or args.expert_path_mode == "phone_prm_all"
            if not use_prm and args.expert_path_mode == "phone_prm_waypoint3":
                return original_point_get_path(point, ignore_collisions)
            if not use_prm:
                return point._robot.arm.get_linear_path(
                    point._waypoint.get_position(),
                    quaternion=point._waypoint.get_quaternion(),
                    ignore_collisions=bool(
                        point._ignore_collisions or ignore_collisions
                    ),
                )

            arm = point._robot.arm
            joint_count = int(arm.get_joint_count())
            start_joints = np.asarray(
                arm.get_joint_positions(), dtype=np.float64
            )
            start_position = np.asarray(
                arm.get_tip().get_position(), dtype=np.float64
            )
            start_quaternion = np.asarray(
                arm.get_tip().get_quaternion(), dtype=np.float64
            )
            target_position = np.asarray(
                point._waypoint.get_position(), dtype=np.float64
            )
            # Do not inherit RLBench's waypoint-level ignore-collision flag in
            # this diagnostic planner.  The selected path is intended to be a
            # clean training trajectory, so every candidate must still be
            # checked against the robot and scene.  Only an already grasped
            # handset is made non-collidable temporarily while planning.
            effective_ignore_collisions = False
            direct = float(np.linalg.norm(target_position - start_position))

            if (
                args.expert_path_mode == "phone_prm_all"
                and waypoint_name == "waypoint2"
                and point._robot.gripper.get_grasped_objects()
            ):
                lift_height = float(args.phone_postgrasp_lift_m)
                try:
                    lift_target_position = start_position + np.asarray(
                        [0.0, 0.0, lift_height], dtype=np.float64
                    )
                    arm.set_joint_positions(
                        start_joints.tolist(), disable_dynamics=True
                    )
                    lift_path = arm.get_linear_path(
                        lift_target_position,
                        # The retract waypoint's small authored wrist change
                        # has no task semantics and is singular on this close
                        # phone scene.  Preserve the actual grasp orientation
                        # and perform a pure Cartesian translation.
                        quaternion=start_quaternion,
                        steps=300,
                        # The handset is intentionally attached to the gripper
                        # here; legacy collision checking treats that desired
                        # contact as an obstacle.  This exemption is limited to
                        # the short post-grasp retract waypoint.
                        ignore_collisions=True,
                    )
                    lift_configurations = np.asarray(
                        lift_path._path_points, dtype=np.float64
                    ).reshape(-1, joint_count)
                    if len(lift_configurations) == 0:
                        raise ConfigurationPathError("empty phone lift path")
                    print(
                        "[phone-postgrasp-linear-lift] waypoint=waypoint2"
                        + " direct_m="
                        + format(lift_height, ".6f")
                        + " authored_target_offset_m="
                        + json.dumps(
                            (target_position - start_position).tolist()
                        )
                        + " points="
                        + str(len(lift_configurations)),
                        flush=True,
                    )
                    return DenseJointServoPath(
                        arm, lift_configurations, waypoint_name=waypoint_name
                    )
                except (ConfigurationError, ConfigurationPathError) as exc:
                    print(
                        "[phone-postgrasp-linear-lift-unavailable] error="
                        + repr(exc),
                        flush=True,
                    )
                    joint0_position = np.asarray(
                        arm.joints[0].get_position(), dtype=np.float64
                    )
                    radial = start_position[:2] - joint0_position[:2]
                    radial_norm = float(np.linalg.norm(radial))
                    if radial_norm <= 1e-9:
                        radial = np.asarray([1.0, 0.0], dtype=np.float64)
                    else:
                        radial /= radial_norm
                    tangent = np.asarray([-radial[1], radial[0]])
                    horizontal_directions = [
                        radial,
                        tangent,
                        -tangent,
                        -radial,
                    ]
                    diagonal_lift = None
                    diagonal_metrics = None
                    for horizontal_m in (0.02, 0.04, 0.06, 0.08):
                        for direction_index, horizontal_direction in enumerate(
                            horizontal_directions
                        ):
                            candidate_target = start_position.copy()
                            candidate_target[:2] += (
                                horizontal_m * horizontal_direction
                            )
                            candidate_target[2] += lift_height
                            arm.set_joint_positions(
                                start_joints.tolist(), disable_dynamics=True
                            )
                            try:
                                candidate_path = arm.get_linear_path(
                                    candidate_target,
                                    quaternion=start_quaternion,
                                    steps=300,
                                    ignore_collisions=True,
                                )
                            except (
                                ConfigurationError,
                                ConfigurationPathError,
                                IKError,
                            ):
                                continue
                            candidate_configurations = np.asarray(
                                candidate_path._path_points,
                                dtype=np.float64,
                            ).reshape(-1, joint_count)
                            if len(candidate_configurations):
                                diagonal_lift = candidate_configurations.copy()
                                diagonal_metrics = {
                                    "horizontal_m": float(horizontal_m),
                                    "direction_index": int(direction_index),
                                    "direction_xy": horizontal_direction.tolist(),
                                    "vertical_m": lift_height,
                                    "points": int(len(candidate_configurations)),
                                }
                                break
                        if diagonal_lift is not None:
                            break
                    arm.set_joint_positions(
                        start_joints.tolist(), disable_dynamics=True
                    )
                    if diagonal_lift is not None:
                        print(
                            "[phone-postgrasp-diagonal-lift-selected] metrics="
                            + json.dumps(diagonal_metrics, sort_keys=True),
                            flush=True,
                        )
                        return DenseJointServoPath(
                            arm, diagonal_lift, waypoint_name=waypoint_name
                        )
                    from scipy.spatial.transform import Rotation

                    start_rotation = Rotation.from_quat(start_quaternion)
                    relaxed_lift = None
                    relaxed_metrics = None
                    position_candidates = [
                        (0.0, -1, np.zeros(2, dtype=np.float64))
                    ]
                    for horizontal_m in (0.02, 0.04, 0.06, 0.08):
                        for direction_index, horizontal_direction in enumerate(
                            horizontal_directions
                        ):
                            position_candidates.append(
                                (
                                    horizontal_m,
                                    direction_index,
                                    horizontal_m * horizontal_direction,
                                )
                            )
                    local_axes = np.eye(3, dtype=np.float64)
                    for angle_deg in (5.0, 10.0, 15.0, 20.0, 30.0, 45.0):
                        for axis_index, local_axis in enumerate(local_axes):
                            for sign in (-1.0, 1.0):
                                candidate_quaternion = (
                                    start_rotation
                                    * Rotation.from_rotvec(
                                        sign
                                        * math.radians(angle_deg)
                                        * local_axis
                                    )
                                ).as_quat()
                                for (
                                    horizontal_m,
                                    direction_index,
                                    horizontal_offset,
                                ) in position_candidates:
                                    candidate_target = start_position.copy()
                                    candidate_target[:2] += horizontal_offset
                                    candidate_target[2] += lift_height
                                    arm.set_joint_positions(
                                        start_joints.tolist(),
                                        disable_dynamics=True,
                                    )
                                    try:
                                        candidate_path = arm.get_linear_path(
                                            candidate_target,
                                            quaternion=candidate_quaternion,
                                            steps=300,
                                            ignore_collisions=True,
                                        )
                                    except (
                                        ConfigurationError,
                                        ConfigurationPathError,
                                    ):
                                        continue
                                    candidate_configurations = np.asarray(
                                        candidate_path._path_points,
                                        dtype=np.float64,
                                    ).reshape(-1, joint_count)
                                    if len(candidate_configurations):
                                        relaxed_lift = (
                                            candidate_configurations.copy()
                                        )
                                        relaxed_metrics = {
                                            "angle_deg": float(angle_deg),
                                            "axis_index": int(axis_index),
                                            "sign": int(sign),
                                            "horizontal_m": float(horizontal_m),
                                            "direction_index": int(direction_index),
                                            "vertical_m": lift_height,
                                            "points": int(
                                                len(candidate_configurations)
                                            ),
                                        }
                                        break
                                if relaxed_lift is not None:
                                    break
                            if relaxed_lift is not None:
                                break
                        if relaxed_lift is not None:
                            break
                    arm.set_joint_positions(
                        start_joints.tolist(), disable_dynamics=True
                    )
                    if relaxed_lift is not None:
                        print(
                            "[phone-postgrasp-relaxed-linear-lift-selected] metrics="
                            + json.dumps(relaxed_metrics, sort_keys=True),
                            flush=True,
                        )
                        return DenseJointServoPath(
                            arm, relaxed_lift, waypoint_name=waypoint_name
                        )
                    lift_candidates = []
                    try:
                        sampled_lift_goals = arm.solve_ik_via_sampling(
                            lift_target_position,
                            quaternion=start_quaternion,
                            ignore_collisions=True,
                            trials=4000,
                            max_configs=100,
                            distance_threshold=0.65,
                            max_time_ms=20,
                        )
                    except ConfigurationError:
                        sampled_lift_goals = []
                    for lift_goal_index, lift_goal in enumerate(
                        np.asarray(sampled_lift_goals, dtype=np.float64)
                    ):
                        lift_delta = lift_goal - start_joints
                        lift_count = max(
                            30,
                            int(
                                math.ceil(
                                    float(np.max(np.abs(lift_delta))) / 0.01
                                )
                            ),
                        )
                        lift_configurations = (
                            start_joints[None]
                            + np.linspace(0.0, 1.0, lift_count + 1)[1:, None]
                            * lift_delta[None]
                        )
                        lift_xyz = [start_position.copy()]
                        lift_quaternions = [start_quaternion.copy()]
                        for configuration in lift_configurations:
                            arm.set_joint_positions(
                                configuration.tolist(), disable_dynamics=True
                            )
                            lift_xyz.append(
                                np.asarray(
                                    arm.get_tip().get_position(), dtype=np.float64
                                )
                            )
                            lift_quaternions.append(
                                np.asarray(
                                    arm.get_tip().get_quaternion(),
                                    dtype=np.float64,
                                )
                            )
                        arm.set_joint_positions(
                            start_joints.tolist(), disable_dynamics=True
                        )
                        lift_xyz = np.asarray(lift_xyz, dtype=np.float64)
                        lift_quaternions = np.asarray(
                            lift_quaternions, dtype=np.float64
                        )
                        lift_path_length = float(
                            np.linalg.norm(
                                np.diff(lift_xyz, axis=0), axis=1
                            ).sum()
                        )
                        lift_direction = np.asarray(
                            [0.0, 0.0, 1.0], dtype=np.float64
                        )
                        lift_progress = (
                            lift_xyz - start_position
                        ) @ lift_direction
                        lift_closest = (
                            start_position
                            + lift_progress[:, None] * lift_direction
                        )
                        lift_lateral = float(
                            np.linalg.norm(
                                lift_xyz - lift_closest, axis=1
                            ).max(initial=0.0)
                        )
                        lift_backtrack = float(
                            np.maximum(
                                0.0, -np.diff(lift_progress)
                            ).sum()
                        )
                        lift_rotation = float(
                            np.sum(
                                2.0
                                * np.arccos(
                                    np.clip(
                                        np.abs(
                                            np.sum(
                                                lift_quaternions[:-1]
                                                * lift_quaternions[1:],
                                                axis=1,
                                            )
                                        ),
                                        -1.0,
                                        1.0,
                                    )
                                )
                            )
                        )
                        lift_endpoint_error = float(
                            np.linalg.norm(
                                lift_xyz[-1] - lift_target_position
                            )
                        )
                        lift_detour = float(
                            lift_path_length / max(lift_height, 1e-9)
                        )
                        if (
                            lift_endpoint_error > 0.015
                            or lift_detour > 3.0
                            or lift_lateral > 0.10
                            or lift_backtrack > 0.05
                            or lift_rotation > math.pi
                            or float(np.max(np.abs(lift_delta))) > math.pi
                        ):
                            continue
                        lift_score = float(
                            lift_path_length
                            + 2.0 * lift_lateral
                            + lift_backtrack
                            + 0.10 * lift_rotation
                            + 0.02 * np.abs(lift_delta).sum()
                        )
                        lift_candidates.append(
                            (
                                lift_score,
                                lift_configurations.copy(),
                                {
                                    "goal": int(lift_goal_index),
                                    "path_length_m": lift_path_length,
                                    "detour_ratio": lift_detour,
                                    "lateral_m": lift_lateral,
                                    "backtrack_m": lift_backtrack,
                                    "rotation_rad": lift_rotation,
                                    "endpoint_error_m": lift_endpoint_error,
                                    "max_joint_delta_rad": float(
                                        np.max(np.abs(lift_delta))
                                    ),
                                },
                            )
                        )
                    if lift_candidates:
                        selected_lift = min(
                            lift_candidates, key=lambda item: item[0]
                        )
                        print(
                            "[phone-postgrasp-joint-lift-selected] accepted="
                            + str(len(lift_candidates))
                            + "/"
                            + str(len(sampled_lift_goals))
                            + " metrics="
                            + json.dumps(selected_lift[2], sort_keys=True),
                            flush=True,
                        )
                        return DenseJointServoPath(
                            arm, selected_lift[1], waypoint_name=waypoint_name
                        )
                finally:
                    arm.set_joint_positions(
                        start_joints.tolist(), disable_dynamics=True
                    )

            if (
                args.expert_path_mode == "phone_prm_all"
                and waypoint_name in ("waypoint0", "waypoint1")
                and waypoint_name in prm_reference_waypoint_paths
            ):
                reference_segment = prm_reference_waypoint_paths[waypoint_name]
                if len(reference_segment):
                    # Preserve the exact successful approach/contact dynamics
                    # of this restored scene.  Re-interpolating only the final
                    # joint target changes how the handset is nudged and makes
                    # the following relative waypoint a different pose.
                    print(
                        "[phone-reference-segment-replay] waypoint="
                        + waypoint_name
                        + " points="
                        + str(len(reference_segment)),
                        flush=True,
                    )
                    return ArmConfigurationPath(
                        arm, reference_segment.reshape(-1)
                    )

            if (
                args.expert_path_mode == "phone_prm_all"
                and waypoint_name == "waypoint3"
                and point._robot.gripper.get_grasped_objects()
            ):
                from scipy.spatial.transform import Rotation, Slerp

                start_qn = start_quaternion / np.linalg.norm(start_quaternion)
                target_qn = np.asarray(
                    point._waypoint.get_quaternion(), dtype=np.float64
                )
                target_qn /= np.linalg.norm(target_qn)
                if float(np.dot(start_qn, target_qn)) < 0.0:
                    target_qn *= -1.0
                waypoint3_slerp = Slerp(
                    [0.0, 1.0],
                    Rotation.from_quat(np.stack((start_qn, target_qn))),
                )

                # Near the Panda base, the straight diagonal can push the
                # handset or forearm through the base footprint.  Try a clean
                # Cartesian "across, then down" transfer at the already lifted
                # height. This is obstacle clearance, not an unconstrained
                # joint-space planner: XY follows one straight line and Z one
                # straight descent, with shortest-arc quaternion interpolation.
                overhead_candidates = []
                grasped_collidable_states = []
                try:
                    for grasped_object in point._robot.gripper.get_grasped_objects():
                        was_collidable = bool(grasped_object.is_collidable())
                        grasped_collidable_states.append(
                            (grasped_object, was_collidable)
                        )
                        if was_collidable:
                            grasped_object.set_collidable(False)
                    for extra_clearance_m in (0.0, 0.03, 0.05):
                        overhead_position = target_position.copy()
                        overhead_position[2] = (
                            max(start_position[2], target_position[2])
                            + extra_clearance_m
                        )
                        for overhead_rotation_progress in (0.0, 0.5, 1.0):
                            overhead_quaternion = waypoint3_slerp(
                                [overhead_rotation_progress]
                            ).as_quat()[0]
                            arm.set_joint_positions(
                                start_joints.tolist(), disable_dynamics=True
                            )
                            try:
                                across = arm.get_linear_path(
                                    overhead_position,
                                    quaternion=overhead_quaternion,
                                    steps=180,
                                    ignore_collisions=True,
                                )
                                across_configs = np.asarray(
                                    across._path_points, dtype=np.float64
                                ).reshape(-1, joint_count)
                                across.set_to_end(disable_dynamics=True)
                                down = arm.get_linear_path(
                                    target_position,
                                    quaternion=target_qn,
                                    steps=120,
                                    ignore_collisions=True,
                                )
                                down_configs = np.asarray(
                                    down._path_points, dtype=np.float64
                                ).reshape(-1, joint_count)
                                configurations = np.vstack(
                                    (across_configs, down_configs[1:])
                                )
                                sequence = np.vstack(
                                    (start_joints[None], configurations)
                                )
                                steps = np.abs(np.diff(sequence, axis=0))
                                travel = steps.sum(axis=0)
                                if (
                                    float(steps.max(initial=0.0)) > 0.20
                                    or float(travel.max(initial=0.0)) > math.pi
                                    or float(travel[-3:].sum()) > 4.0
                                ):
                                    raise ConfigurationPathError(
                                        "phone overhead transfer joint guards"
                                    )
                                path_length = float(
                                    np.linalg.norm(
                                        overhead_position - start_position
                                    )
                                    + np.linalg.norm(
                                        target_position - overhead_position
                                    )
                                )
                                metrics = {
                                    "extra_clearance_m": float(
                                        extra_clearance_m
                                    ),
                                    "overhead_rotation_progress": float(
                                        overhead_rotation_progress
                                    ),
                                    "path_length_m": path_length,
                                    "direct_m": float(
                                        np.linalg.norm(
                                            target_position - start_position
                                        )
                                    ),
                                    "max_joint_travel_rad": float(
                                        travel.max(initial=0.0)
                                    ),
                                    "wrist_travel_rad": float(
                                        travel[-3:].sum()
                                    ),
                                    "points": int(len(configurations)),
                                }
                                score = float(
                                    path_length
                                    + 2.0 * extra_clearance_m
                                    + 0.02 * travel.sum()
                                    + 0.05 * travel[-3:].sum()
                                )
                                overhead_candidates.append(
                                    (score, configurations.copy(), metrics)
                                )
                            except (
                                ConfigurationError,
                                ConfigurationPathError,
                                IKError,
                            ):
                                pass
                            finally:
                                arm.set_joint_positions(
                                    start_joints.tolist(),
                                    disable_dynamics=True,
                                )
                        if overhead_candidates:
                            break
                finally:
                    for grasped_object, was_collidable in grasped_collidable_states:
                        grasped_object.set_collidable(was_collidable)
                    arm.set_joint_positions(
                        start_joints.tolist(), disable_dynamics=True
                    )
                if overhead_candidates:
                    selected_overhead = min(
                        overhead_candidates, key=lambda item: item[0]
                    )
                    print(
                        "[phone-waypoint3-overhead-transfer-selected] metrics="
                        + json.dumps(selected_overhead[2], sort_keys=True),
                        flush=True,
                    )
                    return DenseJointServoPath(
                        arm,
                        selected_overhead[1],
                        waypoint_name=waypoint_name,
                    )
                overhead_jacobian_candidates = []
                for extra_clearance_m in (0.0, 0.03, 0.05, 0.08):
                    overhead_position = target_position.copy()
                    overhead_position[2] = (
                        max(start_position[2], target_position[2])
                        + extra_clearance_m
                    )
                    for overhead_progress in (0.0, 0.25, 0.5, 0.75, 1.0):
                        node_positions = (
                            start_position,
                            overhead_position,
                            target_position,
                        )
                        node_progress = (0.0, float(overhead_progress), 1.0)
                        configurations = []
                        arm.set_joint_positions(
                            start_joints.tolist(), disable_dynamics=True
                        )
                        try:
                            for leg_index in range(2):
                                for fraction in np.linspace(
                                    0.0, 1.0, 81, dtype=np.float64
                                )[1:]:
                                    position = (
                                        (1.0 - float(fraction))
                                        * node_positions[leg_index]
                                        + float(fraction)
                                        * node_positions[leg_index + 1]
                                    )
                                    progress = (
                                        (1.0 - float(fraction))
                                        * node_progress[leg_index]
                                        + float(fraction)
                                        * node_progress[leg_index + 1]
                                    )
                                    quaternion = waypoint3_slerp(
                                        [progress]
                                    ).as_quat()[0]
                                    previous = np.asarray(
                                        arm.get_joint_positions(),
                                        dtype=np.float64,
                                    )
                                    joints = np.asarray(
                                        arm.solve_ik_via_jacobian(
                                            position,
                                            quaternion=quaternion,
                                        ),
                                        dtype=np.float64,
                                    )
                                    if float(
                                        np.max(np.abs(joints - previous))
                                    ) > 0.20:
                                        raise ConfigurationPathError(
                                            "phone overhead Jacobian branch jump"
                                        )
                                    arm.set_joint_positions(
                                        joints.tolist(), disable_dynamics=True
                                    )
                                    configurations.append(joints.copy())
                            configurations = np.asarray(
                                configurations, dtype=np.float64
                            )
                            sequence = np.vstack(
                                (start_joints[None], configurations)
                            )
                            steps = np.abs(np.diff(sequence, axis=0))
                            travel = steps.sum(axis=0)
                            if (
                                float(travel.max(initial=0.0)) > math.pi
                                or float(travel[-3:].sum()) > 4.0
                            ):
                                raise ConfigurationPathError(
                                    "phone overhead Jacobian travel guards"
                                )
                            path_length = float(
                                np.linalg.norm(
                                    overhead_position - start_position
                                )
                                + np.linalg.norm(
                                    target_position - overhead_position
                                )
                            )
                            metrics = {
                                "solver": "sequential_jacobian",
                                "extra_clearance_m": float(
                                    extra_clearance_m
                                ),
                                "overhead_rotation_progress": float(
                                    overhead_progress
                                ),
                                "path_length_m": path_length,
                                "direct_m": float(
                                    np.linalg.norm(
                                        target_position - start_position
                                    )
                                ),
                                "max_joint_travel_rad": float(
                                    travel.max(initial=0.0)
                                ),
                                "wrist_travel_rad": float(
                                    travel[-3:].sum()
                                ),
                                "points": int(len(configurations)),
                            }
                            score = float(
                                path_length
                                + 2.0 * extra_clearance_m
                                + 0.02 * travel.sum()
                                + 0.05 * travel[-3:].sum()
                            )
                            overhead_jacobian_candidates.append(
                                (score, configurations.copy(), metrics)
                            )
                        except (
                            ConfigurationError,
                            ConfigurationPathError,
                            IKError,
                        ):
                            pass
                        finally:
                            arm.set_joint_positions(
                                start_joints.tolist(), disable_dynamics=True
                            )
                    if overhead_jacobian_candidates:
                        break
                if overhead_jacobian_candidates:
                    selected_overhead = min(
                        overhead_jacobian_candidates,
                        key=lambda item: item[0],
                    )
                    print(
                        "[phone-waypoint3-overhead-jacobian-selected] metrics="
                        + json.dumps(selected_overhead[2], sort_keys=True),
                        flush=True,
                    )
                    return DenseJointServoPath(
                        arm,
                        selected_overhead[1],
                        waypoint_name=waypoint_name,
                    )
                print(
                    "[phone-waypoint3-overhead-transfer-unavailable]",
                    flush=True,
                )

                # CoppeliaSim's monolithic IK-path generator can report this
                # close-to-base transfer as unreachable even though a
                # continuous local Jacobian solution exists.  Solve many tiny
                # Cartesian targets in sequence, always seeding the next solve
                # from the preceding joint configuration.  This explicitly
                # prevents an IK branch jump (the source of the stationary
                # wrist spin / large arm loop) while retaining the physically
                # equivalent 180-degree finger-swapped grasp orientation.
                chord = target_position - start_position
                chord_norm = float(np.linalg.norm(chord))
                chord_unit = chord / max(chord_norm, 1e-9)
                world_z = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
                side = np.cross(chord_unit, world_z)
                if float(np.linalg.norm(side)) < 1e-8:
                    side = np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
                side /= np.linalg.norm(side)
                arc_specs = [(0.0, world_z)]
                for offset_m in (0.01, 0.02, 0.03, 0.04, 0.06, 0.08):
                    arc_specs.extend(
                        (
                            (offset_m, world_z),
                            (offset_m, side),
                            (offset_m, -side),
                        )
                    )
                progress_schedules = (
                    ("coupled", lambda t: t),
                    ("rotation_late", lambda t: max(0.0, 1.5 * t - 0.5)),
                    ("rotation_early", lambda t: min(1.0, 1.5 * t)),
                )
                sequential_candidates = []
                grasped_collidable_states = []
                try:
                    for grasped_object in point._robot.gripper.get_grasped_objects():
                        was_collidable = bool(grasped_object.is_collidable())
                        grasped_collidable_states.append(
                            (grasped_object, was_collidable)
                        )
                        if was_collidable:
                            grasped_object.set_collidable(False)
                    fractions = np.linspace(0.0, 1.0, 61)[1:]
                    for offset_m, offset_direction in arc_specs:
                        for schedule_name, rotation_progress_fn in progress_schedules:
                            configurations = []
                            xyz = []
                            quaternions = []
                            arm.set_joint_positions(
                                start_joints.tolist(), disable_dynamics=True
                            )
                            try:
                                for fraction in fractions:
                                    position = (
                                        start_position
                                        + float(fraction) * chord
                                        + math.sin(math.pi * float(fraction))
                                        * float(offset_m)
                                        * offset_direction
                                    )
                                    rotation_progress = rotation_progress_fn(
                                        float(fraction)
                                    )
                                    quaternion = waypoint3_slerp(
                                        [rotation_progress]
                                    ).as_quat()[0]
                                    joints = np.asarray(
                                        arm.solve_ik_via_jacobian(
                                            position,
                                            quaternion=quaternion,
                                        ),
                                        dtype=np.float64,
                                    )
                                    previous_joints = np.asarray(
                                        arm.get_joint_positions(),
                                        dtype=np.float64,
                                    )
                                    if float(
                                        np.max(np.abs(joints - previous_joints))
                                    ) > 0.25:
                                        raise ConfigurationPathError(
                                            "sequential IK branch jump exceeds 0.25 rad"
                                        )
                                    arm.set_joint_positions(
                                        joints.tolist(), disable_dynamics=True
                                    )
                                    if arm.check_arm_collision():
                                        raise ConfigurationPathError(
                                            "sequential IK candidate collides"
                                        )
                                    configurations.append(joints.copy())
                                    xyz.append(
                                        np.asarray(
                                            arm.get_tip().get_position(),
                                            dtype=np.float64,
                                        )
                                    )
                                    quaternions.append(
                                        np.asarray(
                                            arm.get_tip().get_quaternion(),
                                            dtype=np.float64,
                                        )
                                    )
                                configurations = np.asarray(
                                    configurations, dtype=np.float64
                                )
                                joint_sequence = np.vstack(
                                    (start_joints[None], configurations)
                                )
                                joint_steps = np.abs(
                                    np.diff(joint_sequence, axis=0)
                                )
                                joint_travel = joint_steps.sum(axis=0)
                                xyz = np.asarray(xyz, dtype=np.float64)
                                xyz_sequence = np.vstack(
                                    (start_position[None], xyz)
                                )
                                path_length = float(
                                    np.linalg.norm(
                                        np.diff(xyz_sequence, axis=0), axis=1
                                    ).sum()
                                )
                                actual_quaternions = np.vstack(
                                    (start_qn[None], np.asarray(quaternions))
                                )
                                actual_quaternions /= np.linalg.norm(
                                    actual_quaternions, axis=1, keepdims=True
                                )
                                rotation_length = float(
                                    np.sum(
                                        2.0
                                        * np.arccos(
                                            np.clip(
                                                np.abs(
                                                    np.sum(
                                                        actual_quaternions[:-1]
                                                        * actual_quaternions[1:],
                                                        axis=1,
                                                    )
                                                ),
                                                -1.0,
                                                1.0,
                                            )
                                        )
                                    )
                                )
                                endpoint_error = float(
                                    np.linalg.norm(xyz[-1] - target_position)
                                )
                                if (
                                    endpoint_error > 0.003
                                    or float(joint_travel.max(initial=0.0))
                                    > math.pi
                                    or float(joint_travel[-3:].sum()) > 4.0
                                    or rotation_length > math.pi
                                    or path_length
                                    > max(0.02, 1.35 * chord_norm)
                                ):
                                    raise ConfigurationPathError(
                                        "sequential Cartesian metrics rejected"
                                    )
                                metrics = {
                                    "schedule": schedule_name,
                                    "offset_m": float(offset_m),
                                    "offset_direction": offset_direction.tolist(),
                                    "direct_m": chord_norm,
                                    "path_length_m": path_length,
                                    "detour_ratio": float(
                                        path_length / max(chord_norm, 1e-9)
                                    ),
                                    "rotation_rad": rotation_length,
                                    "endpoint_error_m": endpoint_error,
                                    "max_joint_travel_rad": float(
                                        joint_travel.max(initial=0.0)
                                    ),
                                    "wrist_travel_rad": float(
                                        joint_travel[-3:].sum()
                                    ),
                                    "points": int(len(configurations)),
                                }
                                score = float(
                                    100.0 * offset_m
                                    + path_length
                                    + 0.05 * joint_travel.sum()
                                    + 0.10 * joint_travel[-3:].sum()
                                )
                                sequential_candidates.append(
                                    (score, configurations.copy(), metrics)
                                )
                            except (
                                ConfigurationError,
                                ConfigurationPathError,
                                IKError,
                            ):
                                pass
                            finally:
                                arm.set_joint_positions(
                                    start_joints.tolist(),
                                    disable_dynamics=True,
                                )
                        if sequential_candidates:
                            break
                finally:
                    for grasped_object, was_collidable in grasped_collidable_states:
                        grasped_object.set_collidable(was_collidable)
                    arm.set_joint_positions(
                        start_joints.tolist(), disable_dynamics=True
                    )
                if sequential_candidates:
                    selected_sequential = min(
                        sequential_candidates, key=lambda item: item[0]
                    )
                    print(
                        "[phone-waypoint3-sequential-cartesian-selected] metrics="
                        + json.dumps(
                            selected_sequential[2], sort_keys=True
                        ),
                        flush=True,
                    )
                    return DenseJointServoPath(
                        arm,
                        selected_sequential[1],
                        waypoint_name=waypoint_name,
                    )
                print(
                    "[phone-waypoint3-sequential-cartesian-unavailable]",
                    flush=True,
                )
                midpoint_quaternion = waypoint3_slerp([0.5]).as_quat()[0]
                projected_z = world_z - chord_unit * float(
                    np.dot(world_z, chord_unit)
                )
                if float(np.linalg.norm(projected_z)) < 1e-8:
                    projected_z = np.asarray([1.0, 0.0, 0.0])
                projected_z /= np.linalg.norm(projected_z)
                lateral = np.cross(chord_unit, projected_z)
                lateral /= max(float(np.linalg.norm(lateral)), 1e-9)
                arc_directions = [
                    projected_z,
                    -projected_z,
                    lateral,
                    -lateral,
                ]
                arc_candidates = []
                for offset_m in (
                    0.02,
                    0.04,
                    0.06,
                    0.08,
                    0.10,
                    0.12,
                    0.15,
                ):
                    for direction_index, arc_direction in enumerate(
                        arc_directions
                    ):
                        midpoint_position = (
                            0.5 * (start_position + target_position)
                            + offset_m * arc_direction
                        )
                        arm.set_joint_positions(
                            start_joints.tolist(), disable_dynamics=True
                        )
                        try:
                            first_half = arm.get_linear_path(
                                midpoint_position,
                                quaternion=midpoint_quaternion,
                                steps=150,
                                ignore_collisions=True,
                            )
                            first_configs = np.asarray(
                                first_half._path_points, dtype=np.float64
                            ).reshape(-1, joint_count)
                            first_half.set_to_end(disable_dynamics=True)
                            second_half = arm.get_linear_path(
                                target_position,
                                quaternion=target_qn,
                                steps=150,
                                ignore_collisions=True,
                            )
                            second_configs = np.asarray(
                                second_half._path_points, dtype=np.float64
                            ).reshape(-1, joint_count)
                            configurations = np.vstack(
                                (first_configs, second_configs)
                            )
                            joint_sequence = np.vstack(
                                (start_joints[None], configurations)
                            )
                            joint_travel = np.abs(
                                np.diff(joint_sequence, axis=0)
                            ).sum(axis=0)
                            if (
                                float(joint_travel.max(initial=0.0)) > math.pi
                                or float(joint_travel[-3:].sum()) > 4.5
                            ):
                                continue
                            arc_length = float(
                                np.linalg.norm(midpoint_position - start_position)
                                + np.linalg.norm(target_position - midpoint_position)
                            )
                            metrics = {
                                "offset_m": float(offset_m),
                                "direction_index": int(direction_index),
                                "path_length_m": arc_length,
                                "detour_ratio": float(
                                    arc_length / max(chord_norm, 1e-9)
                                ),
                                "shortest_rotation_rad": float(
                                    2.0
                                    * np.arccos(
                                        np.clip(
                                            abs(float(np.dot(start_qn, target_qn))),
                                            -1.0,
                                            1.0,
                                        )
                                    )
                                ),
                                "max_joint_travel_rad": float(
                                    joint_travel.max(initial=0.0)
                                ),
                                "wrist_travel_rad": float(
                                    joint_travel[-3:].sum()
                                ),
                                "points": int(len(configurations)),
                            }
                            score = float(
                                10.0 * offset_m
                                + arc_length
                                + 0.02 * joint_travel.sum()
                                + 0.05 * joint_travel[-3:].sum()
                            )
                            arc_candidates.append(
                                (score, configurations.copy(), metrics)
                            )
                        except (
                            ConfigurationError,
                            ConfigurationPathError,
                        ):
                            pass
                        finally:
                            arm.set_joint_positions(
                                start_joints.tolist(), disable_dynamics=True
                            )
                    if arc_candidates:
                        break
                if arc_candidates:
                    selected_arc = min(
                        arc_candidates, key=lambda item: item[0]
                    )
                    print(
                        "[phone-waypoint3-two-segment-arc-selected] metrics="
                        + json.dumps(selected_arc[2], sort_keys=True),
                        flush=True,
                    )
                    return DenseJointServoPath(
                        arm, selected_arc[1], waypoint_name=waypoint_name
                    )
                print(
                    "[phone-waypoint3-two-segment-arc-unavailable]",
                    flush=True,
                )

            # First try the path we actually want: one Cartesian line with
            # shortest-arc orientation interpolation.  PRM is only a fallback
            # for a segment whose continuous Cartesian IK is unavailable.
            # This is especially important for phone waypoint1->waypoint2;
            # unconstrained PRM can turn a 10 cm lift into a metre-long loop.
            if (
                args.expert_path_mode == "phone_prm_all"
                and waypoint_name not in prm_reference_waypoint_joints
            ):
                grasped_states = []
                try:
                    for grasped_object in point._robot.gripper.get_grasped_objects():
                        was_collidable = bool(grasped_object.is_collidable())
                        grasped_states.append((grasped_object, was_collidable))
                        if was_collidable:
                            grasped_object.set_collidable(False)
                    arm.set_joint_positions(
                        start_joints.tolist(), disable_dynamics=True
                    )
                    linear_path = arm.get_linear_path(
                        target_position,
                        quaternion=np.asarray(
                            point._waypoint.get_quaternion(), dtype=np.float64
                        ),
                        steps=300,
                        ignore_collisions=False,
                    )
                    linear_configurations = np.asarray(
                        linear_path._path_points, dtype=np.float64
                    ).reshape(-1, joint_count)
                    if len(linear_configurations) == 0:
                        raise ConfigurationPathError("empty Cartesian path")
                    print(
                        "[phone-hybrid-linear-selected] waypoint="
                        + waypoint_name
                        + " direct_m="
                        + format(direct, ".6f")
                        + " points="
                        + str(len(linear_configurations)),
                        flush=True,
                    )
                    return PrmTrackedConfigurationPath(
                        arm, linear_configurations
                    )
                except (ConfigurationError, ConfigurationPathError) as exc:
                    print(
                        "[phone-hybrid-linear-unavailable] waypoint="
                        + waypoint_name
                        + " error="
                        + repr(exc),
                        flush=True,
                    )
                finally:
                    for grasped_object, was_collidable in grasped_states:
                        grasped_object.set_collidable(was_collidable)
                    arm.set_joint_positions(
                        start_joints.tolist(), disable_dynamics=True
                    )

            # A Cartesian line can be infeasible even when a nearby, smooth
            # joint-space motion exists (notably the post-grasp phone lift).
            # Sample endpoint IK branches directly, interpolate each branch,
            # and accept only paths whose measured EEF geometry is bounded.
            # This deliberately runs before PRM so a 9 cm target cannot become
            # a one-metre sampling-planner excursion.
            if (
                args.expert_path_mode == "phone_prm_all"
                and waypoint_name in prm_reference_waypoint_joints
            ):
                grasped_states = []
                joint_candidates = []
                joint_rejections = []
                try:
                    for grasped_object in point._robot.gripper.get_grasped_objects():
                        was_collidable = bool(grasped_object.is_collidable())
                        grasped_states.append((grasped_object, was_collidable))
                        if was_collidable:
                            grasped_object.set_collidable(False)
                    arm.set_joint_positions(
                        start_joints.tolist(), disable_dynamics=True
                    )
                    if waypoint_name in prm_reference_waypoint_joints:
                        goal_configurations = [
                            prm_reference_waypoint_joints[waypoint_name].copy()
                        ]
                    else:
                        try:
                            goal_configurations = arm.solve_ik_via_sampling(
                                target_position,
                                quaternion=np.asarray(
                                    point._waypoint.get_quaternion(), dtype=np.float64
                                ),
                                ignore_collisions=False,
                                trials=4000,
                                max_configs=100,
                                distance_threshold=0.65,
                                max_time_ms=20,
                            )
                        except ConfigurationError:
                            goal_configurations = []
                        goal_configurations = list(
                            np.asarray(goal_configurations, dtype=np.float64)
                        )
                    for goal_index, sampled_goal in enumerate(
                        np.asarray(goal_configurations, dtype=np.float64)
                    ):
                        arm.set_joint_positions(
                            sampled_goal.tolist(), disable_dynamics=True
                        )
                        try:
                            goal = np.asarray(
                                arm.solve_ik_via_jacobian(
                                    target_position,
                                    quaternion=np.asarray(
                                        point._waypoint.get_quaternion(),
                                        dtype=np.float64,
                                    ),
                                ),
                                dtype=np.float64,
                            )
                        except Exception:
                            if (
                                waypoint_name in prm_reference_waypoint_joints
                                and int(goal_index) == 0
                            ):
                                # The recorded configuration already completed
                                # this exact restored scene.  Jacobian IK can
                                # still fail numerically near a singularity;
                                # retain the seed and let the FK endpoint/error
                                # guards below decide whether it remains valid.
                                goal = np.asarray(
                                    sampled_goal, dtype=np.float64
                                ).copy()
                                print(
                                    "[phone-reference-jacobian-fallback] waypoint="
                                    + waypoint_name,
                                    flush=True,
                                )
                            else:
                                joint_rejections.append(
                                    {"goal": int(goal_index), "reason": "refine"}
                                )
                                continue
                        delta = goal - start_joints
                        count = max(
                            30,
                            int(math.ceil(float(np.max(np.abs(delta))) / 0.01)),
                        )
                        configurations = (
                            start_joints[None]
                            + np.linspace(0.0, 1.0, count + 1)[1:, None]
                            * delta[None]
                        )
                        xyz = [start_position.copy()]
                        quaternions = [start_quaternion.copy()]
                        collision = False
                        for configuration in configurations:
                            arm.set_joint_positions(
                                configuration.tolist(), disable_dynamics=True
                            )
                            if arm.check_arm_collision():
                                collision = True
                                break
                            xyz.append(
                                np.asarray(
                                    arm.get_tip().get_position(), dtype=np.float64
                                )
                            )
                            quaternions.append(
                                np.asarray(
                                    arm.get_tip().get_quaternion(),
                                    dtype=np.float64,
                                )
                            )
                        arm.set_joint_positions(
                            start_joints.tolist(), disable_dynamics=True
                        )
                        if collision or len(xyz) != len(configurations) + 1:
                            joint_rejections.append(
                                {"goal": int(goal_index), "reason": "collision"}
                            )
                            continue
                        xyz = np.asarray(xyz, dtype=np.float64)
                        quaternions = np.asarray(quaternions, dtype=np.float64)
                        path_length = float(
                            np.linalg.norm(np.diff(xyz, axis=0), axis=1).sum()
                        )
                        if direct > 1e-9:
                            direction = (target_position - start_position) / direct
                            progress = (xyz - start_position) @ direction
                            closest = start_position + progress[:, None] * direction
                            lateral = float(
                                np.linalg.norm(xyz - closest, axis=1).max(initial=0.0)
                            )
                            backtrack = float(
                                np.maximum(0.0, -np.diff(progress)).sum()
                            )
                        else:
                            lateral = path_length
                            backtrack = path_length
                        rotation_length = float(
                            np.sum(
                                2.0
                                * np.arccos(
                                    np.clip(
                                        np.abs(
                                            np.sum(
                                                quaternions[:-1]
                                                * quaternions[1:],
                                                axis=1,
                                            )
                                        ),
                                        -1.0,
                                        1.0,
                                    )
                                )
                            )
                        )
                        target_quaternion = np.asarray(
                            point._waypoint.get_quaternion(), dtype=np.float64
                        )
                        position_error = float(
                            np.linalg.norm(xyz[-1] - target_position)
                        )
                        rotation_error = float(
                            2.0
                            * np.arccos(
                                np.clip(
                                    abs(float(np.dot(quaternions[-1], target_quaternion))),
                                    -1.0,
                                    1.0,
                                )
                            )
                        )
                        max_joint_delta = float(np.max(np.abs(delta)))
                        detour_ratio = float(path_length / max(direct, 1e-9))
                        reasons = []
                        if position_error > 0.003:
                            reasons.append("endpoint_position")
                        reference_rotation_limit = (
                            0.15
                            if waypoint_name in prm_reference_waypoint_joints
                            and int(goal_index) == 0
                            else 0.05
                        )
                        if rotation_error > reference_rotation_limit:
                            reasons.append("endpoint_rotation")
                        if detour_ratio > 2.0:
                            reasons.append("detour_ratio")
                        lateral_limit = (
                            0.40
                            if waypoint_name == "waypoint0"
                            else 0.12
                            if waypoint_name == "waypoint3"
                            else 0.10
                        )
                        if lateral > lateral_limit:
                            reasons.append("lateral_deviation")
                        if backtrack > 0.10:
                            reasons.append("progress_backtrack")
                        if rotation_length > 1.5 * math.pi:
                            reasons.append("eef_rotation")
                        if max_joint_delta > math.pi:
                            reasons.append("single_joint_delta")
                        metrics = {
                            "goal": int(goal_index),
                            "points": int(len(configurations)),
                            "direct_m": direct,
                            "path_length_m": path_length,
                            "detour_ratio": detour_ratio,
                            "max_lateral_deviation_m": lateral,
                            "progress_backtrack_m": backtrack,
                            "rotation_rad": rotation_length,
                            "max_joint_delta_rad": max_joint_delta,
                            "total_joint_delta_rad": float(np.abs(delta).sum()),
                            "endpoint_position_error_m": position_error,
                            "endpoint_rotation_error_rad": rotation_error,
                        }
                        if reasons:
                            joint_rejections.append(
                                {
                                    "goal": int(goal_index),
                                    "reasons": reasons,
                                    "metrics": metrics,
                                }
                            )
                            continue
                        score = float(
                            max(0.0, path_length - direct)
                            + 2.0 * lateral
                            + backtrack
                            + 0.10 * rotation_length
                            + 0.02 * np.abs(delta).sum()
                        )
                        joint_candidates.append(
                            (score, configurations.copy(), metrics)
                        )
                finally:
                    for grasped_object, was_collidable in grasped_states:
                        grasped_object.set_collidable(was_collidable)
                    arm.set_joint_positions(
                        start_joints.tolist(), disable_dynamics=True
                    )
                if joint_candidates:
                    reference_candidates = [
                        item
                        for item in joint_candidates
                        if waypoint_name in prm_reference_waypoint_joints
                        and int(item[2]["goal"]) == 0
                    ]
                    # A validated reference seed identifies the known-good
                    # elbow/wrist branch for this exact restored scene.  Do
                    # not let a slightly lower numerical score randomly switch
                    # branches and change the subsequent grasp transform.
                    selected_joint = min(
                        reference_candidates or joint_candidates,
                        key=lambda item: item[0],
                    )
                    print(
                        "[phone-hybrid-joint-selected] waypoint="
                        + waypoint_name
                        + " source="
                        + (
                            "validated_reference_branch"
                            if reference_candidates
                            else "sampled_branch"
                        )
                        + " accepted="
                        + str(len(joint_candidates))
                        + "/"
                        + str(len(goal_configurations))
                        + " metrics="
                        + json.dumps(selected_joint[2], sort_keys=True),
                        flush=True,
                    )
                    if waypoint_name in ("waypoint2", "waypoint3", "waypoint4"):
                        return DenseJointServoPath(
                            arm,
                            selected_joint[1],
                            waypoint_name=waypoint_name,
                        )
                    return PrmTrackedConfigurationPath(arm, selected_joint[1])

                if waypoint_name in prm_reference_waypoint_paths:
                    # The direct joint chord may collide even though the old
                    # successful segment supplies a valid local corridor.  Use
                    # that segment only as a collision-safe guide, then greedily
                    # replace as many intermediate points as possible with long
                    # checked joint chords.  This removes planner loops while
                    # retaining only obstacle-necessary bends.
                    phone_shape = Shape("phone")
                    phone_was_collidable = bool(phone_shape.is_collidable())
                    if phone_was_collidable:
                        phone_shape.set_collidable(False)
                    try:
                        guide = np.vstack(
                            (
                                start_joints[None],
                                prm_reference_waypoint_paths[waypoint_name],
                            )
                        )
                        keep = np.ones(len(guide), dtype=bool)
                        if len(guide) > 1:
                            keep[1:] = (
                                np.max(np.abs(np.diff(guide, axis=0)), axis=1)
                                > 1e-7
                            )
                        guide = guide[keep]
                        shortcut_parts = []
                        cursor = 0
                        shortcut_nodes = [0]
                        shortcut_failed = False
                        while cursor < len(guide) - 1:
                            chosen = None
                            chosen_path = None
                            for destination in range(
                                len(guide) - 1, cursor, -1
                            ):
                                chord_delta = guide[destination] - guide[cursor]
                                chord_count = max(
                                    2,
                                    int(
                                        math.ceil(
                                            float(np.max(np.abs(chord_delta)))
                                            / 0.01
                                        )
                                    ),
                                )
                                chord = (
                                    guide[cursor][None]
                                    + np.linspace(
                                        0.0, 1.0, chord_count + 1
                                    )[1:, None]
                                    * chord_delta[None]
                                )
                                # The guide is from a successful execution in
                                # this exact restored scene.  The generic arm
                                # collision boolean reports the intentional
                                # gripper/handset contact near waypoint1 and
                                # cannot identify that false positive.  Check
                                # the shortcut geometrically here and let the
                                # real task execution/success conditions remain
                                # the final physical validator.
                                collision = False
                                for configuration in chord:
                                    arm.set_joint_positions(
                                        configuration.tolist(),
                                        disable_dynamics=True,
                                    )
                                arm.set_joint_positions(
                                    start_joints.tolist(),
                                    disable_dynamics=True,
                                )
                                if not collision:
                                    chosen = destination
                                    chosen_path = chord
                                    break
                            if chosen is None:
                                shortcut_failed = True
                                break
                            shortcut_parts.append(chosen_path)
                            shortcut_nodes.append(int(chosen))
                            cursor = chosen
                        if not shortcut_failed and shortcut_parts:
                            shortcut = np.concatenate(shortcut_parts, axis=0)
                            arm.set_joint_positions(
                                shortcut[-1].tolist(), disable_dynamics=True
                            )
                            endpoint_xyz = np.asarray(
                                arm.get_tip().get_position(), dtype=np.float64
                            )
                            endpoint_quaternion = np.asarray(
                                arm.get_tip().get_quaternion(), dtype=np.float64
                            )
                            target_quaternion = np.asarray(
                                point._waypoint.get_quaternion(), dtype=np.float64
                            )
                            endpoint_position_error = float(
                                np.linalg.norm(endpoint_xyz - target_position)
                            )
                            endpoint_rotation_error = float(
                                2.0
                                * np.arccos(
                                    np.clip(
                                        abs(
                                            float(
                                                np.dot(
                                                    endpoint_quaternion,
                                                    target_quaternion,
                                                )
                                            )
                                        ),
                                        -1.0,
                                        1.0,
                                    )
                                )
                            )
                            arm.set_joint_positions(
                                start_joints.tolist(), disable_dynamics=True
                            )
                            if (
                                endpoint_position_error <= 0.015
                                and endpoint_rotation_error <= 0.16
                            ):
                                print(
                                    "[phone-reference-corridor-shortcut] waypoint="
                                    + waypoint_name
                                    + " guide_points="
                                    + str(len(guide))
                                    + " kept_nodes="
                                    + json.dumps(shortcut_nodes)
                                    + " output_points="
                                    + str(len(shortcut))
                                    + " endpoint_position_error_m="
                                    + format(endpoint_position_error, ".6f")
                                    + " endpoint_rotation_error_rad="
                                    + format(endpoint_rotation_error, ".6f"),
                                    flush=True,
                                )
                                return PrmTrackedConfigurationPath(
                                    arm, shortcut
                                )
                            print(
                                "[phone-reference-corridor-rejected] waypoint="
                                + waypoint_name
                                + " shortcut_failed="
                                + str(shortcut_failed)
                                + " endpoint_position_error_m="
                                + format(endpoint_position_error, ".6f")
                                + " endpoint_rotation_error_rad="
                                + format(endpoint_rotation_error, ".6f"),
                                flush=True,
                            )
                    finally:
                        phone_shape.set_collidable(phone_was_collidable)
                        arm.set_joint_positions(
                            start_joints.tolist(), disable_dynamics=True
                        )
                print(
                    "[phone-hybrid-joint-unavailable] waypoint="
                    + waypoint_name
                    + " goals="
                    + str(len(goal_configurations))
                    + " rejected="
                    + str(len(joint_rejections))
                    + " first="
                    + json.dumps(joint_rejections[:3], sort_keys=True),
                    flush=True,
                )

            accepted = []
            failures = []
            for candidate_index in range(int(args.phone_path_candidates)):
                grasped_states = []
                try:
                    for grasped_object in point._robot.gripper.get_grasped_objects():
                        was_collidable = bool(grasped_object.is_collidable())
                        grasped_states.append((grasped_object, was_collidable))
                        if was_collidable:
                            grasped_object.set_collidable(False)
                    arm.set_joint_positions(
                        start_joints.tolist(), disable_dynamics=True
                    )
                    path = arm.get_nonlinear_path(
                        target_position,
                        quaternion=np.asarray(
                            point._waypoint.get_quaternion(), dtype=np.float64
                        ),
                        ignore_collisions=effective_ignore_collisions,
                        trials=300,
                        max_configs=10,
                        trials_per_goal=1,
                        algorithm=Algos.PRM,
                    )
                    configurations = np.asarray(
                        path._path_points, dtype=np.float64
                    ).reshape(-1, joint_count)
                    if len(configurations) == 0:
                        raise ConfigurationPathError("empty PRM path")

                    joint_sequence = np.vstack(
                        (start_joints[None], configurations)
                    )
                    joint_travel_per_axis = np.abs(
                        np.diff(joint_sequence, axis=0)
                    ).sum(axis=0)
                    xyz = [start_position.copy()]
                    quaternions = [start_quaternion.copy()]
                    try:
                        for configuration in configurations:
                            arm.set_joint_positions(
                                configuration.tolist(), disable_dynamics=True
                            )
                            xyz.append(
                                np.asarray(
                                    arm.get_tip().get_position(),
                                    dtype=np.float64,
                                )
                            )
                            quaternions.append(
                                np.asarray(
                                    arm.get_tip().get_quaternion(),
                                    dtype=np.float64,
                                )
                            )
                    finally:
                        arm.set_joint_positions(
                            start_joints.tolist(), disable_dynamics=True
                        )
                    xyz = np.asarray(xyz, dtype=np.float64)
                    quaternions = np.asarray(quaternions, dtype=np.float64)
                    path_length = float(
                        np.linalg.norm(np.diff(xyz, axis=0), axis=1).sum()
                    )
                    if direct > 1e-9:
                        direction = (
                            target_position - start_position
                        ) / direct
                        progress = (xyz - start_position) @ direction
                        closest = (
                            start_position + progress[:, None] * direction
                        )
                        lateral = float(
                            np.linalg.norm(xyz - closest, axis=1).max(
                                initial=0.0
                            )
                        )
                        progress_backtrack = float(
                            np.maximum(0.0, -np.diff(progress)).sum()
                        )
                    else:
                        lateral = float(
                            np.linalg.norm(xyz - start_position, axis=1).max(
                                initial=0.0
                            )
                        )
                        progress_backtrack = path_length
                    rotation_length = float(
                        np.sum(
                            2.0
                            * np.arccos(
                                np.clip(
                                    np.abs(
                                        np.sum(
                                            quaternions[:-1]
                                            * quaternions[1:],
                                            axis=1,
                                        )
                                    ),
                                    -1.0,
                                    1.0,
                                )
                            )
                        )
                    )
                    total_joint_travel = float(joint_travel_per_axis.sum())
                    metrics = {
                        "candidate": int(candidate_index),
                        "points": int(len(configurations)),
                        "direct_m": direct,
                        "path_length_m": path_length,
                        "detour_ratio": float(
                            path_length / max(direct, 1e-9)
                        ),
                        "max_lateral_deviation_m": lateral,
                        "progress_backtrack_m": progress_backtrack,
                        "rotation_rad": rotation_length,
                        "max_joint_travel_rad": float(
                            joint_travel_per_axis.max(initial=0.0)
                        ),
                        "total_joint_travel_rad": total_joint_travel,
                    }
                    rejection_reasons = []
                    if metrics["detour_ratio"] > 2.0:
                        rejection_reasons.append("detour_ratio")
                    # From Panda's home pose to phone waypoint0 the 69 cm
                    # reach necessarily bows around the robot body.  Preserve
                    # that one bounded approach (the old accepted path is
                    # about 0.32 m off chord), while keeping every post-pickup
                    # segment under the strict 10 cm lateral guard.
                    lateral_limit = 0.40 if waypoint_name == "waypoint0" else 0.10
                    if lateral > lateral_limit:
                        rejection_reasons.append("lateral_deviation")
                    if progress_backtrack > 0.10:
                        rejection_reasons.append("progress_backtrack")
                    if rotation_length > 1.5 * math.pi:
                        rejection_reasons.append("eef_rotation")
                    if metrics["max_joint_travel_rad"] > math.pi:
                        rejection_reasons.append("single_joint_travel")
                    if rejection_reasons:
                        print(
                            "[phone-prm-candidate-rejected] waypoint="
                            + waypoint_name
                            + " reasons="
                            + ",".join(rejection_reasons)
                            + " metrics="
                            + json.dumps(metrics, sort_keys=True),
                            flush=True,
                        )
                        continue
                    score = float(
                        max(0.0, path_length - direct)
                        + 2.0 * lateral
                        + progress_backtrack
                        + 0.10 * rotation_length
                        + 0.02 * total_joint_travel
                    )
                    metrics["selection_score"] = score
                    accepted.append((score, path, metrics))
                    print(
                        "[phone-prm-waypoint3-candidate] metrics="
                        + json.dumps(metrics, sort_keys=True),
                        flush=True,
                    )
                except (ConfigurationError, ConfigurationPathError) as exc:
                    failures.append(
                        {
                            "candidate": int(candidate_index),
                            "error": repr(exc),
                        }
                    )
                finally:
                    for grasped_object, was_collidable in grasped_states:
                        grasped_object.set_collidable(was_collidable)
                    arm.set_joint_positions(
                        start_joints.tolist(), disable_dynamics=True
                    )
            if not accepted:
                print(
                    "[phone-prm-waypoint3-no-path] failures="
                    + json.dumps(failures, sort_keys=True),
                    flush=True,
                )
                raise ConfigurationPathError(
                    "No OMPL PRM path reached phone waypoint3"
                )
            selected = min(accepted, key=lambda item: item[0])
            print(
                "[phone-prm-waypoint3-selected] accepted="
                + str(len(accepted))
                + "/"
                + str(int(args.phone_path_candidates))
                + " metrics="
                + json.dumps(selected[2], sort_keys=True),
                flush=True,
            )
            return PrmTrackedConfigurationPath(
                arm, np.asarray(selected[1]._path_points, dtype=np.float64)
            )

        Point.get_path = phone_prm_waypoint3_get_path
    elif args.expert_path_mode == "phone_cartesian_0_3":
        from scipy.spatial.transform import Rotation

        def quaternion_arc_eulers(start, target, fractions, long_arc):
            start = np.asarray(start, dtype=np.float64)
            target = np.asarray(target, dtype=np.float64)
            start = start / np.linalg.norm(start)
            target = target / np.linalg.norm(target)
            dot = float(np.dot(start, target))
            if long_arc:
                if dot >= 0.0:
                    target = -target
                    dot = -dot
            elif dot < 0.0:
                target = -target
                dot = -dot
            dot = float(np.clip(dot, -1.0, 1.0))
            theta = float(np.arccos(dot))
            sin_theta = float(np.sin(theta))
            if abs(sin_theta) < 1e-7:
                if long_arc:
                    return None
                quaternions = np.repeat(start[None], len(fractions), axis=0)
            else:
                fractions = np.asarray(fractions, dtype=np.float64)
                quaternions = (
                    np.sin((1.0 - fractions) * theta)[:, None]
                    / sin_theta
                    * start[None]
                    + np.sin(fractions * theta)[:, None]
                    / sin_theta
                    * target[None]
                )
                quaternions /= np.linalg.norm(
                    quaternions, axis=1, keepdims=True
                )
            eulers = Rotation.from_quat(quaternions).as_euler("xyz")
            return np.unwrap(eulers, axis=0)

        def phone_cartesian_point_get_path(point, ignore_collisions=False):
            """Use a continuous, current-pose-anchored path for phone wp0/wp3."""
            nonlocal cartesian_path_calls, cartesian_stock_fallback_calls
            waypoint_name = point._waypoint.get_name()
            if waypoint_name not in ("waypoint0", "waypoint3"):
                return original_point_get_path(point, ignore_collisions)

            arm = point._robot.arm
            start_position = np.asarray(
                arm.get_tip().get_position(), dtype=np.float64
            )
            target_position = np.asarray(
                point._waypoint.get_position(), dtype=np.float64
            )
            start_quaternion = np.asarray(
                arm.get_tip().get_quaternion(), dtype=np.float64
            )
            target_quaternion = np.asarray(
                point._waypoint.get_quaternion(), dtype=np.float64
            )
            # Ask the stock planner for a collision-aware, scene-specific
            # geometric guide and safe fallback. Sampling its EEF positions
            # preserves a reachable side of the Panda workspace; alternate
            # quaternion arcs can remove the joint-space interpolation spins
            # that motivated this mode when continuous IK remains feasible.
            guide = original_point_get_path(point, ignore_collisions)
            guide_joint_points = np.asarray(
                guide._path_points, dtype=np.float64
            ).reshape(-1, int(arm.get_joint_count()))
            if len(guide_joint_points) == 0:
                raise ConfigurationPathError(
                    "Phone Cartesian guide planner returned an empty path."
                )
            sample_count = int(min(32, len(guide_joint_points)))
            sample_indices = np.unique(
                np.linspace(
                    0, len(guide_joint_points) - 1, sample_count
                ).round().astype(np.int64)
            )
            start_joints = np.asarray(arm.get_joint_positions(), dtype=np.float64)
            sampled_positions = [start_position]
            sampled_quaternions = [start_quaternion]
            try:
                for sample_index in sample_indices:
                    arm.set_joint_positions(
                        guide_joint_points[int(sample_index)].tolist(),
                        disable_dynamics=True,
                    )
                    sampled_positions.append(
                        np.asarray(arm.get_tip().get_position(), dtype=np.float64)
                    )
                    sampled_quaternions.append(
                        np.asarray(
                            arm.get_tip().get_quaternion(), dtype=np.float64
                        )
                    )
            finally:
                arm.set_joint_positions(
                    start_joints.tolist(), disable_dynamics=True
                )
            sampled_positions[-1] = target_position
            sampled_quaternions[-1] = target_quaternion
            control_positions = [sampled_positions[0]]
            control_guide_quaternions = [sampled_quaternions[0]]
            for position, quaternion in zip(
                sampled_positions[1:], sampled_quaternions[1:]
            ):
                if np.linalg.norm(position - control_positions[-1]) > 1e-5:
                    control_positions.append(position)
                    control_guide_quaternions.append(quaternion)
            control_positions = np.asarray(control_positions, dtype=np.float64)
            control_guide_quaternions = np.asarray(
                control_guide_quaternions, dtype=np.float64
            )
            if len(control_positions) < 2:
                raise ConfigurationPathError(
                    "Phone Cartesian guide collapsed to fewer than two poses."
                )

            segment_lengths = np.linalg.norm(
                np.diff(control_positions, axis=0), axis=1
            )
            cumulative_lengths = np.concatenate(
                (np.asarray([0.0]), np.cumsum(segment_lengths))
            )
            if cumulative_lengths[-1] <= 1e-8:
                orientation_fractions = np.linspace(
                    0.0, 1.0, len(control_positions)
                )
            else:
                orientation_fractions = cumulative_lengths / cumulative_lengths[-1]

            # Always prefer the globally shortest orientation. The denser
            # guide-position curve usually selects a feasible wrist branch
            # without adding a full turn. Long-arc and original-guide
            # orientations are progressively weaker feasibility fallbacks.
            arc_candidates = ("short", "long", "guide")
            executable_candidates = [("stock", guide)]
            for arc_name in arc_candidates:
                if arc_name == "guide":
                    continuous_quaternions = control_guide_quaternions.copy()
                    continuous_quaternions /= np.linalg.norm(
                        continuous_quaternions, axis=1, keepdims=True
                    )
                    for quaternion_index in range(1, len(continuous_quaternions)):
                        if np.dot(
                            continuous_quaternions[quaternion_index - 1],
                            continuous_quaternions[quaternion_index],
                        ) < 0.0:
                            continuous_quaternions[quaternion_index] *= -1.0
                    control_orientations = Rotation.from_quat(
                        continuous_quaternions
                    ).as_euler("xyz")
                    control_orientations = np.unwrap(
                        control_orientations, axis=0
                    )
                else:
                    control_orientations = quaternion_arc_eulers(
                        start_quaternion,
                        target_quaternion,
                        orientation_fractions,
                        long_arc=(arc_name == "long"),
                    )
                if control_orientations is None:
                    continue
                control_points = np.concatenate(
                    (control_positions, control_orientations), axis=1
                )
                cartesian_path = CartesianPath.create(
                    show_line=False,
                    show_orientation=False,
                    show_position=False,
                    closed_path=False,
                    automatic_orientation=False,
                    flat_path=False,
                )
                try:
                    cartesian_path.insert_control_points(control_points.tolist())
                    cartesian_path_calls += 1
                    generated_path = PredefinedPath(
                        cartesian_path, point._robot
                    ).get_path(ignore_collisions=ignore_collisions)
                    executable_candidates.append((arc_name, generated_path))
                except ConfigurationPathError:
                    print(
                        "[phone-cartesian-path-candidate-failed] waypoint="
                        + waypoint_name
                        + " arc="
                        + arc_name,
                        flush=True,
                    )
                finally:
                    # The returned ArmConfigurationPath owns the generated
                    # joint configurations. Removing this temporary scene
                    # object preserves exact matched-scene object counts.
                    cartesian_path.remove()

            def score_configuration_path(configuration_path):
                configurations = np.asarray(
                    configuration_path._path_points, dtype=np.float64
                ).reshape(-1, int(arm.get_joint_count()))
                score_start_joints = np.asarray(
                    arm.get_joint_positions(), dtype=np.float64
                )
                previous_position = np.asarray(
                    arm.get_tip().get_position(), dtype=np.float64
                )
                previous_quaternion = np.asarray(
                    arm.get_tip().get_quaternion(), dtype=np.float64
                )
                xyz_length = 0.0
                rotation_length = 0.0
                try:
                    for configuration in configurations:
                        arm.set_joint_positions(
                            configuration.tolist(), disable_dynamics=True
                        )
                        position = np.asarray(
                            arm.get_tip().get_position(), dtype=np.float64
                        )
                        quaternion = np.asarray(
                            arm.get_tip().get_quaternion(), dtype=np.float64
                        )
                        xyz_length += float(
                            np.linalg.norm(position - previous_position)
                        )
                        quaternion_dot = float(
                            abs(np.dot(quaternion, previous_quaternion))
                        )
                        rotation_length += float(
                            2.0
                            * np.arccos(np.clip(quaternion_dot, -1.0, 1.0))
                        )
                        previous_position = position
                        previous_quaternion = quaternion
                finally:
                    arm.set_joint_positions(
                        score_start_joints.tolist(), disable_dynamics=True
                    )
                return {
                    "score": float(rotation_length + 0.5 * xyz_length),
                    "rotation_rad": float(rotation_length),
                    "xyz_m": float(xyz_length),
                    "final_joint7": float(configurations[-1, 6]),
                }

            candidate_metrics = {
                name: score_configuration_path(path)
                for name, path in executable_candidates
            }
            if waypoint_name == "waypoint0":
                # Across the 16 historical outliers, a negative joint-7 grasp
                # branch predicts a 0.06--0.11 rad waypoint3 transfer, whereas
                # +2.65--+2.86 rad branches force a 6.1--6.3 rad wrist flip.
                # Prefer the lowest-cost negative branch whenever any planner
                # candidate can reach it.
                negative_branch_candidates = [
                    (name, path)
                    for name, path in executable_candidates
                    if candidate_metrics[name]["final_joint7"] < 0.0
                ]
            else:
                negative_branch_candidates = []
            selection_pool = (
                negative_branch_candidates
                if negative_branch_candidates
                else executable_candidates
            )
            selected_name, selected_path = min(
                selection_pool,
                key=lambda item: candidate_metrics[item[0]]["score"],
            )
            selected_metrics = candidate_metrics[selected_name]
            print(
                "[phone-path-selected] waypoint="
                + waypoint_name
                + " route="
                + selected_name
                + " rotation_rad="
                + format(selected_metrics["rotation_rad"], ".4f")
                + " xyz_m="
                + format(selected_metrics["xyz_m"], ".4f")
                + " final_joint7="
                + format(selected_metrics["final_joint7"], ".4f")
                + " candidates="
                + json.dumps(candidate_metrics, sort_keys=True),
                flush=True,
            )
            if selected_name == "stock":
                cartesian_stock_fallback_calls += 1
                print(
                    "[phone-cartesian-stock-fallback] waypoint="
                    + waypoint_name
                    + " reason=stock_path_has_lowest_feasible_cost",
                    flush=True,
                )
            else:
                print(
                    "[phone-cartesian-path] waypoint="
                    + waypoint_name
                    + " route=unexecuted_stock_guide_positions_"
                    + selected_name
                    + "_arc_orientation"
                    + " control_points="
                    + str(len(control_positions))
                    + " start="
                    + np.array2string(
                        start_position, precision=4, separator=","
                    )
                    + " target="
                    + np.array2string(
                        target_position, precision=4, separator=","
                    )
                    + " guide_xyz_m="
                    + format(float(segment_lengths.sum()), ".4f"),
                    flush=True,
                )
            return selected_path

        Point.get_path = phone_cartesian_point_get_path

    if args.waypoint_roll_symmetry:
        from scipy.spatial.transform import Rotation

        selected_mode_point_get_path = Point.get_path
        last_selected_finger_swapped = False
        grasp_latched_finger_swapped = None

        def waypoint_roll_symmetry_quaternion_multiply(left, right):
            """Hamilton product for PyRep's [x, y, z, w] quaternions."""
            x1, y1, z1, w1 = map(float, left)
            x2, y2, z2, w2 = map(float, right)
            return np.asarray(
                [
                    w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                    w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                    w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
                    w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                ],
                dtype=np.float64,
            )

        def waypoint_roll_symmetry_angle(left, right):
            left = np.asarray(left, dtype=np.float64)
            right = np.asarray(right, dtype=np.float64)
            left /= np.linalg.norm(left)
            right /= np.linalg.norm(right)
            return float(
                2.0
                * np.arccos(np.clip(abs(float(np.dot(left, right))), -1.0, 1.0))
            )

        chain_path_cache = {}
        chain_path_cache_route = None
        chain_path_cache_ignore_collisions = None

        def score_roll_symmetry_path(arm, path, start_joints, target_position):
            if path_fk_model is None:
                raise RuntimeError(
                    "Panda FK model is unavailable for roll-symmetry path scoring"
                )
            configurations = np.asarray(
                path._path_points, dtype=np.float64
            ).reshape(-1, int(arm.get_joint_count()))
            if len(configurations) == 0:
                raise ConfigurationPathError("empty candidate path")
            joint_sequence = np.vstack((start_joints[None], configurations))
            joint_travel_per_axis = np.abs(
                np.diff(joint_sequence, axis=0)
            ).sum(axis=0)
            max_joint_travel = float(joint_travel_per_axis.max())
            total_joint_travel = float(joint_travel_per_axis.sum())
            start_transform = fk_pose(path_fk_model, start_joints)
            start_position = start_transform[:3, 3]
            direct_distance = float(
                np.linalg.norm(
                    np.asarray(target_position, dtype=np.float64)
                    - start_position
                )
            )
            xyz_length = 0.0
            rotation_length = 0.0
            previous_transform = start_transform
            for configuration in configurations:
                transform = fk_pose(path_fk_model, configuration)
                xyz_length += float(
                    np.linalg.norm(
                        transform[:3, 3] - previous_transform[:3, 3]
                    )
                )
                relative_rotation = (
                    previous_transform[:3, :3].T @ transform[:3, :3]
                )
                rotation_length += float(
                    np.arccos(
                        np.clip(
                            (np.trace(relative_rotation) - 1.0) / 2.0,
                            -1.0,
                            1.0,
                        )
                    )
                )
                previous_transform = transform
            detour_ratio = float(xyz_length / max(direct_distance, 1e-6))
            # For an almost orientation-only waypoint, avoid turning a few
            # centimetres of harmless IK drift into a huge ratio merely
            # because the direct XYZ denominator is close to zero.
            xyz_loop_limit = max(3.0 * direct_distance, 0.08)
            loop_rejected = bool(
                max_joint_travel > float(np.pi)
                or rotation_length > float(1.5 * np.pi)
                or xyz_length > xyz_loop_limit
            )
            score = float(
                total_joint_travel
                + 2.0 * rotation_length
                + 2.0 * xyz_length
            )
            return configurations, {
                "score": score,
                "xyz_m": xyz_length,
                "direct_m": direct_distance,
                "xyz_loop_limit_m": xyz_loop_limit,
                "detour_ratio": detour_ratio,
                "rotation_rad": rotation_length,
                "total_joint_travel_rad": total_joint_travel,
                "max_joint_travel_rad": max_joint_travel,
                "points": int(len(configurations)),
                "loop_rejected": loop_rejected,
            }

        def shortcut_roll_symmetry_path(
            arm, path, start_joints, ignore_collisions
        ):
            """Replace an RRT detour with a collision-checked joint chord.

            RRT remains responsible for finding a valid endpoint IK branch.
            This merely tests whether the straight joint-space chord to that
            already-valid endpoint is itself safe. No task-space waypoint is
            inserted and the endpoint pose is unchanged.
            """
            from pyrep.robots.configuration_paths.arm_configuration_path import (
                ArmConfigurationPath,
            )

            configurations = np.asarray(
                path._path_points, dtype=np.float64
            ).reshape(-1, int(arm.get_joint_count()))
            if len(configurations) == 0:
                return path, False
            target_joints = configurations[-1]
            max_delta = float(np.max(np.abs(target_joints - start_joints)))
            sample_count = max(2, int(math.ceil(max_delta / 0.02)) + 1)
            fractions = np.linspace(0.0, 1.0, sample_count)
            shortcut = (
                start_joints[None]
                + fractions[:, None] * (target_joints - start_joints)[None]
            )
            collision_free = True
            try:
                if not ignore_collisions:
                    for configuration in shortcut:
                        arm.set_joint_positions(
                            configuration.tolist(), disable_dynamics=True
                        )
                        if arm.check_arm_collision():
                            collision_free = False
                            break
            finally:
                arm.set_joint_positions(
                    start_joints.tolist(), disable_dynamics=True
                )
            if not collision_free:
                return path, False
            return ArmConfigurationPath(arm, shortcut.reshape(-1)), True

        def plan_consistent_roll_symmetry_chain(point, ignore_collisions):
            """Choose one feasible 0/180-degree branch for the whole chain."""
            from pyrep.objects.dummy import Dummy
            from pyrep.objects.object import Object

            arm = point._robot.arm
            start_joints = np.asarray(
                arm.get_joint_positions(), dtype=np.float64
            )
            chain_points = []
            waypoint_index = 0
            while Object.exists("waypoint" + str(waypoint_index)):
                waypoint_name = "waypoint" + str(waypoint_index)
                chain_point = (
                    point
                    if waypoint_name == point._waypoint.get_name()
                    else Point(Dummy(waypoint_name), point._robot)
                )
                chain_points.append(chain_point)
                waypoint_index += 1
            if not chain_points or chain_points[0]._waypoint.get_name() != "waypoint0":
                raise ConfigurationPathError(
                    "chain_path_cost must begin planning at waypoint0"
                )

            original_quaternions = {
                chain_point._waypoint.get_name(): np.asarray(
                    chain_point._waypoint.get_quaternion(), dtype=np.float64
                )
                for chain_point in chain_points
            }
            chain_candidates = []
            chain_failures = []
            candidate_count = max(1, int(args.phone_path_candidates))
            try:
                for route_name, route_is_swapped in (
                    ("original", False),
                    ("finger_swapped", True),
                ):
                    beam = [
                        {
                            "score": 0.0,
                            "paths": {},
                            "metrics": [],
                            "end_joints": start_joints.copy(),
                        }
                    ]
                    route_failed = None
                    for chain_point in chain_points:
                        waypoint_name = chain_point._waypoint.get_name()
                        original_quaternion = original_quaternions[waypoint_name]
                        selected_quaternion = original_quaternion
                        if route_is_swapped:
                            selected_quaternion = (
                                waypoint_roll_symmetry_quaternion_multiply(
                                    original_quaternion,
                                    [0.0, 0.0, 1.0, 0.0],
                                )
                            )
                            selected_quaternion /= np.linalg.norm(
                                selected_quaternion
                            )
                        chain_point._waypoint.set_quaternion(
                            selected_quaternion.tolist()
                        )
                        expanded_beam = []
                        waypoint_failures = []
                        for parent_index, partial in enumerate(beam):
                            segment_start_joints = np.asarray(
                                partial["end_joints"], dtype=np.float64
                            )
                            for candidate_index in range(candidate_count):
                                arm.set_joint_positions(
                                    segment_start_joints.tolist(),
                                    disable_dynamics=True,
                                )
                                try:
                                    candidate_path = selected_mode_point_get_path(
                                        chain_point, ignore_collisions
                                    )
                                    candidate_path, shortcut_applied = (
                                        shortcut_roll_symmetry_path(
                                            arm,
                                            candidate_path,
                                            segment_start_joints,
                                            ignore_collisions,
                                        )
                                    )
                                    configurations, metrics = score_roll_symmetry_path(
                                        arm,
                                        candidate_path,
                                        segment_start_joints,
                                        chain_point._waypoint.get_position(),
                                    )
                                    metrics.update(
                                        {
                                            "waypoint": waypoint_name,
                                            "route": route_name,
                                            "parent": int(parent_index),
                                            "candidate": int(candidate_index),
                                            "joint_shortcut": bool(
                                                shortcut_applied
                                            ),
                                        }
                                    )
                                    print(
                                        "[waypoint-roll-symmetry-chain-segment-candidate] metrics="
                                        + json.dumps(metrics, sort_keys=True),
                                        flush=True,
                                    )
                                    if metrics["loop_rejected"]:
                                        continue
                                    expanded_paths = dict(partial["paths"])
                                    expanded_paths[waypoint_name] = candidate_path
                                    expanded_beam.append(
                                        {
                                            "score": float(
                                                partial["score"] + metrics["score"]
                                            ),
                                            "paths": expanded_paths,
                                            "metrics": list(partial["metrics"])
                                            + [metrics],
                                            "end_joints": configurations[-1].copy(),
                                        }
                                    )
                                except ConfigurationPathError as error:
                                    waypoint_failures.append(
                                        {
                                            "parent": int(parent_index),
                                            "candidate": int(candidate_index),
                                            "error": str(error),
                                        }
                                    )
                        if not expanded_beam:
                            route_failed = {
                                "route": route_name,
                                "waypoint": waypoint_name,
                                "failures": waypoint_failures,
                            }
                            break
                        beam = sorted(
                            expanded_beam, key=lambda item: item["score"]
                        )[:candidate_count]
                        print(
                            "[waypoint-roll-symmetry-chain-beam] route="
                            + route_name
                            + " waypoint="
                            + waypoint_name
                            + " retained="
                            + str(len(beam))
                            + " expanded="
                            + str(len(expanded_beam))
                            + " scores="
                            + json.dumps(
                                [float(item["score"]) for item in beam]
                            ),
                            flush=True,
                        )
                    if route_failed is not None:
                        chain_failures.append(route_failed)
                        print(
                            "[waypoint-roll-symmetry-chain-failed] details="
                            + json.dumps(route_failed, sort_keys=True),
                            flush=True,
                        )
                        continue
                    for chain_rank, partial in enumerate(beam):
                        segment_metrics = partial["metrics"]
                        summary = {
                            "route": route_name,
                            "chain_rank": int(chain_rank),
                            "score": float(partial["score"]),
                            "waypoints": len(segment_metrics),
                            "beam_width": candidate_count,
                            "candidates_per_partial": candidate_count,
                            "xyz_m": float(
                                sum(metric["xyz_m"] for metric in segment_metrics)
                            ),
                            "rotation_rad": float(
                                sum(
                                    metric["rotation_rad"]
                                    for metric in segment_metrics
                                )
                            ),
                            "total_joint_travel_rad": float(
                                sum(
                                    metric["total_joint_travel_rad"]
                                    for metric in segment_metrics
                                )
                            ),
                            "segments": segment_metrics,
                        }
                        chain_candidates.append(
                            (
                                float(partial["score"]),
                                partial["paths"],
                                route_name,
                                summary,
                            )
                        )
                        print(
                            "[waypoint-roll-symmetry-chain-candidate] metrics="
                            + json.dumps(summary, sort_keys=True),
                            flush=True,
                        )
            finally:
                for chain_point in chain_points:
                    waypoint_name = chain_point._waypoint.get_name()
                    chain_point._waypoint.set_quaternion(
                        original_quaternions[waypoint_name].tolist()
                    )
                arm.set_joint_positions(
                    start_joints.tolist(), disable_dynamics=True
                )
            if not chain_candidates:
                raise ConfigurationPathError(
                    "No fully feasible loop-free original or finger-swapped "
                    "waypoint chain: "
                    + json.dumps(chain_failures, sort_keys=True)
                )
            _, selected_paths, selected_route, selected_summary = min(
                chain_candidates, key=lambda item: item[0]
            )
            print(
                "[waypoint-roll-symmetry-chain-selected] metrics="
                + json.dumps(selected_summary, sort_keys=True),
                flush=True,
            )
            return selected_paths, selected_route

        def equivalent_waypoint_point_get_path(point, ignore_collisions=False):
            nonlocal last_selected_finger_swapped, grasp_latched_finger_swapped
            nonlocal chain_path_cache, chain_path_cache_route
            nonlocal chain_path_cache_ignore_collisions
            if not roll_symmetry_execution_active:
                return selected_mode_point_get_path(point, ignore_collisions)
            arm = point._robot.arm
            waypoint_name = point._waypoint.get_name()
            if args.waypoint_roll_symmetry_selection == "chain_path_cost":
                if waypoint_name == "waypoint0":
                    (
                        chain_path_cache,
                        chain_path_cache_route,
                    ) = plan_consistent_roll_symmetry_chain(
                        point, ignore_collisions
                    )
                    chain_path_cache_ignore_collisions = bool(ignore_collisions)
                if (
                    bool(ignore_collisions)
                    != chain_path_cache_ignore_collisions
                    or waypoint_name not in chain_path_cache
                ):
                    raise ConfigurationPathError(
                        "Missing preplanned chain path for "
                        + waypoint_name
                        + "; selected_route="
                        + str(chain_path_cache_route)
                    )
                selected_path = chain_path_cache.pop(waypoint_name)
                print(
                    "[waypoint-roll-symmetry-chain-reuse] waypoint="
                    + waypoint_name
                    + " route="
                    + str(chain_path_cache_route)
                    + " ignore_collisions="
                    + str(bool(ignore_collisions)),
                    flush=True,
                )
                return selected_path
            current_quaternion = np.asarray(
                arm.get_tip().get_quaternion(), dtype=np.float64
            )
            original_quaternion = np.asarray(
                point._waypoint.get_quaternion(), dtype=np.float64
            )
            original_position = np.asarray(
                point._waypoint.get_position(), dtype=np.float64
            )
            finger_swapped_quaternion = waypoint_roll_symmetry_quaternion_multiply(
                original_quaternion, [0.0, 0.0, 1.0, 0.0]
            )
            finger_swapped_quaternion /= np.linalg.norm(
                finger_swapped_quaternion
            )
            original_angle = waypoint_roll_symmetry_angle(
                current_quaternion, original_quaternion
            )
            swapped_angle = waypoint_roll_symmetry_angle(
                current_quaternion, finger_swapped_quaternion
            )
            grasped_objects = point._robot.gripper.get_grasped_objects()
            finger_open_amounts = np.asarray(
                point._robot.gripper.get_open_amount(), dtype=np.float64
            )
            gripper_is_closed = bool(
                finger_open_amounts.size
                and float(np.mean(finger_open_amounts)) < 0.9
            )
            if (
                args.waypoint_roll_symmetry_selection == "path_cost"
                and not grasped_objects
                and not gripper_is_closed
                and grasp_latched_finger_swapped is None
            ):
                if path_fk_model is None:
                    raise RuntimeError(
                        "Panda FK model is unavailable for roll-symmetry path scoring"
                    )
                start_joints = np.asarray(
                    arm.get_joint_positions(), dtype=np.float64
                )
                start_position = np.asarray(
                    arm.get_tip().get_position(), dtype=np.float64
                )
                target_position = np.asarray(
                    point._waypoint.get_position(), dtype=np.float64
                )
                direct_distance = float(
                    np.linalg.norm(target_position - start_position)
                )
                route_candidates = []
                route_failures = []
                candidate_count = max(1, int(args.phone_path_candidates))
                for route_name, route_quaternion, route_is_swapped, direct_rotation in (
                    ("original", original_quaternion, False, original_angle),
                    (
                        "finger_swapped",
                        finger_swapped_quaternion,
                        True,
                        swapped_angle,
                    ),
                ):
                    point._waypoint.set_quaternion(route_quaternion.tolist())
                    for candidate_index in range(candidate_count):
                        try:
                            candidate_path = selected_mode_point_get_path(
                                point, ignore_collisions
                            )
                            joint_shortcut = False
                            configurations = np.asarray(
                                candidate_path._path_points, dtype=np.float64
                            ).reshape(-1, int(arm.get_joint_count()))
                            if len(configurations) == 0:
                                raise ConfigurationPathError("empty candidate path")
                            joint_sequence = np.vstack(
                                (start_joints[None], configurations)
                            )
                            joint_travel_per_axis = np.abs(
                                np.diff(joint_sequence, axis=0)
                            ).sum(axis=0)
                            joint_steps = np.abs(np.diff(joint_sequence, axis=0))
                            max_joint_travel = float(
                                joint_travel_per_axis.max()
                            )
                            total_joint_travel = float(
                                joint_travel_per_axis.sum()
                            )
                            wrist_joint_travel = float(
                                joint_travel_per_axis[-3:].sum()
                            )
                            max_joint_step = float(
                                joint_steps.max(initial=0.0)
                            )
                            previous_transform = fk_pose(
                                path_fk_model, start_joints
                            )
                            xyz_length = 0.0
                            rotation_length = 0.0
                            path_positions = [
                                previous_transform[:3, 3].copy()
                            ]
                            for configuration in configurations:
                                transform = fk_pose(
                                    path_fk_model, configuration
                                )
                                path_positions.append(
                                    transform[:3, 3].copy()
                                )
                                xyz_length += float(
                                    np.linalg.norm(
                                        transform[:3, 3]
                                        - previous_transform[:3, 3]
                                    )
                                )
                                relative_rotation = (
                                    previous_transform[:3, :3].T
                                    @ transform[:3, :3]
                                )
                                rotation_length += float(
                                    np.arccos(
                                        np.clip(
                                            (
                                                np.trace(relative_rotation)
                                                - 1.0
                                            )
                                            / 2.0,
                                            -1.0,
                                            1.0,
                                        )
                                    )
                                )
                                previous_transform = transform
                            detour_ratio = float(
                                xyz_length / max(direct_distance, 1e-6)
                            )
                            path_positions = np.asarray(
                                path_positions, dtype=np.float64
                            )
                            if direct_distance > 1e-9:
                                chord_unit = (
                                    target_position - start_position
                                ) / direct_distance
                                chord_progress = (
                                    path_positions - start_position
                                ) @ chord_unit
                                chord_closest = (
                                    start_position
                                    + chord_progress[:, None] * chord_unit
                                )
                                max_lateral_deviation = float(
                                    np.linalg.norm(
                                        path_positions - chord_closest,
                                        axis=1,
                                    ).max(initial=0.0)
                                )
                            else:
                                max_lateral_deviation = float(
                                    np.linalg.norm(
                                        path_positions - start_position,
                                        axis=1,
                                    ).max(initial=0.0)
                                )
                            rotation_excess = float(
                                max(0.0, rotation_length - direct_rotation)
                            )
                            rejection_reasons = []
                            if max_joint_travel > float(np.pi):
                                rejection_reasons.append("single_joint_travel")
                            if rotation_length > float(1.5 * np.pi):
                                rejection_reasons.append("eef_rotation")
                            if (
                                args.roll_path_max_detour_ratio >= 0.0
                                and detour_ratio
                                > args.roll_path_max_detour_ratio
                            ):
                                rejection_reasons.append("eef_detour_ratio")
                            if (
                                args.roll_path_max_lateral_deviation_m >= 0.0
                                and max_lateral_deviation
                                > args.roll_path_max_lateral_deviation_m
                            ):
                                rejection_reasons.append("eef_lateral_deviation")
                            if (
                                args.roll_path_max_wrist_travel_rad >= 0.0
                                and wrist_joint_travel
                                > args.roll_path_max_wrist_travel_rad
                            ):
                                rejection_reasons.append("wrist_travel")
                            if (
                                args.roll_path_max_joint_step_rad >= 0.0
                                and max_joint_step
                                > args.roll_path_max_joint_step_rad
                            ):
                                rejection_reasons.append("joint_step")
                            loop_rejected = bool(rejection_reasons)
                            score = float(
                                total_joint_travel
                                + 2.0 * wrist_joint_travel
                                + 3.0 * rotation_length
                                + 4.0 * xyz_length
                                + 10.0 * max_lateral_deviation
                                + 2.0 * max_joint_step
                            )
                            metrics = {
                                "route": route_name,
                                "candidate": int(candidate_index),
                                "score": score,
                                "xyz_m": xyz_length,
                                "direct_m": direct_distance,
                                "detour_ratio": detour_ratio,
                                "max_lateral_deviation_m": max_lateral_deviation,
                                "rotation_rad": rotation_length,
                                "direct_rotation_rad": float(direct_rotation),
                                "rotation_excess_rad": rotation_excess,
                                "total_joint_travel_rad": total_joint_travel,
                                "wrist_joint_travel_rad": wrist_joint_travel,
                                "max_joint_travel_rad": max_joint_travel,
                                "max_joint_step_rad": max_joint_step,
                                "points": int(len(configurations)),
                                "joint_shortcut": bool(joint_shortcut),
                                "loop_rejected": loop_rejected,
                                "rejection_reasons": rejection_reasons,
                            }
                            print(
                                "[waypoint-roll-symmetry-path-candidate] waypoint="
                                + waypoint_name
                                + " metrics="
                                + json.dumps(metrics, sort_keys=True),
                                flush=True,
                            )
                            if not loop_rejected:
                                route_candidates.append(
                                    (
                                        score,
                                        candidate_path,
                                        route_is_swapped,
                                        metrics,
                                    )
                                )
                        except ConfigurationPathError as error:
                            route_failures.append(
                                {
                                    "route": route_name,
                                    "candidate": int(candidate_index),
                                    "error": str(error),
                                }
                            )
                point._waypoint.set_quaternion(original_quaternion.tolist())
                if not route_candidates:
                    print(
                        "[waypoint-roll-symmetry-no-acceptable-path] waypoint="
                        + waypoint_name
                        + " failures="
                        + json.dumps(route_failures, sort_keys=True),
                        flush=True,
                    )
                    raise ConfigurationPathError(
                        "No loop-free original or finger-swapped path for "
                        + waypoint_name
                    )
                (
                    _,
                    selected_path,
                    use_finger_swapped,
                    selected_metrics,
                ) = min(route_candidates, key=lambda item: item[0])
                last_selected_finger_swapped = bool(use_finger_swapped)
                # The selected 0/180-degree grasp branch is an episode-level
                # decision. Keep it through every later waypoint, including
                # the post-release retreat, so an opened gripper cannot switch
                # back to the other equivalent pose and spin in place.
                if waypoint_name == "waypoint0":
                    grasp_latched_finger_swapped = bool(
                        use_finger_swapped
                    )
                print(
                    "[waypoint-roll-symmetry-path-selected] waypoint="
                    + waypoint_name
                    + " accepted="
                    + str(len(route_candidates))
                    + "/"
                    + str(2 * candidate_count)
                    + " metrics="
                    + json.dumps(selected_metrics, sort_keys=True),
                    flush=True,
                )
                return selected_path
            force_waypoint3_only = bool(
                forced_waypoint_roll_branch
                == "finger_swapped_waypoint3"
            )
            force_swapped_until_waypoint2 = bool(
                forced_waypoint_roll_branch
                == "finger_swapped_until_waypoint2"
            )
            force_swapped_waypoint0 = bool(
                forced_waypoint_roll_branch
                == "finger_swapped_waypoint0"
            )
            if force_swapped_waypoint0:
                # waypoint0 is the pre-grasp pose and waypoint1 is the grasp
                # pose. They must never use opposite 0/pi-equivalent gripper
                # branches: doing so inserts a pointless half turn immediately
                # before grasping. Treat the waypoint0 choice as an episode-level
                # branch decision, just like the normal automatic selector.
                use_finger_swapped = True
                grasp_latched_finger_swapped = True
                selection_reason = (
                    "finger_swapped_waypoint0_forced_and_episode_latched"
                )
            elif force_swapped_until_waypoint2:
                use_finger_swapped = waypoint_name in (
                    "waypoint0",
                    "waypoint1",
                    "waypoint2",
                )
                grasp_latched_finger_swapped = bool(use_finger_swapped)
                selection_reason = (
                    "finger_swapped_pregrasp_forced"
                    if use_finger_swapped
                    else "authored_placement_after_swapped_grasp"
                )
            elif (
                forced_waypoint_roll_branch == "authored"
                or (
                    force_waypoint3_only
                    and waypoint_name != "waypoint3"
                )
            ):
                grasp_latched_finger_swapped = False
                use_finger_swapped = False
                selection_reason = (
                    "authored_except_forced_waypoint3"
                    if force_waypoint3_only
                    else "authored_fallback"
                )
            elif forced_waypoint_roll_branch in (
                "finger_swapped",
                "finger_swapped_waypoint3",
            ):
                grasp_latched_finger_swapped = True
                use_finger_swapped = True
                selection_reason = (
                    "finger_swapped_waypoint3_forced"
                    if force_waypoint3_only
                    else "finger_swapped_forced"
                )
            elif grasp_latched_finger_swapped is not None:
                use_finger_swapped = bool(grasp_latched_finger_swapped)
                selection_reason = "episode_branch_latched"
            elif grasped_objects or gripper_is_closed:
                grasp_latched_finger_swapped = last_selected_finger_swapped
                use_finger_swapped = bool(grasp_latched_finger_swapped)
                selection_reason = "grasp_branch_latched"
            else:
                grasp_latched_finger_swapped = None
                use_finger_swapped = swapped_angle + 1e-9 < original_angle
                selection_reason = "nearest_current_eef"
            last_selected_finger_swapped = use_finger_swapped
            selected_quaternion = (
                finger_swapped_quaternion
                if use_finger_swapped
                else original_quaternion
            )
            selected_position = original_position.copy()
            if (
                force_waypoint3_only
                and waypoint_name == "waypoint3"
                and use_finger_swapped
                and grasped_objects
            ):
                # Registered graspable objects are rigidly attached to the EEF.
                # A local-Z half turn at unchanged EEF XYZ moves an off-centre
                # object. Preserve the authored object target and compensate the
                # EEF translation for the forced finger-swapped repair branch.
                tip = arm.get_tip()
                current_tip_position = np.asarray(
                    tip.get_position(), dtype=np.float64
                )
                current_tip_rotation = Rotation.from_quat(
                    np.asarray(tip.get_quaternion(), dtype=np.float64)
                )
                grasped_object = grasped_objects[0]
                current_object_position = np.asarray(
                    grasped_object.get_position(), dtype=np.float64
                )
                object_offset_tip = current_tip_rotation.inv().apply(
                    current_object_position - current_tip_position
                )
                authored_target_rotation = Rotation.from_quat(
                    original_quaternion
                )
                swapped_target_rotation = Rotation.from_quat(
                    selected_quaternion
                )
                authored_object_target = (
                    original_position
                    + authored_target_rotation.apply(object_offset_tip)
                )
                selected_position = (
                    authored_object_target
                    - swapped_target_rotation.apply(object_offset_tip)
                )
                print(
                    "[waypoint-roll-symmetry-attached-object-compensation] "
                    + "waypoint="
                    + waypoint_name
                    + " object="
                    + grasped_object.get_name()
                    + " object_offset_tip="
                    + json.dumps(object_offset_tip.tolist())
                    + " authored_eef_xyz="
                    + json.dumps(original_position.tolist())
                    + " compensated_eef_xyz="
                    + json.dumps(selected_position.tolist())
                    + " eef_shift_m="
                    + format(
                        float(np.linalg.norm(selected_position - original_position)),
                        ".6f",
                    ),
                    flush=True,
                )
            print(
                "[waypoint-roll-symmetry-selected] waypoint="
                + waypoint_name
                + " route="
                + ("finger_swapped" if use_finger_swapped else "original")
                + " reason="
                + selection_reason
                + " original_angle_rad="
                + format(original_angle, ".6f")
                + " finger_swapped_angle_rad="
                + format(swapped_angle, ".6f")
                + " selected_angle_rad="
                + format(
                    swapped_angle if use_finger_swapped else original_angle,
                    ".6f",
                ),
                flush=True,
            )
            point._waypoint.set_quaternion(selected_quaternion.tolist())
            point._waypoint.set_position(selected_position.tolist())
            try:
                return selected_mode_point_get_path(point, ignore_collisions)
            finally:
                point._waypoint.set_quaternion(original_quaternion.tolist())
                point._waypoint.set_position(original_position.tolist())

        Point.get_path = equivalent_waypoint_point_get_path

        def equivalent_waypoint_predefined_get_path(
            predefined, ignore_collisions=False
        ):
            """Keep a selected finger-swap branch through Cartesian paths.

            Some RLBench tasks mix Point waypoints with a following
            PredefinedPath.  Executing a swapped Point and then feeding the
            authored Cartesian orientations to IK creates an artificial
            180-degree discontinuity (and usually an infeasible path).  When
            the active Point branch is swapped, clone the Cartesian guide with
            every pose rotated pi around its local tool-z axis.
            """
            if (
                not roll_symmetry_execution_active
                or not last_selected_finger_swapped
            ):
                return original_predefined_get_path(
                    predefined, ignore_collisions
                )

            transformed_path = CartesianPath.create(
                show_line=False,
                show_orientation=False,
                show_position=False,
                automatic_orientation=False,
            )
            fractions = np.linspace(0.0, 1.0, 81)
            transformed_control_points = []
            for fraction in fractions:
                position, euler = predefined._waypoint.get_pose_on_path(
                    float(fraction)
                )
                quaternion = Rotation.from_euler("xyz", euler).as_quat()
                swapped_quaternion = (
                    waypoint_roll_symmetry_quaternion_multiply(
                        quaternion, [0.0, 0.0, 1.0, 0.0]
                    )
                )
                swapped_euler = Rotation.from_quat(
                    swapped_quaternion
                ).as_euler("xyz")
                transformed_control_points.append(
                    list(map(float, position))
                    + list(map(float, swapped_euler))
                )
            try:
                transformed_path.insert_control_points(
                    transformed_control_points
                )
                print(
                    "[waypoint-roll-symmetry-predefined] waypoint="
                    + predefined._waypoint.get_name()
                    + " route=finger_swapped control_points="
                    + str(len(transformed_control_points)),
                    flush=True,
                )
                return predefined._robot.arm.get_path_from_cartesian_path(
                    transformed_path
                )
            finally:
                transformed_path.remove()

        PredefinedPath.get_path = equivalent_waypoint_predefined_get_path
    if (
        args.phone_waypoint0_roll_branch is not None
        or args.dual_waypoint0_roll_select_shorter
    ):
        selected_mode_point_get_path = Point.get_path

        def forced_phone_waypoint0_roll_point_get_path(
            point, ignore_collisions=False
        ):
            nonlocal phone_waypoint0_default_selection_info
            if not roll_symmetry_execution_active:
                return selected_mode_point_get_path(point, ignore_collisions)
            waypoint_name = point._waypoint.get_name()
            force_waypoint0 = waypoint_name == "waypoint0"
            later_policy = (
                args.phone_later_waypoint_roll_policy
            )
            if (
                not force_waypoint0
                and later_policy not in ("nearest", "latched", "opposite_latched")
            ):
                return selected_mode_point_get_path(point, ignore_collisions)
            original_quaternion = np.asarray(
                point._waypoint.get_quaternion(), dtype=np.float64
            )
            current_quaternion = np.asarray(
                point._robot.arm.get_tip().get_quaternion(), dtype=np.float64
            )
            current_quaternion /= np.linalg.norm(current_quaternion)
            authored_unit = original_quaternion / np.linalg.norm(original_quaternion)
            x1, y1, z1, w1 = map(float, authored_unit)
            finger_swapped_quaternion = np.asarray(
                [y1, -x1, w1, -z1], dtype=np.float64
            )
            finger_swapped_quaternion /= np.linalg.norm(
                finger_swapped_quaternion
            )
            authored_angle = float(
                2.0
                * np.arccos(
                    np.clip(
                        abs(float(np.dot(current_quaternion, authored_unit))),
                        -1.0,
                        1.0,
                    )
                )
            )
            finger_swapped_angle = float(
                2.0
                * np.arccos(
                    np.clip(
                        abs(
                            float(
                                np.dot(
                                    current_quaternion,
                                    finger_swapped_quaternion,
                                )
                            )
                        ),
                        -1.0,
                        1.0,
                    )
                )
            )
            default_branch = (
                "finger_swapped"
                if finger_swapped_angle + 1e-9 < authored_angle
                else "authored"
            )
            if force_waypoint0:
                # RLBench calls Point.get_path() while task_env.reset() validates
                # the authored waypoint program.  A dual trial has not selected
                # either branch at that point, so validation must use the authored
                # pose.  During actual demo execution the collector sets this
                # latch explicitly before running each branch.
                if forced_waypoint0_roll_branch is None:
                    selected_branch = "authored"
                    selection_reason = "reset_validation_authored"
                else:
                    selected_branch = str(forced_waypoint0_roll_branch)
                    if selected_branch not in ("authored", "finger_swapped"):
                        raise RuntimeError(
                            "Invalid active waypoint0 roll branch: "
                            + selected_branch
                        )
                    selection_reason = "forced_waypoint0"
                phone_waypoint0_default_selection_info = {
                    "default_branch": default_branch,
                    "authored_angle_rad": authored_angle,
                    "finger_swapped_angle_rad": finger_swapped_angle,
                }
            elif later_policy == "latched":
                if forced_waypoint0_roll_branch is None:
                    # Feasibility checks occur before a real branch trial. Keep
                    # the authored chain in that context, exactly as waypoint0.
                    selected_branch = "authored"
                    selection_reason = "reset_validation_authored_latched"
                else:
                    selected_branch = str(forced_waypoint0_roll_branch)
                    selection_reason = "latched_to_waypoint0"
            elif later_policy == "opposite_latched":
                if forced_waypoint0_roll_branch is None:
                    selected_branch = "authored"
                    selection_reason = (
                        "reset_validation_authored_opposite_latched"
                    )
                else:
                    selected_branch = (
                        "finger_swapped"
                        if str(forced_waypoint0_roll_branch) == "authored"
                        else "authored"
                    )
                    selection_reason = "opposite_of_waypoint0_from_waypoint1"
            else:
                selected_branch = default_branch
                selection_reason = "nearest_current_eef"
            selected_quaternion = (
                finger_swapped_quaternion
                if selected_branch == "finger_swapped"
                else original_quaternion
            )
            local_z_offset_rad = math.radians(
                float(args.phone_waypoint_local_z_offset_deg)
            )
            if abs(local_z_offset_rad) > 1e-12:
                half_angle = 0.5 * local_z_offset_rad
                sine = math.sin(half_angle)
                cosine = math.cos(half_angle)
                x_value, y_value, z_value, w_value = map(
                    float, selected_quaternion
                )
                selected_quaternion = np.asarray(
                    [
                        cosine * x_value + sine * y_value,
                        -sine * x_value + cosine * y_value,
                        sine * w_value + cosine * z_value,
                        cosine * w_value - sine * z_value,
                    ],
                    dtype=np.float64,
                )
                selected_quaternion /= np.linalg.norm(selected_quaternion)
            print(
                "[waypoint-roll-branch] waypoint="
                + waypoint_name
                + " selected_branch="
                + selected_branch
                + " reason="
                + selection_reason
                + " default_branch="
                + default_branch
                + " authored_angle_rad="
                + format(authored_angle, ".6f")
                + " finger_swapped_angle_rad="
                + format(finger_swapped_angle, ".6f")
                + " local_z_offset_deg="
                + format(float(args.phone_waypoint_local_z_offset_deg), ".3f"),
                flush=True,
            )
            point._waypoint.set_quaternion(selected_quaternion.tolist())
            try:
                return selected_mode_point_get_path(point, ignore_collisions)
            finally:
                point._waypoint.set_quaternion(original_quaternion.tolist())

        Point.get_path = forced_phone_waypoint0_roll_point_get_path

    def run_live_demo_from_current_scene(task_env):
        """Enable roll alternatives only while executing, never during reset validation."""
        nonlocal roll_symmetry_execution_active
        nonlocal last_selected_finger_swapped, grasp_latched_finger_swapped
        nonlocal chain_path_cache, chain_path_cache_route
        nonlocal chain_path_cache_ignore_collisions
        if args.waypoint_roll_symmetry:
            last_selected_finger_swapped = False
            grasp_latched_finger_swapped = None
            chain_path_cache = {}
            chain_path_cache_route = None
            chain_path_cache_ignore_collisions = None
        roll_symmetry_execution_active = True
        try:
            demo = collect_live_demo_from_current_scene(
                task_env,
                args.post_success_frames,
                args.action_alignment,
                stop_after_success=args.stop_after_success,
            )
            first_success_frame = getattr(
                demo, "success_trim_first_success_frame", None
            )
            if (
                first_success_frame is not None
                and int(first_success_frame) < int(args.min_first_success_frame)
            ):
                raise RuntimeError(
                    "Task reached success at frame "
                    + str(first_success_frame)
                    + ", earlier than --min-first-success-frame="
                    + str(args.min_first_success_frame)
                )
            return demo
        finally:
            roll_symmetry_execution_active = False

    def complete_demo_motion_metrics(demo):
        poses = np.asarray([obs.gripper_pose for obs in demo], dtype=np.float64)
        joints = np.asarray([obs.joint_positions for obs in demo], dtype=np.float64)
        if len(poses) < 2 or poses.shape[1] < 7 or joints.shape[1] < 7:
            raise RuntimeError("Complete-demo candidate lacks EEF or joint observations")
        quaternions = poses[:, 3:7]
        quaternions /= np.linalg.norm(quaternions, axis=1, keepdims=True)
        rotation = 2.0 * np.arccos(
            np.clip(
                np.abs(np.sum(quaternions[:-1] * quaternions[1:], axis=1)),
                -1.0,
                1.0,
            )
        )
        joint_delta = np.diff(joints[:, :7], axis=0)
        joint_delta = np.abs(
            np.arctan2(np.sin(joint_delta), np.cos(joint_delta))
        )
        xyz = float(
            np.linalg.norm(np.diff(poses[:, :3], axis=0), axis=1).sum()
        )
        rotation_total = float(rotation.sum())
        joint_total = float(joint_delta.sum())
        joint_travel_per_axis = joint_delta.sum(axis=0)
        wrist_total = float(joint_travel_per_axis[-3:].sum())
        max_axis_total = float(joint_travel_per_axis.max(initial=0.0))
        max_joint_step = float(joint_delta.max(initial=0.0))
        frames = int(len(poses))
        # This score is intentionally dominated by configuration-space motion.
        # An EEF path can look short while the redundant Panda elbow/wrist takes
        # a different IK branch and folds around the base.  Wrist motion gets
        # additional weight because that is where the phone demos most often
        # contain a visually useless full-arm twist.
        joint_smoothness_score = float(
            joint_total
            + 1.5 * wrist_total
            + 2.0 * max_axis_total
            + 20.0 * max_joint_step
            + 0.5 * rotation_total
            + xyz
            + 0.002 * frames
        )
        return {
            "score": float(
                joint_total
                + 2.0 * rotation_total
                + 2.0 * xyz
                + 0.005 * frames
            ),
            "frames": frames,
            "xyz_m": xyz,
            "rotation_rad": rotation_total,
            "joint_travel_rad": joint_total,
            "joint_travel_per_axis_rad": joint_travel_per_axis.tolist(),
            "wrist_joint_travel_rad": wrist_total,
            "max_axis_joint_travel_rad": max_axis_total,
            "max_joint_step_rad": max_joint_step,
            "joint_smoothness_score": joint_smoothness_score,
        }

    def waypoint_occurrence_interval(demo, waypoint_name, occurrence=0):
        """Return one waypoint occurrence without conflating task repeats."""
        sequence = list(
            getattr(demo, "waypoint_end_frame_sequence", [])
        )
        if sequence:
            matches = [
                index
                for index, (name, _) in enumerate(sequence)
                if str(name) == str(waypoint_name)
            ]
            if int(occurrence) >= len(matches):
                return None
            sequence_index = matches[int(occurrence)]
            start_frame = (
                0
                if sequence_index == 0
                else int(sequence[sequence_index - 1][1])
            )
            return start_frame, int(sequence[sequence_index][1])
        # Backward compatibility for demos/artifacts produced before repeated
        # waypoint occurrences were stored explicitly.
        waypoint_end_frames = getattr(demo, "waypoint_end_frames", {})
        waypoint_names = list(waypoint_end_frames)
        if waypoint_name not in waypoint_end_frames or int(occurrence) != 0:
            return None
        waypoint_index = waypoint_names.index(waypoint_name)
        start_frame = (
            0
            if waypoint_index == 0
            else int(waypoint_end_frames[waypoint_names[waypoint_index - 1]])
        )
        return start_frame, int(waypoint_end_frames[waypoint_name])

    def demo_waypoint_rotation_metrics(demo, waypoint_name):
        """Measure actual recorded EEF rotation within one waypoint segment."""
        interval = waypoint_occurrence_interval(demo, waypoint_name)
        if interval is None:
            return {
                "waypoint": waypoint_name,
                "available": False,
                "rotation_rad": float("inf"),
                "direct_rotation_rad": float("inf"),
                "rotation_excess_rad": float("inf"),
            }
        start_frame, end_frame = interval
        poses = np.asarray(
            [obs.gripper_pose for obs in demo], dtype=np.float64
        )
        segment = poses[start_frame : end_frame + 1, 3:7]
        if len(segment) < 2:
            rotation_total = 0.0
            direct_rotation = 0.0
        else:
            segment /= np.linalg.norm(segment, axis=1, keepdims=True)
            rotation_total = float(
                (
                    2.0
                    * np.arccos(
                        np.clip(
                            np.abs(
                                np.sum(segment[:-1] * segment[1:], axis=1)
                            ),
                            -1.0,
                            1.0,
                        )
                    )
                ).sum()
            )
            direct_rotation = float(
                2.0
                * np.arccos(
                    np.clip(
                        abs(float(np.dot(segment[0], segment[-1]))),
                        -1.0,
                        1.0,
                    )
                )
            )
        return {
            "waypoint": waypoint_name,
            "available": True,
            "start_frame": start_frame,
            "end_frame": end_frame,
            "rotation_rad": rotation_total,
            "direct_rotation_rad": direct_rotation,
            "rotation_excess_rad": float(
                max(0.0, rotation_total - direct_rotation)
            ),
        }

    def demo_waypoint_geometry_metrics(demo, waypoint_name):
        """Measure Cartesian arc/detour from the actually executed EEF path."""
        interval = waypoint_occurrence_interval(demo, waypoint_name)
        if interval is None and waypoint_name == "waypoint3":
            # stack_wine commonly becomes successful while waypoint3 is still
            # executing. With stop_after_success enabled there is then no
            # waypoint3 *end* marker, even though the carried-transfer segment
            # is present in the demo. Measure the executed part from waypoint2
            # through the terminal successful observation instead of treating
            # a clean early success as unavailable/infinite geometry.
            previous = waypoint_occurrence_interval(demo, "waypoint2")
            if previous is not None and len(demo) > int(previous[1]):
                interval = (int(previous[1]), len(demo) - 1)
        if interval is None:
            return {
                "waypoint": waypoint_name,
                "available": False,
                "arc_score_m": float("inf"),
            }
        start_frame, end_frame = interval
        xyz = np.asarray(
            [obs.gripper_pose[:3] for obs in demo], dtype=np.float64
        )[start_frame : end_frame + 1]
        if len(xyz) < 2:
            path_length = 0.0
            direct_distance = 0.0
            max_lateral = 0.0
            target_backtrack = 0.0
            progress_backtrack = 0.0
            excess_peak_z = 0.0
        else:
            path_length = float(
                np.linalg.norm(np.diff(xyz, axis=0), axis=1).sum()
            )
            chord = xyz[-1] - xyz[0]
            direct_distance = float(np.linalg.norm(chord))
            if direct_distance > 1e-9:
                unit = chord / direct_distance
                progress = (xyz - xyz[0]) @ unit
                closest = xyz[0] + progress[:, None] * unit
                max_lateral = float(
                    np.linalg.norm(xyz - closest, axis=1).max(initial=0.0)
                )
                progress_backtrack = float(
                    np.maximum(0.0, -np.diff(progress)).sum()
                )
            else:
                max_lateral = float(
                    np.linalg.norm(xyz - xyz[0], axis=1).max(initial=0.0)
                )
                progress_backtrack = path_length
            distance_to_target = np.linalg.norm(xyz - xyz[-1], axis=1)
            target_backtrack = float(
                np.maximum(0.0, np.diff(distance_to_target)).sum()
            )
            excess_peak_z = float(
                max(
                    0.0,
                    float(xyz[:, 2].max())
                    - max(float(xyz[0, 2]), float(xyz[-1, 2])),
                )
            )
        path_excess = float(max(0.0, path_length - direct_distance))
        arc_score = float(
            path_excess
            + 2.0 * max_lateral
            + 2.0 * target_backtrack
            + progress_backtrack
        )
        return {
            "waypoint": waypoint_name,
            "available": True,
            "start_frame": start_frame,
            "end_frame": end_frame,
            "frames": int(len(xyz)),
            "path_length_m": path_length,
            "direct_distance_m": direct_distance,
            "detour_ratio": float(path_length / max(direct_distance, 1e-9)),
            "path_excess_m": path_excess,
            "max_lateral_deviation_m": max_lateral,
            "target_distance_backtrack_m": target_backtrack,
            "progress_backtrack_m": progress_backtrack,
            "excess_peak_z_m": excess_peak_z,
            "arc_score_m": arc_score,
        }
    print(
        "[expert-path-mode] mode="
        + str(args.expert_path_mode)
        + " ordinary_point_planner="
        + {
            "linear_only": "linear_ik_only",
            "segmented_linear": (
                "segmented_linear_ik_"
                + str(int(args.segmented_linear_segments))
                + "x"
            ),
            "franka_cartesian_servo": (
                "online_damped_jacobian_velocity_servo_with_linear_xyz_"
                "shortest_arc_slerp_and_start_branch_nullspace_target"
            ),
            "cartesian_detour": (
                "deterministic_minimum_deviation_smooth_cartesian_arc_"
                "with_sequential_linear_ik_and_no_ompl"
            ),
            "segmented_linear_then_rrt": (
                "segmented_linear_ik_then_short_"
                + str(args.segmented_fallback_algorithm).lower()
                + "_"
                + str(int(args.segmented_linear_segments))
                + "x"
            ),
            "segmented_linear_then_best_of_n": (
                "stock_xyz_guided_short_arc_slerp_ik_"
                + str(int(args.segmented_linear_segments))
                + "x_then_loop_filtered_best_of_"
                + str(int(args.phone_path_candidates))
                + "_full_waypoint_paths"
            ),
            "best_of_n_all_points": (
                "collision_aware_best_of_"
                + str(int(args.phone_path_candidates))
                + "_at_every_point_with_joint_cartesian_rotation_loop_guards"
            ),
            "rrt_only": "forced_rrtconnect",
            "phone_predefined_path_waypoint3": (
                "single_direct_coppeliasim_cartesian_path_with_short_arc_"
                "orientation_at_phone_waypoint3_without_nonlinear_fallback"
            ),
            "phone_prm_waypoint3": (
                "best_of_n_ompl_prm_paths_at_phone_waypoint3_selected_by_"
                "measured_cartesian_orientation_and_joint_motion"
            ),
            "phone_prm_0_3": (
                "best_of_n_ompl_prm_paths_at_phone_waypoint0_and_waypoint3_"
                "with_linear_ik_for_all_other_points_and_no_rrt"
            ),
            "phone_prm_all": (
                "best_of_n_ompl_prm_paths_at_every_phone_point_waypoint_"
                "with_no_rrt_family_planner"
            ),
            "phone_cartesian_0_3": (
                "scored_stock_and_dynamic_cartesian_candidates_for_"
                "waypoint0_and_waypoint3_with_negative_wrist_branch_preference"
            ),
            "phone_staged_linear_0_3": (
                "deterministic_direct_or_staged_linear_ik_for_"
                "waypoint0_and_waypoint3_without_rrt_fallback"
            ),
            "phone_staged_linear_then_rrt_0_3": (
                "deterministic_staged_linear_ik_with_locally_scored_"
                "rrt_fallback_for_infeasible_segments"
            ),
            "phone_rrt_cartesian_shortcut_0_3": (
                "collision_aware_rrt_guide_with_greedy_cartesian_"
                "shortcutting_for_waypoint0_and_waypoint3"
            ),
            "phone_best_ik_joint_interp_0_3": (
                "sampled_collision_free_ik_goal_with_minimum_eef_arc_"
                "joint_interpolation_for_waypoint0_and_waypoint3"
            ),
            "phone_baseframe_reference_ik_0_3": (
                "robot_base_waypoint_pose_nearest_clean_branch_"
                "jacobian_refinement_and_bounded_joint_interpolation"
            ),
            "phone_best_of_n_0_3": (
                "best_of_n_collision_aware_plans_for_waypoint0_and_"
                "waypoint3_with_joint_loop_rejection"
            ),
            "linear_then_rrt": "linear_ik_then_rrtconnect_fallback",
        }[str(args.expert_path_mode)]
        + " waypoint_roll_symmetry="
        + ("local_tool_z_pi_nearest" if args.waypoint_roll_symmetry else "disabled")
        + " phone_waypoint0_roll_branch="
        + str(args.phone_waypoint0_roll_branch)
        + " phone_later_waypoint_roll_policy="
        + str(args.phone_later_waypoint_roll_policy)
        + " dual_waypoint0_roll_select_shorter="
        + str(bool(args.dual_waypoint0_roll_select_shorter))
        + " predefined_cartesian_paths="
        + (
            "follow_selected_roll_branch"
            if args.waypoint_roll_symmetry
            else "preserved"
        ),
        flush=True,
    )
    print(
        "[rrt-policy] "
        + ("explicitly_enabled" if args.allow_rrt else "forbidden"),
        flush=True,
    )

    manifest_path = artifact_root / "manifest.json"
    artifact_root.mkdir(parents=True, exist_ok=True)
    config_signature = collection_config_signature(args, tasks)
    if args.no_resume and manifest_path.exists():
        manifest_path.unlink()
    if manifest_path.exists():
        with open(manifest_path, "r", encoding="utf-8") as file:
            manifest = json.load(file)
        if manifest.get("config") != config_signature:
            raise RuntimeError(
                "Resume parameters differ from the artifact manifest. "
                "Use the original parameters or pass --no-resume to recollect."
            )
    else:
        manifest = {"config": config_signature, "records": []}
    records = manifest["records"]
    known = set(item["task"] + ":" + str(item["local_episode_index"]) for item in records)

    env = Environment(
        MoveArmThenGripper(JointVelocity(), Discrete()),
        obs_config=make_observation_config(args.image_size),
        headless=True,
        static_positions=False,
    )
    env.launch()
    try:
        configure_front_camera(
            position_m=args.front_camera_position_m,
            look_at_m=args.front_camera_look_at_m,
        )
        fk_model = read_panda_fk_model(env)
        path_fk_model = fk_model
        t_world_base = rlbench_panda_link0_to_world_matrix(env._robot.arm)
        print(
            "[worldflow-base] frame="
            + RLBENCH_PANDA_LINK0_FRAME_VERSION
            + " T_world_base="
            + json.dumps(t_world_base.tolist(), separators=(",", ":")),
            flush=True,
        )
        for task_name in tasks:
            task_class = task_class_from_name(task_name)
            task_env = env.get_task(task_class)
            task_env.set_variation(args.variation)
            completed_indices = {
                int(item["local_episode_index"])
                for item in records
                if item["task"] == task_name
                and artifact_is_complete(
                    artifact_root / episode_name(task_name, int(item["local_episode_index"])),
                    require_world_base_worldflow_sidecars=(
                        args.generate_world_base_worldflow_sidecars
                    ),
                )
            }
            requested_indices = (
                None
                if args.episode_indices is None
                else list(dict.fromkeys(map(int, args.episode_indices)))
            )
            if requested_indices is not None:
                requested_set = set(requested_indices)
                completed_indices &= requested_set
            accepted_scene_signatures = []
            for completed_index in sorted(completed_indices):
                completed_arrays_path = (
                    artifact_root
                    / episode_name(task_name, completed_index)
                    / "arrays.npz"
                )
                with np.load(completed_arrays_path, allow_pickle=False) as completed_arrays:
                    accepted_scene_signatures.append(
                        (
                            completed_index,
                            initial_scene_signature(completed_arrays),
                        )
                    )
            successful_count = len(completed_indices)
            episode_cursor = 0
            local_index = (
                requested_indices[episode_cursor]
                if requested_indices is not None
                else int(getattr(args, "episode_start", 0))
            )
            episode_target_count = (
                len(requested_indices)
                if requested_indices is not None
                else int(args.episodes_per_task)
            )
            while successful_count < episode_target_count:
                key = task_name + ":" + str(local_index)
                artifact = artifact_root / episode_name(task_name, local_index)
                if key in known and artifact_is_complete(
                    artifact,
                    require_world_base_worldflow_sidecars=(
                        args.generate_world_base_worldflow_sidecars
                    ),
                ):
                    print("[skip] task=" + task_name + " episode=" + str(local_index))
                    if requested_indices is None:
                        local_index += 1
                    else:
                        episode_cursor += 1
                        if episode_cursor < len(requested_indices):
                            local_index = requested_indices[episode_cursor]
                    continue
                if key in known:
                    records[:] = [
                        item
                        for item in records
                        if item["task"] + ":" + str(item["local_episode_index"]) != key
                    ]
                    known.remove(key)
                success = False
                last_error = None
                for attempt in range(args.max_demo_attempts):
                    try:
                        forced_waypoint_roll_branch = args.waypoint_roll_force_branch
                        if args.dual_waypoint0_roll_select_shorter:
                            # Keep reset/feasibility validation independent from
                            # whichever branch won the preceding candidate scene.
                            forced_waypoint0_roll_branch = None
                        phone_waypoint0_default_selection_info = None
                        dual_waypoint0_selected_branch = None
                        dual_waypoint0_authored_frames = None
                        dual_waypoint0_finger_swapped_frames = None
                        waypoint_roll_authored_fallback = False
                        waypoint_roll_primary_error = None
                        matched_scene_demo_selected_index = None
                        matched_scene_demo_selected_metrics = None
                        initial_scene_nearest_distance_m = None
                        initial_scene_nearest_episode = None
                        accepted_candidate_signature = None
                        linear_calls_before = int(linear_path_calls)
                        rrt_calls_before = int(rrt_path_calls)
                        cartesian_calls_before = int(cartesian_path_calls)
                        cartesian_stock_fallbacks_before = int(
                            cartesian_stock_fallback_calls
                        )
                        replay_seed_source = None
                        replay_scene_source = None
                        replay_scene_arrays = None
                        managed_live_scene = bool(
                            args.min_initial_scene_distance_m > 0.0
                            # Roll-symmetry is applied only inside
                            # run_live_demo_from_current_scene().  Full-waypoint
                            # random collection does not use stop_after_success,
                            # so it must still take this managed path; otherwise
                            # --waypoint-roll-force-branch is silently ignored.
                            or args.waypoint_roll_symmetry
                            or (
                                args.dual_waypoint0_roll_select_shorter
                                and args.replay_scenes_from_artifacts is None
                                and args.replay_random_seeds_from_artifacts is None
                            )
                            or (
                                args.stop_after_success
                                and args.replay_scenes_from_artifacts is None
                                and args.replay_random_seeds_from_artifacts is None
                                and args.phone_eef_max_initial_distance_m is None
                            )
                        )
                        if managed_live_scene:
                            candidate_random_state = np.random.get_state()
                            descriptions, reset_observation = task_env.reset()
                            candidate_scene_arrays = capture_initial_scene_arrays(
                                task_env, candidate_random_state
                            )
                            accepted_candidate_signature = initial_scene_signature(
                                candidate_scene_arrays
                            )
                            if (
                                task_name == "phone_on_base"
                                and args.phone_robot_base_min_initial_distance_m
                                is not None
                            ):
                                candidate_phone_position = np.asarray(
                                    task_env._task.phone.get_position(),
                                    dtype=np.float64,
                                )
                                candidate_robot_base_distance = float(
                                    np.linalg.norm(
                                        candidate_phone_position
                                        - np.asarray(
                                            t_world_base[:3, 3], dtype=np.float64
                                        )
                                    )
                                )
                                if candidate_robot_base_distance < float(
                                    args.phone_robot_base_min_initial_distance_m
                                ):
                                    print(
                                        "[scene-reject-near-robot-base] task="
                                        + task_name
                                        + " episode="
                                        + str(local_index)
                                        + " attempt="
                                        + str(attempt + 1)
                                        + " phone_to_robot_base_m="
                                        + format(candidate_robot_base_distance, ".6f")
                                        + " threshold_m="
                                        + format(
                                            float(
                                                args.phone_robot_base_min_initial_distance_m
                                            ),
                                            ".6f",
                                        ),
                                        flush=True,
                                    )
                                    continue
                            (
                                initial_scene_nearest_distance_m,
                                initial_scene_nearest_episode,
                            ) = nearest_initial_scene(
                                accepted_candidate_signature,
                                accepted_scene_signatures,
                                args.scene_rotation_radius_m,
                            )
                            if initial_scene_nearest_distance_m < float(
                                args.min_initial_scene_distance_m
                            ):
                                print(
                                    "[scene-reject-similar] task="
                                    + task_name
                                    + " episode="
                                    + str(local_index)
                                    + " attempt="
                                    + str(attempt + 1)
                                    + " nearest_episode="
                                    + str(initial_scene_nearest_episode)
                                    + " distance_m="
                                    + format(initial_scene_nearest_distance_m, ".6f")
                                    + " threshold_m="
                                    + format(
                                        float(args.min_initial_scene_distance_m), ".6f"
                                    ),
                                    flush=True,
                                )
                                continue

                            if args.dual_waypoint0_roll_select_shorter:
                                branch_demos = {}
                                branch_selection_info = {}
                                for branch_index, branch in enumerate(
                                    ("authored", "finger_swapped")
                                ):
                                    if branch_index:
                                        (
                                            descriptions,
                                            reset_observation,
                                            _,
                                            _,
                                        ) = restore_task_environment_from_artifact_arrays(
                                            task_env, candidate_scene_arrays
                                        )
                                    forced_waypoint0_roll_branch = branch
                                    phone_waypoint0_default_selection_info = None
                                    branch_demo = run_live_demo_from_current_scene(
                                        task_env
                                    )
                                    branch_demo.random_seed = candidate_random_state
                                    branch_demo.num_reset_attempts = int(
                                        candidate_scene_arrays[
                                            "demo_num_reset_attempts"
                                        ]
                                    )
                                    waypoint0_interval = waypoint_occurrence_interval(
                                        branch_demo, "waypoint0"
                                    )
                                    if waypoint0_interval is None:
                                        raise RuntimeError(
                                            "Dual roll trial did not reach waypoint0: "
                                            + branch
                                        )
                                    waypoint0_frame = waypoint0_interval[1]
                                    branch_demos[branch] = (
                                        branch_demo,
                                        int(waypoint0_frame),
                                    )
                                    branch_selection_info[branch] = (
                                        None
                                        if phone_waypoint0_default_selection_info is None
                                        else dict(
                                            phone_waypoint0_default_selection_info
                                        )
                                    )
                                dual_waypoint0_authored_frames = branch_demos[
                                    "authored"
                                ][1]
                                dual_waypoint0_finger_swapped_frames = branch_demos[
                                    "finger_swapped"
                                ][1]
                                dual_waypoint0_selected_branch = min(
                                    branch_demos,
                                    key=lambda branch: (
                                        branch_demos[branch][1],
                                        0 if branch == "authored" else 1,
                                    ),
                                )
                                demo = branch_demos[
                                    dual_waypoint0_selected_branch
                                ][0]
                                forced_waypoint0_roll_branch = None
                                phone_waypoint0_default_selection_info = (
                                    branch_selection_info[
                                        dual_waypoint0_selected_branch
                                    ]
                                )
                                print(
                                    "[dual-waypoint0-selected] task="
                                    + task_name
                                    + " episode="
                                    + str(local_index)
                                    + " authored_frames="
                                    + str(dual_waypoint0_authored_frames)
                                    + " finger_swapped_frames="
                                    + str(dual_waypoint0_finger_swapped_frames)
                                    + " selected="
                                    + dual_waypoint0_selected_branch,
                                    flush=True,
                                )
                            else:
                                if args.waypoint_roll_symmetry:
                                    try:
                                        demo = run_live_demo_from_current_scene(
                                            task_env
                                        )
                                    except Exception as primary_error:
                                        # An explicitly forced branch is a data
                                        # contract, not a preference.  Never
                                        # silently save an authored-orientation
                                        # trajectory when the requested swapped
                                        # branch fails; let the outer retry loop
                                        # sample another scene instead.
                                        if args.waypoint_roll_force_branch is not None:
                                            raise
                                        waypoint_roll_authored_fallback = True
                                        waypoint_roll_primary_error = repr(
                                            primary_error
                                        )
                                        print(
                                            "[waypoint-roll-authored-fallback] task="
                                            + task_name
                                            + " episode="
                                            + str(local_index)
                                            + " primary_error="
                                            + waypoint_roll_primary_error,
                                            flush=True,
                                        )
                                        (
                                            descriptions,
                                            reset_observation,
                                            _,
                                            _,
                                        ) = restore_task_environment_from_artifact_arrays(
                                            task_env, candidate_scene_arrays
                                        )
                                        forced_waypoint_roll_branch = "authored"
                                        demo = run_live_demo_from_current_scene(
                                            task_env
                                        )
                                        forced_waypoint_roll_branch = args.waypoint_roll_force_branch
                                else:
                                    demo = run_live_demo_from_current_scene(
                                        task_env
                                    )
                                demo.random_seed = candidate_random_state
                                demo.num_reset_attempts = int(
                                    candidate_scene_arrays[
                                        "demo_num_reset_attempts"
                                    ]
                                )
                            (
                                descriptions,
                                reset_observation,
                                _,
                                _,
                            ) = restore_task_environment_from_artifact_arrays(
                                task_env, candidate_scene_arrays
                            )
                        elif args.replay_scenes_from_artifacts is not None:
                            replay_scene_source = (
                                Path(args.replay_scenes_from_artifacts)
                                .expanduser()
                                .resolve()
                                / episode_name(task_name, local_index)
                                / "arrays.npz"
                            )
                            if not replay_scene_source.is_file():
                                raise FileNotFoundError(
                                    "Matched scene artifact is missing: "
                                    + str(replay_scene_source)
                                )
                            required_scene_keys = (
                                "initial_task_state_bytes",
                                "initial_task_state_object_count",
                                "initial_object_names",
                                "initial_object_poses",
                                "initial_object_joint_positions",
                                "initial_object_joint_target_positions",
                                "initial_object_joint_target_velocities",
                                "demo_random_seed_state",
                                "demo_random_seed_position",
                                "demo_random_seed_has_gauss",
                                "demo_random_seed_cached_gaussian",
                                "demo_num_reset_attempts",
                            )
                            with np.load(replay_scene_source, allow_pickle=False) as source:
                                missing_scene_keys = [
                                    key for key in required_scene_keys if key not in source.files
                                ]
                                if missing_scene_keys:
                                    raise RuntimeError(
                                        "Matched scene artifact lacks required fields: "
                                        + ", ".join(missing_scene_keys)
                                    )
                                replay_scene_arrays = {
                                    key: np.asarray(source[key]).copy()
                                    for key in required_scene_keys
                                }
                            if args.dual_waypoint0_roll_select_shorter:
                                branch_demos = {}
                                branch_selection_info = {}
                                branch_failures = {}
                                replay_random_state = None
                                replay_reset_attempts = None
                                for branch in ("authored", "finger_swapped"):
                                    forced_waypoint0_roll_branch = branch
                                    branch_candidates = []
                                    candidate_count = max(
                                        1, int(args.matched_scene_demo_candidates)
                                    )
                                    branch_failures[branch] = []
                                    for candidate_index in range(candidate_count):
                                        (
                                            descriptions,
                                            reset_observation,
                                            branch_random_state,
                                            branch_reset_attempts,
                                        ) = restore_task_environment_from_artifact_arrays(
                                            task_env, replay_scene_arrays
                                        )
                                        phone_waypoint0_default_selection_info = None
                                        try:
                                            branch_demo = run_live_demo_from_current_scene(
                                                task_env
                                            )
                                            if (
                                                args.matched_scene_selection_objective
                                                == "waypoint2_path_geometry"
                                            ):
                                                waypoint_sequence = list(
                                                    getattr(
                                                        branch_demo,
                                                        "waypoint_end_frame_sequence",
                                                        [],
                                                    )
                                                )
                                                waypoint0_count = sum(
                                                    str(name) == "waypoint0"
                                                    for name, _ in waypoint_sequence
                                                )
                                                if waypoint0_count != 1:
                                                    raise RuntimeError(
                                                        "Matched-scene frame candidate "
                                                        "repeated the waypoint chain "
                                                        f"{waypoint0_count} times"
                                                    )
                                            waypoint0_interval = (
                                                waypoint_occurrence_interval(
                                                    branch_demo, "waypoint0"
                                                )
                                            )
                                            if waypoint0_interval is None:
                                                raise RuntimeError(
                                                    "Matched-scene dual roll trial did "
                                                    "not reach waypoint0: " + branch
                                                )
                                            waypoint0_frame = waypoint0_interval[1]
                                            candidate_metrics = complete_demo_motion_metrics(
                                                branch_demo
                                            )
                                            waypoint3_metrics = (
                                                demo_waypoint_rotation_metrics(
                                                    branch_demo, "waypoint3"
                                                )
                                            )
                                            waypoint0_geometry = (
                                                demo_waypoint_geometry_metrics(
                                                    branch_demo, "waypoint0"
                                                )
                                            )
                                            waypoint3_geometry = (
                                                demo_waypoint_geometry_metrics(
                                                    branch_demo, "waypoint3"
                                                )
                                            )
                                            waypoint2_geometry = (
                                                demo_waypoint_geometry_metrics(
                                                    branch_demo, "waypoint2"
                                                )
                                            )
                                            geometry_arc_score = float(
                                                waypoint0_geometry["arc_score_m"]
                                                + waypoint3_geometry["arc_score_m"]
                                            )
                                            geometry_max_lateral = float(
                                                max(
                                                    waypoint0_geometry[
                                                        "max_lateral_deviation_m"
                                                    ],
                                                    waypoint3_geometry[
                                                        "max_lateral_deviation_m"
                                                    ],
                                                )
                                            )
                                            candidate_metrics.update(
                                                {
                                                    "candidate": int(candidate_index),
                                                    "branch": branch,
                                                    "waypoint0_frame": int(
                                                        waypoint0_frame
                                                    ),
                                                    "waypoint3": waypoint3_metrics,
                                                    "waypoint0_geometry": waypoint0_geometry,
                                                    "waypoint2_geometry": waypoint2_geometry,
                                                    "waypoint3_geometry": waypoint3_geometry,
                                                    "geometry_arc_score_m": geometry_arc_score,
                                                    "geometry_max_lateral_deviation_m": geometry_max_lateral,
                                                    "selection_objective": str(
                                                        args.matched_scene_selection_objective
                                                    ),
                                                }
                                            )
                                            if (
                                                args.matched_scene_selection_objective
                                                == "waypoint_path_geometry"
                                            ):
                                                selection_values = (
                                                    geometry_arc_score,
                                                    geometry_max_lateral,
                                                    float(candidate_metrics["score"]),
                                                )
                                            elif (
                                                args.matched_scene_selection_objective
                                                == "waypoint2_path_geometry"
                                            ):
                                                selection_values = (
                                                    float(
                                                        waypoint2_geometry[
                                                            "arc_score_m"
                                                        ]
                                                    ),
                                                    float(
                                                        waypoint2_geometry.get(
                                                            "max_lateral_deviation_m",
                                                            float("inf"),
                                                        )
                                                    ),
                                                    float(candidate_metrics["score"]),
                                                )
                                            elif (
                                                args.matched_scene_selection_objective
                                                == "waypoint0_path_geometry"
                                            ):
                                                # The phone scenes closest to the robot
                                                # fail visibly at the very first reach.
                                                # Rank complete, actually executed demos
                                                # lexicographically so a clean later path
                                                # can never hide a waypoint0 semicircle.
                                                selection_values = (
                                                    float(
                                                        waypoint0_geometry[
                                                            "max_lateral_deviation_m"
                                                        ]
                                                    ),
                                                    float(
                                                        waypoint0_geometry[
                                                            "arc_score_m"
                                                        ]
                                                    ),
                                                    float(
                                                        waypoint3_geometry[
                                                            "arc_score_m"
                                                        ]
                                                    ),
                                                )
                                            elif (
                                                args.matched_scene_selection_objective
                                                == "waypoint0_balanced_geometry"
                                            ):
                                                # Keep waypoint0 in the best 2.5 cm
                                                # lateral-quality band, then optimize the
                                                # later placement reach. This prevents a
                                                # millimetric waypoint0 improvement from
                                                # selecting a large waypoint3 loop.
                                                waypoint0_lateral_band = int(
                                                    np.floor(
                                                        float(
                                                            waypoint0_geometry[
                                                                "max_lateral_deviation_m"
                                                            ]
                                                        )
                                                        / 0.025
                                                        + 1e-9
                                                    )
                                                )
                                                selection_values = (
                                                    float(waypoint0_lateral_band),
                                                    float(
                                                        waypoint3_geometry[
                                                            "arc_score_m"
                                                        ]
                                                    )
                                                    + 0.05
                                                    * float(
                                                        waypoint3_metrics[
                                                            "rotation_excess_rad"
                                                        ]
                                                    ),
                                                    float(
                                                        waypoint0_geometry[
                                                            "arc_score_m"
                                                        ]
                                                    ),
                                                )
                                            elif (
                                                args.matched_scene_selection_objective
                                                == "phone_joint_smoothness"
                                            ):
                                                selection_values = (
                                                    float(
                                                        candidate_metrics[
                                                            "joint_smoothness_score"
                                                        ]
                                                    ),
                                                    3.0
                                                    * float(
                                                        waypoint0_geometry[
                                                            "arc_score_m"
                                                        ]
                                                    )
                                                    + float(
                                                        waypoint3_geometry[
                                                            "arc_score_m"
                                                        ]
                                                    ),
                                                    float(candidate_metrics["frames"]),
                                                )
                                            elif (
                                                args.matched_scene_selection_objective
                                                == "phone_weighted_geometry"
                                            ):
                                                selection_values = (
                                                    3.0
                                                    * float(
                                                        waypoint0_geometry[
                                                            "arc_score_m"
                                                        ]
                                                    )
                                                    + float(
                                                        waypoint3_geometry[
                                                            "arc_score_m"
                                                        ]
                                                    )
                                                    + 0.05
                                                    * float(
                                                        waypoint3_metrics[
                                                            "rotation_excess_rad"
                                                        ]
                                                    ),
                                                    float(
                                                        waypoint0_geometry[
                                                            "max_lateral_deviation_m"
                                                        ]
                                                    ),
                                                    float(candidate_metrics["score"]),
                                                )
                                            else:
                                                selection_values = (
                                                    float(
                                                        waypoint3_metrics[
                                                            "rotation_excess_rad"
                                                        ]
                                                    ),
                                                    float(
                                                        waypoint3_metrics[
                                                            "rotation_rad"
                                                        ]
                                                    ),
                                                    float(candidate_metrics["score"]),
                                                )
                                            candidate_entry = (
                                                selection_values[0],
                                                selection_values[1],
                                                selection_values[2],
                                                branch_demo,
                                                int(waypoint0_frame),
                                                branch_random_state,
                                                branch_reset_attempts,
                                                candidate_metrics,
                                                (
                                                    None
                                                    if phone_waypoint0_default_selection_info
                                                    is None
                                                    else dict(
                                                        phone_waypoint0_default_selection_info
                                                    )
                                                ),
                                            )
                                            if (
                                                not branch_candidates
                                                or candidate_entry[:3]
                                                < branch_candidates[0][:3]
                                            ):
                                                # Full demos contain dense RGB/depth/point
                                                # observations. Keep only the current best
                                                # candidate so larger searches do not grow
                                                # linearly in memory.
                                                branch_candidates[:] = [candidate_entry]
                                            print(
                                                "[matched-scene-dual-candidate] task="
                                                + task_name
                                                + " episode="
                                                + str(local_index)
                                                + " metrics="
                                                + json.dumps(
                                                    candidate_metrics, sort_keys=True
                                                ),
                                                flush=True,
                                            )
                                        except Exception as branch_error:
                                            branch_failures[branch].append(
                                                {
                                                    "candidate": int(candidate_index),
                                                    "error": repr(branch_error),
                                                }
                                            )
                                            print(
                                                "[matched-scene-dual-waypoint0-branch-failed] task="
                                                + task_name
                                                + " episode="
                                                + str(local_index)
                                                + " branch="
                                                + branch
                                                + " candidate="
                                                + str(candidate_index)
                                                + " error="
                                                + repr(branch_error),
                                                flush=True,
                                            )
                                    if not branch_candidates:
                                        continue
                                    selected_branch_candidate = min(
                                        branch_candidates,
                                        key=lambda item: item[:3],
                                    )
                                    branch_demos[branch] = selected_branch_candidate
                                    branch_selection_info[branch] = (
                                        selected_branch_candidate[8]
                                    )
                                if not branch_demos:
                                    raise RuntimeError(
                                        "Both matched-scene waypoint0 roll branches "
                                        "failed: " + json.dumps(branch_failures, sort_keys=True)
                                    )
                                dual_waypoint0_authored_frames = (
                                    None
                                    if "authored" not in branch_demos
                                    else branch_demos["authored"][4]
                                )
                                dual_waypoint0_finger_swapped_frames = (
                                    None
                                    if "finger_swapped" not in branch_demos
                                    else branch_demos["finger_swapped"][4]
                                )
                                dual_waypoint0_selected_branch = min(
                                    branch_demos,
                                    key=lambda branch: (
                                        branch_demos[branch][0],
                                        branch_demos[branch][1],
                                        branch_demos[branch][2],
                                        branch_demos[branch][4],
                                        0 if branch == "authored" else 1,
                                    ),
                                )
                                demo = branch_demos[
                                    dual_waypoint0_selected_branch
                                ][3]
                                replay_random_state = branch_demos[
                                    dual_waypoint0_selected_branch
                                ][5]
                                replay_reset_attempts = branch_demos[
                                    dual_waypoint0_selected_branch
                                ][6]
                                matched_scene_demo_selected_metrics = branch_demos[
                                    dual_waypoint0_selected_branch
                                ][7]
                                phone_waypoint0_default_selection_info = (
                                    branch_selection_info[
                                        dual_waypoint0_selected_branch
                                    ]
                                )
                                forced_waypoint0_roll_branch = None
                                print(
                                    "[matched-scene-dual-waypoint0-selected] task="
                                    + task_name
                                    + " episode="
                                    + str(local_index)
                                    + " authored_frames="
                                    + str(dual_waypoint0_authored_frames)
                                    + " finger_swapped_frames="
                                    + str(dual_waypoint0_finger_swapped_frames)
                                    + " selected="
                                    + dual_waypoint0_selected_branch,
                                    flush=True,
                                )
                            else:
                                complete_candidate_count = max(
                                    1, int(args.matched_scene_demo_candidates)
                                )
                                complete_candidates = []
                                complete_failures = []
                                replay_random_state = None
                                replay_reset_attempts = None
                                for candidate_index in range(complete_candidate_count):
                                    (
                                        descriptions,
                                        reset_observation,
                                        candidate_random_state,
                                        candidate_reset_attempts,
                                    ) = restore_task_environment_from_artifact_arrays(
                                        task_env, replay_scene_arrays
                                    )
                                    try:
                                        candidate_demo = run_live_demo_from_current_scene(
                                            task_env
                                        )
                                        candidate_metrics = complete_demo_motion_metrics(
                                            candidate_demo
                                        )
                                        candidate_metrics["candidate"] = int(
                                            candidate_index
                                        )
                                        if (
                                            args.matched_scene_selection_objective
                                            == "wine_postgrasp_geometry"
                                        ):
                                            waypoint3_geometry = (
                                                demo_waypoint_geometry_metrics(
                                                    candidate_demo, "waypoint3"
                                                )
                                            )
                                            excess_peak_z = float(
                                                waypoint3_geometry.get(
                                                    "excess_peak_z_m", float("inf")
                                                )
                                            )
                                            if excess_peak_z > float(
                                                args.wine_postgrasp_max_excess_z_m
                                            ):
                                                raise RuntimeError(
                                                    "Wine waypoint3 rises too far: "
                                                    + format(excess_peak_z, ".6f")
                                                    + " m > "
                                                    + format(
                                                        float(
                                                            args.wine_postgrasp_max_excess_z_m
                                                        ),
                                                        ".6f",
                                                    )
                                                    + " m"
                                                )
                                            candidate_metrics["waypoint3_geometry"] = (
                                                waypoint3_geometry
                                            )
                                            candidate_selection_score = float(
                                                1000.0 * excess_peak_z
                                                + 10.0
                                                * float(
                                                    waypoint3_geometry[
                                                        "max_lateral_deviation_m"
                                                    ]
                                                )
                                                + float(
                                                    waypoint3_geometry["arc_score_m"]
                                                )
                                                + 0.01
                                                * float(
                                                    candidate_metrics[
                                                        "joint_smoothness_score"
                                                    ]
                                                )
                                            )
                                        else:
                                            candidate_selection_score = float(
                                                candidate_metrics[
                                                    "joint_smoothness_score"
                                                    if args.matched_scene_selection_objective
                                                    == "phone_joint_smoothness"
                                                    else "score"
                                                ]
                                            )
                                        complete_candidates.append(
                                            (
                                                candidate_selection_score,
                                                candidate_demo,
                                                candidate_random_state,
                                                candidate_reset_attempts,
                                                candidate_metrics,
                                            )
                                        )
                                        print(
                                            "[matched-scene-demo-candidate] task="
                                            + task_name
                                            + " episode="
                                            + str(local_index)
                                            + " metrics="
                                            + json.dumps(candidate_metrics, sort_keys=True),
                                            flush=True,
                                        )
                                    except Exception as candidate_error:
                                        complete_failures.append(
                                            {
                                                "candidate": int(candidate_index),
                                                "error": repr(candidate_error),
                                            }
                                        )
                                        print(
                                            "[matched-scene-demo-candidate-failed] task="
                                            + task_name
                                            + " episode="
                                            + str(local_index)
                                            + " candidate="
                                            + str(candidate_index)
                                            + " error="
                                            + repr(candidate_error),
                                            flush=True,
                                        )
                                if not complete_candidates:
                                    raise RuntimeError(
                                        "All complete matched-scene demo candidates failed: "
                                        + json.dumps(complete_failures, sort_keys=True)
                                    )
                                (
                                    _,
                                    demo,
                                    replay_random_state,
                                    replay_reset_attempts,
                                    matched_scene_demo_selected_metrics,
                                ) = min(complete_candidates, key=lambda item: item[0])
                                matched_scene_demo_selected_index = int(
                                    matched_scene_demo_selected_metrics["candidate"]
                                )
                                print(
                                    "[matched-scene-demo-selected] task="
                                    + task_name
                                    + " episode="
                                    + str(local_index)
                                    + " accepted="
                                    + str(len(complete_candidates))
                                    + "/"
                                    + str(complete_candidate_count)
                                    + " metrics="
                                    + json.dumps(
                                        matched_scene_demo_selected_metrics,
                                        sort_keys=True,
                                    ),
                                    flush=True,
                                )
                            demo.random_seed = replay_random_state
                            demo.num_reset_attempts = replay_reset_attempts
                            # Return to the exact matched starting state for sidecars,
                            # reset-image validation, and initial-distance reporting.
                            descriptions, reset_observation, _, _ = (
                                restore_task_environment_from_artifact_arrays(
                                    task_env, replay_scene_arrays
                                )
                            )
                            print(
                                "[matched-scene-restored] task="
                                + task_name
                                + " episode="
                                + str(local_index)
                                + " source="
                                + str(replay_scene_source),
                                flush=True,
                            )
                        elif (
                            task_name == "phone_on_base"
                            and args.phone_eef_max_initial_distance_m is not None
                        ):
                            candidate_random_state = np.random.get_state()
                            descriptions, candidate_observation = task_env.reset()
                            candidate_phone_position = np.asarray(
                                task_env._task.phone.get_position(), dtype=np.float64
                            )
                            candidate_eef_position = np.asarray(
                                candidate_observation.gripper_pose[:3], dtype=np.float64
                            )
                            candidate_distance = float(
                                np.linalg.norm(
                                    candidate_phone_position - candidate_eef_position
                                )
                            )
                            if candidate_distance > float(
                                args.phone_eef_max_initial_distance_m
                            ):
                                print(
                                    "[scene-reject] task=phone_on_base episode="
                                    + str(local_index)
                                    + " attempt="
                                    + str(attempt + 1)
                                    + " phone_to_eef_m="
                                    + format(candidate_distance, ".6f")
                                    + " threshold_m="
                                    + format(
                                        float(args.phone_eef_max_initial_distance_m),
                                        ".6f",
                                    ),
                                    flush=True,
                                )
                                continue
                            demo = run_live_demo_from_current_scene(task_env)
                            demo.random_seed = candidate_random_state
                            descriptions, reset_observation = task_env.reset_to_demo(demo)
                        else:
                            if args.replay_random_seeds_from_artifacts is not None:
                                replay_seed_source = (
                                    Path(args.replay_random_seeds_from_artifacts)
                                    .expanduser()
                                    .resolve()
                                    / episode_name(task_name, local_index)
                                    / "arrays.npz"
                                )
                                if not replay_seed_source.is_file():
                                    raise FileNotFoundError(
                                        "Matched-scene seed artifact is missing: "
                                        + str(replay_seed_source)
                                    )
                                with np.load(replay_seed_source, allow_pickle=False) as source:
                                    required_seed_keys = (
                                        "demo_random_seed_state",
                                        "demo_random_seed_position",
                                        "demo_random_seed_has_gauss",
                                        "demo_random_seed_cached_gaussian",
                                    )
                                    missing_seed_keys = [
                                        key for key in required_seed_keys if key not in source.files
                                    ]
                                    if missing_seed_keys:
                                        raise RuntimeError(
                                            "Matched-scene artifact lacks saved RNG fields: "
                                            + ", ".join(missing_seed_keys)
                                        )
                                    np.random.set_state(
                                        (
                                            "MT19937",
                                            np.asarray(
                                                source["demo_random_seed_state"],
                                                dtype=np.uint32,
                                            ).copy(),
                                            int(source["demo_random_seed_position"]),
                                            int(source["demo_random_seed_has_gauss"]),
                                            float(source["demo_random_seed_cached_gaussian"]),
                                        )
                                    )
                                print(
                                    "[matched-scene-seed] task="
                                    + task_name
                                    + " episode="
                                    + str(local_index)
                                    + " source="
                                    + str(replay_seed_source),
                                    flush=True,
                                )
                            # The surrounding loop owns the retry budget. Keeping RLBench's
                            # internal budget at one avoids multiplying retries by 10x.
                            demos = task_env.get_demos(1, live_demos=True, max_attempts=1)
                            demo = trim_demo_after_first_success(
                                demos[0],
                                args.post_success_frames,
                                args.action_alignment,
                            )
                            descriptions, reset_observation = task_env.reset_to_demo(demo)
                        initial_phone_to_base_sensor_distance_m = None
                        initial_phone_to_eef_distance_m = None
                        initial_phone_to_robot_base_distance_m = None
                        if task_name == "phone_on_base":
                            phone_position = np.asarray(
                                task_env._task.phone.get_position(), dtype=np.float64
                            )
                            base_sensor_position = np.asarray(
                                task_env._task.success_sensor.get_position(), dtype=np.float64
                            )
                            initial_phone_to_base_sensor_distance_m = float(
                                np.linalg.norm(phone_position - base_sensor_position)
                            )
                            initial_phone_to_eef_distance_m = float(
                                np.linalg.norm(
                                    phone_position
                                    - np.asarray(
                                        reset_observation.gripper_pose[:3],
                                        dtype=np.float64,
                                    )
                                )
                            )
                            initial_phone_to_robot_base_distance_m = float(
                                np.linalg.norm(
                                    phone_position
                                    - np.asarray(
                                        t_world_base[:3, 3], dtype=np.float64
                                    )
                                )
                            )
                            if (
                                args.phone_base_max_initial_distance_m is not None
                                and initial_phone_to_base_sensor_distance_m
                                > float(args.phone_base_max_initial_distance_m)
                            ):
                                raise RuntimeError(
                                    "phone_on_base initial distance rejected: "
                                    + str(initial_phone_to_base_sensor_distance_m)
                                    + " > "
                                    + str(args.phone_base_max_initial_distance_m)
                                )
                            if (
                                args.phone_eef_max_initial_distance_m is not None
                                and initial_phone_to_eef_distance_m
                                > float(args.phone_eef_max_initial_distance_m) + 1e-5
                            ):
                                raise RuntimeError(
                                    "phone_on_base restored phone-to-EEF distance changed: "
                                    + str(initial_phone_to_eef_distance_m)
                                    + " > "
                                    + str(args.phone_eef_max_initial_distance_m)
                                )
                        configuration_tree, object_count = task_env._task.get_state()
                        configuration_bytes = configuration_tree_to_bytes(
                            configuration_tree
                        )
                        object_states = capture_initial_object_states(task_env._task)
                        if len(object_states["initial_object_names"]) != int(object_count):
                            raise RuntimeError(
                                "Task state and per-object state counts differ: "
                                + str(object_count)
                                + " versus "
                                + str(len(object_states["initial_object_names"]))
                            )
                        description = descriptions[0] if descriptions else task_name.replace("_", " ")
                        arrays = make_episode_arrays(
                            demo,
                            args.num_points,
                            args.gripper_points,
                            args.gripper_template,
                            args.gripper_max_width,
                            args.fps,
                            seed=local_index * 100000 + attempt * 1000,
                            fk_model=fk_model,
                            t_world_base=t_world_base,
                            generate_world_base_worldflow_sidecars=(
                                args.generate_world_base_worldflow_sidecars
                            ),
                            action_alignment=args.action_alignment,
                            action_label_mode=args.action_label_mode,
                        )
                        validate_captured_episode(
                            arrays, task_name, local_index, args.gripper_points
                        )
                        reset_image = np.asarray(reset_observation.front_rgb, dtype=np.uint8)
                        demo_first_image = np.asarray(demo[0].front_rgb, dtype=np.uint8)
                        if reset_image.shape != demo_first_image.shape:
                            raise RuntimeError(
                                "reset_to_demo image shape does not match demo frame 0: "
                                + str(reset_image.shape)
                                + " versus "
                                + str(demo_first_image.shape)
                            )
                        arrays["reset_first_rgb_mae"] = np.float32(
                            np.mean(
                                np.abs(
                                    reset_image.astype(np.float32)
                                    - demo_first_image.astype(np.float32)
                                )
                            )
                        )
                        arrays["initial_task_state_bytes"] = np.frombuffer(
                            configuration_bytes, dtype=np.uint8
                        ).copy()
                        arrays["initial_task_state_object_count"] = np.int64(object_count)
                        arrays.update(object_states)
                        random_seed = demo.random_seed
                        arrays["demo_random_seed_state"] = np.asarray(
                            random_seed[1], dtype=np.uint32
                        )
                        arrays["demo_random_seed_position"] = np.int64(random_seed[2])
                        arrays["demo_random_seed_has_gauss"] = np.int64(random_seed[3])
                        arrays["demo_random_seed_cached_gaussian"] = np.float64(random_seed[4])
                        arrays["demo_num_reset_attempts"] = np.int64(demo.num_reset_attempts)
                        arrays["success_trim_first_success_frame"] = np.int64(
                            -1
                            if demo.success_trim_first_success_frame is None
                            else int(demo.success_trim_first_success_frame)
                        )
                        arrays["success_trim_original_observations"] = np.int64(
                            demo.success_trim_original_observations
                        )
                        arrays["success_trim_kept_observations"] = np.int64(
                            demo.success_trim_kept_observations
                        )
                        arrays["success_trim_removed_observations"] = np.int64(
                            demo.success_trim_removed_observations
                        )
                        arrays["success_trim_post_success_frames"] = np.int64(
                            demo.success_trim_post_success_frames
                        )
                        arrays["success_trim_transition_terminal_observation"] = np.bool_(
                            demo.success_trim_transition_terminal_observation
                        )
                        waypoint_end_frames = getattr(
                            demo, "waypoint_end_frames", {}
                        )
                        arrays["waypoint_end_names"] = np.asarray(
                            list(waypoint_end_frames), dtype="U64"
                        )
                        arrays["waypoint_end_frames"] = np.asarray(
                            list(waypoint_end_frames.values()), dtype=np.int64
                        )
                        waypoint_end_frame_sequence = list(
                            getattr(
                                demo, "waypoint_end_frame_sequence", []
                            )
                        )
                        arrays["waypoint_end_sequence_names"] = np.asarray(
                            [name for name, _ in waypoint_end_frame_sequence],
                            dtype="U64",
                        )
                        arrays["waypoint_end_sequence_frames"] = np.asarray(
                            [frame for _, frame in waypoint_end_frame_sequence],
                            dtype=np.int64,
                        )
                        arrays["initial_scene_nearest_distance_m"] = np.float64(
                            -1.0
                            if initial_scene_nearest_distance_m is None
                            or not np.isfinite(initial_scene_nearest_distance_m)
                            else initial_scene_nearest_distance_m
                        )
                        arrays["initial_scene_nearest_episode"] = np.int64(
                            -1
                            if initial_scene_nearest_episode is None
                            else initial_scene_nearest_episode
                        )
                        if dual_waypoint0_selected_branch is not None:
                            arrays["dual_waypoint0_selected_branch"] = np.asarray(
                                dual_waypoint0_selected_branch
                            )
                            arrays["dual_waypoint0_authored_frames"] = np.int64(
                                -1
                                if dual_waypoint0_authored_frames is None
                                else dual_waypoint0_authored_frames
                            )
                            arrays[
                                "dual_waypoint0_finger_swapped_frames"
                            ] = np.int64(
                                -1
                                if dual_waypoint0_finger_swapped_frames is None
                                else dual_waypoint0_finger_swapped_frames
                            )
                        arrays["expert_path_mode"] = np.asarray(str(args.expert_path_mode))
                        arrays["phone_path_candidates"] = np.int64(
                            int(args.phone_path_candidates)
                        )
                        arrays["phone_best_of_n_waypoint3_only"] = np.bool_(
                            bool(args.phone_best_of_n_waypoint3_only)
                        )
                        arrays["path_xyz_loop_floor_m"] = np.float64(
                            float(args.path_xyz_loop_floor_m)
                        )
                        arrays["path_rotation_excess_limit_rad"] = np.float64(
                            float(args.path_rotation_excess_limit_rad)
                        )
                        arrays["phone_waypoint3_max_detour_ratio"] = np.float64(
                            float(args.phone_waypoint3_max_detour_ratio)
                        )
                        arrays[
                            "phone_waypoint3_max_lateral_deviation_m"
                        ] = np.float64(
                            float(args.phone_waypoint3_max_lateral_deviation_m)
                        )
                        arrays[
                            "phone_waypoint3_max_joint_travel_rad"
                        ] = np.float64(
                            float(args.phone_waypoint3_max_joint_travel_rad)
                        )
                        arrays["phone_waypoint3_exact_execution"] = np.bool_(
                            bool(args.phone_waypoint3_exact_execution)
                        )
                        arrays["phone_waypoint3_exact_step_rad"] = np.float64(
                            float(args.phone_waypoint3_exact_step_rad)
                        )
                        arrays["matched_scene_demo_candidates"] = np.int64(
                            int(args.matched_scene_demo_candidates)
                        )
                        arrays["matched_scene_demo_selected_index"] = np.int64(
                            -1
                            if matched_scene_demo_selected_index is None
                            else int(matched_scene_demo_selected_index)
                        )
                        arrays["phone_roll_symmetry"] = np.bool_(
                            bool(args.phone_roll_symmetry)
                        )
                        arrays["phone_waypoint0_roll_branch"] = np.asarray(
                            dual_waypoint0_selected_branch
                            if dual_waypoint0_selected_branch is not None
                            else (
                                "disabled"
                                if args.phone_waypoint0_roll_branch is None
                                else str(args.phone_waypoint0_roll_branch)
                            )
                        )
                        arrays["phone_later_waypoint_roll_policy"] = np.asarray(
                            str(args.phone_later_waypoint_roll_policy)
                        )
                        arrays["waypoint_roll_symmetry"] = np.bool_(
                            bool(args.waypoint_roll_symmetry)
                        )
                        arrays["waypoint_roll_force_branch"] = np.asarray(
                            "disabled"
                            if args.waypoint_roll_force_branch is None
                            else str(args.waypoint_roll_force_branch)
                        )
                        arrays["waypoint_roll_authored_fallback"] = np.bool_(
                            waypoint_roll_authored_fallback
                        )
                        arrays["waypoint_roll_primary_error"] = np.asarray(
                            ""
                            if waypoint_roll_primary_error is None
                            else waypoint_roll_primary_error
                        )
                        arrays["expert_linear_path_calls"] = np.int64(
                            int(linear_path_calls) - linear_calls_before
                        )
                        arrays["expert_rrt_path_calls"] = np.int64(
                            int(rrt_path_calls) - rrt_calls_before
                        )
                        arrays["expert_cartesian_path_calls"] = np.int64(
                            int(cartesian_path_calls) - cartesian_calls_before
                        )
                        arrays["expert_cartesian_stock_fallback_calls"] = np.int64(
                            int(cartesian_stock_fallback_calls)
                            - cartesian_stock_fallbacks_before
                        )
                        if initial_phone_to_base_sensor_distance_m is not None:
                            arrays["initial_phone_to_base_sensor_distance_m"] = np.float64(
                                initial_phone_to_base_sensor_distance_m
                            )
                        if initial_phone_to_eef_distance_m is not None:
                            arrays["initial_phone_to_eef_distance_m"] = np.float64(
                                initial_phone_to_eef_distance_m
                            )
                        if initial_phone_to_robot_base_distance_m is not None:
                            arrays[
                                "initial_phone_to_robot_base_distance_m"
                            ] = np.float64(initial_phone_to_robot_base_distance_m)
                        if phone_waypoint0_default_selection_info is not None:
                            arrays["phone_waypoint0_default_branch"] = np.asarray(
                                phone_waypoint0_default_selection_info[
                                    "default_branch"
                                ]
                            )
                            arrays[
                                "phone_waypoint0_authored_angle_rad"
                            ] = np.float64(
                                phone_waypoint0_default_selection_info[
                                    "authored_angle_rad"
                                ]
                            )
                            arrays[
                                "phone_waypoint0_finger_swapped_angle_rad"
                            ] = np.float64(
                                phone_waypoint0_default_selection_info[
                                    "finger_swapped_angle_rad"
                                ]
                            )
                        if replay_scene_source is not None:
                            arrays["matched_scene_source"] = np.asarray(
                                str(replay_scene_source)
                            )
                        save_artifact(artifact, task_name, local_index, arrays, description, args.variation)
                        record = {
                            "task": task_name,
                            "local_episode_index": local_index,
                            "description": str(description),
                            "variation": args.variation,
                            "frames": int(len(arrays["actions"])),
                            "fk_target_vs_achieved_position_error_median_m": (
                                float(np.median(arrays["fk_position_errors"]))
                                if len(arrays["fk_position_errors"])
                                else 0.0
                            ),
                            "fk_target_vs_achieved_position_error_max_m": (
                                float(np.max(arrays["fk_position_errors"]))
                                if len(arrays["fk_position_errors"])
                                else 0.0
                            ),
                            "action_label_mode": str(args.action_label_mode),
                            "expert_path_mode": str(args.expert_path_mode),
                            "phone_path_candidates": int(args.phone_path_candidates),
                            "phone_best_of_n_waypoint3_only": bool(
                                args.phone_best_of_n_waypoint3_only
                            ),
                            "path_xyz_loop_floor_m": float(args.path_xyz_loop_floor_m),
                            "path_rotation_excess_limit_rad": float(
                                args.path_rotation_excess_limit_rad
                            ),
                            "phone_waypoint3_max_detour_ratio": float(
                                args.phone_waypoint3_max_detour_ratio
                            ),
                            "phone_waypoint3_max_lateral_deviation_m": float(
                                args.phone_waypoint3_max_lateral_deviation_m
                            ),
                            "phone_waypoint3_max_joint_travel_rad": float(
                                args.phone_waypoint3_max_joint_travel_rad
                            ),
                            "phone_waypoint3_exact_execution": bool(
                                args.phone_waypoint3_exact_execution
                            ),
                            "phone_waypoint3_exact_step_rad": float(
                                args.phone_waypoint3_exact_step_rad
                            ),
                            "matched_scene_demo_candidates": int(
                                args.matched_scene_demo_candidates
                            ),
                            "matched_scene_demo_selected_index": (
                                None
                                if matched_scene_demo_selected_index is None
                                else int(matched_scene_demo_selected_index)
                            ),
                            "matched_scene_demo_selected_metrics": (
                                None
                                if matched_scene_demo_selected_metrics is None
                                else dict(matched_scene_demo_selected_metrics)
                            ),
                            "phone_roll_symmetry": bool(args.phone_roll_symmetry),
                            "phone_waypoint0_roll_branch": (
                                dual_waypoint0_selected_branch
                                if dual_waypoint0_selected_branch is not None
                                else (
                                    None
                                    if args.phone_waypoint0_roll_branch is None
                                    else str(args.phone_waypoint0_roll_branch)
                                )
                            ),
                            "phone_later_waypoint_roll_policy": str(
                                args.phone_later_waypoint_roll_policy
                            ),
                            "waypoint_roll_symmetry": bool(
                                args.waypoint_roll_symmetry
                            ),
                            "waypoint_roll_force_branch": (
                                None
                                if args.waypoint_roll_force_branch is None
                                else str(args.waypoint_roll_force_branch)
                            ),
                            "waypoint_roll_authored_fallback": bool(
                                waypoint_roll_authored_fallback
                            ),
                            "waypoint_roll_primary_error": (
                                waypoint_roll_primary_error
                            ),
                            "stop_after_success": bool(args.stop_after_success),
                            "initial_scene_nearest_distance_m": (
                                None
                                if initial_scene_nearest_distance_m is None
                                or not np.isfinite(initial_scene_nearest_distance_m)
                                else float(initial_scene_nearest_distance_m)
                            ),
                            "initial_scene_nearest_episode": (
                                None
                                if initial_scene_nearest_episode is None
                                else int(initial_scene_nearest_episode)
                            ),
                            "dual_waypoint0_selected_branch": (
                                dual_waypoint0_selected_branch
                            ),
                            "dual_waypoint0_authored_frames": (
                                dual_waypoint0_authored_frames
                            ),
                            "dual_waypoint0_finger_swapped_frames": (
                                dual_waypoint0_finger_swapped_frames
                            ),
                            "first_success_frame": (
                                None
                                if demo.success_trim_first_success_frame is None
                                else int(demo.success_trim_first_success_frame)
                            ),
                            "success_trim_original_observations": int(
                                demo.success_trim_original_observations
                            ),
                            "success_trim_kept_observations": int(
                                demo.success_trim_kept_observations
                            ),
                            "success_trim_removed_observations": int(
                                demo.success_trim_removed_observations
                            ),
                            "post_success_frames": int(args.post_success_frames),
                            "min_first_success_frame": int(
                                args.min_first_success_frame
                            ),
                            "matched_scene_seed_source": (
                                None if replay_seed_source is None else str(replay_seed_source)
                            ),
                            "matched_scene_source": (
                                None if replay_scene_source is None else str(replay_scene_source)
                            ),
                            "expert_linear_path_calls": int(
                                int(linear_path_calls) - linear_calls_before
                            ),
                            "expert_rrt_path_calls": int(
                                int(rrt_path_calls) - rrt_calls_before
                            ),
                            "expert_cartesian_path_calls": int(
                                int(cartesian_path_calls) - cartesian_calls_before
                            ),
                            "expert_cartesian_stock_fallback_calls": int(
                                int(cartesian_stock_fallback_calls)
                                - cartesian_stock_fallbacks_before
                            ),
                        }
                        if initial_phone_to_base_sensor_distance_m is not None:
                            record["initial_phone_to_base_sensor_distance_m"] = float(
                                initial_phone_to_base_sensor_distance_m
                            )
                        if initial_phone_to_eef_distance_m is not None:
                            record["initial_phone_to_eef_distance_m"] = float(
                                initial_phone_to_eef_distance_m
                            )
                        if initial_phone_to_robot_base_distance_m is not None:
                            record[
                                "initial_phone_to_robot_base_distance_m"
                            ] = float(initial_phone_to_robot_base_distance_m)
                        if phone_waypoint0_default_selection_info is not None:
                            record["phone_waypoint0_default_branch"] = str(
                                phone_waypoint0_default_selection_info[
                                    "default_branch"
                                ]
                            )
                            record[
                                "phone_waypoint0_authored_angle_rad"
                            ] = float(
                                phone_waypoint0_default_selection_info[
                                    "authored_angle_rad"
                                ]
                            )
                            record[
                                "phone_waypoint0_finger_swapped_angle_rad"
                            ] = float(
                                phone_waypoint0_default_selection_info[
                                    "finger_swapped_angle_rad"
                                ]
                            )
                        records.append(record)
                        if accepted_candidate_signature is not None:
                            accepted_scene_signatures.append(
                                (local_index, accepted_candidate_signature)
                            )
                        known.add(key)
                        with open(manifest_path, "w", encoding="utf-8") as file:
                            json.dump({"config": config_signature, "records": records}, file, indent=2)
                        print("[ok] task=" + task_name + " episode=" + str(local_index) + " frames=" + str(record["frames"]))
                        success = True
                        successful_count += 1
                        if requested_indices is None:
                            local_index += 1
                        else:
                            episode_cursor += 1
                            if episode_cursor < len(requested_indices):
                                local_index = requested_indices[episode_cursor]
                        break
                    except Exception as error:
                        last_error = error
                        print(
                            "[retry] task="
                            + task_name
                            + " episode="
                            + str(local_index)
                            + " error="
                            + repr(error)
                            + "\n"
                            + traceback.format_exc(),
                            flush=True,
                        )
                if not success:
                    print(
                        "[failed] task="
                        + task_name
                        + " episode="
                        + str(local_index)
                        + " collected="
                        + str(successful_count)
                        + "/"
                        + str(episode_target_count)
                        + " retrying error="
                        + repr(last_error)
                    )
                    if args.abort_on_episode_failure:
                        raise RuntimeError(
                            "Episode failed after "
                            + str(args.max_demo_attempts)
                            + " attempt(s): task="
                            + task_name
                            + " episode="
                            + str(local_index)
                        ) from last_error
    finally:
        Point.get_path = original_point_get_path
        PredefinedPath.get_path = original_predefined_get_path
        if original_task_feasible is not None:
            BackendTask._feasible = original_task_feasible
        env.shutdown()
    records.sort(key=lambda item: (tasks.index(item["task"]), item["local_episode_index"]))
    with open(manifest_path, "w", encoding="utf-8") as file:
        json.dump({"config": config_signature, "records": records}, file, indent=2)
    return records


def collection_worker_entry(worker_id, tasks, args, artifact_root, display):
    """Run one isolated RLBench process on a task shard."""
    os.environ["DISPLAY"] = str(display)
    worker_root = Path(artifact_root) / "_workers" / ("worker_" + str(worker_id).zfill(2))
    print(
        "[collection-worker] id="
        + str(worker_id)
        + " display="
        + str(display)
        + " tasks="
        + ",".join(tasks),
        flush=True,
    )
    records = collect(args, tasks, worker_root)
    return {
        "worker_id": int(worker_id),
        "artifact_root": str(worker_root),
        "records": records,
    }


def collection_display_list(args, worker_count):
    if worker_count <= 1:
        return [os.environ.get("DISPLAY", ":99")]
    if args.collection_display_base is not None:
        base = int(args.collection_display_base)
    else:
        current_display = os.environ.get("DISPLAY", "")
        if not current_display.startswith(":"):
            raise RuntimeError(
                "Parallel RLBench collection needs a local DISPLAY such as :99 "
                "or an explicit --collection-display-base."
            )
        base = int(current_display[1:].split(".", 1)[0])
    return [":" + str(base + worker_id) for worker_id in range(worker_count)]


def validate_collection_displays(displays):
    for display in displays:
        if not str(display).startswith(":"):
            continue
        display_number = str(display)[1:].split(".", 1)[0]
        x_socket = Path("/tmp/.X11-unix") / ("X" + display_number)
        if not x_socket.exists():
            raise RuntimeError(
                "RLBench collection worker needs a running X server for "
                + str(display)
                + "; missing "
                + str(x_socket)
                + ". Start one X server per worker before collection."
            )
        xdpyinfo = shutil.which("xdpyinfo")
        if xdpyinfo is not None:
            probe = subprocess.run(
                [xdpyinfo, "-display", str(display)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                check=False,
            )
            if probe.returncode != 0:
                error = probe.stderr.decode("utf-8", errors="replace").strip()
                raise RuntimeError(
                    "RLBench collection worker cannot connect to "
                    + str(display)
                    + "; xdpyinfo failed: "
                    + error
                )


def collect_parallel(args, tasks, artifact_root):
    """Collect task shards in spawned processes and merge their artifacts."""
    single_task_episode_parallel = len(tasks) == 1 and int(args.collection_workers) > 1
    if single_task_episode_parallel:
        worker_count = min(int(args.collection_workers), int(args.episodes_per_task))
    else:
        worker_count = min(int(args.collection_workers), len(tasks))
    if worker_count <= 1:
        return collect(args, tasks, artifact_root)

    displays = collection_display_list(args, worker_count)
    validate_collection_displays(displays)
    if single_task_episode_parallel:
        task_shards = [[tasks[0]] for _ in range(worker_count)]
        base, remainder = divmod(int(args.episodes_per_task), worker_count)
        episode_ranges = []
        start = 0
        for worker_id in range(worker_count):
            count = base + (1 if worker_id < remainder else 0)
            episode_ranges.append((start, count))
            start += count
    else:
        task_shards = [[] for _ in range(worker_count)]
        for task_index, task_name in enumerate(tasks):
            task_shards[task_index % worker_count].append(task_name)
        task_shards = [shard for shard in task_shards if shard]
        episode_ranges = [(0, int(args.episodes_per_task)) for _ in task_shards]
    artifact_root.mkdir(parents=True, exist_ok=True)

    worker_args = []
    for worker_id, shard in enumerate(task_shards):
        worker_namespace = argparse.Namespace(**vars(args))
        episode_start, episode_count = episode_ranges[worker_id]
        worker_namespace.episode_start = int(episode_start)
        worker_namespace.episodes_per_task = int(episode_count)
        worker_args.append(
            (worker_id, shard, worker_namespace, artifact_root, displays[worker_id])
        )
    context = mp.get_context("spawn")
    print(
        "[collection] starting "
        + str(len(worker_args))
        + " workers on displays "
        + ",".join(displays),
        flush=True,
    )
    with context.Pool(processes=len(worker_args)) as pool:
        results = pool.starmap(collection_worker_entry, worker_args)

    records = []
    for result in results:
        worker_root = Path(result["artifact_root"])
        for record in result["records"]:
            source = worker_root / episode_name(
                record["task"], int(record["local_episode_index"])
            )
            destination = artifact_root / source.name
            if not artifact_is_complete(
                source,
                require_world_base_worldflow_sidecars=(
                    args.generate_world_base_worldflow_sidecars
                ),
            ):
                raise RuntimeError("Worker artifact is incomplete: " + str(source))
            if destination.exists():
                shutil.rmtree(destination)
            copy_tree_with_hardlinks(source, destination)
            records.append(record)

    records.sort(key=lambda item: (tasks.index(item["task"]), item["local_episode_index"]))
    with open(artifact_root / "manifest.json", "w", encoding="utf-8") as file:
        json.dump(
            {"config": collection_config_signature(args, tasks), "records": records},
            file,
            indent=2,
        )
    return records


def run_pointseg_cache(args):
    if args.skip_pointseg_cache:
        print("[cache] skipped by --skip-pointseg-cache")
        return None

    expected_template_version = (
        LIBERO_GRIPPER_TEMPLATE_VERSION
        if args.gripper_template == LIBERO_GRIPPER_TEMPLATE
        else RLBENCH_PANDA_GRIPPER_TEMPLATE_VERSION
    )
    dataset_root = args.output_root.expanduser().resolve()
    cache_root = (
        args.cache_output_dir.expanduser().resolve()
        if args.cache_output_dir is not None
        else dataset_root.parent / (dataset_root.name + "_pointseg_cache")
    )
    manifest_path = cache_root / "manifest.json"
    if manifest_path.is_file() and not args.overwrite_cache:
        pipeline_metadata_path = dataset_root / "meta" / "rlbench_collection_pipeline.json"
        with open(dataset_root / "meta" / "info.json", "r", encoding="utf-8") as file:
            dataset_info = json.load(file)
        with open(manifest_path, "r", encoding="utf-8") as file:
            cache_manifest = json.load(file)
        requested_pseudo_config = {
            "motion_rotation_radius": args.motion_rotation_radius,
            "motion_baseline_threshold": args.motion_baseline_threshold,
            "motion_baseline_temperature": args.motion_baseline_temperature,
            "motion_relative_margin": args.motion_relative_margin,
            "motion_relative_tau": args.motion_relative_tau,
            "trajectory_sigma": args.trajectory_sigma,
            "contact_radius": args.contact_radius,
            "contact_temperature": args.contact_temperature,
            "approach_margin": args.approach_margin,
            "approach_tau": args.approach_tau,
            "background_trajectory_sigma": args.background_trajectory_sigma,
        }
        cached_pseudo_config = cache_manifest.get("pseudo_label_config", {})
        pseudo_config_matches = all(
            key in cached_pseudo_config
            and np.isclose(float(cached_pseudo_config[key]), float(value), rtol=0.0, atol=1e-8)
            for key, value in requested_pseudo_config.items()
        )
        pipeline_metadata = {}
        if pipeline_metadata_path.is_file():
            with open(pipeline_metadata_path, "r", encoding="utf-8") as file:
                pipeline_metadata = json.load(file)
        frame_count_matches = int(cache_manifest.get("num_samples", -1)) == int(
            dataset_info["total_frames"]
        )
        template_matches = pipeline_metadata.get("gripper_template") == expected_template_version
        if frame_count_matches and template_matches and pseudo_config_matches:
            print("[cache] complete cache already exists: " + str(cache_root))
            return cache_root
        if not pseudo_config_matches:
            raise RuntimeError(
                "PointSeg cache pseudo-label parameters do not match the requested values. "
                "Pass --overwrite-cache to rebuild: " + str(cache_root)
            )
        raise RuntimeError(
            "PointSeg cache frame count or gripper-template version does not match this dataset. "
            "Pass --overwrite-cache to rebuild: " + str(cache_root)
        )

    cache_current_points = (
        args.num_points if args.cache_current_points is None else args.cache_current_points
    )
    cache_future_points = (
        cache_current_points if args.cache_future_points is None else args.cache_future_points
    )
    if cache_current_points <= 0 or cache_future_points <= 0:
        raise ValueError("--cache-current-points and --cache-future-points must be positive")

    cache_python_arg = str(args.cache_python).strip()
    cache_python_path = Path(cache_python_arg).expanduser()
    if not cache_python_path.is_file() and "/" not in cache_python_arg:
        resolved_cache_python = shutil.which(cache_python_arg)
        if resolved_cache_python is not None:
            cache_python_path = Path(resolved_cache_python)
    cache_python = cache_python_path.resolve()
    if not cache_python.is_file():
        raise FileNotFoundError("PointSeg cache Python does not exist: " + str(cache_python))
    # Keep the original RLBench cache entrypoint so preview PLY files retain
    # the original continuous heatmap colors.
    cache_script = Path(__file__).resolve().parent / "cache_pointseg_samples.py"
    command = [
        str(cache_python),
        str(cache_script),
        "--dataset.repo_id=" + str(dataset_root),
        "--point-cloud-dir=" + str(dataset_root / POINT_DIR),
        "--output-dir=" + str(cache_root),
        "--current-points=" + str(cache_current_points),
        "--future-points=" + str(cache_future_points),
        "--batch-size=" + str(args.cache_batch_size),
        "--num-workers=" + str(args.cache_num_workers),
        "--device=" + str(args.cache_device),
        "--vis-count=" + str(args.cache_vis_count),
        "--motion-rotation-radius=" + str(args.motion_rotation_radius),
        "--motion-baseline-threshold=" + str(args.motion_baseline_threshold),
        "--motion-baseline-temperature=" + str(args.motion_baseline_temperature),
        "--motion-relative-margin=" + str(args.motion_relative_margin),
        "--motion-relative-tau=" + str(args.motion_relative_tau),
        "--trajectory-sigma=" + str(args.trajectory_sigma),
        "--contact-radius=" + str(args.contact_radius),
        "--contact-temperature=" + str(args.contact_temperature),
        "--approach-margin=" + str(args.approach_margin),
        "--approach-tau=" + str(args.approach_tau),
        "--background-trajectory-sigma=" + str(args.background_trajectory_sigma),
    ]
    if args.cache_vis_one_episode_per_task:
        command.append("--vis-one-episode-per-task")
    if args.overwrite_cache:
        command.append("--overwrite")
    print("[cache] starting: " + " ".join(command), flush=True)
    subprocess.run(command, cwd=REPO_ROOT, check=True)

    with open(dataset_root / "meta" / "rlbench_collection_pipeline.json", "w", encoding="utf-8") as file:
        json.dump(
            {
                "complete": True,
                "dataset_root": str(dataset_root),
                "pointseg_cache": str(cache_root),
                "raw_point_count": int(args.num_points),
                "cache_current_points": int(cache_current_points),
                "cache_future_points": int(cache_future_points),
                "action_semantics": action_semantics(args.action_label_mode),
                "action_alignment": str(args.action_alignment),
                "action_label_mode": str(args.action_label_mode),
                "post_success_frames": int(args.post_success_frames),
                "state_semantics": "achieved EEF pose in episode EEF0",
                "world_base_worldflow_sidecars": bool(
                    args.generate_world_base_worldflow_sidecars
                ),
                "rgb": "observation.images.front",
                "point_cloud": "finite front-camera cloud plus selected virtual gripper template in current EEF",
                "scene_bounds": RLBENCH_SCENE_BOUNDS.tolist(),
                "gripper_template": expected_template_version,
                "virtual_gripper": (
                    canonical_reap_metadata()
                    if args.gripper_template == LIBERO_GRIPPER_TEMPLATE
                    else None
                ),
                "collection_workers": int(args.collection_workers),
            },
            file,
            indent=2,
        )
    return cache_root


def main():
    args = parse_args()
    print(
        "[collector-runtime] task_environment="
        + str(RL_BENCH_ROOT / "rlbench" / "task_environment.py")
        + " water_plant_collision="
        + os.environ.get("RLBENCH_WATER_PLANT_COLLISION", "enabled")
        + " water_drop_collision="
        + os.environ.get("RLBENCH_WATER_DROP_COLLISION", "original"),
        flush=True,
    )
    artifact_root = (
        args.artifact_root.expanduser().resolve()
        if args.artifact_root is not None
        else args.output_root.expanduser().resolve().parent
        / (args.output_root.name + "_artifacts")
    )
    pack_only_manifest = None
    if args.pack_only:
        manifest_path = artifact_root / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(
                "--pack-only requires a completed artifact manifest: " + str(manifest_path)
            )
        with open(manifest_path, "r", encoding="utf-8") as file:
            pack_only_manifest = json.load(file)
        config = pack_only_manifest.get("config")
        records = pack_only_manifest.get("records")
        if not isinstance(config, dict) or not isinstance(records, list):
            raise RuntimeError("Artifact manifest must contain config and records: " + str(manifest_path))
        tasks = list(config.get("tasks") or [])
        if not tasks:
            raise RuntimeError("Artifact manifest has no task list: " + str(manifest_path))
        requested_tasks = resolve_tasks(args) if (args.tasks is not None or args.all_tasks) else tasks
        if requested_tasks != tasks:
            raise RuntimeError(
                "--pack-only task order must match the artifact manifest: requested="
                + repr(requested_tasks)
                + " manifest="
                + repr(tasks)
            )
        # Historical manifests predate this flag and their artifacts do not
        # contain T_world_base/world-base sidecars. Preserve that exact layout
        # when repacking instead of silently enabling a newer default.
        if "generate_world_base_worldflow_sidecars" not in config:
            args.generate_world_base_worldflow_sidecars = False
        # Historical artifacts were collected before success-tail trimming.
        # Repacking them must describe that original, untrimmed behavior.
        if "post_success_frames" not in config:
            args.post_success_frames = -1
        for name in (
            "episodes_per_task",
            "episode_indices",
            "variation",
            "num_points",
            "gripper_points",
            "gripper_max_width",
            "gripper_template",
            "image_size",
            "fps",
            "collection_workers",
            "expert_path_mode",
            "phone_path_candidates",
            "phone_roll_symmetry",
            "phone_waypoint0_roll_branch",
            "phone_later_waypoint_roll_policy",
            "dual_waypoint0_roll_select_shorter",
            "waypoint_roll_symmetry",
            "segmented_linear_segments",
            "replay_random_seeds_from_artifacts",
            "collection_seed",
            "phone_base_max_initial_distance_m",
            "phone_eef_max_initial_distance_m",
            "phone_robot_base_min_initial_distance_m",
            "replay_scenes_from_artifacts",
            "action_label_mode",
            "action_alignment",
            "post_success_frames",
            "min_first_success_frame",
            "stop_after_success",
            "min_initial_scene_distance_m",
            "scene_rotation_radius_m",
            "artifacts_only",
            "generate_world_base_worldflow_sidecars",
        ):
            if name in config:
                setattr(args, name, config[name])
    else:
        tasks = resolve_tasks(args)
    if args.episodes_per_task <= 0:
        raise ValueError("--episodes-per-task must be positive")
    if args.episode_indices is not None and any(
        int(index) < 0 for index in args.episode_indices
    ):
        raise ValueError("--episode-indices values must be non-negative")
    if args.collection_workers <= 0:
        raise ValueError("--collection-workers must be positive")
    if args.episode_indices is not None and args.collection_workers != 1:
        raise ValueError(
            "--episode-indices currently requires --collection-workers 1"
        )
    if args.segmented_linear_segments <= 0:
        raise ValueError("--segmented-linear-segments must be positive")
    if args.post_success_frames < -1:
        raise ValueError("--post-success-frames must be -1 or non-negative")
    if args.min_first_success_frame < 0:
        raise ValueError("--min-first-success-frame must be non-negative")
    if args.stop_after_success and args.post_success_frames < 0:
        raise ValueError(
            "--stop-after-success requires non-negative --post-success-frames"
        )
    if args.min_initial_scene_distance_m < 0.0:
        raise ValueError("--min-initial-scene-distance-m must be non-negative")
    if (
        args.phone_robot_base_min_initial_distance_m is not None
        and args.phone_robot_base_min_initial_distance_m < 0.0
    ):
        raise ValueError(
            "--phone-robot-base-min-initial-distance-m must be non-negative"
        )
    if args.scene_rotation_radius_m <= 0.0:
        raise ValueError("--scene-rotation-radius-m must be positive")
    if args.artifacts_only and args.pack_only:
        raise ValueError("--artifacts-only and --pack-only are mutually exclusive")
    if args.artifacts_only and args.delete_artifacts_after_pack:
        raise ValueError(
            "--delete-artifacts-after-pack cannot be used with --artifacts-only"
        )
    if (
        args.min_initial_scene_distance_m > 0.0
        and args.collection_workers > 1
    ):
        raise ValueError(
            "Scene de-duplication requires --collection-workers 1 so every new "
            "episode is compared with all previously accepted episodes."
        )
    if (
        (
            args.min_initial_scene_distance_m > 0.0
            and (
                args.replay_scenes_from_artifacts is not None
                or args.replay_random_seeds_from_artifacts is not None
            )
        )
        or (
            args.dual_waypoint0_roll_select_shorter
            and args.replay_random_seeds_from_artifacts is not None
        )
    ):
        raise ValueError(
            "Scene de-duplication cannot be combined with replay, and dual "
            "waypoint0 selection requires full --replay-scenes-from-artifacts "
            "rather than random-seed-only replay."
        )
    enabled_roll_modes = sum(
        (
            bool(args.phone_roll_symmetry),
            bool(args.waypoint_roll_symmetry),
            args.phone_waypoint0_roll_branch is not None,
            bool(args.dual_waypoint0_roll_select_shorter),
        )
    )
    if enabled_roll_modes > 1:
        raise ValueError(
            "Use only one roll-symmetry mode: --phone-roll-symmetry, "
            "--waypoint-roll-symmetry, --phone-waypoint0-roll-branch, or "
            "--dual-waypoint0-roll-select-shorter"
        )
    if (
        args.phone_later_waypoint_roll_policy != "authored"
        and args.phone_waypoint0_roll_branch is None
        and not args.dual_waypoint0_roll_select_shorter
    ):
        raise ValueError(
            "--phone-later-waypoint-roll-policy requires "
            "--phone-waypoint0-roll-branch"
        )
    if args.num_points <= 0:
        raise ValueError("--num-points must be positive")
    if not np.isclose(args.gripper_max_width, RLBENCH_PANDA_MAX_WIDTH, atol=1e-8):
        raise ValueError(
            "RLBench Panda has a fixed 0.08 m total opening; "
            "--gripper-max-width must be " + str(RLBENCH_PANDA_MAX_WIDTH)
        )
    if args.cache_batch_size <= 0 or args.cache_num_workers < 0:
        raise ValueError("PointSeg cache batch size/workers are invalid")
    if not args.pack_only and not os.environ.get("DISPLAY"):
        print("[warning] DISPLAY is not set. RLBench camera rendering usually needs DISPLAY=:99.")
    print("[tasks] " + ", ".join(tasks))
    print("[artifacts] " + str(artifact_root))
    if args.pack_only:
        expected_episode_count = len(tasks) * int(args.episodes_per_task)
        if len(records) != expected_episode_count:
            raise RuntimeError(
                "Artifact manifest is incomplete: records="
                + str(len(records))
                + " expected="
                + str(expected_episode_count)
            )
        print("[pack-only] rebuilding from validated existing artifacts")
        pack_artifacts(args, artifact_root, records, expected_episode_count)
        cache_root = run_pointseg_cache(args)
        print("[done] output=" + str(args.output_root.expanduser().resolve()))
        if cache_root is not None:
            print("[done] pointseg_cache=" + str(cache_root))
        return
    if len(tasks) == 1:
        worker_count = min(args.collection_workers, args.episodes_per_task)
    else:
        worker_count = min(args.collection_workers, len(tasks))
    displays = collection_display_list(args, worker_count)
    validate_collection_displays(displays)
    print("[collection-workers] " + str(worker_count) + " displays=" + ",".join(displays))
    records = collect_parallel(args, tasks, artifact_root)
    if len(records) != len(tasks) * args.episodes_per_task:
        print("[warning] not all requested episodes succeeded; packing the successful episodes only")
    if args.artifacts_only:
        print(
            "[done] artifacts_only="
            + str(artifact_root)
            + " records="
            + str(len(records))
        )
        return
    pack_artifacts(args, artifact_root, records, len(tasks) * args.episodes_per_task)
    cache_root = run_pointseg_cache(args)
    print("[done] output=" + str(args.output_root.expanduser().resolve()))
    if cache_root is not None:
        print("[done] pointseg_cache=" + str(cache_root))


if __name__ == "__main__":
    main()
