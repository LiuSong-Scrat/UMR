#!/usr/bin/env python3
"""Robot-base WorldFlow sidecar math shared by RLBench collection and backfill."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


WORLD_BASE_EE_POSE_DIR = "world_base_ee_poses"
WORLD_BASE_ACTION_TARGET_EE_POSE_DIR = "world_base_action_target_ee_poses"
RLBENCH_PANDA_LINK0_FROM_JOINT1_Z_M = -0.333
RLBENCH_PANDA_LINK0_FRAME_VERSION = "panda_link0_from_joint1_minus_0p333m_v1"
RLBENCH_PANDA_LINK0_TRANSFORM_SOURCE = (
    "RLBench Panda link0 mounting frame: Panda_joint1.get_matrix() @ "
    "Translation(0, 0, -0.333 m), matching robosuite/LIBERO Panda base==link0"
)


def rlbench_panda_link0_to_world_matrix(arm) -> np.ndarray:
    """Return ``^world T_link0`` with LIBERO-compatible Panda base semantics.

    RLBench's top-level ``Panda`` scene object is a model/visual root whose
    object axes and origin are not the Franka kinematic link0 frame.  The fixed
    first-joint frame is 0.333 m above link0 along link0 +Z, matching the Panda
    geometry used by robosuite/LIBERO.
    """

    joints = list(getattr(arm, "joints", []))
    if not joints:
        raise ValueError("RLBench arm exposes no joints; cannot resolve Panda link0")
    joint1 = joints[0]
    joint_name = str(joint1.get_name())
    if joint_name != "Panda_joint1":
        raise ValueError(
            "Robot-base WorldFlow requires the RLBench Panda; first joint is "
            + repr(joint_name)
        )
    t_world_joint1 = validate_rigid_transform(
        np.asarray(joint1.get_matrix(), dtype=np.float64),
        "T_world_Panda_joint1",
    )
    t_joint1_link0 = np.eye(4, dtype=np.float64)
    t_joint1_link0[2, 3] = RLBENCH_PANDA_LINK0_FROM_JOINT1_Z_M
    return validate_rigid_transform(
        t_world_joint1 @ t_joint1_link0,
        "T_world_Panda_link0",
    )


def pose9_to_homo(pose9: np.ndarray) -> np.ndarray:
    """Convert xyz + first two rotation columns to a proper homogeneous pose."""

    pose = np.asarray(pose9, dtype=np.float64)
    if pose.shape[-1] < 9:
        raise ValueError(f"pose9 requires at least 9 values, got {pose.shape}")
    first = pose[..., 3:6]
    second = pose[..., 6:9]
    first_norm = np.linalg.norm(first, axis=-1, keepdims=True)
    if np.any(first_norm < 1e-10):
        raise ValueError("pose9 first rotation column has near-zero norm")
    first = first / first_norm
    second = second - np.sum(first * second, axis=-1, keepdims=True) * first
    second_norm = np.linalg.norm(second, axis=-1, keepdims=True)
    if np.any(second_norm < 1e-10):
        raise ValueError("pose9 rotation columns are linearly dependent")
    second = second / second_norm
    third = np.cross(first, second)

    output = np.zeros(pose.shape[:-1] + (4, 4), dtype=np.float64)
    output[..., :3, 0] = first
    output[..., :3, 1] = second
    output[..., :3, 2] = third
    output[..., :3, 3] = pose[..., :3]
    output[..., 3, 3] = 1.0
    return output


def homo_to_pose9(transform: np.ndarray) -> np.ndarray:
    matrix = np.asarray(transform, dtype=np.float64)
    if matrix.shape[-2:] != (4, 4):
        raise ValueError(f"Expected (...,4,4) homogeneous matrices, got {matrix.shape}")
    return np.concatenate(
        (matrix[..., :3, 3], matrix[..., :3, 0], matrix[..., :3, 1]),
        axis=-1,
    ).astype(np.float32)


def validate_rigid_transform(transform: np.ndarray, name: str, atol: float = 2e-5) -> np.ndarray:
    matrix = np.asarray(transform, dtype=np.float64)
    if matrix.shape != (4, 4):
        raise ValueError(f"{name} must have shape (4,4), got {matrix.shape}")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{name} contains NaN or Inf")
    if not np.allclose(matrix[3], np.array([0.0, 0.0, 0.0, 1.0]), atol=atol, rtol=0.0):
        raise ValueError(f"{name} has an invalid homogeneous bottom row: {matrix[3].tolist()}")
    rotation = matrix[:3, :3]
    orthogonality_error = float(np.max(np.abs(rotation.T @ rotation - np.eye(3))))
    determinant = float(np.linalg.det(rotation))
    if orthogonality_error > atol or abs(determinant - 1.0) > atol:
        raise ValueError(
            f"{name} is not rigid: orthogonality_error={orthogonality_error}, det={determinant}"
        )
    return matrix


def rotation_validation(pose9: np.ndarray) -> dict[str, float]:
    transforms = pose9_to_homo(pose9)
    rotation = transforms[..., :3, :3]
    gram = np.swapaxes(rotation, -1, -2) @ rotation
    orthogonality = np.max(np.abs(gram - np.eye(3)), axis=(-2, -1))
    determinants = np.linalg.det(rotation)
    return {
        "rotation_orthogonality_max_abs": float(np.max(orthogonality)),
        "rotation_determinant_min": float(np.min(determinants)),
        "rotation_determinant_max": float(np.max(determinants)),
    }


def build_robot_base_episode_sidecars(
    world_ee_poses: np.ndarray,
    action_labels: np.ndarray,
    t_world_base: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, float | int]]:
    """Build achieved and commanded EEF trajectories in the Panda base frame.

    ``action_labels[:, :9]`` are commanded targets in episode EEF0. This is
    independent of the gripper scalar and preserves transition alignment.
    """

    world_pose9 = np.asarray(world_ee_poses, dtype=np.float32)
    actions = np.asarray(action_labels, dtype=np.float32)
    if world_pose9.ndim != 2 or world_pose9.shape[1] != 9:
        raise ValueError(f"world_ee_poses must have shape (T,9), got {world_pose9.shape}")
    if actions.ndim != 2 or actions.shape[1] < 9:
        raise ValueError(f"action labels must have shape (T,D>=9), got {actions.shape}")
    if len(world_pose9) != len(actions):
        raise ValueError(
            f"world/action length mismatch: {len(world_pose9)} versus {len(actions)}"
        )
    if len(world_pose9) == 0:
        raise ValueError("Cannot build sidecars for an empty episode")
    if not np.isfinite(world_pose9).all() or not np.isfinite(actions[:, :9]).all():
        raise ValueError("World poses or action targets contain NaN/Inf")

    t_world_base = validate_rigid_transform(t_world_base, "T_world_base")
    t_base_world = validate_rigid_transform(np.linalg.inv(t_world_base), "T_base_world")
    t_world_ee = pose9_to_homo(world_pose9)
    t_world_eef0 = t_world_ee[0]
    t_eef0_target = pose9_to_homo(actions[:, :9])

    t_base_ee = t_base_world[None] @ t_world_ee
    t_base_target = t_base_world[None] @ t_world_eef0[None] @ t_eef0_target
    base_ee_pose9 = homo_to_pose9(t_base_ee)
    base_target_pose9 = homo_to_pose9(t_base_target)

    recovered_world_ee = t_world_base[None] @ pose9_to_homo(base_ee_pose9)
    recovered_eef0_target = (
        np.linalg.inv(t_world_eef0)[None]
        @ t_world_base[None]
        @ pose9_to_homo(base_target_pose9)
    )
    achieved_roundtrip = float(np.max(np.abs(recovered_world_ee - t_world_ee)))
    target_roundtrip = float(np.max(np.abs(recovered_eef0_target - t_eef0_target)))
    achieved_rotation = rotation_validation(base_ee_pose9)
    target_rotation = rotation_validation(base_target_pose9)
    metrics: dict[str, float | int] = {
        "frames": int(len(world_pose9)),
        "achieved_roundtrip_max_abs": achieved_roundtrip,
        "action_target_roundtrip_max_abs": target_roundtrip,
        "achieved_rotation_orthogonality_max_abs": achieved_rotation[
            "rotation_orthogonality_max_abs"
        ],
        "achieved_rotation_determinant_min": achieved_rotation[
            "rotation_determinant_min"
        ],
        "achieved_rotation_determinant_max": achieved_rotation[
            "rotation_determinant_max"
        ],
        "target_rotation_orthogonality_max_abs": target_rotation[
            "rotation_orthogonality_max_abs"
        ],
        "target_rotation_determinant_min": target_rotation["rotation_determinant_min"],
        "target_rotation_determinant_max": target_rotation["rotation_determinant_max"],
    }
    if achieved_roundtrip > 5e-5 or target_roundtrip > 5e-5:
        raise ValueError(f"Robot-base sidecar roundtrip validation failed: {metrics}")
    if not np.isfinite(base_ee_pose9).all() or not np.isfinite(base_target_pose9).all():
        raise ValueError("Generated robot-base sidecars contain NaN/Inf")
    return base_ee_pose9, base_target_pose9, metrics


def sidecar_metadata(
    t_world_base: np.ndarray,
    transform_source: str,
    action_alignment: str = "transition",
    base_frame_definition: str | None = None,
) -> tuple[dict, dict]:
    t_world_base = validate_rigid_transform(t_world_base, "T_world_base")
    t_base_world = np.linalg.inv(t_world_base)
    common = {
        "shape": [9],
        "dtype": "float32",
        "layout": "episode_npy",
        "coordinate_frame": "robot_base",
        "transform_convention": "^A T_B maps B-frame coordinates into A-frame coordinates",
        "T_world_base": t_world_base.tolist(),
        "T_base_world": t_base_world.tolist(),
        "transform_source": str(transform_source),
    }
    if base_frame_definition is not None:
        common["base_frame_definition"] = str(base_frame_definition)
    achieved = {
        **common,
        "key": "worldflow.current_ee_pose",
        "target_semantics": "achieved_eef_pose",
        "source": "world_ee_poses",
        "formula": "T_base_ee = T_base_world @ T_world_ee",
        "path_format": f"{WORLD_BASE_EE_POSE_DIR}/episode_{{episode_index:06d}}.npy",
    }
    target = {
        **common,
        "key": "worldflow.eef_trajectory",
        "target_semantics": "commanded_eef_pose",
        "alignment": str(action_alignment),
        "source": (
            "commanded expert target pose9 in episode EEF0; collector derives it "
            "from the RLBench expert joint command by FK, while dataset backfill "
            "requires LeRobot action_label_mode=expert_target"
        ),
        "formula": "T_base_target = T_base_world @ T_world_eef0 @ T_eef0_target",
        "path_format": (
            f"{WORLD_BASE_ACTION_TARGET_EE_POSE_DIR}/episode_{{episode_index:06d}}.npy"
        ),
    }
    return achieved, target


def write_sidecar_metadata(
    root: Path,
    t_world_base: np.ndarray,
    transform_source: str,
    action_alignment: str = "transition",
    base_frame_definition: str | None = None,
) -> None:
    root = Path(root)
    achieved, target = sidecar_metadata(
        t_world_base,
        transform_source,
        action_alignment=action_alignment,
        base_frame_definition=base_frame_definition,
    )
    for directory_name, payload in (
        (WORLD_BASE_EE_POSE_DIR, achieved),
        (WORLD_BASE_ACTION_TARGET_EE_POSE_DIR, target),
    ):
        directory = root / directory_name
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "meta.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
