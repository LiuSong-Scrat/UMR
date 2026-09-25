#!/usr/bin/env python3
"""Shared RLBench adapters for physical-width gripper action labels.

``delta_width_initial_sync`` follows the evaluator protocol used by the
LIBERO benchmark code in this repository:

* synchronize the physical gripper to the first prediction exactly once;
* self-reference the first executed row of every newly predicted chunk; and
* decode only within-chunk changes after that first row.

The legacy ``libero_delta`` mode is intentionally not aliased to this mode:
old RLBench results anchored the first row to a measured width and carried the
last prediction across replanned chunks.  Keeping the names distinct prevents
historical results from silently changing meaning.
"""

from __future__ import annotations

from typing import Any

import numpy as np


DELTA_WIDTH_INITIAL_SYNC = "delta_width_initial_sync"
LEGACY_LIBERO_DELTA = "libero_delta"
ABSOLUTE_WIDTH = "absolute_width"
GRIPPER_CONTROL_MODES = (
    DELTA_WIDTH_INITIAL_SYNC,
    LEGACY_LIBERO_DELTA,
    ABSOLUTE_WIDTH,
)
DELTA_GRIPPER_EVENTS = frozenset({"delta_open", "delta_close"})


def recover_discrete_gripper_command_after_control_failure(
    previous_command_open: bool,
    observed_open_amount: float,
) -> bool:
    """Preserve command intent when arm control fails before a gripper event.

    ``observed_open_amount`` is a continuous mechanical aperture, not the last
    discrete OPEN/CLOSE command.  In particular, fingers holding an object can
    remain more than half open after a valid CLOSE command.  Keep the previous
    logical command across controller recovery instead of thresholding that
    physical aperture and fabricating a new gripper event.
    """
    # Validate the diagnostic input without using it to infer command intent.
    if not np.isfinite(float(observed_open_amount)):
        raise ValueError("Observed gripper opening amount must be finite.")
    return bool(previous_command_open)


def libero_style_gripper_target(
    previous_width: float,
    predicted_width: float,
    commanded_gripper_open: bool,
    threshold: float,
    alignment: str = "current_minus_previous",
    next_width: float | None = None,
    open_threshold: float | None = None,
    close_threshold: float | None = None,
) -> tuple[bool, str, float]:
    """Decode one physical-width change into a persistent discrete command.

    ``threshold`` remains the backwards-compatible symmetric deadband.  The
    optional directional thresholds allow opening sensitivity to be changed
    without also making closing more sensitive (or vice versa).
    """
    previous_width = float(previous_width)
    predicted_width = float(predicted_width)
    if alignment == "current_minus_previous":
        width_change = predicted_width - previous_width
    elif alignment == "next_minus_current":
        width_change = (
            float(next_width) - predicted_width if next_width is not None else 0.0
        )
    else:
        raise ValueError("Unsupported gripper delta alignment: " + str(alignment))

    threshold = float(threshold)
    effective_open_threshold = (
        threshold if open_threshold is None else float(open_threshold)
    )
    effective_close_threshold = (
        threshold if close_threshold is None else float(close_threshold)
    )
    if effective_open_threshold < 0.0 or effective_close_threshold < 0.0:
        raise ValueError("Gripper delta thresholds must be non-negative.")
    if width_change > effective_open_threshold:
        return True, "delta_open", width_change
    if width_change < -effective_close_threshold:
        return False, "delta_close", width_change
    return bool(commanded_gripper_open), "delta_keep", width_change


def resolve_pending_delta_gripper_target(
    decoded_target_open: bool,
    decoded_event: str,
    pending_target_open: bool | None,
) -> tuple[bool, str, bool | None, bool]:
    """Carry an unexecuted delta event across a newly predicted chunk.

    A fresh chunk self-references its first row, so its first decoded event is
    normally ``delta_keep``.  If the previous chunk requested a gripper
    transition but the arm never reached that waypoint, replay the pending
    transition against the new pose target.  A new explicit and opposite delta
    event cancels/supersedes the stale pending event.

    Returns ``(target_open, event, pending_target_open, replayed_pending)``.
    The caller clears ``pending_target_open`` only after the deferred gripper
    command has actually been sent to the environment.
    """
    decoded_target_open = bool(decoded_target_open)
    if pending_target_open is None:
        return decoded_target_open, str(decoded_event), None, False

    pending_target_open = bool(pending_target_open)
    if decoded_event in DELTA_GRIPPER_EVENTS:
        if decoded_target_open != pending_target_open:
            return decoded_target_open, str(decoded_event), None, False
        return decoded_target_open, str(decoded_event), pending_target_open, False

    pending_event = "pending_delta_open" if pending_target_open else "pending_delta_close"
    return pending_target_open, pending_event, pending_target_open, True


def absolute_width_gripper_target(
    predicted_width: float,
    threshold: float,
) -> tuple[bool, str]:
    """Convert an absolute physical-width label into RLBench Discrete."""
    is_open = float(predicted_width) > float(threshold)
    return is_open, ("absolute_open" if is_open else "absolute_close")


def initial_delta_reference_for_chunk(
    action_chunk: np.ndarray,
    start_index: int,
) -> float:
    """Self-reference the first executed row of a newly predicted chunk."""
    chunk = np.asarray(action_chunk)
    start = int(start_index)
    if chunk.ndim != 2 or chunk.shape[1] < 10:
        raise ValueError("Expected action chunk with shape (T, >=10).")
    if start < 0 or start >= len(chunk):
        raise IndexError(
            f"Chunk start index {start} is outside a chunk with {len(chunk)} rows."
        )
    return float(chunk[start, 9])


def _resolve_gripper(task_env: Any) -> Any:
    robot = getattr(task_env, "_robot", None)
    if robot is None:
        scene = getattr(task_env, "_scene", None)
        robot = getattr(scene, "robot", None)
    if robot is None or getattr(robot, "gripper", None) is None:
        raise AttributeError("Could not resolve an RLBench robot gripper.")
    return robot.gripper


def set_gripper_absolute_width_position_target(
    task_env: Any,
    predicted_width: float,
    mechanical_max_width: float,
) -> dict[str, Any]:
    """Kinematically synchronize an RLBench gripper to one physical width.

    RLBench evaluation uses a Discrete gripper action mode, but the model label
    is total physical finger opening in metres.  The one-time episode-start
    synchronization therefore writes the two gripper joint positions directly,
    without inserting an environment step or fabricating an OPEN/CLOSE event.
    Subsequent policy rows continue to use the normal discrete action mode.
    """
    requested_width = float(predicted_width)
    max_width = float(mechanical_max_width)
    if not np.isfinite(requested_width):
        raise ValueError("Predicted gripper width must be finite.")
    if not np.isfinite(max_width) or max_width <= 0.0:
        raise ValueError("Mechanical gripper max width must be positive and finite.")

    target_width = float(np.clip(requested_width, 0.0, max_width))
    opening_fraction = target_width / max_width
    gripper = _resolve_gripper(task_env)
    _cyclic, joint_intervals_list = gripper.get_joint_intervals()
    joint_intervals = np.asarray(joint_intervals_list, dtype=np.float64)
    if joint_intervals.ndim != 2 or joint_intervals.shape[1] != 2:
        raise RuntimeError(
            "Unexpected RLBench gripper joint interval shape: "
            + str(joint_intervals.shape)
        )
    lower = joint_intervals[:, 0]
    upper = joint_intervals[:, 1]
    targets = lower + (upper - lower) * opening_fraction

    before_positions = np.asarray(gripper.get_joint_positions(), dtype=np.float64)
    gripper.set_joint_positions(targets.tolist(), disable_dynamics=True)
    gripper.set_joint_target_positions(targets.tolist())
    gripper.set_joint_target_velocities([0.0] * len(targets))
    # Gripper.actuate() maintains short-lived oscillation-detection state.
    # A kinematic synchronization starts a new command history.
    if hasattr(gripper, "_prev_positions"):
        gripper._prev_positions = [None] * len(targets)
    if hasattr(gripper, "_prev_vels"):
        gripper._prev_vels = [None] * len(targets)
    after_positions = np.asarray(gripper.get_joint_positions(), dtype=np.float64)
    after_open_amounts = np.asarray(gripper.get_open_amount(), dtype=np.float64)
    command_gripper_open = bool(np.all(after_open_amounts > 0.9))

    return {
        "requested_width": requested_width,
        "target_width": target_width,
        "mechanical_max_width": max_width,
        "opening_fraction": opening_fraction,
        "joint_positions_before": before_positions.tolist(),
        "joint_position_targets": targets.tolist(),
        "joint_positions_after": after_positions.tolist(),
        "open_amounts_after": after_open_amounts.tolist(),
        "command_gripper_open": command_gripper_open,
        "clipped": bool(target_width != requested_width),
        "inserted_environment_steps": 0,
    }
