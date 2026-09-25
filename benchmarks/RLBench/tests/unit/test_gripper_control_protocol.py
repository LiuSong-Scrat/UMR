import sys
from pathlib import Path

import numpy as np


SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from gripper_control import (  # noqa: E402
    initial_delta_reference_for_chunk,
    libero_style_gripper_target,
    recover_discrete_gripper_command_after_control_failure,
    resolve_pending_delta_gripper_target,
    set_gripper_absolute_width_position_target,
)


class FakeGripper:
    def __init__(self):
        self.positions = [0.04, 0.04]
        self.targets = None
        self.velocities = None
        self._prev_positions = [1.0, 1.0]
        self._prev_vels = [1.0, 1.0]

    def get_joint_intervals(self):
        return [False, False], [[0.0, 0.04], [0.0, 0.04]]

    def get_joint_positions(self):
        return list(self.positions)

    def set_joint_positions(self, values, disable_dynamics=False):
        assert disable_dynamics is True
        self.positions = list(values)

    def set_joint_target_positions(self, values):
        self.targets = list(values)

    def set_joint_target_velocities(self, values):
        self.velocities = list(values)

    def get_open_amount(self):
        return [position / 0.04 for position in self.positions]


class FakeRobot:
    def __init__(self):
        self.gripper = FakeGripper()


class FakeTaskEnvironment:
    def __init__(self):
        self._robot = FakeRobot()


def test_new_chunk_first_executed_row_is_its_own_delta_reference():
    chunk = np.zeros((4, 10), dtype=np.float32)
    chunk[:, 9] = [0.08, 0.06, 0.025, 0.07]

    previous = initial_delta_reference_for_chunk(chunk, start_index=2)
    target_open, event, delta = libero_style_gripper_target(
        previous, chunk[2, 9], True, threshold=0.003
    )

    assert previous == float(chunk[2, 9])
    assert target_open is True
    assert event == "delta_keep"
    assert delta == 0.0


def test_control_failure_does_not_turn_blocked_close_aperture_into_open_command():
    command_open = recover_discrete_gripper_command_after_control_failure(
        previous_command_open=False,
        observed_open_amount=0.68,
    )

    assert command_open is False


def test_control_failure_does_not_turn_lagging_open_aperture_into_close_command():
    command_open = recover_discrete_gripper_command_after_control_failure(
        previous_command_open=True,
        observed_open_amount=0.25,
    )

    assert command_open is True


def test_only_subsequent_rows_trigger_within_chunk_open_close_events():
    command_open = True
    command_open, event, delta = libero_style_gripper_target(
        0.06, 0.025, command_open, threshold=0.003
    )
    assert command_open is False
    assert event == "delta_close"
    assert delta < -0.003

    command_open, event, delta = libero_style_gripper_target(
        0.025, 0.07, command_open, threshold=0.003
    )
    assert command_open is True
    assert event == "delta_open"
    assert delta > 0.003


def test_directional_delta_thresholds_only_change_the_requested_direction():
    command_open, event, delta = libero_style_gripper_target(
        0.03,
        0.0326,
        False,
        threshold=0.003,
        open_threshold=0.0025,
        close_threshold=0.003,
    )
    assert command_open is True
    assert event == "delta_open"
    assert delta > 0.0025

    command_open, event, delta = libero_style_gripper_target(
        0.0326,
        0.03,
        command_open,
        threshold=0.003,
        open_threshold=0.0025,
        close_threshold=0.003,
    )
    assert command_open is True
    assert event == "delta_keep"
    assert delta < -0.0025


def test_directional_delta_thresholds_fall_back_to_symmetric_threshold():
    command_open, event, _delta = libero_style_gripper_target(
        0.03, 0.0326, False, threshold=0.003
    )
    assert command_open is False
    assert event == "delta_keep"


def test_pending_delta_event_replays_on_new_chunk_keep_row():
    target_open, event, pending, replayed = resolve_pending_delta_gripper_target(
        decoded_target_open=False,
        decoded_event="delta_keep",
        pending_target_open=True,
    )

    assert target_open is True
    assert event == "pending_delta_open"
    assert pending is True
    assert replayed is True


def test_new_opposite_delta_event_cancels_pending_event():
    target_open, event, pending, replayed = resolve_pending_delta_gripper_target(
        decoded_target_open=False,
        decoded_event="delta_close",
        pending_target_open=True,
    )

    assert target_open is False
    assert event == "delta_close"
    assert pending is None
    assert replayed is False


def test_episode_initial_sync_sets_physical_width_without_environment_step():
    task_env = FakeTaskEnvironment()

    info = set_gripper_absolute_width_position_target(
        task_env, predicted_width=0.03, mechanical_max_width=0.08
    )

    assert np.allclose(task_env._robot.gripper.positions, [0.015, 0.015])
    assert np.allclose(task_env._robot.gripper.targets, [0.015, 0.015])
    assert task_env._robot.gripper.velocities == [0.0, 0.0]
    assert task_env._robot.gripper._prev_positions == [None, None]
    assert task_env._robot.gripper._prev_vels == [None, None]
    assert info["opening_fraction"] == 0.375
    assert info["command_gripper_open"] is False
    assert info["inserted_environment_steps"] == 0


def test_episode_initial_sync_clips_width_to_mechanical_range():
    task_env = FakeTaskEnvironment()

    info = set_gripper_absolute_width_position_target(
        task_env, predicted_width=0.2, mechanical_max_width=0.08
    )

    assert np.allclose(task_env._robot.gripper.positions, [0.04, 0.04])
    assert info["target_width"] == 0.08
    assert info["clipped"] is True
    assert info["command_gripper_open"] is True
