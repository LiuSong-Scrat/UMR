import numpy as np
import pytest

from pyrep.errors import ConfigurationPathError

from rlbench.action_modes.arm_action_modes import (
    EndEffectorPoseViaLinearPlanning,
)
from rlbench.backend.exceptions import InvalidActionError


class _Path:
    def __init__(self, configurations=None):
        if configurations is None:
            configurations = [np.zeros(7), np.full(7, 0.01)]
        self._path_points = np.asarray(configurations, dtype=np.float64).reshape(-1)
        self.steps = 0

    def step(self):
        self.steps += 1
        return self.steps >= 2


class _Arm:
    def __init__(self, fail_linear=False, nonlinear_configurations=None):
        self.fail_linear = fail_linear
        self.linear_calls = []
        self.nonlinear_calls = []
        self.path = _Path()
        self.nonlinear_path = _Path(nonlinear_configurations)

    def get_joint_count(self):
        return 7

    def get_joint_positions(self):
        return np.zeros(7)

    def get_linear_path(self, *args, **kwargs):
        self.linear_calls.append((args, kwargs))
        if self.fail_linear:
            raise ConfigurationPathError("linear IK failed")
        return self.path

    def get_nonlinear_path(self, *args, **kwargs):
        self.nonlinear_calls.append((args, kwargs))
        return self.nonlinear_path


class _Robot:
    def __init__(self, arm):
        self.arm = arm


class _Task:
    def success(self):
        return False, False


class _Scene:
    def __init__(self, arm):
        self.robot = _Robot(arm)
        self.task = _Task()
        self.steps = 0

    def check_target_in_workspace(self, _position):
        return True

    def step(self):
        self.steps += 1


def _target():
    return np.array([0.4, 0.0, 0.9, 0.0, 0.0, 0.0, 1.0])


def test_linear_planning_executes_only_linear_path():
    arm = _Arm()
    scene = _Scene(arm)

    EndEffectorPoseViaLinearPlanning(collision_checking=False).action(
        scene, _target()
    )

    assert len(arm.linear_calls) == 1
    assert len(arm.nonlinear_calls) == 0
    assert scene.steps == 2
    _, kwargs = arm.linear_calls[0]
    assert kwargs["ignore_collisions"] is True


def test_linear_planning_accepts_short_bounded_fallback():
    arm = _Arm(
        fail_linear=True,
        nonlinear_configurations=[
            np.zeros(7),
            np.array([0.10, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            np.array([0.20, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        ],
    )
    scene = _Scene(arm)

    EndEffectorPoseViaLinearPlanning(collision_checking=False).action(
        scene, _target()
    )

    assert len(arm.linear_calls) == 1
    assert len(arm.nonlinear_calls) == 1
    assert scene.steps == 2


def test_linear_planning_rejects_large_joint_detour():
    arm = _Arm(
        fail_linear=True,
        nonlinear_configurations=[
            np.zeros(7),
            np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            np.array([-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            np.array([0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        ],
    )
    scene = _Scene(arm)

    with pytest.raises(InvalidActionError, match="excessive joint-space detour"):
        EndEffectorPoseViaLinearPlanning(collision_checking=False).action(
            scene, _target()
        )

    assert len(arm.linear_calls) == 1
    assert len(arm.nonlinear_calls) == 1
    assert scene.steps == 0
