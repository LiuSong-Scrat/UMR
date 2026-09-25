import numpy as np

from rlbench.action_modes.arm_action_modes import EndEffectorPoseViaPlanning


class _Path:
    def __init__(self, arm, joint_target):
        self._arm = arm
        self._path_points = np.concatenate((arm.current, joint_target))
        self._joint_target = np.asarray(joint_target, dtype=np.float64)
        self._done = False

    def step(self):
        self._arm.target = self._joint_target.copy()
        self._arm.current += 0.9 * (self._arm.target - self._arm.current)
        if self._done:
            return True
        self._done = True
        return False


class _Arm:
    def __init__(self):
        self.current = np.zeros(7, dtype=np.float64)
        self.target = self.current.copy()
        self.get_path_calls = 0

    def get_tip(self):
        return None

    def get_joint_count(self):
        return 7

    def get_joint_positions(self):
        return self.current.tolist()

    def set_joint_target_positions(self, target):
        self.target = np.asarray(target, dtype=np.float64)

    def get_path(self, *_args, **_kwargs):
        self.get_path_calls += 1
        return _Path(self, np.full(7, 0.1, dtype=np.float64))


class _Gripper:
    def get_grasped_objects(self):
        return []


class _Robot:
    def __init__(self, arm):
        self.arm = arm
        self.gripper = _Gripper()


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
        self.robot.arm.current += 0.5 * (
            self.robot.arm.target - self.robot.arm.current
        )


def test_repeated_planning_target_settles_cached_joint_goal_without_replanning():
    arm = _Arm()
    scene = _Scene(arm)
    mode = EndEffectorPoseViaPlanning(
        collision_checking=False,
        settle_same_target_without_replanning=True,
    )
    target = np.array([0.4, 0.0, 0.9, 0.0, 0.0, 0.0, 1.0])

    mode.action(scene, target)
    calls_after_first_execution = arm.get_path_calls
    error_after_first_execution = np.max(np.abs(arm.target - arm.current))

    mode.action(scene, target.copy())

    assert calls_after_first_execution == 1
    assert arm.get_path_calls == 1
    assert np.max(np.abs(arm.target - arm.current)) < error_after_first_execution


def test_same_target_is_replanned_when_arm_is_far_from_cached_joint_goal():
    arm = _Arm()
    scene = _Scene(arm)
    mode = EndEffectorPoseViaPlanning(
        collision_checking=False,
        settle_same_target_without_replanning=True,
    )
    target = np.array([0.4, 0.0, 0.9, 0.0, 0.0, 0.0, 1.0])

    mode.action(scene, target)
    arm.current = np.full(7, -1.0, dtype=np.float64)
    mode.action(scene, target.copy())

    assert arm.get_path_calls == 2


def test_same_target_settling_is_opt_in_for_non_song_callers():
    arm = _Arm()
    scene = _Scene(arm)
    mode = EndEffectorPoseViaPlanning(collision_checking=False)
    target = np.array([0.4, 0.0, 0.9, 0.0, 0.0, 0.0, 1.0])

    mode.action(scene, target)
    mode.action(scene, target.copy())

    assert arm.get_path_calls == 2
