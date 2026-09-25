import numpy as np

from rlbench.action_modes.gripper_action_modes import Discrete


class _AlreadyOpenAttachedGripper:
    def __init__(self):
        self.release_calls = 0
        self.actuate_calls = 0

    def get_open_amount(self):
        return [1.0, 1.0]

    def release(self):
        self.release_calls += 1

    def actuate(self, _action, velocity):
        self.actuate_calls += 1
        return True


class _Robot:
    def __init__(self):
        self.gripper = _AlreadyOpenAttachedGripper()


class _Scene:
    def __init__(self):
        self.robot = _Robot()


class _OpeningGripper:
    def __init__(self, events):
        self._events = events
        self._open = False

    def get_open_amount(self):
        return [0.0, 0.0]

    def release(self):
        self._events.append("release")

    def actuate(self, _action, velocity):
        self._events.append("actuate")
        if self._open:
            return True
        self._open = True
        return False


class _OpeningTask:
    def __init__(self, events):
        self._events = events

    def step(self):
        self._events.append("task_step")

    def success(self):
        return False, False


class _OpeningPyRep:
    def __init__(self, events):
        self._events = events

    def step(self):
        self._events.append("sim_step")


class _OpeningScene:
    def __init__(self):
        self.events = []
        self.robot = type("Robot", (), {})()
        self.robot.gripper = _OpeningGripper(self.events)
        self.task = _OpeningTask(self.events)
        self.pyrep = _OpeningPyRep(self.events)


class _AsymmetricContactGripper:
    def __init__(self, events):
        self._events = events

    def get_open_amount(self):
        # One finger has been displaced by light object contact, while the
        # combined physical aperture is still more than 90% open.
        return [1.0, 0.85]

    def get_grasped_objects(self):
        return []

    def grasp(self, _obj):
        self._events.append("grasp")

    def release(self):
        self._events.append("release")

    def actuate(self, action, velocity):
        self._events.append(("actuate", float(action), float(velocity)))
        return True


class _AsymmetricContactTask(_OpeningTask):
    def get_physical_graspable_objects(self):
        return []

    def get_graspable_objects(self):
        return []


class _AsymmetricContactScene:
    def __init__(self):
        self.events = []
        self.robot = type("Robot", (), {})()
        self.robot.gripper = _AsymmetricContactGripper(self.events)
        self.task = _AsymmetricContactTask(self.events)
        self.pyrep = _OpeningPyRep(self.events)


def test_discrete_open_releases_fake_grasp_when_joints_are_already_open():
    scene = _Scene()

    Discrete().action(scene, np.asarray([1.0], dtype=np.float32))

    assert scene.robot.gripper.release_calls == 1
    assert scene.robot.gripper.actuate_calls == 0


def test_discrete_open_detaches_after_fingers_open_by_default():
    scene = _OpeningScene()

    Discrete().action(scene, np.asarray([1.0], dtype=np.float32))

    assert scene.events.count("actuate") == 2
    assert scene.events.index("release") > max(
        index for index, event in enumerate(scene.events)
        if event == "actuate")


def test_discrete_can_preserve_legacy_detach_before_open_order():
    scene = _OpeningScene()

    Discrete(detach_before_open=True).action(
        scene, np.asarray([1.0], dtype=np.float32))

    assert scene.events.index("release") < scene.events.index("actuate")


def test_discrete_close_uses_combined_two_finger_aperture():
    scene = _AsymmetricContactScene()

    Discrete().action(scene, np.asarray([0.0], dtype=np.float32))

    assert ("actuate", 0.0, 0.2) in scene.events
