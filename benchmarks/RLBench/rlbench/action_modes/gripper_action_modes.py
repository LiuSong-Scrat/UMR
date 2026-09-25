from abc import abstractmethod

import numpy as np

from rlbench.backend.exceptions import InvalidActionError
from rlbench.backend.scene import Scene


def assert_action_shape(action: np.ndarray, expected_shape: tuple):
    if np.shape(action) != expected_shape:
        raise InvalidActionError(
            'Expected the action shape to be: %s, but was shape: %s' % (
                str(expected_shape), str(np.shape(action))))


class GripperActionMode(object):

    @abstractmethod
    def action(self, scene: Scene, action: np.ndarray):
        pass

    def action_step(self, scene: Scene, action: np.ndarray):
        pass

    def action_pre_step(self, scene: Scene, action: np.ndarray):
        pass

    def action_post_step(self, scene: Scene, action: np.ndarray):
        pass

    @abstractmethod
    def action_shape(self, scene: Scene):
        pass

    @abstractmethod
    def action_bounds(self):
        pass


class Discrete(GripperActionMode):
    """Control if the gripper is open or closed in a discrete manner.

    Action values > 0.5 will be discretised to 1 (open), and values < 0.5
    will be  discretised to 0 (closed).
    """

    def __init__(self, attach_grasped_objects: bool = True,
                 detach_before_open: bool = False):
        self._attach_grasped_objects = attach_grasped_objects
        self._detach_before_open = detach_before_open

    def _actuate(self, action, scene):
        done = False
        while not done:
            done = scene.robot.gripper.actuate(action, velocity=0.2)
            scene.pyrep.step()
            scene.task.step()
            success, terminate = scene.task.success()
            if success or terminate:
                return True
        return False

    def action(self, scene: Scene, action: np.ndarray):
        assert_action_shape(action, self.action_shape(scene.robot))
        if 0.0 > action[0] > 1.0:
            raise InvalidActionError(
                'Gripper action expected to be within 0 and 1.')
        # Use the joint pair as one physical aperture.  Looking at each finger
        # as an independent binary switch makes a small contact displacement
        # on either side look like an already-closed gripper, which can cause
        # a real CLOSE transition to be skipped while the other finger is
        # still fully open.
        finger_open_amounts = np.asarray(
            scene.robot.gripper.get_open_amount(), dtype=np.float64)
        if finger_open_amounts.size == 0:
            raise InvalidActionError(
                'Gripper returned no finger opening amounts.')
        open_condition = float(np.mean(finger_open_amounts)) > 0.9
        current_ee = 1.0 if open_condition else 0.0
        action = float(action[0] > 0.5)

        # A fake grasp is represented by parenting the grasped object to the
        # gripper attach point.  Mechanical opening and attachment state can
        # become inconsistent (for example, the fingers can already report
        # > 0.9 open while an object is still parented to the gripper).  In
        # that state the old current_ee != action guard skipped release(), so
        # repeated OPEN commands could leave the object attached forever.
        # release() is idempotent, therefore an OPEN command must always clear
        # any fake-grasp attachment even when no joint actuation is required.
        if action == 1.0 and current_ee == action:
            scene.robot.gripper.release()
            return

        if current_ee != action:
            if action == 0.0 and self._attach_grasped_objects:
                # If gripper close action, the check for grasp.
                physical_objects = scene.task.get_physical_graspable_objects()
                for g_obj in scene.task.get_graspable_objects():
                    if g_obj not in physical_objects:
                        scene.robot.gripper.grasp(g_obj)

            # By default, keep a fake-grasped object attached while the
            # fingers open, then detach it.  Detaching first can expose an
            # object that is still intersecting the fingers or nearby scene
            # geometry to a large contact-resolution impulse.
            if action == 1.0 and self._detach_before_open:
                scene.robot.gripper.release()

            task_finished = self._actuate(action, scene)

            if action == 1.0 and not self._detach_before_open:
                scene.robot.gripper.release()

            if task_finished:
                return
            if action == 1.0:
                # Step a few more times to allow objects to drop
                for _ in range(10):
                    scene.pyrep.step()
                    scene.task.step()
                    success, terminate = scene.task.success()
                    if success or terminate:
                        break

    def action_shape(self, scene: Scene) -> tuple:
        return 1,

    def action_bounds(self):
        """Get the action bounds.

        Returns: Returns the min and max of the action.
        """
        return np.array([0]), np.array([1])


class GripperJointPosition(GripperActionMode):
    """Control the target joint positions absolute or delta) of the gripper.

    The action mode opoerates in absolute mode or delta mode, where delta
    mode takes the current joint positions and adds the new joint positions
    to get a set of target joint positions. The robot uses a simple control
    loop to execute until the desired poses have been reached.
    It os the users responsibility to ensure that the action lies within
    a usuable range.

    Control if the gripper is open or closed in a discrete manner.

    Action values > 0.5 will be discretised to 1 (open), and values < 0.5
    will be  discretised to 0 (closed).
    """

    def __init__(self, attach_grasped_objects: bool = True,
                 detach_before_open: bool = True,
                 absolute_mode: bool = True):
        self._attach_grasped_objects = attach_grasped_objects
        self._detach_before_open = detach_before_open
        self._absolute_mode = absolute_mode
        self._control_mode_set = False

    def action(self, scene: Scene, action: np.ndarray):
        self.action_pre_step(scene, action)
        self.action_step(scene, action)
        self.action_post_step(scene, action)

    def action_pre_step(self, scene: Scene, action: np.ndarray):
        if not self._control_mode_set:
            scene.robot.gripper.set_control_loop_enabled(True)
            self._control_mode_set = True
        assert_action_shape(action, self.action_shape(scene.robot))
        action = action.repeat(2)  # use same action for both joints
        a = action if self._absolute_mode else np.array(
            scene.robot.gripper.get_joint_positions())
        scene.robot.gripper.set_joint_target_positions(a)

    def action_step(self, scene: Scene, action: np.ndarray):
        scene.step()

    def action_post_step(self, scene: Scene, action: np.ndarray):
        scene.robot.gripper.set_joint_target_positions(
            scene.robot.gripper.get_joint_positions())

    def action_shape(self, scene: Scene) -> tuple:
        return 1,

    def action_bounds(self):
        """Get the action bounds.

        Returns: Returns the min and max of the action.
        """
        return np.array([0]), np.array([0.04])
