from abc import abstractmethod
import os

import numpy as np
from enum import Enum
from typing import List, Union
from pyquaternion import Quaternion
from pyrep.const import ConfigurationPathAlgorithms as Algos, ObjectType
from pyrep.errors import ConfigurationPathError, IKError
from pyrep.objects import Object, Dummy
from scipy.spatial.transform import Rotation

from rlbench.backend.exceptions import InvalidActionError
from rlbench.backend.robot import Robot
from rlbench.backend.scene import Scene
from rlbench.const import SUPPORTED_ROBOTS


def assert_action_shape(action: np.ndarray, expected_shape: tuple):
    if np.shape(action) != expected_shape:
        raise InvalidActionError(
            'Expected the action shape to be: %s, but was shape: %s' % (
                str(expected_shape), str(np.shape(action))))


def assert_unit_quaternion(quat):
    if not np.isclose(np.linalg.norm(quat), 1.0):
        raise InvalidActionError('Action contained non unit quaternion!')


def calculate_delta_pose(robot: Robot, action: np.ndarray):
    a_x, a_y, a_z, a_qx, a_qy, a_qz, a_qw = action
    x, y, z, qx, qy, qz, qw = robot.arm.get_tip().get_pose()
    new_rot = Quaternion(
        a_qw, a_qx, a_qy, a_qz) * Quaternion(qw, qx, qy, qz)
    qw, qx, qy, qz = list(new_rot)
    pose = [a_x + x, a_y + y, a_z + z] + [qx, qy, qz, qw]
    return pose


class RelativeFrame(Enum):
    WORLD = 0
    EE = 1


class ArmActionMode(object):

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

    def set_control_mode(self, robot: Robot):
        robot.arm.set_control_loop_enabled(True)


class JointVelocity(ArmActionMode):
    """Control the joint velocities of the arm.

    Similar to the action space in many continious control OpenAI Gym envs.
    """

    def action(self, scene: Scene, action: np.ndarray):
        self.action_pre_step(scene, action)
        self.action_step(scene, action)
        self.action_post_step(scene, action)
    
    def action_pre_step(self, scene: Scene, action: np.ndarray):
        assert_action_shape(action, self.action_shape(scene))
        scene.robot.arm.set_joint_target_velocities(action)

    def action_step(self, scene: Scene, action: np.ndarray):
        scene.step()

    def action_post_step(self, scene: Scene, action: np.ndarray):
        scene.robot.arm.set_joint_target_velocities(np.zeros_like(action))

    def action_shape(self, scene: Scene) -> tuple:
        return SUPPORTED_ROBOTS[scene.robot_setup][2],

    def set_control_mode(self, robot: Robot):
        robot.arm.set_control_loop_enabled(False)
        robot.arm.set_motor_locked_at_zero_velocity(True)


class JointPosition(ArmActionMode):
    """Control the target joint positions (absolute or delta) of the arm.

    The action mode opoerates in absolute mode or delta mode, where delta
    mode takes the current joint positions and adds the new joint positions
    to get a set of target joint positions. The robot uses a simple control
    loop to execute until the desired poses have been reached.
    It os the users responsibility to ensure that the action lies within
    a usuable range.
    """

    def __init__(self, absolute_mode: bool = True):
        """
        Args:
            absolute_mode: If we should opperate in 'absolute', or 'delta' mode.
        """
        self._absolute_mode = absolute_mode

    def action(self, scene: Scene, action: np.ndarray):
        self.action_pre_step(scene, action)
        self.action_step(scene, action)
        self.action_post_step(scene, action)

    def action_pre_step(self, scene: Scene, action: np.ndarray):
        assert_action_shape(action, self.action_shape(scene))
        a = action if self._absolute_mode else np.array(
            scene.robot.arm.get_joint_positions()) + action
        scene.robot.arm.set_joint_target_positions(a)

    def action_step(self, scene: Scene, action: np.ndarray):
        scene.step()

    def action_post_step(self, scene: Scene, action: np.ndarray):
        scene.robot.arm.set_joint_target_positions(
            scene.robot.arm.get_joint_positions())

    def action_shape(self, scene: Scene) -> tuple:
        return SUPPORTED_ROBOTS[scene.robot_setup][2],


class JointTorque(ArmActionMode):
    """Control the joint torques of the arm.
    """

    TORQUE_MAX_VEL = 9999

    def _torque_action(self, robot, action):
        tml = JointTorque.TORQUE_MAX_VEL
        robot.arm.set_joint_target_velocities(
            [(tml if t < 0 else -tml) for t in action])
        robot.arm.set_joint_forces(np.abs(action))

    def action(self, scene: Scene, action: np.ndarray):
        self.action_pre_step(scene, action)
        self.action_step(scene, action)
        self.action_post_step(scene, action)

    def action_pre_step(self, scene: Scene, action: np.ndarray):
        assert_action_shape(action, self.action_shape(scene))
        self._torque_action(scene.robot, action)

    def action_step(self, scene: Scene, action: np.ndarray):
        scene.step()

    def action_post_step(self, scene: Scene, action: np.ndarray):
        self._torque_action(scene.robot, scene.robot.arm.get_joint_forces())
        scene.robot.arm.set_joint_target_velocities(np.zeros_like(action))

    def action_shape(self, scene: Scene) -> tuple:
        return SUPPORTED_ROBOTS[scene.robot_setup][2],

    def set_control_mode(self, robot: Robot):
        robot.arm.set_control_loop_enabled(False)


class EndEffectorPoseViaPlanning(ArmActionMode):
    """High-level action where target pose is given and reached via planning.

    Given a target pose, a linear path is first planned (via IK). If that fails,
    sample-based planning will be used. The decision to apply collision
    checking is a crucial trade off! With collision checking enabled, you
    are guaranteed collision free paths, but this may not be applicable for task
    that do require some collision. E.g. using this mode on pushing object will
    mean that the generated path will actively avoid not pushing the object.

    Note that path planning can be slow, often taking a few seconds in the worst
    case.

    This was the action mode used in:
    James, Stephen, and Andrew J. Davison. "Q-attention: Enabling Efficient
    Learning for Vision-based Robotic Manipulation."
    arXiv preprint arXiv:2105.14829 (2021).
    """

    def __init__(self,
                 absolute_mode: bool = True,
                 frame: RelativeFrame = RelativeFrame.WORLD,
                 collision_checking: bool = False,
                 settle_same_target_without_replanning: bool = False):
        """
        If collision check is enbled, and an object is grasped, then we

        Args:
            absolute_mode: If we should opperate in 'absolute', or 'delta' mode.
            frame: Either WORLD or EE.
            collision_checking: IF collision checking is enabled.
        """
        self._absolute_mode = absolute_mode
        self._frame = frame
        self._collision_checking = collision_checking
        self._robot_shapes = None
        self._settle_same_target_without_replanning = bool(
            settle_same_target_without_replanning
        )
        # PointACT's Mover can send the exact same EEF target again when the
        # first execution misses a millimetre-level tolerance. Re-running
        # get_path() for that retry can sample a different redundant Panda IK
        # branch and produce a large elbow/wrist jump. Cache the first path's
        # joint goal so a nearby same-target retry only lets the existing joint
        # servo settle instead of invoking IK/RRT again.
        self._last_planned_action = None
        self._last_planned_joint_target = None
        self._same_target_settle_max_start_error = float(os.environ.get(
            "RLBENCH_PLANNING_RETRY_MAX_JOINT_ERROR_RAD", "0.25"
        ))
        self._same_target_settle_joint_tolerance = float(os.environ.get(
            "RLBENCH_PLANNING_RETRY_JOINT_TOLERANCE_RAD", "0.001"
        ))
        self._same_target_settle_max_steps = int(os.environ.get(
            "RLBENCH_PLANNING_RETRY_SETTLE_STEPS", "20"
        ))
        self._same_target_settle_stall_steps = int(os.environ.get(
            "RLBENCH_PLANNING_RETRY_STALL_STEPS", "3"
        ))

    def _try_settle_same_target(self, scene: Scene, action: np.ndarray) -> bool:
        """Settle a repeated target without sampling a new IK/RRT branch."""
        if not self._settle_same_target_without_replanning:
            return False
        if (
                self._last_planned_action is None
                or self._last_planned_joint_target is None
                or not np.allclose(
                    action,
                    self._last_planned_action,
                    atol=1e-8,
                    rtol=0.0,
                )):
            return False

        arm = scene.robot.arm
        current = np.asarray(arm.get_joint_positions(), dtype=np.float64)
        joint_target = np.asarray(
            self._last_planned_joint_target, dtype=np.float64
        )
        if current.shape != joint_target.shape:
            return False
        start_error = float(np.max(np.abs(joint_target - current)))
        # Only treat this as a Mover settling retry when the previous path has
        # already brought the arm close to its cached joint goal. This prevents
        # a coincidentally identical action after an episode reset from driving
        # directly to a stale configuration without path planning.
        if start_error > self._same_target_settle_max_start_error:
            return False

        arm.set_joint_target_positions(joint_target.tolist())
        previous = current
        stalled_steps = 0
        steps = 0
        final_error = start_error
        for steps in range(1, self._same_target_settle_max_steps + 1):
            scene.step()
            success, terminate = scene.task.success()
            current = np.asarray(arm.get_joint_positions(), dtype=np.float64)
            final_error = float(np.max(np.abs(joint_target - current)))
            if success or terminate or final_error <= (
                    self._same_target_settle_joint_tolerance):
                break
            movement = float(np.max(np.abs(current - previous)))
            stalled_steps = stalled_steps + 1 if movement < 1e-5 else 0
            previous = current
            if stalled_steps >= self._same_target_settle_stall_steps:
                break

        print(
            "[planning-same-target-settle] "
            + str({
                "replanned": False,
                "start_joint_error_rad": start_error,
                "final_joint_error_rad": final_error,
                "physics_steps": steps,
                "stalled": (
                    stalled_steps >= self._same_target_settle_stall_steps
                ),
            }),
            flush=True,
        )
        return True

    def _remember_planned_target(self, arm, action, path) -> None:
        """Remember the exact joint endpoint selected for a new EEF target."""
        num_joints = int(arm.get_joint_count())
        path_points = np.asarray(
            getattr(path, "_path_points", []), dtype=np.float64
        ).reshape(-1)
        if path_points.size < num_joints:
            self._last_planned_action = None
            self._last_planned_joint_target = None
            return
        self._last_planned_action = np.asarray(action, dtype=np.float64).copy()
        self._last_planned_joint_target = path_points[-num_joints:].copy()

    def _quick_boundary_check(self, scene: Scene, action: np.ndarray):
        pos_to_check = action[:3]
        relative_to = None if self._frame == RelativeFrame.WORLD else scene.robot.arm.get_tip()
        if relative_to is not None:
            scene.target_workspace_check.set_position(pos_to_check, relative_to)
            pos_to_check = scene.target_workspace_check.get_position()
        else:
            # SmolVLA's quantized targets can land a fraction below the
            # simulator's strict z lower bound (e.g. 0.750 vs 0.7505).
            # Correct only this small numerical boundary mismatch; retain
            # hard failures for genuinely unreachable targets.
            z_tolerance = float(os.environ.get("RLBENCH_WORKSPACE_Z_TOLERANCE", "0.001"))
            min_z = float(getattr(scene, "_workspace_minz", float("-inf")))
            if min_z - z_tolerance < float(pos_to_check[2]) <= min_z:
                action[2] = min_z + max(z_tolerance * 0.1, 1e-4)
                pos_to_check = action[:3]
                if os.environ.get("RLBENCH_LOG_WORKSPACE_CLAMPS", "1") == "1":
                    print(
                        "[workspace-clamp] z target "
                        f"{float(action[2]):.6f} raised above lower bound {min_z:.6f}",
                        flush=True,
                    )
        if not scene.check_target_in_workspace(pos_to_check):
            raise InvalidActionError('A path could not be found because the '
                                     'target is outside of workspace.')

    def _pose_in_end_effector_frame(self, robot: Robot, action: np.ndarray):
        a_x, a_y, a_z, a_qx, a_qy, a_qz, a_qw = action
        x, y, z, qx, qy, qz, qw = robot.arm.get_tip().get_pose()
        new_rot = Quaternion(
            a_qw, a_qx, a_qy, a_qz) * Quaternion(qw, qx, qy, qz)
        qw, qx, qy, qz = list(new_rot)
        pose = [a_x + x, a_y + y, a_z + z] + [qx, qy, qz, qw]
        return pose

    def action(self, scene: Scene, action: np.ndarray):
        assert_action_shape(action, (7,))
        assert_unit_quaternion(action[3:])
        if not self._absolute_mode and self._frame != RelativeFrame.EE:
            action = calculate_delta_pose(scene.robot, action)
        relative_to = None if self._frame == RelativeFrame.WORLD else scene.robot.arm.get_tip()
        self._quick_boundary_check(scene, action)

        if self._try_settle_same_target(scene, action):
            return

        # This is a genuinely new target (or the arm is too far from a cached
        # goal for safe settling). Do not retain stale state if planning fails.
        self._last_planned_action = None
        self._last_planned_joint_target = None

        colliding_shapes = []
        if self._collision_checking:
            if self._robot_shapes is None:
                self._robot_shapes = scene.robot.arm.get_objects_in_tree(
                    object_type=ObjectType.SHAPE)
            # First check if we are colliding with anything
            colliding = scene.robot.arm.check_arm_collision()
            if colliding:
                # Disable collisions with the objects that we are colliding with
                grasped_objects = scene.robot.gripper.get_grasped_objects()
                colliding_shapes = [
                    s for s in scene.pyrep.get_objects_in_tree(
                        object_type=ObjectType.SHAPE) if (
                            s.is_collidable() and
                            s not in self._robot_shapes and
                            s not in grasped_objects and
                            scene.robot.arm.check_arm_collision(
                                s))]
                [s.set_collidable(False) for s in colliding_shapes]

        try:
            path = scene.robot.arm.get_path(
                action[:3],
                quaternion=action[3:],
                ignore_collisions=not self._collision_checking,
                relative_to=relative_to,
                trials=100,
                max_configs=10,
                max_time_ms=int(os.environ.get("RLBENCH_PLANNER_MAX_TIME_MS", "10")),
                trials_per_goal=5,
                algorithm=Algos.RRTConnect
            )
            self._remember_planned_target(scene.robot.arm, action, path)
            [s.set_collidable(True) for s in colliding_shapes]
        except ConfigurationPathError as e:
            [s.set_collidable(True) for s in colliding_shapes]
            raise InvalidActionError(
                'A path could not be found. Most likely due to the target '
                'being inaccessible or a collison was detected.') from e
        done = False
        while not done:
            done = path.step()
            scene.step()
            success, terminate = scene.task.success()
            # Do not continue a path after either terminal outcome. The
            # compound action mode also short-circuits its following gripper
            # action when this happens.
            if success or terminate:
                break

    def action_shape(self, scene: Scene) -> tuple:
        return 7,


class EndEffectorPoseViaLinearPlanning(EndEffectorPoseViaPlanning):
    """Prefer Cartesian-linear IK and reject excessive joint-space detours.

    A small non-linear path is allowed when Cartesian-linear IK is infeasible,
    but every candidate is checked before execution.  This differs from
    :class:`EndEffectorPoseViaPlanning`, which executes any path returned by
    RRTConnect.  The limits here keep a redundant-arm IK solution from sending
    the Panda around a large loop merely to reach a nearby EEF target.
    """

    def __init__(
            self,
            absolute_mode: bool = True,
            frame: RelativeFrame = RelativeFrame.WORLD,
            collision_checking: bool = False,
            max_joint_goal_distance: float = 2.2,
            max_joint_path_length: float = 3.5,
            max_joint_path_ratio: float = 2.0,
            max_single_joint_travel: float = 1.5):
        super().__init__(absolute_mode, frame, collision_checking)
        self._max_joint_goal_distance = float(max_joint_goal_distance)
        self._max_joint_path_length = float(max_joint_path_length)
        self._max_joint_path_ratio = float(max_joint_path_ratio)
        self._max_single_joint_travel = float(max_single_joint_travel)

    @staticmethod
    def _path_metrics(arm, path):
        num_joints = int(arm.get_joint_count())
        flat_path = np.asarray(path._path_points, dtype=np.float64).reshape(-1)
        if flat_path.size == 0 or flat_path.size % num_joints != 0:
            raise InvalidActionError(
                "Planner returned a malformed joint path; refusing unchecked execution."
            )
        configurations = flat_path.reshape(-1, num_joints)
        current = np.asarray(arm.get_joint_positions(), dtype=np.float64)
        if current.shape != (num_joints,):
            raise InvalidActionError(
                "Could not read the current joint configuration for detour checking."
            )
        if not np.allclose(configurations[0], current, atol=1e-7, rtol=0.0):
            configurations = np.concatenate((current[None], configurations), axis=0)
        differences = np.diff(configurations, axis=0)
        segment_lengths = np.linalg.norm(differences, axis=1)
        path_length = float(segment_lengths.sum())
        goal_distance = float(np.linalg.norm(configurations[-1] - current))
        # Near an unchanged target, harmless IK/interpolation noise can be
        # several times larger than an almost-zero direct distance.  A 0.05 rad
        # denominator floor still rejects a real closed loop while avoiding an
        # unstable ratio for hold-position commands.
        path_ratio = path_length / max(goal_distance, 0.05)
        per_joint_travel = np.abs(differences).sum(axis=0)
        return {
            "joint_goal_distance_rad": goal_distance,
            "joint_path_length_rad": path_length,
            "joint_path_ratio": path_ratio,
            "max_single_joint_travel_rad": float(per_joint_travel.max(initial=0.0)),
            "path_configurations": int(len(configurations)),
        }

    def _validate_path(self, arm, path, path_kind):
        metrics = self._path_metrics(arm, path)
        limits = (
            (
                "joint_goal_distance_rad",
                self._max_joint_goal_distance,
            ),
            (
                "joint_path_length_rad",
                self._max_joint_path_length,
            ),
            (
                "joint_path_ratio",
                self._max_joint_path_ratio,
            ),
            (
                "max_single_joint_travel_rad",
                self._max_single_joint_travel,
            ),
        )
        violations = [
            "%s=%.6f>%.6f" % (name, metrics[name], limit)
            for name, limit in limits
            if limit > 0.0 and metrics[name] > limit
        ]
        if violations:
            raise InvalidActionError(
                "Rejected %s path because it would make an excessive joint-space "
                "detour: %s" % (path_kind, ", ".join(violations))
            )
        if path_kind != "linear":
            print(
                "[linear-planning-bounded-fallback] "
                + str(metrics),
                flush=True,
            )
        return path

    def action(self, scene: Scene, action: np.ndarray):
        assert_action_shape(action, (7,))
        assert_unit_quaternion(action[3:])
        if not self._absolute_mode and self._frame != RelativeFrame.EE:
            action = calculate_delta_pose(scene.robot, action)
        relative_to = (
            None
            if self._frame == RelativeFrame.WORLD
            else scene.robot.arm.get_tip()
        )
        self._quick_boundary_check(scene, action)

        colliding_shapes = []
        if self._collision_checking:
            if self._robot_shapes is None:
                self._robot_shapes = scene.robot.arm.get_objects_in_tree(
                    object_type=ObjectType.SHAPE
                )
            if scene.robot.arm.check_arm_collision():
                grasped_objects = scene.robot.gripper.get_grasped_objects()
                colliding_shapes = [
                    shape
                    for shape in scene.pyrep.get_objects_in_tree(
                        object_type=ObjectType.SHAPE
                    )
                    if (
                        shape.is_collidable()
                        and shape not in self._robot_shapes
                        and shape not in grasped_objects
                        and scene.robot.arm.check_arm_collision(shape)
                    )
                ]
                for shape in colliding_shapes:
                    shape.set_collidable(False)

        arm = scene.robot.arm
        try:
            path = arm.get_linear_path(
                action[:3],
                quaternion=action[3:],
                ignore_collisions=not self._collision_checking,
                relative_to=relative_to,
            )
            path_kind = "linear"
        except ConfigurationPathError:
            try:
                path = arm.get_nonlinear_path(
                    action[:3],
                    quaternion=action[3:],
                    ignore_collisions=not self._collision_checking,
                    relative_to=relative_to,
                    trials=100,
                    max_configs=10,
                    max_time_ms=int(os.environ.get(
                        "RLBENCH_PLANNER_MAX_TIME_MS", "10"
                    )),
                    trials_per_goal=5,
                    algorithm=Algos.RRTConnect,
                )
                path_kind = "bounded_rrt"
            except ConfigurationPathError as error:
                raise InvalidActionError(
                    "Neither a Cartesian-linear path nor a bounded RRT "
                    "candidate could be found."
                ) from error
        finally:
            for shape in colliding_shapes:
                shape.set_collidable(True)

        path = self._validate_path(arm, path, path_kind)

        done = False
        while not done:
            done = path.step()
            scene.step()
            success, terminate = scene.task.success()
            if success or terminate:
                break

    def action_shape(self, scene: Scene) -> tuple:
        return 7,


class EndEffectorPoseViaIK(ArmActionMode):
    """High-level action where target pose is given and reached via IK.

    Given a target pose, IK via inverse Jacobian is performed. This requires
    the target pose to be close to the current pose, otherwise the action
    will fail. It is up to the user to constrain the action to
    meaningful values.

    The decision to apply collision checking is a crucial trade off!
    With collision checking enabled, you are guaranteed collision free paths,
    but this may not be applicable for task that do require some collision.
    E.g. using this mode on pushing object will mean that the generated
    path will actively avoid not pushing the object.
    """

    def __init__(self,
                 absolute_mode: bool = True,
                 frame: RelativeFrame = RelativeFrame.WORLD,
                 collision_checking: bool = False):
        """
        Args:
            absolute_mode: If we should opperate in 'absolute', or 'delta' mode.
            frame: Either WORLD or EE.
            collision_checking: IF collision checking is enabled.
        """
        self._absolute_mode = absolute_mode
        self._frame = frame
        self._collision_checking = collision_checking

    def action(self, scene: Scene, action: np.ndarray):
        assert_action_shape(action, (7,))
        assert_unit_quaternion(action[3:])
        if not self._absolute_mode and self._frame != RelativeFrame.EE:
            action = calculate_delta_pose(scene.robot, action)
        relative_to = None if self._frame == RelativeFrame.WORLD else scene.robot.arm.get_tip()

        try:
            joint_positions = scene.robot.arm.solve_ik_via_jacobian(
                action[:3], quaternion=action[3:], relative_to=relative_to)
            scene.robot.arm.set_joint_target_positions(joint_positions)
        except IKError as e:
            raise InvalidActionError(
                'Could not perform IK via Jacobian; most likely due to current '
                'end-effector pose being too far from the given target pose. '
                'Try limiting/bounding your action space.') from e
        done = False
        prev_values = None
        # Move until reached target joint positions or until we stop moving
        # (e.g. when we collide wth something)
        while not done:
            scene.step()
            cur_positions = scene.robot.arm.get_joint_positions()
            reached = np.allclose(cur_positions, joint_positions, atol=0.01)
            not_moving = False
            if prev_values is not None:
                not_moving = np.allclose(
                    cur_positions, prev_values, atol=0.001)
            prev_values = cur_positions
            done = reached or not_moving

    def action_shape(self, scene: Scene) -> tuple:
        return 7,

class ERJointViaIK(ArmActionMode):
    """High-level action where target EE pose + Elbow angle is given in ER 
    space (End-Effector and Elbow) and reached via IK.

    Given a target pose, IK via inverse Jacobian is performed. This requires
    the target pose to be close to the current pose, otherwise the action
    will fail. It is up to the user to constrain the action to
    meaningful values.

    The decision to apply collision checking is a crucial trade off!
    With collision checking enabled, you are guaranteed collision free paths,
    but this may not be applicable for task that do require some collision.
    E.g. using this mode on pushing object will mean that the generated
    path will actively avoid not pushing the object.
    """

    def __init__(self,
            absolute_mode: bool = True,
            frame: RelativeFrame = RelativeFrame.WORLD,
            collision_checking: bool = False,
            commanded_joints : List[int] = [0],
            eps : float = 1e-3,
            delta_angle : bool = False,
    ):
        self._absolute_mode = absolute_mode
        self._frame = frame
        self._collision_checking = collision_checking
        self._excl_j_idx = commanded_joints
        self._action_shape = (7 + len(commanded_joints),)
        self.EPS = eps
        self.delta_angle = delta_angle

    def action(self, scene: Scene, action: np.ndarray):
        """Performs action using IK.

        :param scene: CoppeliaSim scene.
        :param action: Must be in the form [ee_pose, elbow_angle] with a len of 8.
        """
        arm = scene.robot.arm
        n_excl_joints = len(self._excl_j_idx)
        assert_action_shape(action, self._action_shape)
        assert_unit_quaternion(action[3:-n_excl_joints])
        angles = action[-n_excl_joints:]
        ee_action = action[:-n_excl_joints]
        if self.delta_angle:
            assert not self._absolute_mode, 'Cannot use delta_angle_mode'
            if n_excl_joints == 1:
                angle = angles[0]
                if self._excl_j_idx[0] == 0:
                    c = arm.joints[0].get_position()[:2]
                    p = arm.get_tip().get_position()[:2]
                    a = angle

                    new_p = [c[0] + (p[0] - c[0]) * np.cos(a) - (p[1] - c[1]) * np.sin(a),
                        c[1] + (p[0] - c[0]) * np.sin(a) + (p[1] - c[1]) * np.cos(a)]

                    angle_delta_ee = new_p - p
                    ee_action[:2] += angle_delta_ee

                    rot = Rotation.from_quat(ee_action[-4:]).as_euler('xyz')
                    rot[-1] += angle
                    ee_action[-4:] = Rotation.from_euler('xyz', rot).as_quat(canonical=True)
                elif self._excl_j_idx[0] == 6:
                    # Get axis for z axis wrt the gripper
                    v = np.array([0.,0.,1.])
                    axis = Rotation.from_quat(arm.get_tip().get_quaternion()).apply(v)
                    # Get rotation given by joint
                    quat_new = Quaternion(axis=axis, angle=angle)
                    w_new, x_new, y_new, z_new = quat_new
                    quat_new = Rotation.from_quat([x_new, y_new, z_new, w_new])
                    # Add rotation to original rotation
                    rot = Rotation.from_quat(ee_action[-4:])
                    rot_new = rot * quat_new
                    ee_action[-4:] = rot_new.as_quat(canonical=True)
                else:
                    raise NotImplementedError(f'Cannot use delta_angle_mode with joint {self._excl_j_idx}')

        if not self._absolute_mode and self._frame != RelativeFrame.EE:
            ee_action = calculate_delta_pose(scene.robot, ee_action)

        if self._frame == RelativeFrame.WORLD:
            relative_to = None
        elif self._frame == RelativeFrame.EE:
            relative_to = Dummy.create()
            relative_to.set_position(arm.get_tip().get_position())
            relative_to.set_quaternion(arm.get_tip().get_quaternion())

        try:
            # Constrain joint to final position
            new_joint_pos = dict()
            for ei, excl_j in enumerate(self._excl_j_idx):
                prev_joint_pos = arm.get_joint_positions()[excl_j]
                new_joint_pos[excl_j] = angles[ei] if self._absolute_mode else prev_joint_pos + angles[ei]
                # NOTE(pmazzaglia): setting eps=0 -> i.e. we want the joint to be absolutely there, but we may set eps to a small value 
                # as well (noticed that it could lead to drifting though, cause the IK will be computed without knowing how much the 
                # the joint will be off the wanted position)
                eps = self.EPS
                orig_cyclic, orig_interval = arm.joints[excl_j].get_joint_interval()
                # Making the joint angle valid
                if new_joint_pos[excl_j] - eps < orig_interval[0]:
                    new_joint_pos[excl_j] = orig_interval[0] + eps
                if new_joint_pos[excl_j] + eps > orig_interval[0] + orig_interval[1]:
                    new_joint_pos[excl_j] = orig_interval[0] + orig_interval[1] - eps
                # Set target joint interval
                arm.joints[excl_j].set_joint_interval(orig_cyclic, [new_joint_pos[excl_j]-eps, 2 * eps])

            joint_positions = scene.robot.arm.solve_ik_via_jacobian(
                position=ee_action[:3], quaternion=ee_action[3:],
                relative_to=relative_to,
                locked_joints=self._excl_j_idx,
            )
            # Restore joint constraints
            for excl_j in self._excl_j_idx:
                arm.joints[excl_j].set_joint_interval(orig_cyclic, orig_interval)
            # Combine targets
            ik_joint_positions, joint_positions = joint_positions, []
            counter = 0
            for idx in range( len(arm.get_joint_positions() )):
                if idx in self._excl_j_idx:
                    joint_positions.append(new_joint_pos[idx])
                else:
                    joint_positions.append(ik_joint_positions[counter])
                    counter += 1
            arm.set_joint_target_positions(joint_positions)
        except IKError as e:
            # TODO: may add configuration path for failure, to reduce number of failures
            # though this may slowdown things significantly
            # Restoring joint constraints if there was an error (also restoring to prev_pos first to avoid internal accumulating error)
            for excl_j in self._excl_j_idx:
                arm.joints[excl_j].set_joint_interval(orig_cyclic, [prev_joint_pos, 2 * eps])
                arm.joints[excl_j].set_joint_interval(orig_cyclic, orig_interval)
            raise InvalidActionError(
                'Could not perform IK via Jacobian; most likely due to current '
                'end-effector pose being too far from the given target pose. '
                'Try limiting/bounding your action space.') from e

        done = False
        prev_values = None
        max_steps = 50
        steps = 0

        # Move until reached target joint positions or until we stop moving
        # (e.g. when we collide wth something)

        while not done and steps < max_steps:
            scene.step()
            cur_positions = arm.get_joint_positions()
            reached = np.allclose(cur_positions, joint_positions, atol=1e-3)
            not_moving = False
            if prev_values is not None:
                not_moving = np.allclose(
                    cur_positions, prev_values, atol=1e-3)
            prev_values = cur_positions
            done = reached or not_moving
            steps += 1

    def action_shape(self, _: Scene) -> tuple:
        return self._action_shape
