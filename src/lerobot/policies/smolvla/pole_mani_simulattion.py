import argparse
from dataclasses import dataclass

import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R


np.set_printoptions(precision=5, suppress=True)


@dataclass
class PoleScenario:
    pole_length: float
    pole_radius: float
    pole_initial: np.ndarray
    hand_tail_pose: np.ndarray
    hand_head_pose: np.ndarray
    hand_tail_delta: np.ndarray


def make_transform(translation, rotation_euler_xyz=(0.0, 0.0, 0.0), degrees=True):
    transform = np.eye(4)
    transform[:3, :3] = R.from_euler("xyz", rotation_euler_xyz, degrees=degrees).as_matrix()
    transform[:3, 3] = np.asarray(translation, dtype=float)
    return transform


def inverse_transform(transform):
    inverse = np.eye(4)
    inverse[:3, :3] = transform[:3, :3].T
    inverse[:3, 3] = -inverse[:3, :3] @ transform[:3, 3]
    return inverse


def apply_hand_frame_delta(object_world_pose, hand_world_pose, hand_frame_delta):
    return hand_world_pose @ hand_frame_delta @ inverse_transform(hand_world_pose) @ object_world_pose


def solve_head_grasp_delta(hand_tail_pose, hand_head_pose, hand_tail_delta):
    return (
        inverse_transform(hand_head_pose)
        @ hand_tail_pose
        @ hand_tail_delta
        @ inverse_transform(hand_tail_pose)
        @ hand_head_pose
    )


def build_scenario():
    pole_length = 0.6
    pole_radius = 0.018

    # Pole local frame l0: origin at tail A, x-axis points from A to B.
    pole_initial = make_transform(
        translation=(0.25, -0.10, 0.35),
        rotation_euler_xyz=(0.0, 0.0, 25.0),
    )

    # Hand h1 grasps tail A. Its origin is placed at A with a different orientation.
    hand_tail_pose = pole_initial @ make_transform(
        translation=(0.0, 0.0, 0.0),
        rotation_euler_xyz=(0.0, 90.0, 0.0),
    )

    # Hand h2 grasps head B. Its origin is placed at B and faces the opposite end.
    hand_head_pose = pole_initial @ make_transform(
        translation=(pole_length, 0.0, 0.0),
        rotation_euler_xyz=(0.0, -90.0, 180.0),
    )

    # Desired motion when grasping A, expressed in h1 coordinates.
    hand_tail_delta = make_transform(
        translation=(0.08, -0.04, 0.10),
        rotation_euler_xyz=(18.0, -12.0, 35.0),
    )

    return PoleScenario(
        pole_length=pole_length,
        pole_radius=pole_radius,
        pole_initial=pole_initial,
        hand_tail_pose=hand_tail_pose,
        hand_head_pose=hand_head_pose,
        hand_tail_delta=hand_tail_delta,
    )


def sample_pole_points(length, radius, axial_count=90, radial_count=18):
    xs = np.linspace(0.0, length, axial_count)
    angles = np.linspace(0.0, 2.0 * np.pi, radial_count, endpoint=False)
    points = []
    for x in xs:
        for angle in angles:
            points.append([x, radius * np.cos(angle), radius * np.sin(angle)])
    return np.asarray(points)


def transform_points(transform, points):
    points_h = np.hstack([points, np.ones((points.shape[0], 1))])
    return (transform @ points_h.T).T[:, :3]


def make_point_cloud(points, color):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(np.tile(color, (points.shape[0], 1)))
    return pcd


def make_frame(transform, size):
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)
    frame.transform(transform)
    return frame


def make_line_set(points, lines, colors):
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(np.asarray(points))
    line_set.lines = o3d.utility.Vector2iVector(np.asarray(lines, dtype=np.int32))
    line_set.colors = o3d.utility.Vector3dVector(np.asarray(colors))
    return line_set


def build_pole_point_clouds(scenario, pole_final_from_tail, pole_final_from_head):
    pole_local_points = sample_pole_points(scenario.pole_length, scenario.pole_radius)
    initial_points = transform_points(scenario.pole_initial, pole_local_points)
    tail_result_points = transform_points(pole_final_from_tail, pole_local_points)
    head_result_points = transform_points(pole_final_from_head, pole_local_points)

    return {
        "initial": make_point_cloud(initial_points, color=(0.72, 0.72, 0.72)),
        "tail_result": make_point_cloud(tail_result_points, color=(0.05, 0.45, 1.0)),
        "head_result": make_point_cloud(head_result_points, color=(1.0, 0.25, 0.15)),
    }


def build_motion_lines(scenario, pole_final_from_tail):
    tail_start = scenario.pole_initial[:3, 3]
    head_start = (scenario.pole_initial @ np.array([scenario.pole_length, 0.0, 0.0, 1.0]))[:3]
    tail_end = pole_final_from_tail[:3, 3]
    head_end = (pole_final_from_tail @ np.array([scenario.pole_length, 0.0, 0.0, 1.0]))[:3]

    return make_line_set(
        points=[tail_start, head_start, tail_end, head_end],
        lines=[[0, 1], [2, 3], [0, 2], [1, 3]],
        colors=[
            [0.45, 0.45, 0.45],
            [0.10, 0.55, 1.00],
            [0.10, 0.80, 0.30],
            [0.10, 0.80, 0.30],
        ],
    )


def build_open3d_scene(scenario, pole_final_from_tail, pole_final_from_head):
    pole_clouds = build_pole_point_clouds(scenario, pole_final_from_tail, pole_final_from_head)

    return [
        pole_clouds["initial"],
        pole_clouds["tail_result"],
        pole_clouds["head_result"],
        make_frame(np.eye(4), size=0.18),
        make_frame(scenario.pole_initial, size=0.12),
        make_frame(scenario.hand_tail_pose, size=0.10),
        make_frame(scenario.hand_head_pose, size=0.10),
        make_frame(pole_final_from_tail, size=0.14),
        build_motion_lines(scenario, pole_final_from_tail),
    ]


def visualize_scene(geometries, window_name):
    o3d.visualization.draw_geometries(
        geometries,
        window_name=window_name,
        width=1280,
        height=800,
    )


def visualize_step_by_step(scenario, pole_final_from_tail, pole_final_from_head):
    pole_clouds = build_pole_point_clouds(scenario, pole_final_from_tail, pole_final_from_head)

    common_frames = [
        make_frame(np.eye(4), size=0.18),
        make_frame(scenario.pole_initial, size=0.12),
    ]

    visualize_scene(
        [
            pole_clouds["initial"],
            *common_frames,
        ],
        window_name="1. Original pole point cloud",
    )

    visualize_scene(
        [
            pole_clouds["initial"],
            pole_clouds["tail_result"],
            *common_frames,
            make_frame(scenario.hand_tail_pose, size=0.10),
            make_frame(pole_final_from_tail, size=0.14),
            build_motion_lines(scenario, pole_final_from_tail),
        ],
        window_name="2. Case 1: original + tail grasp T1 result",
    )

    visualize_scene(
        [
            pole_clouds["initial"],
            pole_clouds["head_result"],
            *common_frames,
            make_frame(scenario.hand_head_pose, size=0.10),
            make_frame(pole_final_from_head, size=0.14),
            build_motion_lines(scenario, pole_final_from_head),
        ],
        window_name="3. Case 2: original + head grasp solved T2 result",
    )

    visualize_scene(
        [
            pole_clouds["initial"],
            pole_clouds["tail_result"],
            pole_clouds["head_result"],
            *common_frames,
            make_frame(pole_final_from_tail, size=0.14),
            build_motion_lines(scenario, pole_final_from_tail),
        ],
        window_name="4. Original + case 1 result + case 2 result",
    )


def main():
    parser = argparse.ArgumentParser(
        description="Verify equivalent pole manipulation from tail A and head B grasps."
    )
    parser.add_argument("--no-vis", action="store_true", help="Run numeric verification only.")
    args = parser.parse_args()

    scenario = build_scenario()

    h1 = apply_hand_frame_delta(
        scenario.pole_initial,
        scenario.hand_tail_pose,
        scenario.hand_tail_delta,
    )
    hand_head_delta = solve_head_grasp_delta(
        scenario.hand_tail_pose,
        scenario.hand_head_pose,
        scenario.hand_tail_delta,
    )
    h2 = apply_hand_frame_delta(
        scenario.pole_initial,
        scenario.hand_head_pose,
        hand_head_delta,
    )

    print("Initial pole pose ^wP_l0:")
    print(scenario.pole_initial)
    print("\nTail-grasp hand pose ^wP_h1:")
    print(scenario.hand_tail_pose)
    print("\nHead-grasp hand pose ^wP_h2:")
    print(scenario.hand_head_pose)
    print("\nTail-grasp delta ^h1T1:")
    print(scenario.hand_tail_delta)
    print("\nSolved head-grasp delta ^h2T2 = inv(P_h2) @ P_h1 @ T1 @ inv(P_h1) @ P_h2:")
    print(hand_head_delta)
    print("\nFinal pole pose from case 1 H1:")
    print(h1)
    print("\nFinal pole pose from case 2 H2:")
    print(h2)
    print("\nmax(|H1 - H2|) =", np.max(np.abs(h1 - h2)))

    if not np.allclose(h1, h2, atol=1e-9):
        raise RuntimeError("Case 1 and case 2 did not produce the same final pole pose.")

    if not args.no_vis:
        visualize_step_by_step(scenario, h1, h2)


if __name__ == "__main__":
    main()
