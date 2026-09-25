from __future__ import annotations

import json
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R

from benchmarks.song_real_libero.scripts.real_setting.build_humanhand_hdf5_dataset import (
    CameraIntrinsics,
    attach_external_camera_poses,
    canonicalize_gripper_pose_sequence,
    distort_normalized_points,
    filter_gripper_pose_sequence,
    normalized_pixel_rays,
)
from benchmarks.song_real_libero.scripts.real_setting.camera_motion_utils import (
    camera_motion_metrics,
    camera_pose_values_to_matrices,
    camera_to_model_world_transforms,
    invert_transform_sequence,
    matrix_to_pose9,
    pose9_to_matrix,
    stationary_pose_report,
    transform_points,
    transform_pose9_sequence,
)
from benchmarks.song_real_libero.scripts.real_setting.real_hdf5_to_dataset import (
    convert_hdf5_episode,
    load_camera_motion,
    write_dataset_sidecar_meta,
)
from benchmarks.song_real_libero.scripts.real_setting.record_bestman_rgbd import (
    RGBDOdometryDebugger,
    _camera_axes_lines,
    _camera_candidates,
    _external_camera_trajectory,
    _resolve_aligned_playback_poses,
    _static_camera_trajectory,
    _write_camera_trajectory,
)


def _translation(x: float, y: float = 0.0, z: float = 0.0) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, 3] = [x, y, z]
    return transform


def _camera_cfg(**overrides) -> dict:
    cfg = {
        "camera": "overhead",
        "camera_motion_compensation": "auto",
        "camera_pose_key": None,
        "camera_pose_format": "auto",
        "camera_pose_direction": "auto",
        "camera_pose_translation_scale": None,
        "camera_reference_mode": "auto",
        "canonical_camera_to_tracking_matrix": None,
    }
    cfg.update(overrides)
    return cfg


def test_hdf5_builder_brown_conrady_pixels_roundtrip_to_camera_rays() -> None:
    intrinsics = CameraIntrinsics(
        width=1280,
        height=720,
        fx=902.048828125,
        fy=902.8289794921875,
        ppx=645.163330078125,
        ppy=364.6947326660156,
        coeffs=(0.1653282642, -0.5026872158, -0.0019255957, -0.0003119746, 0.4571717680),
        model="distortion.brown_conrady",
    )
    pixels = np.asarray([[20.0, 30.0], [645.0, 365.0], [1200.0, 680.0]])
    rays = normalized_pixel_rays(pixels, intrinsics)
    distorted = distort_normalized_points(rays, intrinsics)
    reconstructed = np.column_stack(
        [
            distorted[:, 0] * intrinsics.fx + intrinsics.ppx,
            distorted[:, 1] * intrinsics.fy + intrinsics.ppy,
        ]
    )
    np.testing.assert_allclose(reconstructed, pixels, atol=2e-5)


def test_hdf5_builder_canonicalizes_rotation_representation_without_changing_pose() -> None:
    physical_euler = np.asarray(
        [[3.12, 0.1, -0.2], [-3.13, 0.1, -0.2], [-3.10, 0.1, -0.2]],
        dtype=np.float64,
    )
    quaternions = R.from_euler("zyx", physical_euler).as_quat()
    quaternions[1] *= -1.0
    poses = np.column_stack([np.zeros((3, 3)), physical_euler])

    continuous_pose, continuous_quaternion = canonicalize_gripper_pose_sequence(
        poses,
        quaternions,
    )

    assert np.all(np.sum(continuous_quaternion[:-1] * continuous_quaternion[1:], axis=1) > 0.0)
    assert np.max(np.abs(np.diff(continuous_pose[:, 3], axis=0))) < 0.1
    np.testing.assert_allclose(
        R.from_quat(continuous_quaternion).as_matrix(),
        R.from_euler("zyx", physical_euler).as_matrix(),
        atol=1e-10,
    )


def test_hdf5_builder_zero_phase_se3_filter_reduces_high_frequency_pose_noise() -> None:
    frame_count = 121
    sampling_hz = 30.0
    time_s = np.arange(frame_count, dtype=np.float64) / sampling_hz
    slow_position = np.column_stack(
        [
            0.12 * time_s / time_s[-1],
            0.03 * np.sin(2.0 * np.pi * 0.35 * time_s),
            0.55 + 0.02 * np.cos(2.0 * np.pi * 0.25 * time_s),
        ]
    )
    alternating = np.where(np.arange(frame_count) % 2 == 0, 1.0, -1.0)
    noisy_position = slow_position.copy()
    noisy_position[:, 0] += 0.004 * alternating
    noisy_position[:, 2] += 0.003 * alternating

    slow_angle = 0.45 * time_s / time_s[-1]
    noisy_angle = slow_angle + np.deg2rad(3.0) * alternating
    rotations = R.from_euler("z", noisy_angle)
    poses = np.column_stack([noisy_position, rotations.as_euler("zyx")])

    filtered_pose, filtered_quat, metrics = filter_gripper_pose_sequence(
        poses,
        rotations.as_quat(),
        time_s * 1000.0,
        cutoff_hz=4.0,
        order=3,
        parallel_jaw_symmetry=False,
        preserve_endpoints=True,
    )

    noisy_step = np.linalg.norm(np.diff(noisy_position, axis=0), axis=1)
    filtered_step = np.linalg.norm(np.diff(filtered_pose[:, :3], axis=0), axis=1)
    assert np.percentile(filtered_step, 95) < 0.6 * np.percentile(noisy_step, 95)
    assert metrics["translation_path_after_m"] < metrics["translation_path_before_m"]
    assert metrics["rotation_path_after_deg"] < metrics["rotation_path_before_deg"]
    np.testing.assert_allclose(filtered_pose[[0, -1], :3], noisy_position[[0, -1]], atol=1e-10)
    np.testing.assert_allclose(
        R.from_quat(filtered_quat[[0, -1]]).as_matrix(),
        rotations[[0, -1]].as_matrix(),
        atol=1e-10,
    )
    np.testing.assert_allclose(np.linalg.norm(filtered_quat, axis=1), 1.0, atol=1e-10)


def test_hdf5_builder_filter_removes_parallel_jaw_basis_flip_and_repairs_timestamp() -> None:
    frame_count = 90
    time_s = np.arange(frame_count, dtype=np.float64) / 30.0
    rotations = R.from_euler(
        "zyx",
        np.column_stack(
            [
                np.linspace(0.0, 0.5, frame_count),
                0.08 * np.sin(time_s),
                0.04 * np.cos(time_s),
            ]
        ),
    )
    raw_matrices = rotations.as_matrix()
    raw_matrices[25:65] = raw_matrices[25:65] @ np.diag([-1.0, -1.0, 1.0])
    raw_rotations = R.from_matrix(raw_matrices)
    positions = np.column_stack(
        [0.03 * time_s, np.zeros(frame_count), np.full(frame_count, 0.5)]
    )
    poses = np.column_stack([positions, raw_rotations.as_euler("zyx")])
    timestamps_ms = time_s * 1000.0
    timestamps_ms[50] = timestamps_ms[49]

    filtered_pose, filtered_quat, metrics = filter_gripper_pose_sequence(
        poses,
        raw_rotations.as_quat(),
        timestamps_ms,
        cutoff_hz=4.0,
        order=3,
        parallel_jaw_symmetry=True,
        preserve_endpoints=True,
    )

    filtered_rotations = R.from_quat(filtered_quat)
    filtered_steps_deg = np.rad2deg(
        (filtered_rotations[:-1].inv() * filtered_rotations[1:]).magnitude()
    )
    assert filtered_steps_deg.max() < 2.0
    assert metrics["parallel_jaw_corrected_frame_count"] == 40
    assert metrics["parallel_jaw_state_switch_count"] == 2
    assert metrics["timestamp_repair_count"] == 1
    assert np.isfinite(filtered_pose).all()


def test_hdf5_builder_filter_stabilizes_persistent_orientation_frame_reset() -> None:
    frame_count = 100
    time_s = np.arange(frame_count, dtype=np.float64) / 30.0
    physical_rotations = R.from_euler(
        "zyx",
        np.column_stack(
            [
                0.35 * time_s / time_s[-1],
                0.06 * np.sin(time_s),
                0.03 * np.cos(time_s),
            ]
        ),
    )
    reset = R.from_euler("xyz", [70.0, -35.0, 20.0], degrees=True).as_matrix()
    observed_matrices = physical_rotations.as_matrix()
    observed_matrices[45:] = reset @ observed_matrices[45:]
    observed_rotations = R.from_matrix(observed_matrices)
    positions = np.column_stack(
        [0.04 * time_s, np.zeros(frame_count), np.full(frame_count, 0.5)]
    )
    poses = np.column_stack([positions, observed_rotations.as_euler("zyx")])

    _filtered_pose, filtered_quat, metrics = filter_gripper_pose_sequence(
        poses,
        observed_rotations.as_quat(),
        time_s * 1000.0,
        cutoff_hz=4.0,
        order=3,
        max_angular_speed_deg_s=900.0,
        parallel_jaw_symmetry=False,
        preserve_endpoints=True,
    )

    filtered_rotations = R.from_quat(filtered_quat)
    filtered_steps_deg = np.rad2deg(
        (filtered_rotations[:-1].inv() * filtered_rotations[1:]).magnitude()
    )
    assert metrics["orientation_frame_reset_count"] == 1
    assert metrics["largest_rejected_angular_speed_deg_s"] > 900.0
    assert filtered_steps_deg.max() < 1.0


def test_overview_and_hand_camera_requests_support_l515_and_d435i() -> None:
    camera_cfg = SimpleNamespace(
        L515=SimpleNamespace(dev_name="L515"),
        D435I=SimpleNamespace(dev_name="D435I"),
    )
    assert [name for name, _cfg in _camera_candidates(camera_cfg, "overhead")] == [
        "L515",
        "D435I",
    ]
    assert [name for name, _cfg in _camera_candidates(camera_cfg, "hand")] == [
        "D435I",
        "L515",
    ]
    assert [name for name, _cfg in _camera_candidates(camera_cfg, "L515")] == ["L515"]
    assert [name for name, _cfg in _camera_candidates(camera_cfg, "D435I")] == ["D435I"]


def test_aligned_playback_rebases_diagnostic_pose_to_first_saved_frame() -> None:
    frame_records = [{"index": 5}, {"index": 6}, {"index": 9}]
    diagnostic_poses = {
        5: _translation(1.2, -0.1, 0.3),
        6: _translation(1.25, -0.1, 0.3),
        9: _translation(1.35, -0.08, 0.3),
    }
    args = SimpleNamespace(
        aligned_point_cloud_pose_source="diagnostic",
        aligned_point_cloud_pose_jsonl=None,
    )
    poses, source = _resolve_aligned_playback_poses(frame_records, args, diagnostic_poses)
    np.testing.assert_allclose(poses[0], np.eye(4), atol=1e-8)
    np.testing.assert_allclose(poses[:, :3, 3], [[0, 0, 0], [0.05, 0, 0], [0.15, 0.02, 0]])
    assert source.startswith("diagnostic_")


def test_aligned_playback_loads_external_pose_by_record_index(tmp_path) -> None:
    pose_path = tmp_path / "camera_pose.jsonl"
    records = [
        {
            "record_index": index,
            "camera_to_tracking": _translation(x).tolist(),
            "tracking_source": "rtabmap_rgbd_imu",
            "valid": True,
        }
        for index, x in [(12, 2.0), (18, 2.3)]
    ]
    pose_path.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )
    args = SimpleNamespace(
        aligned_point_cloud_pose_source="external",
        aligned_point_cloud_pose_jsonl=str(pose_path),
    )
    poses, source = _resolve_aligned_playback_poses([{"index": 12}, {"index": 18}], args, {})
    np.testing.assert_allclose(poses[:, 0, 3], [0.0, 0.3], atol=1e-8)
    assert source == f"external:{pose_path}"


def test_camera_axes_lines_follow_camera_se3() -> None:
    pose = _translation(0.3, -0.2, 0.7)
    points, lines, colors = _camera_axes_lines(pose, axis_size=0.1)
    np.testing.assert_allclose(points[0], [0.3, -0.2, 0.7])
    np.testing.assert_allclose(points[1:], [[0.4, -0.2, 0.7], [0.3, -0.1, 0.7], [0.3, -0.2, 0.8]])
    np.testing.assert_array_equal(lines, [[0, 1], [0, 2], [0, 3]])
    assert colors.shape == (3, 3)


def test_rgbd_world_anchor_rejects_independently_moving_foreground() -> None:
    pytest.importorskip("open3d")
    rng = np.random.default_rng(11)

    table_xy = rng.uniform([-0.55, -0.4], [0.55, 0.4], size=(2500, 2))
    table = np.column_stack(
        [
            table_xy,
            0.85 + 0.004 * np.sin(7.0 * table_xy[:, 0]) * np.cos(6.0 * table_xy[:, 1]),
        ]
    )
    table_color = np.column_stack(
        [
            120.0 + 80.0 * (table_xy[:, 0] + 0.55) / 1.1,
            80.0 + 100.0 * (table_xy[:, 1] + 0.4) / 0.8,
            np.full(len(table_xy), 60.0),
        ]
    )
    wall_xz = rng.uniform([-0.55, 0.55], [0.55, 1.25], size=(1000, 2))
    wall = np.column_stack([wall_xz[:, 0], np.full(len(wall_xz), 0.42), wall_xz[:, 1]])
    wall_color = np.column_stack(
        [
            np.full(len(wall_xz), 180.0),
            90.0 + 80.0 * (wall_xz[:, 1] - 0.55) / 0.7,
            80.0 + 80.0 * (wall_xz[:, 0] + 0.55) / 1.1,
        ]
    )
    static_xyz = np.concatenate([table, wall])
    static_bgr = np.clip(np.concatenate([table_color, wall_color]), 0, 255).astype(np.uint8)

    moving_shape = rng.uniform([-0.08, -0.06, -0.05], [0.08, 0.06, 0.05], size=(1200, 3))
    moving_bgr = np.tile(np.asarray([[20, 20, 245]], dtype=np.uint8), (len(moving_shape), 1))
    odometry = RGBDOdometryDebugger(
        voxel_size=0.02,
        max_correspondence=0.06,
        robust_kernel=0.01,
        anchor_min_fitness=0.30,
        anchor_max_correction_m=0.10,
        anchor_max_correction_deg=10.0,
        bootstrap_frames=5,
    )
    for frame_index in range(9):
        moving_center = np.asarray([-0.18 + 0.045 * frame_index, -0.05, 0.58])
        odometry.update(
            np.concatenate([static_xyz, moving_shape + moving_center]),
            np.concatenate([static_bgr, moving_bgr]),
        )

    poses = np.asarray(odometry.pose_history)
    translation_drift = np.linalg.norm(poses[:, :3, 3], axis=-1)
    rotation_drift = np.asarray(
        [np.degrees(np.arccos(np.clip((np.trace(pose[:3, :3]) - 1.0) * 0.5, -1.0, 1.0))) for pose in poses]
    )
    assert translation_drift.max() < 0.01
    assert rotation_drift.max() < 1.0
    assert any(metrics.get("temporal_static_filter_active", False) for metrics in odometry.metrics_history)


def test_camera_to_model_world_uses_explicit_transform_direction() -> None:
    camera_to_tracking = np.stack([_translation(2.0), _translation(2.1), _translation(2.3)])
    camera_to_world = camera_to_model_world_transforms(camera_to_tracking)
    np.testing.assert_allclose(camera_to_world[0], np.eye(4), atol=1e-7)
    np.testing.assert_allclose(camera_to_world[:, 0, 3], [0.0, 0.1, 0.3], atol=1e-7)

    tracking_to_camera = invert_transform_sequence(camera_to_tracking)
    parsed = camera_pose_values_to_matrices(
        tracking_to_camera,
        pose_format="matrix",
        direction="tracking_to_camera",
    )
    np.testing.assert_allclose(parsed, camera_to_tracking, atol=1e-7)


def test_camera_to_model_world_supports_cross_episode_canonical_camera() -> None:
    camera_to_tracking = np.stack([_translation(2.1), _translation(2.2), _translation(2.4)])
    canonical_camera_to_tracking = _translation(2.0)

    camera_to_world = camera_to_model_world_transforms(
        camera_to_tracking,
        reference_camera_to_tracking=canonical_camera_to_tracking,
    )

    np.testing.assert_allclose(camera_to_world[:, 0, 3], [0.1, 0.2, 0.4], atol=1e-7)
    assert not np.allclose(camera_to_world[0], np.eye(4))


def test_head_following_hand_no_longer_produces_constant_world_ee_pose() -> None:
    # The hand stays one metre in front of the moving head camera.
    camera_to_tracking = np.stack([_translation(0.0), _translation(0.1), _translation(0.2)])
    camera_to_world = camera_to_model_world_transforms(camera_to_tracking)
    camera_to_ee = np.repeat(np.eye(4, dtype=np.float64)[None], 3, axis=0)
    camera_to_ee[:, 2, 3] = 1.0
    raw_pose9 = matrix_to_pose9(camera_to_ee)

    corrected_pose9 = transform_pose9_sequence(raw_pose9, camera_to_world)
    np.testing.assert_allclose(raw_pose9[:, :3], [[0.0, 0.0, 1.0]] * 3, atol=1e-7)
    np.testing.assert_allclose(
        corrected_pose9[:, :3],
        [[0.0, 0.0, 1.0], [0.1, 0.0, 1.0], [0.2, 0.0, 1.0]],
        atol=1e-7,
    )


def test_static_tracking_point_aligns_to_model_world() -> None:
    camera_to_tracking = np.stack([_translation(0.0), _translation(0.15), _translation(-0.08)])
    camera_to_world = camera_to_model_world_transforms(camera_to_tracking)
    point_tracking = np.asarray([[0.4, -0.2, 1.3]], dtype=np.float64)
    points_current_camera = []
    for pose in camera_to_tracking:
        tracking_to_camera = invert_transform_sequence(pose)[0]
        points_current_camera.append(transform_points(point_tracking, tracking_to_camera))
    points_aligned = [
        transform_points(point, transform)
        for point, transform in zip(points_current_camera, camera_to_world, strict=True)
    ]
    for point in points_aligned:
        np.testing.assert_allclose(point, point_tracking, atol=1e-7)


def test_current_eff_point_cloud_is_invariant_when_cloud_and_ee_are_both_corrected() -> None:
    camera_to_world = _translation(0.25, -0.04, 0.02)
    camera_to_ee = _translation(0.1, 0.2, 0.8)
    point_camera = np.asarray([[0.3, -0.1, 1.2]], dtype=np.float64)

    direct_eff = transform_points(point_camera, invert_transform_sequence(camera_to_ee)[0])
    point_world = transform_points(point_camera, camera_to_world)
    world_to_ee = camera_to_world @ camera_to_ee
    corrected_eff = transform_points(point_world, invert_transform_sequence(world_to_ee)[0])
    np.testing.assert_allclose(corrected_eff, direct_eff, atol=1e-7)


def test_hdf5_camera_pose_auto_schema_and_required_mode(tmp_path) -> None:
    hdf5_path = tmp_path / "episode.hdf5"
    camera_to_tracking = np.stack([_translation(0.0), _translation(0.05), _translation(0.10)])
    with h5py.File(hdf5_path, "w") as root:
        pose = root.create_dataset("observations/camera_tracking_pose/overhead", data=camera_to_tracking)
        pose.attrs["transform_direction"] = "camera_to_tracking"
        pose.attrs["translation_unit"] = "meter"
        pose.attrs["pose_format"] = "matrix"
        pose.attrs["tracking_source"] = "synthetic_vio"

    with h5py.File(hdf5_path, "r") as root:
        camera_to_world, metadata = load_camera_motion(root, _camera_cfg(), frame_count=3)
    assert camera_to_world is not None
    np.testing.assert_allclose(camera_to_world[:, 0, 3], [0.0, 0.05, 0.10], atol=1e-7)
    assert metadata["enabled"] is True
    assert metadata["tracking_source"] == "synthetic_vio"
    assert metadata["translation_span_m"] == pytest.approx(0.10)

    empty_path = tmp_path / "empty.hdf5"
    with h5py.File(empty_path, "w"):
        pass
    with h5py.File(empty_path, "r") as root:
        camera_to_world, metadata = load_camera_motion(root, _camera_cfg(), frame_count=3)
    assert camera_to_world is None
    assert metadata == {
        "enabled": False,
        "mode": "auto",
        "reason": "camera_pose_missing",
    }
    with h5py.File(empty_path, "r") as root, pytest.raises(KeyError, match="required"):
        load_camera_motion(
            root,
            _camera_cfg(camera_motion_compensation="required"),
            frame_count=3,
        )


def test_hdf5_camera_pose_uses_stored_canonical_reference(tmp_path) -> None:
    hdf5_path = tmp_path / "canonical.hdf5"
    camera_to_tracking = np.stack([_translation(1.1), _translation(1.2), _translation(1.4)])
    with h5py.File(hdf5_path, "w") as root:
        pose = root.create_dataset("observations/camera_tracking_pose/overhead", data=camera_to_tracking)
        pose.attrs["transform_direction"] = "camera_to_tracking"
        pose.attrs["translation_unit"] = "meter"
        pose.attrs["pose_format"] = "matrix"
        pose.attrs["tracking_source"] = "persistent_map_localization"
        pose.attrs["camera_reference_mode"] = "canonical"
        pose.attrs["canonical_camera_to_tracking"] = _translation(1.0)

    with h5py.File(hdf5_path, "r") as root:
        camera_to_world, metadata = load_camera_motion(root, _camera_cfg(), frame_count=3)

    np.testing.assert_allclose(camera_to_world[:, 0, 3], [0.1, 0.2, 0.4], atol=1e-7)
    assert metadata["camera_reference_mode"] == "canonical"
    assert metadata["camera_reference_source"] == "hdf5"
    assert metadata["model_world_definition"] == "canonical_fixed_overview_camera"


def test_recorder_static_trajectory_embeds_identity_in_every_saved_frame(
    tmp_path,
) -> None:
    frame_records = [
        {"index": 4, "timestamp_ms": 100.0, "color_path": "color_4.jpg"},
        {"index": 8, "timestamp_ms": 133.0, "color_path": "color_8.jpg"},
    ]
    frames_path = tmp_path / "frames.jsonl"
    frames_path.write_text(
        "".join(json.dumps(record) + "\n" for record in frame_records),
        encoding="utf-8",
    )
    trajectory = _static_camera_trajectory(frame_records)
    pose_path = tmp_path / "camera_pose.jsonl"

    _write_camera_trajectory(tmp_path, pose_path, frame_records, trajectory)

    saved_poses = [json.loads(line) for line in pose_path.read_text(encoding="utf-8").splitlines()]
    saved_frames = [json.loads(line) for line in frames_path.read_text(encoding="utf-8").splitlines()]
    assert [record["record_index"] for record in saved_poses] == [4, 8]
    assert [record["index"] for record in saved_frames] == [4, 8]
    for pose_record, frame_record in zip(saved_poses, saved_frames, strict=True):
        np.testing.assert_allclose(pose_record["camera_to_tracking"], np.eye(4))
        np.testing.assert_allclose(frame_record["camera_to_tracking"], np.eye(4))
        assert frame_record["camera_pose_source"] == "static_identity_camera"


def test_recorder_external_trajectory_requires_complete_synchronized_se3(
    tmp_path,
) -> None:
    pose_path = tmp_path / "external.jsonl"
    pose_path.write_text(
        "\n".join(
            json.dumps(
                {
                    "record_index": index,
                    "timestamp_ms": timestamp_ms,
                    "camera_to_tracking": _translation(translation).tolist(),
                    "tracking_source": "rtabmap_localization",
                }
            )
            for index, timestamp_ms, translation in [
                (3, 100.5, 0.0),
                (4, 133.5, 0.1),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    frame_records = [
        {"index": 3, "timestamp_ms": 100.0},
        {"index": 4, "timestamp_ms": 133.0},
    ]

    trajectory = _external_camera_trajectory(frame_records, pose_path, max_sync_error_ms=1.0)

    assert len(trajectory) == 2
    np.testing.assert_allclose(trajectory[1]["camera_to_tracking"], _translation(0.1))
    assert trajectory[0]["rgb_pose_sync_error_ms"] == pytest.approx(0.5)
    with pytest.raises(ValueError, match="out of sync"):
        _external_camera_trajectory(frame_records, pose_path, max_sync_error_ms=0.1)
    with pytest.raises(RuntimeError, match="missing"):
        _external_camera_trajectory(
            [*frame_records, {"index": 5, "timestamp_ms": 166.0}],
            pose_path,
            max_sync_error_ms=1.0,
        )


def test_external_pose_jsonl_is_matched_only_by_record_index(tmp_path) -> None:
    pose_path = tmp_path / "camera_poses.jsonl"
    lines = [
        {
            "record_index": 10,
            "timestamp_ms": 101.0,
            "camera_to_tracking": _translation(0.0).tolist(),
            "tracking_source": "vio",
        },
        {
            "record_index": 11,
            "timestamp_ms": 197.0,
            "camera_to_tracking": _translation(0.2).tolist(),
            "tracking_source": "vio",
        },
    ]
    pose_path.write_text("\n".join(json.dumps(line) for line in lines), encoding="utf-8")
    frames = [
        {"index": 10, "timestamp_ms": 100.0},
        {"index": 11, "timestamp_ms": 200.0},
        {"index": 12, "timestamp_ms": 300.0},
    ]
    attached = attach_external_camera_poses(frames, pose_path, max_sync_error_ms=5.0)
    assert attached[0]["camera_pose_valid"] is True
    assert attached[0]["camera_pose_sync_error_ms"] == pytest.approx(1.0)
    assert attached[1]["camera_pose_sync_error_ms"] == pytest.approx(-3.0)
    assert attached[1]["camera_pose_source"] == "vio"
    assert "camera_to_tracking" not in attached[2]
    with pytest.raises(ValueError, match="out of sync"):
        attach_external_camera_poses(frames, pose_path, max_sync_error_ms=2.0)


def test_camera_motion_metrics_are_auditable() -> None:
    camera_to_world = np.stack([_translation(0.0), _translation(0.03), _translation(0.08)])
    metrics = camera_motion_metrics(camera_to_world)
    assert metrics["frame_count"] == 3
    assert metrics["translation_span_m"] == pytest.approx(0.08)
    assert metrics["max_translation_step_m"] == pytest.approx(0.05)

    pose9 = matrix_to_pose9(camera_to_world)
    np.testing.assert_allclose(pose9_to_matrix(pose9), camera_to_world, atol=1e-7)


def test_stationary_pose_report_checks_drift_and_tracking_acceptance() -> None:
    stable = np.stack([_translation(0.0), _translation(0.001), _translation(0.002)])
    report = stationary_pose_report(
        stable,
        min_frames=3,
        max_translation_drift_m=0.005,
        max_rotation_drift_deg=1.0,
        accepted=np.asarray([True, True, True]),
        min_accepted_ratio=0.8,
    )
    assert report["passed"] is True
    assert report["translation"]["max_drift_m"] == pytest.approx(0.002)

    rejected = stationary_pose_report(
        stable,
        min_frames=3,
        max_translation_drift_m=0.005,
        max_rotation_drift_deg=1.0,
        accepted=np.asarray([True, False, False]),
        min_accepted_ratio=0.8,
    )
    assert rejected["passed"] is False
    assert rejected["accepted_ratio_ok"] is False


def test_real_episode_conversion_restores_motion_without_model_changes(
    tmp_path,
) -> None:
    hdf5_path = tmp_path / "moving_head.hdf5"
    camera_to_tracking = np.stack([_translation(0.0), _translation(0.1), _translation(0.2)])
    camera_to_ee = np.repeat(_translation(0.0, 0.0, 0.5)[None], 3, axis=0)
    raw_pose9 = matrix_to_pose9(camera_to_ee)
    tracking_points = np.asarray(
        [[x, y, 1.0, 120.0, 80.0, 40.0] for x in np.linspace(-0.2, 0.2, 5) for y in (-0.1, 0.1)],
        dtype=np.float32,
    )
    camera_clouds = []
    for pose in camera_to_tracking:
        cloud = tracking_points.copy()
        cloud[:, :3] = transform_points(tracking_points[:, :3], invert_transform_sequence(pose)[0])
        camera_clouds.append(cloud)

    with h5py.File(hdf5_path, "w") as root:
        root.attrs["pose_frame"] = "camera"
        root.attrs["task"] = "synthetic pick and place"
        root.create_dataset("observations/cloud_rgb/overhead", data=np.asarray(camera_clouds))
        root.create_dataset("observations/pose_eular", data=raw_pose9)
        root.create_dataset("observations/eff_angular", data=np.asarray([0.08, 0.08, 0.04]))
        root.create_dataset("timestamp_ms", data=np.asarray([0.0, 33.0, 66.0]))
        camera_pose = root.create_dataset(
            "observations/camera_tracking_pose/overhead",
            data=camera_to_tracking,
        )
        camera_pose.attrs["transform_direction"] = "camera_to_tracking"
        camera_pose.attrs["translation_unit"] = "meter"
        camera_pose.attrs["pose_format"] = "matrix"
        camera_pose.attrs["tracking_source"] = "synthetic_vio"

    cfg = {
        **_camera_cfg(camera_motion_compensation="required"),
        "point_cloud_key": None,
        "image_key": "none",
        "image_feature_key": None,
        "pose_key": "observations/pose_eular",
        "pose_format": "pose9",
        "gripper_key": "observations/eff_angular",
        "timestamp_key": "timestamp_ms",
        "timestamp_mode": "source",
        "cloud_frame": "auto",
        "task": None,
        "fps": 30,
        "max_frames": None,
        "num_points": len(tracking_points),
        "add_gripper_cloud": False,
        "input_has_gripper_cloud": False,
        "gripper_points": 0,
        "gripper_len": 0.06,
        "gripper_template": "reap",
        "gripper_drop_strategy": "random",
        "gripper_shuffle_points": False,
        "gripper_widths_are_normalized": False,
        "gripper_max_width": 0.08,
        "seed": 7,
        "camera_motion_debug": True,
        "camera_motion_debug_frames": 3,
        "camera_motion_debug_max_points": len(tracking_points),
    }
    episode, record = convert_hdf5_episode(hdf5_path, cfg)
    assert record["reference_frame"] == "world"
    assert record["world_definition"] == "episode_first_overview_camera"
    assert record["uses_camera_tracking_pose"] is True
    np.testing.assert_allclose(episode["world_ee_poses"][:, 0], [0.0, 0.1, 0.2], atol=1e-6)
    assert np.max(np.abs(episode["actions"][1:, :3])) > 0.05
    assert episode["camera_motion_debug"]["frame_indices"] == [0, 1, 2]


def test_dataset_sidecars_make_world_canonical_and_overhead_a_camera_name(
    tmp_path,
) -> None:
    write_dataset_sidecar_meta(tmp_path, "npy")
    world_meta = json.loads((tmp_path / "world_ee_poses" / "meta.json").read_text())
    camera_meta = json.loads((tmp_path / "camera_motion" / "meta.json").read_text())

    assert world_meta["key"] == "worldflow.ee_poses"
    assert world_meta["coordinate_frame"] == "world"
    assert not (tmp_path / "overhead_ee_poses").exists()
    assert camera_meta["key"] == "camera_motion.camera_to_world"
    assert camera_meta["notation"] == "T_model_world<-current_camera"


def test_fixed_camera_episode_keeps_legacy_real_conversion_semantics(tmp_path) -> None:
    hdf5_path = tmp_path / "fixed_camera.hdf5"
    raw_pose9 = matrix_to_pose9(
        np.stack(
            [
                _translation(0.0, 0.0, 0.5),
                _translation(0.05, 0.0, 0.5),
            ]
        )
    )
    cloud = np.asarray(
        [
            [0.0, 0.0, 0.8, 120.0, 80.0, 40.0],
            [0.1, 0.0, 0.9, 120.0, 80.0, 40.0],
        ],
        dtype=np.float32,
    )
    with h5py.File(hdf5_path, "w") as root:
        root.attrs["pose_frame"] = "camera"
        root.attrs["task"] = "fixed camera"
        root.create_dataset("observations/cloud_rgb/overhead", data=np.stack([cloud, cloud]))
        root.create_dataset("observations/pose_eular", data=raw_pose9)
        root.create_dataset("observations/eff_angular", data=np.asarray([0.08, 0.04]))
        root.create_dataset("timestamp_ms", data=np.asarray([0.0, 33.0]))

    cfg = {
        **_camera_cfg(camera_motion_compensation="auto"),
        "point_cloud_key": None,
        "image_key": "none",
        "image_feature_key": None,
        "pose_key": "observations/pose_eular",
        "pose_format": "pose9",
        "gripper_key": "observations/eff_angular",
        "timestamp_key": "timestamp_ms",
        "timestamp_mode": "source",
        "cloud_frame": "auto",
        "task": None,
        "fps": 30,
        "max_frames": None,
        "num_points": len(cloud),
        "add_gripper_cloud": False,
        "input_has_gripper_cloud": False,
        "gripper_points": 0,
        "gripper_len": 0.06,
        "gripper_template": "reap",
        "gripper_drop_strategy": "random",
        "gripper_shuffle_points": False,
        "gripper_widths_are_normalized": False,
        "gripper_max_width": 0.08,
        "seed": 7,
        "camera_motion_debug": False,
        "camera_motion_debug_frames": 2,
        "camera_motion_debug_max_points": len(cloud),
    }
    episode, record = convert_hdf5_episode(hdf5_path, cfg)

    np.testing.assert_allclose(episode["world_ee_poses"], raw_pose9, atol=1e-7)
    assert "camera_to_world" not in episode
    assert record["reference_frame"] == "world"
    assert record["world_definition"] == "fixed_overview_camera"
    assert record["uses_camera_tracking_pose"] is False


def test_episode_first_camera_world_is_accepted_without_second_alignment(tmp_path) -> None:
    """Aligned human-hand/StageGen files need no rewrite or camera reapplication."""

    hdf5_path = tmp_path / "episode_first_camera_world.hdf5"
    raw_pose9 = matrix_to_pose9(
        np.stack(
            [
                _translation(0.0, 0.0, 0.5),
                _translation(0.05, 0.0, 0.5),
            ]
        )
    )
    cloud = np.asarray(
        [
            [0.0, 0.0, 0.8, 120.0, 80.0, 40.0],
            [0.1, 0.0, 0.9, 120.0, 80.0, 40.0],
        ],
        dtype=np.float32,
    )
    with h5py.File(hdf5_path, "w") as root:
        root.attrs["pose_frame"] = "world"
        root.attrs["source_pose_frame"] = "camera"
        root.attrs["reference_frame"] = "episode_first_camera"
        root.attrs["world_frame_definition"] = "camera frame at first source_record_index"
        root.attrs["episode_first_alignment_applied"] = True
        root.attrs["uses_camera_extrinsic"] = False
        root.attrs["camera_reference_mode"] = "episode_first"
        root.attrs["task"] = "already aligned camera world"
        cloud_dataset = root.create_dataset(
            "observations/cloud_rgb/overhead",
            data=np.stack([cloud, cloud]),
        )
        cloud_dataset.attrs["coordinate_frame"] = "episode_first_camera"
        cloud_dataset.attrs["alignment_applied"] = True
        pose_dataset = root.create_dataset("observations/pose_eular", data=raw_pose9)
        pose_dataset.attrs["coordinate_frame"] = "episode_first_camera"
        root.create_dataset("observations/eff_angular", data=np.asarray([0.08, 0.04]))
        root.create_dataset("timestamp_ms", data=np.asarray([0.0, 33.0]))

        # This is retained audit metadata, not a transform to apply again. Its
        # direction is intentionally outside load_camera_motion's raw-input
        # contract so the test fails if the converter attempts a second pass.
        camera_pose = root.create_dataset(
            "observations/camera_tracking_pose/overhead",
            data=np.stack([np.eye(4), _translation(0.4)]),
        )
        camera_pose.attrs["transform_direction"] = "camera_to_episode_first_camera"
        camera_pose.attrs["translation_unit"] = "meter"
        camera_pose.attrs["pose_format"] = "matrix"

    cfg = {
        **_camera_cfg(camera_motion_compensation="required"),
        "point_cloud_key": None,
        "image_key": "none",
        "image_feature_key": None,
        "pose_key": "observations/pose_eular",
        "pose_format": "pose9",
        "gripper_key": "observations/eff_angular",
        "timestamp_key": "timestamp_ms",
        "timestamp_mode": "source",
        "cloud_frame": "auto",
        "task": None,
        "fps": 30,
        "max_frames": None,
        "num_points": len(cloud),
        "add_gripper_cloud": False,
        "input_has_gripper_cloud": False,
        "gripper_points": 0,
        "gripper_len": 0.06,
        "gripper_template": "reap",
        "gripper_drop_strategy": "random",
        "gripper_shuffle_points": False,
        "gripper_widths_are_normalized": False,
        "gripper_max_width": 0.08,
        "seed": 7,
        "camera_motion_debug": False,
        "camera_motion_debug_frames": 2,
        "camera_motion_debug_max_points": len(cloud),
    }
    episode, record = convert_hdf5_episode(hdf5_path, cfg)

    np.testing.assert_allclose(episode["world_ee_poses"], raw_pose9, atol=1e-7)
    assert "camera_to_world" not in episode
    assert record["source_cloud_frame"] == "camera"
    assert record["world_definition"] == "episode_first_overview_camera"
    assert record["source_already_camera_reference_aligned"] is True
    assert record["uses_camera_tracking_pose"] is False
    assert record["camera_motion"]["reason"] == "source_already_camera_reference_aligned"
