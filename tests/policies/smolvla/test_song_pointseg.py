#!/usr/bin/env python

import json

import numpy as np
import pytest
import torch

from lerobot.policies.smolvla.song_pointseg import (
    POINTSEG_CACHE_FIELDS,
    POINTSEG_CACHE_LABEL_FIELDS,
    POINTSEG_CACHE_VERSION,
    ROLE_FOREGROUND,
    ROLE_IGNORE,
    PseudoLabelConfig,
    SongPointSegCachedDataset,
    SongPointSegLoss,
    SongPointSegLossConfig,
    SongPointSegNet,
    SongTemporalPointCloudDataset,
    _aggregate_motion_hypotheses,
    build_litept_grid_coord,
    compose_point_cloud_views,
    consensus_multiscale_novelty_union_sample_fused_point_cloud,
    fps_sample_fused_point_cloud,
    infer_single_view_point_count,
    multiscale_novelty_union_sample_fused_point_cloud,
    voxel_fps_sample_fused_point_cloud,
    voxel_cover_fps_sample_fused_point_cloud,
    novelty_union_sample_fused_point_cloud,
    transport_novelty_union_sample_fused_point_cloud,
    force_small_current_clouds_foreground,
    generate_pseudo_labels,
    generate_pseudo_labels_from_priors,
    matrix_to_pose9,
    parse_camera_views,
    parse_camera_view_weights,
    paired_view_augmentation_index,
    pose9_to_matrix,
    refine_pseudo_labels_with_teacher,
    relative_poses_to_first,
    song_pointseg_collate,
    use_primary_view_for_training_index,
)


@pytest.mark.parametrize(
    ("single_view_points", "num_views", "gripper_points"),
    [(10_000, 2, 500), (50_000, 2, 500), (50_000, 3, 0), (12_345, 1, 321)],
)
def test_infer_single_view_point_count_restores_native_view_length(
    single_view_points, num_views, gripper_points
):
    fused_points = num_views * (single_view_points - gripper_points) + gripper_points

    assert infer_single_view_point_count(
        fused_points,
        num_views=num_views,
        gripper_points=gripper_points,
    ) == single_view_points


def test_infer_single_view_point_count_rejects_malformed_union():
    with pytest.raises(ValueError, match="not divisible"):
        infer_single_view_point_count(19_501, num_views=2, gripper_points=500)


def test_litept_grid_coord_uses_an_independent_integer_origin_per_sample():
    coord = torch.tensor(
        [
            [-0.011, 0.009, 0.000],
            [0.019, 0.021, -0.001],
            [100.001, -50.001, 7.999],
            [100.031, -49.979, 8.009],
        ],
        dtype=torch.float32,
    )
    batch = torch.tensor([0, 0, 1, 1], dtype=torch.long)

    grid_coord = build_litept_grid_coord(coord, batch, grid_size=0.01)

    assert grid_coord.dtype == torch.int32
    assert torch.equal(
        grid_coord,
        torch.tensor([[0, 0, 1], [3, 2, 0], [0, 0, 0], [3, 3, 1]], dtype=torch.int32),
    )


def test_litept_grid_coord_is_independent_of_other_samples_in_the_batch():
    sample = torch.tensor(
        [[-0.011, 0.009, 0.000], [0.019, 0.021, -0.001]], dtype=torch.float32
    )
    other_near = torch.tensor([[1.0, 2.0, 3.0], [1.1, 2.1, 3.1]], dtype=torch.float32)
    other_far = other_near + torch.tensor([1000.0, -2000.0, 3000.0])
    batch = torch.tensor([0, 0, 1, 1], dtype=torch.long)

    near_grid = build_litept_grid_coord(torch.cat([sample, other_near]), batch, 0.01)
    far_grid = build_litept_grid_coord(torch.cat([sample, other_far]), batch, 0.01)

    assert torch.equal(near_grid[:2], far_grid[:2])


def test_multiview_point_cloud_composition_keeps_one_gripper_tail():
    first = np.zeros((10, 6), dtype=np.float32)
    second = np.ones((10, 6), dtype=np.float32)
    first[-2:, :3] = 7.0
    second[-2:, :3] = 9.0

    fused = compose_point_cloud_views([first, second], gripper_points=2, seed=11)

    assert fused.shape == first.shape
    assert np.all(fused[-2:, :3] == 7.0)
    assert np.any(np.all(fused[:-2] == 0.0, axis=1))
    assert np.any(np.all(fused[:-2] == 1.0, axis=1))


def test_multiview_point_cloud_composition_respects_primary_view_weight():
    first = np.zeros((12, 6), dtype=np.float32)
    second = np.ones((12, 6), dtype=np.float32)
    first[-2:, :3] = 7.0

    fused = compose_point_cloud_views(
        [first, second],
        gripper_points=2,
        seed=11,
        view_weights="9,1",
    )

    assert np.count_nonzero(np.all(fused[:-2] == 0.0, axis=1)) == 9
    assert np.count_nonzero(np.all(fused[:-2] == 1.0, axis=1)) == 1
    assert np.all(fused[-2:, :3] == 7.0)


def test_fps_multiview_composition_uses_equal_union_and_one_gripper_tail():
    first = np.zeros((8, 6), dtype=np.float32)
    second = np.ones((8, 6), dtype=np.float32)
    first[:6, 0] = np.arange(6, dtype=np.float32)
    second[:6, 0] = np.arange(6, dtype=np.float32) + 20.0
    first[-2:, :3] = 7.0
    second[-2:, :3] = 9.0

    union = compose_point_cloud_views(
        [first, second],
        gripper_points=2,
        fusion="fps",
    )

    assert union.shape == (14, 6)
    np.testing.assert_array_equal(union[:6], first[:6])
    np.testing.assert_array_equal(union[6:12], second[:6])
    np.testing.assert_array_equal(union[-2:], first[-2:])


def test_full_union_preserves_every_scene_point_and_one_primary_gripper_tail():
    first = np.arange(48, dtype=np.float32).reshape(8, 6)
    second = first + 100.0

    union = compose_point_cloud_views(
        [first, second],
        gripper_points=2,
        fusion="full_union",
    )

    assert union.shape == (14, 6)
    np.testing.assert_array_equal(union[:6], first[:6])
    np.testing.assert_array_equal(union[6:12], second[:6])
    np.testing.assert_array_equal(union[-2:], first[-2:])


def test_uniform_union_samples_joint_scene_without_fixed_view_quota_and_keeps_primary_gripper():
    first = np.zeros((102, 6), dtype=np.float32)
    second = np.ones((102, 6), dtype=np.float32)
    first[-2:, :3] = 7.0
    second[-2:, :3] = 9.0

    fused = compose_point_cloud_views(
        [first, second],
        gripper_points=2,
        seed=17,
        fusion="uniform_union",
    )
    repeated = compose_point_cloud_views(
        [first, second],
        gripper_points=2,
        seed=17,
        fusion="random_union",
    )

    assert fused.shape == first.shape
    np.testing.assert_array_equal(fused, repeated)
    np.testing.assert_array_equal(fused[-2:], first[-2:])
    primary_count = int(np.count_nonzero(np.all(fused[:-2] == 0.0, axis=1)))
    secondary_count = int(np.count_nonzero(np.all(fused[:-2] == 1.0, axis=1)))
    assert primary_count + secondary_count == 100
    assert primary_count != 50  # no per-view quota; this seed draws a stochastic split


def test_paired_primary_and_full_union_collate_masks_only_the_padded_primary_tail():
    primary = torch.arange(48, dtype=torch.float32).reshape(8, 6)
    full_union = torch.arange(84, dtype=torch.float32).reshape(14, 6)
    batch = song_pointseg_collate(
        [
            {"observation.point_cloud": primary},
            {"observation.point_cloud": full_union},
        ]
    )

    assert batch["observation.point_cloud"].shape == (2, 14, 6)
    assert not batch["observation.point_cloud_is_pad"][0, :8].any()
    assert batch["observation.point_cloud_is_pad"][0, 8:].all()
    assert not batch["observation.point_cloud_is_pad"][1].any()
    torch.testing.assert_close(batch["observation.point_cloud"][0, :8], primary)


def test_primary_residual_composition_preserves_ordered_views_and_primary_gripper():
    first = np.arange(48, dtype=np.float32).reshape(8, 6)
    second = first + 100.0

    union = compose_point_cloud_views(
        [first, second],
        gripper_points=2,
        fusion="primary_residual",
    )

    assert union.shape == (14, 6)
    np.testing.assert_array_equal(union[:6], first[:6])
    np.testing.assert_array_equal(union[6:12], second[:6])
    np.testing.assert_array_equal(union[-2:], first[-2:])


def test_fps_sampler_is_deterministic_and_preserves_gripper_indices():
    cloud = torch.zeros(1, 14, 6)
    cloud[0, :12, 0] = torch.tensor([0, 1, 2, 3, 4, 5, 20, 21, 22, 23, 24, 25])
    cloud[0, -2:, :3] = 7.0

    sampled, sampled_pad, indices = fps_sample_fused_point_cloud(
        cloud, target_points=8, gripper_points=2
    )
    repeated, _, repeated_indices = fps_sample_fused_point_cloud(
        cloud, target_points=8, gripper_points=2
    )

    assert sampled_pad is None
    assert sampled.shape == (1, 8, 6)
    assert indices[0, -2:].tolist() == [12, 13]
    assert torch.equal(indices, repeated_indices)
    assert torch.equal(sampled, repeated)
    assert torch.equal(
        sampled,
        torch.gather(cloud, 1, indices.unsqueeze(-1).expand(-1, -1, 6)),
    )


def test_voxel_fps_deduplicates_overlap_and_preserves_gripper_indices():
    cloud = torch.zeros(1, 14, 6)
    cloud[0, :12, 0] = torch.tensor(
        [0.0, 0.001, 0.010, 0.020, 0.030, 0.040, 0.0005, 0.0015, 0.011, 0.021, 0.031, 0.050]
    )
    cloud[0, -2:, :3] = 7.0

    sampled, sampled_pad, indices = voxel_fps_sample_fused_point_cloud(
        cloud, target_points=8, gripper_points=2, voxel_size=0.005
    )
    repeated, _, repeated_indices = voxel_fps_sample_fused_point_cloud(
        cloud, target_points=8, gripper_points=2, voxel_size=0.005
    )

    assert sampled_pad is None
    assert sampled.shape == (1, 8, 6)
    assert indices[0, -2:].tolist() == [12, 13]
    assert torch.equal(indices, repeated_indices)
    assert torch.equal(sampled, repeated)
    # The first configured view is the deterministic representative for the
    # occupied overlap voxels; duplicate wrist samples are not candidates.
    assert not ({6, 7, 8, 9, 10} & set(indices[0, :-2].tolist()))


def test_voxel_cover_fps_keeps_voxel_cover_then_adds_fps_detail():
    cloud = torch.zeros(1, 14, 6)
    cloud[0, :12, 0] = torch.tensor(
        [0.0, 0.001, 0.011, 0.021, 0.0005, 0.012, 0.022, 0.031, 0.032, 0.033, 0.034, 0.035]
    )
    cloud[0, -2:, :3] = 7.0

    sampled, sampled_pad, indices = voxel_cover_fps_sample_fused_point_cloud(
        cloud, target_points=8, gripper_points=2, voxel_size=0.01
    )
    repeated, _, repeated_indices = voxel_cover_fps_sample_fused_point_cloud(
        cloud, target_points=8, gripper_points=2, voxel_size=0.01
    )

    assert sampled_pad is None
    assert sampled.shape == (1, 8, 6)
    assert indices[0, -2:].tolist() == [12, 13]
    assert {0, 2, 3, 7}.issubset(set(indices[0, :-2].tolist()))
    assert len(set(indices[0, :-2].tolist())) == 6
    assert torch.equal(indices, repeated_indices)
    assert torch.equal(sampled, repeated)


def test_voxel_cover_fps_is_identity_for_target_sized_single_view():
    cloud = torch.randn(2, 8, 6)
    sampled, sampled_pad, indices = voxel_cover_fps_sample_fused_point_cloud(
        cloud, target_points=8, gripper_points=2, voxel_size=0.01
    )

    assert sampled_pad is None
    assert torch.equal(sampled, cloud)
    assert indices.tolist() == [list(range(8)), list(range(8))]


def test_novelty_union_replaces_only_redundant_primary_voxel_points():
    cloud = torch.zeros(1, 14, 6)
    cloud[0, :12, 0] = torch.tensor(
        [0.0, 0.001, 0.010, 0.020, 0.030, 0.040, 0.0005, 0.011, 0.021, 0.031, 0.050, 0.060]
    )
    cloud[0, -2:, :3] = 7.0

    sampled, sampled_pad, indices = novelty_union_sample_fused_point_cloud(
        cloud, target_points=8, gripper_points=2, voxel_size=0.005
    )

    assert sampled_pad is None
    assert indices.tolist() == [[0, 10, 2, 3, 4, 5, 12, 13]]
    assert torch.equal(
        sampled,
        torch.gather(cloud, 1, indices.unsqueeze(-1).expand(-1, -1, 6)),
    )


def test_multiscale_novelty_union_keeps_fine_primary_cover_and_only_coarse_novel_secondary():
    cloud = torch.zeros(1, 14, 6)
    cloud[0, :12, 0] = torch.tensor(
        [0.000, 0.001, 0.010, 0.011, 0.040, 0.041,
         0.002, 0.012, 0.050, 0.091, 0.092, 0.130]
    )
    cloud[0, -2:, :3] = 7.0

    sampled, sampled_pad, indices = multiscale_novelty_union_sample_fused_point_cloud(
        cloud, target_points=8, gripper_points=2, voxel_size=0.01
    )
    repeated, _, repeated_indices = multiscale_novelty_union_sample_fused_point_cloud(
        cloud, target_points=8, gripper_points=2, voxel_size=0.01
    )

    assert sampled_pad is None
    assert indices.tolist() == [[0, 9, 2, 11, 4, 5, 12, 13]]
    assert torch.equal(indices, repeated_indices)
    assert torch.equal(sampled, repeated)
    assert {0, 2, 4}.issubset(set(indices[0, :-2].tolist()))
    assert set(indices[0, :-2].tolist()) & {6, 7, 8, 10} == set()
    assert torch.equal(
        sampled,
        torch.gather(cloud, 1, indices.unsqueeze(-1).expand(-1, -1, 6)),
    )


def test_multiscale_novelty_union_is_exact_identity_for_single_view():
    cloud = torch.randn(2, 8, 6)
    sampled, sampled_pad, indices = multiscale_novelty_union_sample_fused_point_cloud(
        cloud, target_points=8, gripper_points=2, voxel_size=0.01
    )

    assert sampled_pad is None
    assert torch.equal(sampled, cloud)
    assert indices.tolist() == [list(range(8)), list(range(8))]


def test_multiscale_novelty_union_supports_more_conservative_coarse_scale():
    cloud = torch.zeros(1, 14, 6)
    cloud[0, :12, 0] = torch.tensor(
        [0.000, 0.001, 0.010, 0.011, 0.080, 0.081,
         0.002, 0.012, 0.035, 0.091, 0.092, 0.130]
    )
    cloud[0, -2:, :3] = 7.0

    _, _, default_indices = multiscale_novelty_union_sample_fused_point_cloud(
        cloud, target_points=8, gripper_points=2, voxel_size=0.01
    )
    sampled, sampled_pad, conservative_indices = (
        multiscale_novelty_union_sample_fused_point_cloud(
            cloud,
            target_points=8,
            gripper_points=2,
            voxel_size=0.01,
            coarse_novelty_scale=4.0,
        )
    )

    assert sampled_pad is None
    assert default_indices.tolist() == [[0, 8, 2, 9, 4, 11, 12, 13]]
    assert conservative_indices.tolist() == [[0, 11, 2, 3, 4, 5, 12, 13]]
    assert torch.equal(
        sampled,
        torch.gather(
            cloud,
            1,
            conservative_indices.unsqueeze(-1).expand(-1, -1, 6),
        ),
    )


def test_consensus_multiscale_uses_real_union_medoids_and_preserves_fine_cover():
    cloud = torch.zeros(1, 14, 6)
    cloud[0, :12, 0] = torch.tensor(
        [0.000, 0.001, 0.020, 0.040, 0.060, 0.080,
         0.004, 0.005, 0.006, 0.021, 0.160, 0.161]
    )
    cloud[0, -2:, :3] = 7.0

    sampled, sampled_pad, indices = (
        consensus_multiscale_novelty_union_sample_fused_point_cloud(
            cloud,
            target_points=8,
            gripper_points=2,
            voxel_size=0.01,
            coarse_novelty_scale=4.0,
        )
    )

    assert sampled_pad is None
    # Index 6 is the real union medoid of the primary-occupied first fine
    # voxel. Index 10 is the stable medoid of the secondary-only 4 cm cell.
    assert indices.tolist() == [[6, 10, 9, 3, 4, 5, 12, 13]]
    assert indices.unique().numel() == 8
    assert torch.equal(
        sampled,
        torch.gather(cloud, 1, indices.unsqueeze(-1).expand(-1, -1, 6)),
    )
    primary_fine_cells = set(torch.floor(cloud[0, :6, 0] / 0.01).long().tolist())
    sampled_fine_cells = set(torch.floor(sampled[0, :-2, 0] / 0.01).long().tolist())
    assert primary_fine_cells.issubset(sampled_fine_cells)
    assert torch.equal(sampled[0, -2:], cloud[0, -2:])


def test_consensus_multiscale_is_deterministic_and_single_view_exact_identity():
    single = torch.randn(2, 8, 6)
    sampled, sampled_pad, indices = (
        consensus_multiscale_novelty_union_sample_fused_point_cloud(
            single,
            target_points=8,
            gripper_points=2,
            voxel_size=0.01,
            coarse_novelty_scale=4.0,
        )
    )

    assert sampled_pad is None
    assert torch.equal(sampled, single)
    assert indices.tolist() == [list(range(8)), list(range(8))]


def test_consensus_multiscale_keeps_dual_view_batches_isolated():
    cloud = torch.zeros(2, 14, 6)
    pattern = torch.tensor(
        [0.000, 0.001, 0.020, 0.040, 0.060, 0.080,
         0.004, 0.005, 0.006, 0.021, 0.160, 0.161]
    )
    cloud[0, :12, 0] = pattern
    cloud[1, :12, 0] = pattern
    cloud[1, :12, 3] = 1.0
    cloud[:, -2:, :3] = 7.0

    sampled, _, indices = consensus_multiscale_novelty_union_sample_fused_point_cloud(
        cloud,
        target_points=8,
        gripper_points=2,
        voxel_size=0.01,
        coarse_novelty_scale=4.0,
    )

    assert indices.tolist() == [[6, 10, 9, 3, 4, 5, 12, 13]] * 2
    assert torch.equal(sampled[0, :-2, 3], torch.zeros(6))
    assert torch.equal(sampled[1, :-2, 3], torch.ones(6))


def test_transport_novelty_union_uses_local_one_to_one_replacements():
    cloud = torch.zeros(1, 14, 6)
    cloud[0, :12, 0] = torch.tensor(
        [0.0, 0.001, 0.100, 0.101, 0.200, 0.201,
         0.002, 0.102, 0.202, 0.215, 0.015, 0.115]
    )
    cloud[0, -2:, :3] = 7.0

    sampled, sampled_pad, indices = transport_novelty_union_sample_fused_point_cloud(
        cloud, target_points=8, gripper_points=2, voxel_size=0.01
    )

    assert sampled_pad is None
    assert indices.tolist() == [[0, 10, 2, 11, 4, 9, 12, 13]]
    assert torch.equal(
        sampled,
        torch.gather(cloud, 1, indices.unsqueeze(-1).expand(-1, -1, 6)),
    )



def test_parse_camera_view_weights_requires_one_positive_weight_per_view():
    assert parse_camera_view_weights("9,1", num_views=2) == (9.0, 1.0)
    with pytest.raises(ValueError, match="one value per view"):
        parse_camera_view_weights("1", num_views=2)
    with pytest.raises(ValueError, match="finite and positive"):
        parse_camera_view_weights("1,0", num_views=2)


def test_parse_camera_views_accepts_overhead_and_hand_pair():
    assert parse_camera_views("agentview,robot0_eye_in_hand") == (
        "agentview",
        "robot0_eye_in_hand",
    )


def test_training_view_dropout_assignment_is_balanced_and_worker_independent():
    first = [use_primary_view_for_training_index(index, seed=17) for index in range(1000)]
    repeated = [use_primary_view_for_training_index(index, seed=17) for index in range(1000)]

    assert first == repeated
    assert 450 <= sum(first) <= 550
    assert first != [use_primary_view_for_training_index(index, seed=18) for index in range(1000)]


def test_paired_view_augmentation_covers_both_modes_for_every_frame():
    num_samples = 101
    assignments = [
        paired_view_augmentation_index(index, num_samples, seed=17)
        for index in range(2 * num_samples)
    ]

    for base_index in range(num_samples):
        modes = [mode for index, mode in assignments if index == base_index]
        assert sorted(modes) == [False, True]

    with pytest.raises(IndexError):
        paired_view_augmentation_index(2 * num_samples, num_samples, seed=17)


def _pose_from_translation(xyz):
    transform = torch.eye(4, dtype=torch.float32)
    transform[:3, 3] = torch.tensor(xyz, dtype=torch.float32)
    return matrix_to_pose9(transform)


def test_relative_poses_to_first_has_identity_anchor():
    poses = torch.stack(
        [
            _pose_from_translation([0.2, -0.1, 0.0]),
            _pose_from_translation([0.5, -0.1, 0.1]),
            _pose_from_translation([0.5, 0.2, 0.1]),
        ],
        dim=0,
    )

    relative = relative_poses_to_first(poses.unsqueeze(0)).squeeze(0)
    first_matrix = pose9_to_matrix(relative[0])

    assert torch.allclose(first_matrix, torch.eye(4), atol=1e-5)
    assert torch.allclose(relative[1, :3], torch.tensor([0.3, 0.0, 0.1]), atol=1e-5)


def test_motion_residuals_separate_held_and_static_points():
    held_current = torch.tensor([[0.02, 0.0, 0.0]], dtype=torch.float32)
    static_current = torch.tensor([[0.40, 0.0, 0.0]], dtype=torch.float32)
    current_xyz = torch.cat([held_current, static_current], dim=0)
    current_pc = torch.cat([current_xyz, torch.zeros(2, 3)], dim=-1).unsqueeze(0)

    future_pose = _pose_from_translation([0.20, 0.0, 0.0])
    future_poses = torch.stack([_pose_from_translation([0.0, 0.0, 0.0]), future_pose], dim=0).unsqueeze(0)
    static_future = torch.tensor([[0.20, 0.0, 0.0]], dtype=torch.float32)
    future_xyz = torch.cat([held_current, static_future], dim=0)
    future_pc = torch.stack(
        [
            current_pc.squeeze(0),
            torch.cat([future_xyz, torch.zeros(2, 3)], dim=-1),
        ],
        dim=0,
    ).unsqueeze(0)
    future_is_pad = torch.zeros(1, 2, dtype=torch.bool)

    pseudo = generate_pseudo_labels(
        current_pc,
        future_pc,
        future_poses,
        future_is_pad,
        config=PseudoLabelConfig(nn_chunk_size=8, min_confidence=0.0),
    )

    held_res = pseudo["held_residual"].squeeze(0)
    static_res = pseudo["static_residual"].squeeze(0)
    assert held_res[0] < static_res[0]
    assert static_res[1] < held_res[1]


def test_motion_evidence_compares_hypotheses_in_the_same_future_frames():
    held = torch.tensor([[[0.01], [0.20], [0.01]]], dtype=torch.float32)
    static = torch.tensor([[[0.20], [0.01], [0.15]]], dtype=torch.float32)
    motion_weights = torch.tensor([[1.0, 1.0, 0.1]], dtype=torch.float32)
    valid = torch.ones(1, 3, dtype=torch.bool)

    held_residual, static_residual, residual_gap = _aggregate_motion_hypotheses(
        held,
        static,
        motion_weights,
        valid,
        relative_gap_eps=0.005,
        topk=2,
    )

    # Independent minima would produce zero evidence because both minima are 0.01,
    # even though two same-frame comparisons favor the held hypothesis.
    assert torch.allclose(static.amin(dim=1) - held.amin(dim=1), torch.zeros(1, 1))
    assert residual_gap.item() > 0.4
    assert held_residual.item() < static_residual.item()


def test_motion_evidence_is_suppressed_when_pose_motion_is_uninformative():
    held = torch.tensor([[[0.01], [0.01]]], dtype=torch.float32)
    static = torch.tensor([[[0.20], [0.20]]], dtype=torch.float32)
    valid = torch.ones(1, 2, dtype=torch.bool)

    _held, _static, strong_gap = _aggregate_motion_hypotheses(
        held,
        static,
        torch.ones(1, 2),
        valid,
        relative_gap_eps=0.005,
        topk=2,
    )
    _held, _static, weak_gap = _aggregate_motion_hypotheses(
        held,
        static,
        torch.full((1, 2), 0.01),
        valid,
        relative_gap_eps=0.005,
        topk=2,
    )

    assert strong_gap.item() > 0.8
    assert weak_gap.item() < 0.02


def test_generate_pseudo_labels_from_existing_priors():
    priors = torch.zeros(2, 16, 8, dtype=torch.float32)
    priors[..., 0] = torch.linspace(0.0, 0.4, 16)
    priors[..., 1] = priors[..., 0]
    priors[..., 3] = 0.05
    priors[..., 4] = 0.05
    priors[..., 6] = torch.exp(-priors[..., 3] / 0.05)
    priors[..., 7] = torch.exp(-priors[..., 4] / 0.05)

    pseudo = generate_pseudo_labels_from_priors(
        priors,
        config=PseudoLabelConfig(min_confidence=0.0, forced_foreground_min_score=0.0),
    )

    assert pseudo["priors"].shape == priors.shape
    assert pseudo["labels"].shape == priors.shape[:2]
    assert pseudo["weights"].shape == priors.shape[:2]
    assert pseudo["class_scores"].shape == (*priors.shape[:2], 2)


def test_approached_target_does_not_require_static_motion():
    priors = torch.zeros(1, 2, 8, dtype=torch.float32)
    priors[..., 0] = 0.20
    priors[..., 1] = torch.tensor([[0.03, 0.40]])
    priors[..., 2] = torch.tensor([[0.12, 0.0]])
    priors[..., 3] = 0.20
    # Deliberately make the approached point inconsistent with the static
    # hypothesis.  Approach/contact evidence must still make it foreground.
    priors[..., 4] = 1.0
    priors[..., 5] = 0.0
    priors[..., 6] = 0.5
    priors[..., 7] = 1.0

    pseudo = generate_pseudo_labels_from_priors(priors)

    assert pseudo["foreground_score"][0, 0] > 0.7
    assert pseudo["foreground_score"][0, 0] > pseudo["foreground_score"][0, 1]
    assert torch.equal(pseudo["labels"], torch.full_like(pseudo["labels"], ROLE_IGNORE))


def test_near_contact_soft_score_survives_terminal_future_padding():
    current_xyz = torch.tensor([[[0.05, 0.0, 0.0], [0.50, 0.0, 0.0]]], dtype=torch.float32)
    current_pc = torch.cat([current_xyz, torch.zeros(1, 2, 3)], dim=-1)
    # The context cloud is padded, reproducing the final frame of an episode.
    future_pc = current_pc[:, None].repeat(1, 2, 1, 1)
    future_poses = torch.stack(
        [_pose_from_translation([0.0, 0.0, 0.0]), _pose_from_translation([0.0, 0.0, 0.0])]
    ).unsqueeze(0)
    future_is_pad = torch.tensor([[False, True]])

    pseudo = generate_pseudo_labels(
        current_pc,
        future_pc,
        future_poses,
        future_is_pad,
        config=PseudoLabelConfig(nn_chunk_size=8),
    )

    assert pseudo["foreground_score"][0, 0] > 0.3
    assert pseudo["foreground_score"][0, 0] > pseudo["foreground_score"][0, 1]


def test_small_cloud_fallback_does_not_harden_soft_labels():
    role_scores = torch.tensor(
        [
            [
                [0.9, 0.1, 0.0],
                [0.1, 0.8, 0.2],
                [0.0, 0.2, 0.7],
            ]
        ],
        dtype=torch.float32,
    )
    pseudo = {
        "labels": torch.full((1, 3), ROLE_IGNORE, dtype=torch.long),
        "weights": torch.zeros(1, 3),
        "class_scores": torch.zeros(1, 3, 2),
        "role_scores": role_scores.clone(),
        "foreground_score": role_scores.amax(dim=-1),
    }
    current_pc = torch.zeros(1, 3, 6)

    out = force_small_current_clouds_foreground(pseudo, current_pc, configured_current_points=8)

    assert torch.equal(out["labels"], torch.full((1, 3), ROLE_IGNORE, dtype=torch.long))
    assert torch.equal(out["weights"], pseudo["weights"])
    assert torch.equal(out["foreground_score"], pseudo["foreground_score"])
    assert torch.allclose(out["role_scores"], role_scores)


class _FakeBaseDataset(torch.utils.data.Dataset):
    def __init__(self, episode_lengths):
        self.episode_lengths = episode_lengths
        self.index = []
        for episode_index, length in enumerate(episode_lengths):
            for frame_index in range(length):
                self.index.append((episode_index, frame_index))
        self.actions = []
        for episode_index, length in enumerate(episode_lengths):
            poses = []
            for frame_index in range(length):
                poses.append(torch.cat([_pose_from_translation([episode_index + frame_index * 0.1, 0.0, 0.0]), torch.zeros(1)]))
            self.actions.append(torch.stack(poses, dim=0))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        episode_index, frame_index = self.index[idx]
        poses = self.actions[episode_index]
        chunk = []
        for offset in range(3):
            chunk.append(poses[min(frame_index + offset, poses.shape[0] - 1)])
        return {
            "episode_index": torch.tensor(episode_index),
            "frame_index": torch.tensor(frame_index),
            "action": torch.stack(chunk, dim=0),
        }


def test_temporal_point_cloud_dataset_shapes_and_episode_clamping(tmp_path):
    point_cloud_dir = tmp_path / "point_clouds"
    point_cloud_dir.mkdir()
    for episode_index, length in enumerate([3, 2]):
        clouds = np.zeros((length, 10, 6), dtype=np.float32)
        for frame_index in range(length):
            clouds[frame_index, :, 0] = episode_index * 10 + frame_index
        np.save(point_cloud_dir / f"episode_{episode_index:06d}.npy", clouds)

    dataset = SongTemporalPointCloudDataset(
        _FakeBaseDataset([3, 2]),
        point_cloud_dir=point_cloud_dir,
        future_offsets=(1, 2),
        current_points=4,
        future_points=5,
        seed=123,
    )

    last_ep0 = dataset[2]
    assert tuple(last_ep0["observation.point_cloud"].shape) == (4, 6)
    assert last_ep0["pointseg_source_num_points"].item() == 10
    assert last_ep0["pointseg_sample_num_points"].item() == 4
    assert tuple(last_ep0["observation.point_cloud_future"].shape) == (5, 5, 6)
    assert tuple(last_ep0["future_ee_poses"].shape) == (5, 9)
    assert last_ep0["future_offsets"].tolist() == [0, -2, -1, 1, 2]
    assert last_ep0["future_is_pad"].tolist() == [False, False, False, True, True]
    assert torch.allclose(pose9_to_matrix(last_ep0["future_ee_poses"][0]), torch.eye(4), atol=1e-5)
    assert tuple(last_ep0["pointseg_trajectory_ee_poses"].shape) == (33, 9)
    assert tuple(last_ep0["pointseg_trajectory_offsets"].shape) == (33,)
    assert last_ep0["pointseg_trajectory_offsets"][0].item() == 0
    assert last_ep0["pointseg_trajectory_offsets"].min().item() == -2
    assert last_ep0["pointseg_trajectory_offsets"].max().item() == 0
    assert torch.allclose(
        pose9_to_matrix(last_ep0["pointseg_trajectory_ee_poses"][0]), torch.eye(4), atol=1e-5
    )

    first_ep1 = dataset[3]
    assert float(first_ep1["observation.point_cloud"][0, 0]) >= 10.0


def test_temporal_point_cloud_motion_prior_uses_achieved_state_not_action_target(tmp_path):
    point_cloud_dir = tmp_path / "point_clouds"
    point_cloud_dir.mkdir()
    np.save(
        point_cloud_dir / "episode_000000.npy",
        np.zeros((3, 4, 6), dtype=np.float32),
    )
    base = _FakeBaseDataset([3])
    base.observation_states = [
        torch.stack(
            [
                torch.cat([_pose_from_translation([x, 0.0, 0.0]), torch.zeros(1)])
                for x in (0.0, 0.1, 0.2)
            ]
        )
    ]
    base.actions = [
        torch.stack(
            [
                torch.cat([_pose_from_translation([x, 0.0, 0.0]), torch.zeros(1)])
                for x in (0.0, 1.0, 2.0)
            ]
        )
    ]
    dataset = SongTemporalPointCloudDataset(
        base,
        point_cloud_dir=point_cloud_dir,
        future_offsets=(1,),
        current_points=4,
        future_points=4,
    )

    relative = dataset._relative_temporal_poses(0, [0, 1])

    assert torch.allclose(relative[1, :3], torch.tensor([0.1, 0.0, 0.0]), atol=1e-6)


def test_full_episode_pose_trajectory_extends_target_evidence_without_changing_local_motion():
    # One gripper-connected rigid point stays fixed in the Ego frame while two
    # world-static targets shift by -0.10 m as the EEF advances.  This is the
    # minimum physically consistent fixture for a conditioned-tool sweep.
    current_xyz = torch.tensor(
        [[[0.05, 0.0, 0.0], [0.50, 0.0, 0.0], [0.80, 0.0, 0.0]]],
        dtype=torch.float32,
    )
    current_pc = torch.cat([current_xyz, torch.zeros(1, 3, 3)], dim=-1)
    future_xyz = torch.tensor(
        [
            [
                [[0.05, 0.0, 0.0], [0.50, 0.0, 0.0], [0.80, 0.0, 0.0]],
                [[0.05, 0.0, 0.0], [0.40, 0.0, 0.0], [0.70, 0.0, 0.0]],
            ]
        ],
        dtype=torch.float32,
    )
    future_pc = torch.cat([future_xyz, torch.zeros(1, 2, 3, 3)], dim=-1)
    future_poses = torch.stack(
        [_pose_from_translation([0.0, 0.0, 0.0]), _pose_from_translation([0.10, 0.0, 0.0])]
    ).unsqueeze(0)
    future_is_pad = torch.zeros(1, 2, dtype=torch.bool)
    trajectory_poses = torch.stack(
        [
            _pose_from_translation([0.0, 0.0, 0.0]),
            _pose_from_translation([0.25, 0.0, 0.0]),
            _pose_from_translation([0.50, 0.0, 0.0]),
        ]
    ).unsqueeze(0)

    local = generate_pseudo_labels(current_pc, future_pc, future_poses, future_is_pad)
    full = generate_pseudo_labels(
        current_pc,
        future_pc,
        future_poses,
        future_is_pad,
        trajectory_poses=trajectory_poses,
        trajectory_offsets=torch.tensor([[0, 1, 2]], dtype=torch.long),
    )

    assert torch.allclose(local["held_residual"], full["held_residual"])
    assert torch.allclose(local["static_residual"], full["static_residual"])
    assert torch.allclose(local["min_traj_dist"], full["min_traj_dist"])
    assert full["tool_sweep_dist"][0, 1] < local["tool_sweep_dist"][0, 1]
    assert full["foreground_score"][0, 1] > local["foreground_score"][0, 1]
    assert full["foreground_score"][0, 1] > full["foreground_score"][0, 2]


def test_cached_pointseg_dataset_reads_sharded_memmap(tmp_path):
    cache_dir = tmp_path / "cache"
    shard_dir = cache_dir / "shard_000000"
    shard_dir.mkdir(parents=True)
    num_samples = 2
    n_points = 4

    np.save(shard_dir / "point_cloud.npy", np.ones((num_samples, n_points, 6), dtype=np.float16))
    np.save(shard_dir / "priors.npy", np.ones((num_samples, n_points, 8), dtype=np.float16) * 2)
    np.save(shard_dir / "labels.npy", np.array([[0, 1, -100, 0], [1, 1, 0, -100]], dtype=np.int16))
    np.save(shard_dir / "weights.npy", np.ones((num_samples, n_points), dtype=np.float16))
    np.save(shard_dir / "class_scores.npy", np.ones((num_samples, n_points, 2), dtype=np.float16))
    np.save(shard_dir / "role_scores.npy", np.ones((num_samples, n_points, 3), dtype=np.float16))
    np.save(shard_dir / "foreground_score.npy", np.ones((num_samples, n_points), dtype=np.float16))
    np.save(shard_dir / "episode_index.npy", np.array([3, 4], dtype=np.int64))
    np.save(shard_dir / "frame_index.npy", np.array([10, 11], dtype=np.int64))
    np.save(shard_dir / "dataset_index.npy", np.array([0, 1], dtype=np.int64))
    with open(cache_dir / "manifest.json", "w") as f:
        json.dump(
            {
                "version": POINTSEG_CACHE_VERSION,
                "fields": list(POINTSEG_CACHE_FIELDS),
                "shards": [{"path": "shard_000000", "length": num_samples}],
            },
            f,
        )

    dataset = SongPointSegCachedDataset(cache_dir)
    sample = dataset[1]

    assert len(dataset) == num_samples
    assert tuple(sample["observation.point_cloud"].shape) == (n_points, 6)
    assert sample["observation.point_cloud"].dtype == torch.float32
    assert sample["pointseg.labels"].dtype == torch.int64
    assert sample["pointseg.labels"].tolist() == [1, 1, 0, -100]
    assert tuple(sample["pointseg.role_scores"].shape) == (n_points, 3)
    assert sample["episode_index"].item() == 4
    assert sample["dataset_index"].item() == 1

    # V51 writes schema 12, while the immutable primary-view and V46 caches
    # remain schema 11. Both contain the same label arrays and stay readable.
    legacy_manifest = json.loads((cache_dir / "manifest.json").read_text())
    legacy_manifest["version"] = 11
    (cache_dir / "manifest.json").write_text(json.dumps(legacy_manifest))
    legacy_sample = SongPointSegCachedDataset(cache_dir)[1]
    assert legacy_sample["pointseg.labels"].tolist() == [1, 1, 0, -100]


def test_cached_pointseg_dataset_reads_index_cache_role_scores(tmp_path):
    cache_dir = tmp_path / "cache"
    shard_dir = cache_dir / "shard_000000"
    shard_dir.mkdir(parents=True)
    n_points = 3
    np.save(shard_dir / "sample_offsets.npy", np.array([0, n_points], dtype=np.int64))
    np.save(shard_dir / "point_indices.npy", np.array([1, 3, 5], dtype=np.int64))
    np.save(shard_dir / "labels.npy", np.array([1, 0, -100], dtype=np.int16))
    np.save(shard_dir / "weights.npy", np.ones(n_points, dtype=np.float16))
    np.save(shard_dir / "class_scores.npy", np.ones((n_points, 2), dtype=np.float16))
    role_scores = np.arange(n_points * 3, dtype=np.float16).reshape(n_points, 3)
    np.save(shard_dir / "role_scores.npy", role_scores)
    np.save(shard_dir / "foreground_score.npy", np.ones(n_points, dtype=np.float16))
    np.save(shard_dir / "episode_index.npy", np.array([0], dtype=np.int64))
    np.save(shard_dir / "frame_index.npy", np.array([2], dtype=np.int64))
    np.save(shard_dir / "dataset_index.npy", np.array([7], dtype=np.int64))
    with open(cache_dir / "manifest.json", "w") as f:
        json.dump(
            {
                "version": POINTSEG_CACHE_VERSION,
                "fields": list(POINTSEG_CACHE_LABEL_FIELDS),
                "cache_mode": "indices",
                "variable_num_points": True,
                "shards": [{"path": "shard_000000", "length": 1}],
            },
            f,
        )

    sample = SongPointSegCachedDataset(cache_dir)[0]

    assert sample["observation.point_cloud_indices"].tolist() == [1, 3, 5]
    assert torch.allclose(sample["pointseg.role_scores"], torch.as_tensor(role_scores, dtype=torch.float32))


def test_song_pointseg_mlp_smoke_backward():
    bsize, n_points, future_points = 2, 64, 128
    current_pc = torch.rand(bsize, n_points, 6)
    current_pc[..., 3:] *= 255.0
    future_pc = torch.rand(bsize, 2, future_points, 6)
    future_pc[..., 3:] *= 255.0
    future_poses = torch.stack(
        [_pose_from_translation([0.0, 0.0, 0.0]), _pose_from_translation([0.1, 0.0, 0.0])],
        dim=0,
    )[None].repeat(bsize, 1, 1)
    future_is_pad = torch.zeros(bsize, 2, dtype=torch.bool)

    pseudo = generate_pseudo_labels(
        current_pc,
        future_pc,
        future_poses,
        future_is_pad,
        config=PseudoLabelConfig(
            nn_chunk_size=32,
            min_confidence=0.0,
            background_min_confidence=0.0,
            background_foreground_max=1.0,
            forced_foreground_min_score=0.0,
        ),
    )
    assert pseudo["labels"].shape == (bsize, n_points)
    assert torch.equal(pseudo["labels"], torch.full_like(pseudo["labels"], ROLE_IGNORE))
    assert torch.allclose(pseudo["class_scores"].sum(dim=-1), torch.ones(bsize, n_points), atol=1e-6)
    assert bool(((pseudo["foreground_score"] > 0) & (pseudo["foreground_score"] < 1)).any())

    model = SongPointSegNet(backbone_type="mlp", hidden_dim=32)
    outputs = model(current_pc, future_pc, future_poses, future_is_pad, priors=pseudo["priors"])
    assert outputs["role_logits"].shape == (bsize, n_points, 2)
    assert outputs["operation_prob"].shape == (bsize, n_points)

    criterion = SongPointSegLoss(SongPointSegLossConfig(smooth_voxel_size=0.2))
    loss, _metrics = criterion(outputs, pseudo, current_pc)
    loss.backward()
    assert any(param.grad is not None for param in model.parameters())


def test_teacher_refinement_preserves_soft_labels_and_never_hardens_targets():
    pseudo = {
        "labels": torch.tensor([[ROLE_FOREGROUND, ROLE_IGNORE, 0]], dtype=torch.long),
        "weights": torch.tensor([[0.7, 0.0, 0.1]], dtype=torch.float32),
        "foreground_score": torch.tensor([[0.6, 0.4, 0.01]], dtype=torch.float32),
        "class_scores": torch.tensor(
            [
                [
                    [0.1, 0.6],
                    [0.2, 0.4],
                    [0.9, 0.0],
                ]
            ],
            dtype=torch.float32,
        ),
    }
    teacher_logits = torch.tensor([[[8.0, 0.0], [8.0, 0.0], [8.0, 0.0]]])

    refined = refine_pseudo_labels_with_teacher(
        pseudo,
        teacher_logits,
        config=PseudoLabelConfig(teacher_confidence=0.8),
    )

    assert torch.equal(refined["labels"], pseudo["labels"])
    assert torch.allclose(refined["class_scores"].sum(dim=-1), torch.ones(1, 3), atol=1e-6)
    assert torch.all((refined["foreground_score"] >= 0) & (refined["foreground_score"] <= 1))
