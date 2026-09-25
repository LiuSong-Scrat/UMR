from __future__ import annotations

from contextlib import nullcontext
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from benchmarks.song_real_libero.scripts.song_cache_pointseg_samples import (
    _fps_sample_cache_batch,
)
from benchmarks.song_real_libero.scripts.train_song_benchmark import (
    PointCloudMemmapDataset,
    PointSegCacheInjectedDataset,
    WorldFlowMemmapDataset as BenchmarkWorldFlowMemmapDataset,
    _paired_pointseg_cache_contract_mismatches,
    canonical_rgb_camera_name,
    make_policy_on_accelerator_device,
    update_policy,
    worldflow_ego_priority_projection_statistics,
)
from lerobot.policies.smolvla.song_pointseg import POINTSEG_CACHE_LABEL_FIELDS
from lerobot.scripts.train_song import WorldFlowMemmapDataset
from lerobot.scripts.train_song_libero import (
    WorldFlowMemmapDataset as LiberoWorldFlowMemmapDataset,
)


class _TinyDataset(torch.utils.data.Dataset):
    def __init__(self, frame_index: int = 1, chunk_size: int = 4) -> None:
        self.frame_index = frame_index
        self.chunk_size = chunk_size

    def __len__(self) -> int:
        return 1

    def __getitem__(self, index: int) -> dict:
        assert index == 0
        return {
            "episode_index": torch.tensor(0),
            "frame_index": torch.tensor(self.frame_index),
            "action": torch.zeros(self.chunk_size, 10),
        }


@pytest.mark.parametrize("fusion", ["fps", "voxel_fps"])
@pytest.mark.parametrize("single_view_points", [10_000, 50_000])
def test_cacheless_multiview_fps_training_keeps_native_single_view_point_count(
    tmp_path, monkeypatch, fusion, single_view_points
):
    gripper_points = 500
    fused_points = 2 * (single_view_points - gripper_points) + gripper_points
    source = torch.arange(fused_points * 6, dtype=torch.float32).reshape(fused_points, 6).numpy()
    observed = {}

    def fake_sampler(point_cloud, *, target_points, gripper_points, **kwargs):
        observed.update(
            target_points=target_points,
            gripper_points=gripper_points,
            kwargs=kwargs,
        )
        indices = torch.arange(target_points).unsqueeze(0)
        sampled = point_cloud[:, :target_points]
        return sampled, None, indices

    sampler_name = (
        "fps_sample_fused_point_cloud"
        if fusion == "fps"
        else "voxel_fps_sample_fused_point_cloud"
    )
    monkeypatch.setattr(
        f"benchmarks.song_real_libero.scripts.train_song_benchmark.{sampler_name}",
        fake_sampler,
    )
    wrapped = PointCloudMemmapDataset(
        _TinyDataset(),
        point_cloud_dir=tmp_path / "point_clouds",
        camera_views="agentview,robot0_eye_in_hand",
        camera_view_fusion=fusion,
        camera_view_voxel_size=0.007,
        gripper_points=gripper_points,
    )
    monkeypatch.setattr(wrapped, "_point_cloud_frame", lambda *_args: source)

    item = wrapped[0]

    assert item["observation.point_cloud"].shape == (1, single_view_points, 6)
    assert observed["target_points"] == single_view_points
    assert observed["gripper_points"] == gripper_points
    assert observed["kwargs"] == ({} if fusion == "fps" else {"voxel_size": 0.007})


def test_cacheless_single_view_is_not_resampled(tmp_path, monkeypatch):
    source = np.arange(50_000 * 6, dtype=np.float32).reshape(50_000, 6)
    wrapped = PointCloudMemmapDataset(
        _TinyDataset(),
        point_cloud_dir=tmp_path / "point_clouds",
        camera_views="agentview",
        camera_view_fusion="fps",
        gripper_points=500,
    )
    monkeypatch.setattr(wrapped, "_point_cloud_frame", lambda *_args: source)
    monkeypatch.setattr(
        "benchmarks.song_real_libero.scripts.train_song_benchmark.fps_sample_fused_point_cloud",
        lambda *_args, **_kwargs: pytest.fail("single-view input must not be resampled"),
    )

    item = wrapped[0]

    assert item["observation.point_cloud"].shape == (1, 50_000, 6)
    assert np.array_equal(item["observation.point_cloud"].squeeze(0).numpy(), source)


def test_policy_materialization_uses_explicit_local_cuda_then_restores_portable_config(monkeypatch):
    class _Config:
        device = "cuda"

    class _Policy:
        pass

    config = _Config()
    policy = _Policy()
    policy.config = config
    observed = {}

    def fake_make_policy(*, cfg, ds_meta, rename_map):
        observed["device_during_load"] = cfg.device
        observed["ds_meta"] = ds_meta
        observed["rename_map"] = rename_map
        return policy

    monkeypatch.setattr(
        "benchmarks.song_real_libero.scripts.train_song_benchmark.make_policy",
        fake_make_policy,
    )

    result = make_policy_on_accelerator_device(
        policy_cfg=config,
        ds_meta="metadata",
        rename_map={"old": "new"},
        accelerator_device=torch.device("cuda:3"),
    )

    assert result is policy
    assert observed == {
        "device_during_load": "cuda:3",
        "ds_meta": "metadata",
        "rename_map": {"old": "new"},
    }
    assert config.device == "cuda"
    assert policy.config.device == "cuda"


def _pose_sequence(offset: float) -> np.ndarray:
    poses = np.zeros((3, 9), dtype=np.float32)
    poses[:, 0] = np.arange(3, dtype=np.float32) + offset
    poses[:, 3] = 1.0
    poses[:, 7] = 1.0
    return poses


def test_training_rgb_camera_validation_uses_semantic_aliases() -> None:
    assert canonical_rgb_camera_name("observation.images.overhead") == "agentview"
    assert canonical_rgb_camera_name("overview") == "agentview"
    assert canonical_rgb_camera_name("external") == "agentview"
    assert canonical_rgb_camera_name("observation.images.hand") == "robot0_eye_in_hand"
    assert canonical_rgb_camera_name("wrist") == "robot0_eye_in_hand"
    assert canonical_rgb_camera_name("custom_camera") == "custom_camera"


def _write_robot_base_worldflow_meta(base_dir, target_dir) -> None:
    (base_dir / "meta.json").write_text(
        json.dumps({"coordinate_frame": "robot_base"}),
        encoding="utf-8",
    )
    (target_dir / "meta.json").write_text(
        json.dumps(
            {
                "coordinate_frame": "robot_base",
                "target_semantics": "commanded_eef_pose",
            }
        ),
        encoding="utf-8",
    )


def test_worldflow_dataset_uses_achieved_current_and_commanded_future_targets(tmp_path):
    achieved = _pose_sequence(offset=10.0)
    commanded = _pose_sequence(offset=20.0)
    achieved_dir = tmp_path / "world_ee_poses"
    target_dir = tmp_path / "action_target_ee_poses"
    achieved_dir.mkdir()
    target_dir.mkdir()
    np.save(achieved_dir / "episode_000000.npy", achieved)
    np.save(target_dir / "episode_000000.npy", commanded)

    wrapped = WorldFlowMemmapDataset(_TinyDataset(), tmp_path, chunk_size=4)
    item = wrapped[0]

    assert torch.equal(
        item["worldflow.current_ee_pose"],
        torch.from_numpy(achieved[1]),
    )
    assert torch.equal(
        item["worldflow.ee_poses"],
        torch.from_numpy(commanded[[1, 2, 2, 2]]),
    )
    assert torch.equal(
        item["worldflow.step_is_pad"],
        torch.tensor([False, False, True, True]),
    )


@pytest.mark.parametrize(
    "dataset_cls",
    [WorldFlowMemmapDataset, BenchmarkWorldFlowMemmapDataset, LiberoWorldFlowMemmapDataset],
)
def test_worldflow_dataset_applies_same_causal_action_offset_as_ego_chunk(tmp_path, dataset_cls):
    achieved = _pose_sequence(offset=10.0)
    commanded = _pose_sequence(offset=20.0)
    achieved_dir = tmp_path / "world_ee_poses"
    target_dir = tmp_path / "action_target_ee_poses"
    achieved_dir.mkdir()
    target_dir.mkdir()
    np.save(achieved_dir / "episode_000000.npy", achieved)
    np.save(target_dir / "episode_000000.npy", commanded)

    wrapped = dataset_cls(
        _TinyDataset(frame_index=0),
        tmp_path,
        chunk_size=4,
        action_start_offset=1,
    )
    item = wrapped[0]

    # Current observation remains frame 0; only Ego/World action targets shift.
    assert torch.equal(item["worldflow.current_ee_pose"], torch.from_numpy(achieved[0]))
    assert torch.equal(
        item["worldflow.ee_poses"],
        torch.from_numpy(commanded[[1, 2, 2, 2]]),
    )
    assert torch.equal(
        item["worldflow.step_is_pad"],
        torch.tensor([False, False, True, True]),
    )


def test_worldflow_dataset_rejects_mismatched_achieved_and_target_lengths(tmp_path):
    achieved_dir = tmp_path / "world_ee_poses"
    target_dir = tmp_path / "action_target_ee_poses"
    achieved_dir.mkdir()
    target_dir.mkdir()
    np.save(achieved_dir / "episode_000000.npy", _pose_sequence(offset=10.0))
    np.save(target_dir / "episode_000000.npy", _pose_sequence(offset=20.0)[:2])

    wrapped = WorldFlowMemmapDataset(_TinyDataset(), tmp_path, chunk_size=4)
    with pytest.raises(ValueError, match="achieved/target lengths differ"):
        wrapped[0]


def test_world_eef_trajectory_uses_robot_base_current_and_commanded_targets(tmp_path):
    base_poses = _pose_sequence(offset=30.0)
    base_targets = _pose_sequence(offset=40.0)
    base_dir = tmp_path / "world_base_ee_poses"
    target_dir = tmp_path / "world_base_action_target_ee_poses"
    base_dir.mkdir()
    target_dir.mkdir()
    _write_robot_base_worldflow_meta(base_dir, target_dir)
    np.save(base_dir / "episode_000000.npy", base_poses)
    np.save(target_dir / "episode_000000.npy", base_targets)

    wrapped = BenchmarkWorldFlowMemmapDataset(
        _TinyDataset(frame_index=1, chunk_size=4),
        tmp_path,
        chunk_size=4,
        target_type="world_eef_trajectory",
    )
    item = wrapped[0]

    assert torch.equal(item["worldflow.current_ee_pose"], torch.from_numpy(base_poses[1]))
    assert torch.equal(
        item["worldflow.eef_trajectory"],
        torch.from_numpy(base_targets[[1, 2, 2, 2]]),
    )
    assert "worldflow.ee_poses" not in item
    assert torch.equal(
        item["worldflow.step_is_pad"],
        torch.tensor([False, False, True, True]),
    )


def test_world_eef_trajectory_rejects_missing_base_target_sidecar(tmp_path):
    base_dir = tmp_path / "world_base_ee_poses"
    base_dir.mkdir()
    np.save(base_dir / "episode_000000.npy", _pose_sequence(offset=30.0))

    with pytest.raises(FileNotFoundError, match="fallbacks are forbidden"):
        BenchmarkWorldFlowMemmapDataset(
            _TinyDataset(),
            tmp_path,
            chunk_size=4,
            target_type="world_eef_trajectory",
        )


@pytest.mark.parametrize(
    "dataset_cls",
    [WorldFlowMemmapDataset, BenchmarkWorldFlowMemmapDataset, LiberoWorldFlowMemmapDataset],
)
def test_worldflow_dataset_can_require_command_target_sidecar(tmp_path, dataset_cls):
    achieved_dir = tmp_path / "world_ee_poses"
    achieved_dir.mkdir()
    np.save(achieved_dir / "episode_000000.npy", _pose_sequence(offset=10.0))

    with pytest.raises(FileNotFoundError, match="action_target_ee_poses"):
        dataset_cls(
            _TinyDataset(),
            tmp_path,
            chunk_size=4,
            require_action_target_sidecar=True,
        )


def _pointseg_manifest(points: int, *, nn_chunk_size: int = 512) -> dict:
    return {
        "version": 7,
        "num_samples": 20_744,
        "future_offsets": [1, 2, 4, 8, 16, 31],
        "temporal_offsets": [0, 1, 2, 4, 8, 16, 31],
        "trajectory_mode": "full_episode",
        "trajectory_offset_filtering": "future_only",
        "current_points": points,
        "future_points": points,
        "gripper_points": 500,
        "pseudo_label_policy": "soft_geometric_prior",
        "pseudo_label_config": {"held_sigma": 0.025, "nn_chunk_size": nn_chunk_size},
    }


def test_full_union_paired_cache_contract_allows_native_primary_point_count_and_compute_chunking():
    all_view = _pointseg_manifest(19_500, nn_chunk_size=512)
    primary = _pointseg_manifest(10_000, nn_chunk_size=1024)

    mismatches = _paired_pointseg_cache_contract_mismatches(
        all_view,
        primary,
        camera_view_fusion="full_union",
        num_views=2,
        gripper_points=500,
    )

    assert mismatches == {}


def test_paired_cache_contract_allows_compatible_storage_schema_versions():
    all_view = _pointseg_manifest(10_000)
    primary = _pointseg_manifest(10_000)
    all_view["version"] = 12
    primary["version"] = 11

    mismatches = _paired_pointseg_cache_contract_mismatches(
        all_view,
        primary,
        camera_view_fusion="multiscale_novelty_union",
        num_views=2,
        gripper_points=500,
    )

    assert mismatches == {}


def test_full_union_paired_cache_contract_rejects_point_loss_or_semantic_prior_drift():
    all_view = _pointseg_manifest(19_499, nn_chunk_size=512)
    primary = _pointseg_manifest(10_000, nn_chunk_size=1024)
    primary["pseudo_label_config"]["held_sigma"] = 0.03

    mismatches = _paired_pointseg_cache_contract_mismatches(
        all_view,
        primary,
        camera_view_fusion="full_union",
        num_views=2,
        gripper_points=500,
    )

    assert mismatches["current_points"] == (19_499, 10_000)
    assert mismatches["future_points"] == (19_499, 10_000)
    assert "pseudo_label_config" in mismatches


def test_worldflow_ego_priority_projection_preserves_aligned_gradients():
    ego = [torch.tensor([1.0, 2.0]), None]
    world = [torch.tensor([3.0, 4.0]), torch.tensor([5.0])]

    stats = worldflow_ego_priority_projection_statistics(ego, world)

    assert stats["dot"].item() == pytest.approx(11.0)
    assert stats["coefficient"].item() == pytest.approx(0.0)
    assert stats["conflict"].item() == pytest.approx(0.0)
    assert stats["overlap_parameter_count"] == 1


def test_worldflow_ego_priority_projection_removes_only_conflicting_component():
    ego = [torch.tensor([2.0, 0.0]), None]
    world = [torch.tensor([-3.0, 4.0]), torch.tensor([7.0])]

    stats = worldflow_ego_priority_projection_statistics(ego, world)
    coefficient = stats["coefficient"]
    assert torch.is_tensor(coefficient)
    projected_shared = world[0] - coefficient * ego[0]

    assert stats["dot"].item() == pytest.approx(-6.0)
    assert coefficient.item() == pytest.approx(-1.5)
    assert stats["conflict"].item() == pytest.approx(1.0)
    assert torch.dot(projected_shared, ego[0]).item() == pytest.approx(0.0)
    # A World-only tensor never enters the projection inner product.
    assert torch.equal(world[1], torch.tensor([7.0]))


class _GradientRecordingAccelerator:
    def autocast(self):
        return nullcontext()

    def unwrap_model(self, policy, keep_fp32_wrapper=True):
        assert keep_fp32_wrapper is True
        return policy

    def backward(self, loss, **kwargs):
        loss.backward(**kwargs)

    def clip_grad_norm_(self, parameters, max_norm):
        return torch.nn.utils.clip_grad_norm_(parameters, max_norm)


class _MixedPointRolePolicy(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.ego_point = torch.nn.Parameter(torch.tensor(1.0))
        self.world_point = torch.nn.Parameter(torch.tensor(1.0))
        self.config = SimpleNamespace(
            worldflow_training_ego_priority_gradient_projection=False,
            worldflow_training_shared_gradient_ego_tangent_projection=True,
        )
        self.model = SimpleNamespace(
            last_worldflow_world_to_ego_keep_mask=torch.tensor([False, True])
        )

    def get_worldflow_ego_tangent_world_only_parameter_ids(self) -> set[int]:
        return {id(self.world_point)}

    def forward(self, batch, reduction="mean"):
        assert batch == {}
        per_sample = torch.stack((self.ego_point.square(), self.world_point.square()))
        assert reduction == "none"
        return per_sample, {}


def test_ego_tangent_merge_retains_world_gradient_inside_mixed_point_lr_group():
    policy = _MixedPointRolePolicy()
    optimizer = torch.optim.SGD(
        [
            {
                "params": [policy.ego_point, policy.world_point],
                "lr": 5e-8,
                "group_name": "point_input_adaptation_path",
            }
        ]
    )
    captured = {}

    def record_gradients_without_updating(closure=None):
        assert closure is None
        captured["ego"] = policy.ego_point.grad.detach().clone()
        captured["world"] = policy.world_point.grad.detach().clone()

    optimizer.step = record_gradients_without_updating
    metrics = SimpleNamespace(loss=None, grad_norm=None, lr=None, update_s=None)

    _, output = update_policy(
        metrics,
        policy,
        {},
        optimizer,
        grad_clip_norm=0.0,
        accelerator=_GradientRecordingAccelerator(),
    )

    assert captured["ego"].item() == pytest.approx(1.0)
    assert captured["world"].item() == pytest.approx(1.0)
    assert output["worldflow_ego_tangent_gradient_world_only_point_parameter_count"] == 1
    # The optimizer step above only records gradients; the weights are unchanged.
    assert policy.ego_point.item() == pytest.approx(1.0)
    assert policy.world_point.item() == pytest.approx(1.0)


def test_cache_sampling_applies_same_conservative_scale_to_current_and_future():
    cloud = torch.zeros(1, 14, 6)
    cloud[0, :12, 0] = torch.tensor(
        [
            0.000,
            0.001,
            0.010,
            0.011,
            0.080,
            0.081,
            0.002,
            0.012,
            0.035,
            0.091,
            0.092,
            0.130,
        ]
    )
    cloud[0, -2:, :3] = 7.0
    batch = {
        "observation.point_cloud": cloud.clone(),
        "observation.point_cloud_indices": torch.arange(14).unsqueeze(0),
        "observation.point_cloud_future": cloud[:, None].clone(),
    }

    _fps_sample_cache_batch(
        batch,
        target_current_points=8,
        target_future_points=8,
        gripper_points=2,
        fusion="multiscale_novelty_union",
        voxel_size=0.01,
        coarse_novelty_scale=4.0,
    )

    expected_indices = [0, 11, 2, 3, 4, 5, 12, 13]
    assert batch["observation.point_cloud_indices"].tolist() == [expected_indices]
    assert torch.equal(
        batch["observation.point_cloud"], cloud[:, expected_indices]
    )
    assert torch.equal(
        batch["observation.point_cloud_future"],
        cloud[:, None, expected_indices],
    )


def test_multiscale_training_rejects_legacy_scale3_cache_for_scale4(tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    manifest = {
        "version": 12,
        "fields": list(POINTSEG_CACHE_LABEL_FIELDS),
        "cache_mode": "indices",
        "variable_num_points": True,
        "shards": [{"path": "shard_000000", "length": 1}],
        "camera_views": ["agentview", "robot0_eye_in_hand"],
        "camera_view_weights": None,
        "camera_view_fusion": "multiscale_novelty_union",
        "camera_view_voxel_size": 0.01,
    }
    (cache / "manifest.json").write_text(json.dumps(manifest))
    dataset = SimpleNamespace(root=tmp_path)

    with pytest.raises(ValueError, match="coarse novelty scale 3.0.*training scale 4.0"):
        PointSegCacheInjectedDataset(
            dataset,
            cache,
            strict=False,
            camera_views="agentview,robot0_eye_in_hand",
            camera_view_fusion="multiscale_novelty_union",
            camera_view_voxel_size=0.01,
            camera_view_coarse_novelty_scale=4.0,
        )

    manifest["camera_view_coarse_novelty_scale"] = 4.0
    (cache / "manifest.json").write_text(json.dumps(manifest))
    wrapped = PointSegCacheInjectedDataset(
        dataset,
        cache,
        strict=False,
        camera_views="agentview,robot0_eye_in_hand",
        camera_view_fusion="multiscale_novelty_union",
        camera_view_voxel_size=0.01,
        camera_view_coarse_novelty_scale=4.0,
    )
    assert wrapped.camera_view_coarse_novelty_scale == 4.0
