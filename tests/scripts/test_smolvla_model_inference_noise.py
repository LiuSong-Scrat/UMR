from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from benchmarks.song_real_libero.scripts.smolvla_model_inference import SmolVLA_ModelInference


class _NoiseModel:
    def __init__(
        self,
        *,
        se3_enable: bool,
        pose9_enable: bool,
        worldflow_enable: bool = False,
        worldflow_noise_coupling: str = "independent",
    ) -> None:
        self.config = SimpleNamespace(
            chunk_size=3,
            max_action_dim=10,
            se3_enable=se3_enable,
            pose9_action_noise_enable=pose9_enable,
            worldflow_enable=worldflow_enable,
            worldflow_noise_coupling=worldflow_noise_coupling,
        )
        self.calls: list[str] = []

    def sample_noise(self, shape, device):
        self.calls.append("legacy")
        return torch.full(shape, 1.0, device=device)

    def sample_pose9_action_noise(self, shape, device):
        self.calls.append("pose9")
        return torch.full(shape, 2.0, device=device)

    def sample_se3_action_noise(self, actions):
        self.calls.append("se3")
        return None, None, torch.full_like(actions, 3.0)


def _inference_stub(model: _NoiseModel) -> SmolVLA_ModelInference:
    inference = object.__new__(SmolVLA_ModelInference)
    inference.policy = SimpleNamespace(
        model=model,
        prepare_state=lambda _batch: torch.zeros(1, 10),
    )
    return inference


def test_seeded_noise_uses_pose9_sampler_when_pose9_prior_is_enabled():
    model = _NoiseModel(se3_enable=False, pose9_enable=True)
    noise = _inference_stub(model)._make_seeded_action_noise({}, 7)

    assert model.calls == ["pose9"]
    assert torch.equal(noise, torch.full((1, 3, 10), 2.0))


def test_seeded_noise_keeps_se3_and_legacy_dispatch():
    se3_model = _NoiseModel(se3_enable=True, pose9_enable=True)
    se3_noise = _inference_stub(se3_model)._make_seeded_action_noise({}, 7)
    assert se3_model.calls == ["se3"]
    assert torch.equal(se3_noise, torch.full((1, 3, 10), 3.0))

    legacy_model = _NoiseModel(se3_enable=False, pose9_enable=False)
    legacy_noise = _inference_stub(legacy_model)._make_seeded_action_noise({}, 7)
    assert legacy_model.calls == ["legacy"]
    assert torch.equal(legacy_noise, torch.full((1, 3, 10), 1.0))


def test_zero_noise_is_valid_identity_pose9_for_pose_checkpoints():
    model = _NoiseModel(se3_enable=False, pose9_enable=True)
    noise = _inference_stub(model)._make_zero_action_noise({})

    expected = torch.zeros(1, 3, 10)
    expected[..., 3] = 1.0
    expected[..., 7] = 1.0
    assert torch.equal(noise, expected)


@pytest.mark.parametrize(
    "coupling",
    ["left_compose_ego", "conjugate_ego", "projected_ego_chart", "projected_ego_path"],
)
def test_seeded_world_noise_is_deferred_to_model_when_ego_coupling_is_enabled(coupling):
    model = _NoiseModel(
        se3_enable=False,
        pose9_enable=True,
        worldflow_enable=True,
        worldflow_noise_coupling=coupling,
    )

    world_noise = _inference_stub(model)._make_seeded_worldflow_noise({}, 7)

    assert world_noise is None


def test_worldflow_batch_rejects_missing_fixed_reference_pose():
    inference = object.__new__(SmolVLA_ModelInference)
    inference.device = torch.device("cpu")
    inference.camera_view_fusion = "legacy_budget"
    inference.policy = SimpleNamespace(
        config=SimpleNamespace(
            worldflow_enable=True,
            robot_state_feature=None,
            max_state_dim=10,
        )
    )
    observation = {
        "point_cloud": np.zeros((8, 6), dtype=np.float32),
        "state": np.asarray([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.02], dtype=np.float32),
    }

    with pytest.raises(ValueError, match="explicit 'worldflow.current_ee_pose'.*fixed world reference"):
        inference.build_model_batch(observation, state_pose_mode="identity")
