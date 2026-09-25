#!/usr/bin/env python

from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file
from torch import nn

from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.smolvlm_with_expert import load_pretrained_action_expert_weights


class _DummyVLMWithExpert(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lm_expert = nn.Linear(3, 4)


class _DummyFlowMatching(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.vlm_with_expert = _DummyVLMWithExpert()
        self.action_time_mlp_in = nn.Linear(8, 4)
        self.action_time_mlp_out = nn.Linear(4, 4)
        self.action_in_proj = nn.Linear(2, 4)
        self.action_out_proj = nn.Linear(4, 2)


def _source_state_for(target: _DummyFlowMatching, source_action_dim: int = 5) -> dict[str, torch.Tensor]:
    state: dict[str, torch.Tensor] = {}
    value = 1.0
    for key, tensor in target.state_dict().items():
        if key.startswith("action_in_proj.") or key.startswith("action_out_proj."):
            continue
        count = tensor.numel()
        state[f"model.{key}"] = torch.arange(value, value + count).reshape(tensor.shape)
        value += count

    state["model.action_in_proj.weight"] = torch.arange(
        4 * source_action_dim, dtype=torch.float32
    ).reshape(4, source_action_dim)
    state["model.action_in_proj.bias"] = torch.arange(4, dtype=torch.float32) + 100
    state["model.action_out_proj.weight"] = torch.arange(
        source_action_dim * 4, dtype=torch.float32
    ).reshape(source_action_dim, 4)
    state["model.action_out_proj.bias"] = torch.arange(source_action_dim, dtype=torch.float32) + 200
    return state


def test_pretrained_action_expert_loads_exact_layers_and_slices_action_head(tmp_path: Path):
    target = _DummyFlowMatching()
    source_state = _source_state_for(target)
    checkpoint = tmp_path / "model.safetensors"
    save_file(source_state, checkpoint)

    report = load_pretrained_action_expert_weights(
        target,
        str(checkpoint),
        load_action_projections=True,
    )
    loaded = target.state_dict()

    assert report["expert_and_time_tensors"] == 6
    assert report["projection_tensors"] == 4
    assert report["total_tensors"] == 10
    for key in (
        "vlm_with_expert.lm_expert.weight",
        "vlm_with_expert.lm_expert.bias",
        "action_time_mlp_in.weight",
        "action_time_mlp_in.bias",
        "action_time_mlp_out.weight",
        "action_time_mlp_out.bias",
    ):
        assert torch.equal(loaded[key], source_state[f"model.{key}"])

    assert torch.equal(
        loaded["action_in_proj.weight"],
        source_state["model.action_in_proj.weight"][:, :2],
    )
    assert torch.equal(
        loaded["action_out_proj.weight"],
        source_state["model.action_out_proj.weight"][:2, :],
    )
    assert torch.equal(
        loaded["action_out_proj.bias"],
        source_state["model.action_out_proj.bias"][:2],
    )


def test_pretrained_action_expert_validates_before_mutating_target(tmp_path: Path):
    target = _DummyFlowMatching()
    for parameter in target.parameters():
        nn.init.zeros_(parameter)
    before = {key: value.clone() for key, value in target.state_dict().items()}

    source_state = _source_state_for(target)
    source_state.pop("model.action_time_mlp_out.bias")
    checkpoint = tmp_path / "incomplete.safetensors"
    save_file(source_state, checkpoint)

    with pytest.raises(RuntimeError, match="missing"):
        load_pretrained_action_expert_weights(target, str(checkpoint))

    for key, value in target.state_dict().items():
        assert torch.equal(value, before[key])


def test_pretrained_action_expert_leaves_task_specific_projections_unchanged_by_default(
    tmp_path: Path,
):
    target = _DummyFlowMatching()
    projections_before = {
        key: value.clone()
        for key, value in target.state_dict().items()
        if key.startswith(("action_in_proj.", "action_out_proj."))
    }
    checkpoint = tmp_path / "model.safetensors"
    save_file(_source_state_for(target), checkpoint)

    report = load_pretrained_action_expert_weights(target, str(checkpoint))

    assert report["projection_tensors"] == 0
    for key, expected in projections_before.items():
        assert torch.equal(target.state_dict()[key], expected)


def test_action_expert_initialization_requires_policy_source():
    with pytest.raises(ValueError, match="complete SmolVLA policy checkpoint"):
        SmolVLAConfig(load_action_expert_weights=True)
