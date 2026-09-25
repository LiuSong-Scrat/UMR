#!/usr/bin/env python

from types import SimpleNamespace

import torch
from torch import nn

from lerobot.policies.smolvla.modeling_smolvla import (
    PointActionSelfAttention,
    VLAFlowMatching,
    make_att_2d_masks,
)


def _make_suffix_embedder(action_dim: int, hidden_dim: int, chunk_size: int) -> VLAFlowMatching:
    model = VLAFlowMatching.__new__(VLAFlowMatching)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(
        chunk_size=chunk_size,
        min_period=4e-3,
        max_period=4.0,
    )
    model.vlm_with_expert = SimpleNamespace(expert_hidden_size=hidden_dim)
    model.action_in_proj = nn.Linear(action_dim, hidden_dim)
    model.action_time_mlp_in = nn.Linear(hidden_dim * 2, hidden_dim)
    model.action_time_mlp_out = nn.Linear(hidden_dim, hidden_dim)
    return model


def test_action_suffix_is_one_bidirectional_block_and_respects_padding():
    chunk_size = 4
    model = _make_suffix_embedder(action_dim=3, hidden_dim=8, chunk_size=chunk_size)
    noisy_actions = torch.randn(1, chunk_size, 3)
    timestep = torch.tensor([0.5])
    actions_is_pad = torch.tensor([[False, False, True, True]])

    _, suffix_pad_mask, suffix_block_mask = model.embed_suffix(
        noisy_actions,
        timestep,
        actions_is_pad=actions_is_pad,
    )
    prefix_pad_mask = torch.tensor([[True, True]])
    prefix_block_mask = torch.tensor([[False, False]])
    full_pad_mask = torch.cat([prefix_pad_mask, suffix_pad_mask], dim=1)
    full_block_mask = torch.cat([prefix_block_mask, suffix_block_mask.bool()], dim=1)
    attention_mask = make_att_2d_masks(full_pad_mask, full_block_mask)

    assert torch.equal(suffix_pad_mask, ~actions_is_pad)
    assert torch.equal(suffix_block_mask.bool(), torch.tensor([[True, False, False, False]]))
    assert torch.equal(
        attention_mask,
        torch.tensor(
            [
                [
                    [True, True, False, False, False, False],
                    [True, True, False, False, False, False],
                    [True, True, True, True, False, False],
                    [True, True, True, True, False, False],
                    [False, False, False, False, False, False],
                    [False, False, False, False, False, False],
                ]
            ]
        ),
    )


def test_point_action_attention_uses_step_positions_and_isolates_padded_actions():
    torch.manual_seed(7)
    module = PointActionSelfAttention(
        action_dim=8,
        point_dim=8,
        max_action_steps=4,
        num_heads=2,
    ).eval()
    point_tokens = torch.randn(1, 5, 8)
    point_mask = torch.tensor([[True, True, True, False, False]])
    actions_is_pad = torch.tensor([[False, False, True, True]])
    action_tokens = torch.zeros(1, 4, 8)

    output = module(
        action_tokens,
        point_tokens,
        point_mask=point_mask,
        actions_is_pad=actions_is_pad,
    )

    # Equal action values at different valid steps remain distinguishable.
    assert not torch.allclose(output[:, 0], output[:, 1])
    # Padding tokens receive no PointAction residual update.
    assert torch.equal(output[:, 2:], action_tokens[:, 2:])

    changed_padding = action_tokens.clone()
    changed_padding[:, 2:] = 100.0
    changed_output = module(
        changed_padding,
        point_tokens,
        point_mask=point_mask,
        actions_is_pad=actions_is_pad,
    )

    # Padded action values cannot affect valid action outputs.
    assert torch.allclose(output[:, :2], changed_output[:, :2], atol=1e-6, rtol=1e-6)
    assert torch.equal(changed_output[:, 2:], changed_padding[:, 2:])
