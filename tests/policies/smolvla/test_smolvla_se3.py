#!/usr/bin/env python

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig


def test_action_chunk_start_offset_shifts_dataset_action_indices():
    cfg = SmolVLAConfig(chunk_size=4, n_action_steps=4, action_chunk_start_offset=1)

    assert cfg.action_delta_indices == [1, 2, 3, 4]
from lerobot.policies.smolvla.modeling_smolvla import (
    _left_compose_ego_pose9_chart_to_world,
    _worldflow_carrier_matrix,
    PointActionSelfAttention,
    SmolVLAPolicy,
    VLAFlowMatching,
    WorldFlowActionBranch,
    body_twist_to_pose9_velocity,
    conjugate_world_ego_flow_states_for_shared_expert,
    decoupled_base_pose_apply,
    decoupled_base_pose_flow_state,
    physically_aligned_world_ego_trajectory_proposals,
    make_att_2d_masks,
    matrix_to_pose9,
    pose9_endpoint_velocity_to_spatial_twist,
    pose9_velocity_to_spatial_twist,
    pose9_to_matrix,
    se3_exp,
    se3_geodesic_flow_state,
    se3_left_apply,
    se3_log,
    symmetric_world_ego_twist_fusion,
    so3_exp,
    so3_log,
    transform_se3_twist,
    world_eef_rigid_probe_losses,
    world_trajectory_to_current_ego_pose9,
    _ego_point_cloud_to_world,
)


def test_so3_and_se3_exp_log_round_trip():
    torch.manual_seed(7)
    omega = torch.randn(16, 3) * 0.25
    rot = so3_exp(omega)
    assert torch.allclose(so3_log(rot), omega, atol=2e-4, rtol=2e-4)

    xi = torch.randn(16, 6) * 0.20
    transform = se3_exp(xi)
    assert torch.allclose(se3_log(transform), xi, atol=3e-4, rtol=3e-4)


def test_se3_relative_update_recovers_target():
    torch.manual_seed(11)
    a = se3_exp(torch.randn(8, 6) * 0.2)
    b = se3_exp(torch.randn(8, 6) * 0.2)
    recovered = se3_exp(se3_log(a @ torch.linalg.inv(b))) @ b
    assert torch.allclose(recovered, a, atol=3e-4, rtol=3e-4)


def test_decoupled_base_pose_rotation_has_no_global_origin_translation():
    transform = torch.eye(4)
    transform[:3, 3] = torch.tensor([0.6, -0.2, 0.4])
    delta = torch.tensor([0.01, -0.02, 0.03, 0.0, 0.0, torch.pi / 2])

    actual = decoupled_base_pose_apply(delta, transform)
    legacy = se3_left_apply(delta, transform)

    assert torch.allclose(actual[:3, 3], torch.tensor([0.61, -0.22, 0.43]), atol=1e-6)
    assert not torch.allclose(actual[:3, 3], legacy[:3, 3], atol=1e-3)


def test_decoupled_base_pose_flow_reaches_endpoint_without_rate_amplification():
    torch.manual_seed(13)
    noise = se3_exp(torch.randn(2, 4, 6) * 0.2)
    target = se3_exp(torch.randn(2, 4, 6) * 0.2)
    time = torch.tensor([0.2, 0.85])

    state, velocity = decoupled_base_pose_flow_state(noise, target, time)
    remaining = (1.0 - time)[:, None, None]
    recovered = decoupled_base_pose_apply(remaining * velocity, state)

    assert torch.allclose(recovered, target, atol=4e-5, rtol=4e-5)
    assert torch.allclose(velocity[..., :3], target[..., :3, 3] - noise[..., :3, 3])


def test_world_eef_rigid_probe_loss_has_physical_metre_units_and_no_origin_coupling():
    state = torch.eye(4).reshape(1, 1, 4, 4)
    state[..., :3, 3] = torch.tensor([0.7, -0.4, 0.5])
    target_velocity = torch.zeros(1, 1, 6)
    pred_velocity = torch.tensor([[[0.02, 0.0, 0.0, 0.0, 0.0, 0.0]]])
    target_endpoint = state.clone()
    pred_endpoint = state.clone()
    pred_endpoint[..., 0, 3] += 0.02

    losses = world_eef_rigid_probe_losses(
        pred_velocity,
        target_velocity,
        state,
        pred_endpoint,
        target_endpoint,
        probe_radius_m=0.10,
    )

    assert torch.allclose(losses["flow"], torch.tensor([[0.02]]), atol=1e-7)
    assert torch.allclose(losses["endpoint"], torch.tensor([[0.02]]), atol=1e-7)
    assert torch.allclose(losses["flow_translation_m"], torch.tensor([[0.02]]), atol=1e-7)
    assert torch.count_nonzero(losses["flow_rotation_probe_m"]) == 0
    assert torch.count_nonzero(losses["endpoint_rotation_probe_m"]) == 0


def test_world_eef_rigid_probe_loss_supervises_rotation_without_moving_eef_origin():
    state = torch.eye(4).reshape(1, 1, 4, 4)
    state[..., :3, 3] = torch.tensor([0.8, 0.3, -0.2])
    pred_velocity = torch.zeros(1, 1, 6, requires_grad=True)
    target_velocity = torch.tensor([[[0.0, 0.0, 0.0, 0.0, 0.0, 0.10]]])
    pred_endpoint = state.clone()
    target_endpoint = state.clone()
    target_endpoint[..., :3, :3] = so3_exp(torch.tensor([[[0.0, 0.0, 0.10]]]))

    losses = world_eef_rigid_probe_losses(
        pred_velocity,
        target_velocity,
        state,
        pred_endpoint,
        target_endpoint,
        probe_radius_m=0.10,
    )
    (losses["flow"].mean() + losses["endpoint"].mean()).backward()

    assert torch.count_nonzero(losses["flow_translation_m"]) == 0
    assert torch.count_nonzero(losses["endpoint_translation_m"]) == 0
    assert losses["flow_rotation_probe_m"].item() > 0
    assert losses["endpoint_rotation_probe_m"].item() > 0
    assert torch.allclose(pred_endpoint[..., :3, 3], target_endpoint[..., :3, 3])
    assert pred_velocity.grad is not None
    assert pred_velocity.grad[..., 3:6].abs().sum().item() > 0


def test_twist_adjoint_matches_exact_se3_conjugation():
    torch.manual_seed(111)
    carrier = se3_exp(torch.randn(4, 1, 6) * 0.2)
    twist = torch.randn(4, 7, 6) * 0.1
    transformed = transform_se3_twist(twist, carrier)
    lhs = se3_exp(transformed)
    rhs = carrier @ se3_exp(twist) @ torch.linalg.inv(carrier)
    assert torch.allclose(lhs, rhs, atol=4e-5, rtol=4e-5)


def test_symmetric_world_ego_fusion_recovers_equal_conjugate_twists():
    torch.manual_seed(112)
    carrier = se3_exp(torch.randn(3, 1, 6) * 0.2)
    carrier_inv = torch.linalg.inv(carrier)
    ego_twist = torch.randn(3, 6, 6) * 0.1
    world_twist = transform_se3_twist(ego_twist, carrier)
    gripper = torch.randn(3, 6, 1)
    ego_velocity = torch.cat([ego_twist, gripper], dim=-1)
    fused = symmetric_world_ego_twist_fusion(ego_velocity, world_twist, carrier_inv)
    assert torch.allclose(fused, ego_velocity, atol=3e-6, rtol=3e-6)


def test_conjugate_residual_zero_init_preserves_ego_and_conjugates_world_twist():
    model = VLAFlowMatching.__new__(VLAFlowMatching)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(se3_enable=True, worldflow_action_fusion="conjugate_residual")
    model.world_twist_residual_out_proj = nn.Linear(8, 6)
    nn.init.zeros_(model.world_twist_residual_out_proj.weight)
    nn.init.zeros_(model.world_twist_residual_out_proj.bias)
    model._inject_point_action_features = lambda tokens: tokens

    ego_tokens = torch.randn(2, 4, 8)
    ego_mask = torch.ones(2, 4, dtype=torch.bool)
    ego_att = torch.zeros(2, 4)
    world_out = torch.randn(2, 4, 8)
    ego_velocity = torch.randn(2, 4, 7)
    model.embed_suffix = lambda *_args, **_kwargs: (ego_tokens, ego_mask, ego_att)
    model._run_world_ego_joint_expert = lambda *_args, **_kwargs: (ego_tokens, world_out)
    model._predict_ego_se3_velocity = lambda *_args, **_kwargs: ego_velocity

    class _WorldBranch:
        @staticmethod
        def embed_action_tokens(*_args, **_kwargs):
            return torch.randn(2, 4, 8), torch.ones(2, 4, dtype=torch.bool)

    model.worldflow_branch = _WorldBranch()
    carrier = se3_exp(torch.randn(2, 1, 6) * 0.2)
    carrier_inv = torch.linalg.inv(carrier)
    actual_ego, actual_world = model.denoise_step_world_ego(
        prefix_pad_masks=torch.ones(2, 3, dtype=torch.bool),
        past_key_values=object(),
        ego_x_t=torch.randn(2, 4, 10),
        world_x_t=torch.randn(2, 4, 9),
        timestep=torch.tensor([0.25, 0.75]),
        world_scene={},
        lang_tokens=torch.ones(2, 2, dtype=torch.long),
        lang_masks=torch.ones(2, 2, dtype=torch.bool),
        ego_to_world_transform=carrier,
        world_to_ego_transform=carrier_inv,
    )

    assert torch.equal(actual_ego, ego_velocity)
    assert torch.allclose(
        actual_world,
        transform_se3_twist(ego_velocity[..., :6], carrier),
        atol=3e-6,
        rtol=3e-6,
    )


def test_conjugate_residual_world_decoder_is_exact_and_jointly_trainable():
    torch.manual_seed(113)
    model = VLAFlowMatching.__new__(VLAFlowMatching)
    nn.Module.__init__(model)
    model.world_twist_residual_out_proj = nn.Linear(8, 6)

    ego_velocity = torch.randn(2, 4, 7, requires_grad=True)
    world_features = torch.randn(2, 4, 8, requires_grad=True)
    carrier = se3_exp(torch.randn(2, 1, 6) * 0.2)
    world_velocity, residual = model._compose_conjugate_residual_world_velocity(
        ego_velocity,
        world_features,
        carrier,
    )

    expected_residual = model.world_twist_residual_out_proj(world_features)
    expected_world = transform_se3_twist(ego_velocity[..., :6], carrier) + expected_residual
    assert torch.allclose(residual, expected_residual, atol=1e-7, rtol=1e-7)
    assert torch.allclose(world_velocity, expected_world, atol=1e-7, rtol=1e-7)

    world_velocity.square().mean().backward()
    assert ego_velocity.grad is not None and ego_velocity.grad.abs().sum() > 0
    assert world_features.grad is not None and world_features.grad.abs().sum() > 0
    assert model.world_twist_residual_out_proj.weight.grad.abs().sum() > 0


def test_conjugate_residual_consensus_is_exact_bidirectional_midpoint():
    torch.manual_seed(114)
    model = VLAFlowMatching.__new__(VLAFlowMatching)
    nn.Module.__init__(model)
    model.world_twist_residual_out_proj = nn.Linear(8, 6)

    ego_velocity = torch.randn(2, 4, 7, requires_grad=True)
    world_features = torch.randn(2, 4, 8, requires_grad=True)
    carrier = se3_exp(torch.randn(2, 1, 6) * 0.2)
    carrier_inv = torch.linalg.inv(carrier)
    corrected_ego, consensus_world, residual = model._compose_conjugate_residual_consensus(
        ego_velocity,
        world_features,
        carrier,
        carrier_inv,
    )

    expected_residual = model.world_twist_residual_out_proj(world_features)
    expected_ego_twist = ego_velocity[..., :6] + 0.5 * transform_se3_twist(
        expected_residual,
        carrier_inv,
    )
    assert torch.allclose(residual, expected_residual, atol=1e-7, rtol=1e-7)
    assert torch.allclose(corrected_ego[..., :6], expected_ego_twist, atol=1e-7, rtol=1e-7)
    assert torch.equal(corrected_ego[..., 6:], ego_velocity[..., 6:])
    assert torch.allclose(
        consensus_world,
        transform_se3_twist(corrected_ego[..., :6], carrier),
        atol=2e-6,
        rtol=2e-6,
    )
    assert torch.allclose(
        consensus_world,
        transform_se3_twist(ego_velocity[..., :6], carrier) + 0.5 * expected_residual,
        atol=2e-6,
        rtol=2e-6,
    )

    (corrected_ego.square().mean() + consensus_world.square().mean()).backward()
    assert ego_velocity.grad is not None and ego_velocity.grad.abs().sum() > 0
    assert world_features.grad is not None and world_features.grad.abs().sum() > 0
    assert model.world_twist_residual_out_proj.weight.grad.abs().sum() > 0


def test_pose9_endpoint_geodesic_consensus_is_exact_and_jointly_trainable():
    torch.manual_seed(1141)
    model = VLAFlowMatching.__new__(VLAFlowMatching)
    nn.Module.__init__(model)

    batch, steps = 3, 5
    time = torch.tensor([0.1, 0.35, 0.8])
    remaining = (1.0 - time)[:, None, None]
    ego_state_matrix = se3_exp(torch.randn(batch, steps, 6) * 0.12)
    ego_endpoint_matrix = se3_exp(torch.randn(batch, steps, 6) * 0.12)
    world_in_ego_endpoint_matrix = se3_exp(torch.randn(batch, steps, 6) * 0.12)
    carrier = se3_exp(torch.randn(batch, 1, 6) * 0.2)
    carrier_inv = torch.linalg.inv(carrier)

    ego_x_t = torch.cat(
        [matrix_to_pose9(ego_state_matrix), torch.randn(batch, steps, 1)],
        dim=-1,
    )
    ego_velocity = torch.cat(
        [
            (matrix_to_pose9(ego_endpoint_matrix) - ego_x_t[..., :9]) / remaining,
            torch.randn(batch, steps, 1),
        ],
        dim=-1,
    ).requires_grad_()
    world_state_matrix = carrier @ ego_state_matrix @ carrier_inv
    world_endpoint_matrix = carrier @ world_in_ego_endpoint_matrix @ carrier_inv
    world_x_t = matrix_to_pose9(world_state_matrix)
    world_velocity = (
        (matrix_to_pose9(world_endpoint_matrix) - world_x_t) / remaining
    ).requires_grad_()

    fused = model._compose_endpoint_geodesic_consensus(
        ego_x_t,
        ego_velocity,
        world_x_t,
        world_velocity,
        time,
        carrier_inv,
        carrier,
    )
    fused_endpoint = pose9_to_matrix(ego_x_t[..., :9] + remaining * fused[..., :9])
    expected_midpoint = (
        se3_exp(
            0.5
            * se3_log(world_in_ego_endpoint_matrix @ torch.linalg.inv(ego_endpoint_matrix))
        )
        @ ego_endpoint_matrix
    )
    assert torch.allclose(fused_endpoint, expected_midpoint, atol=5e-5, rtol=5e-5)
    assert torch.equal(fused[..., 9:], ego_velocity[..., 9:])

    fused.square().mean().backward()
    assert ego_velocity.grad is not None and ego_velocity.grad.abs().sum() > 0
    assert world_velocity.grad is not None and world_velocity.grad.abs().sum() > 0


def test_pose9_endpoint_consensus_requires_exact_projected_ego_path_contract():
    cfg = SmolVLAConfig(
        worldflow_enable=True,
        worldflow_noise_coupling="projected_ego_path",
        worldflow_action_fusion="endpoint_geodesic_consensus",
    )
    assert not cfg.se3_enable

    with pytest.raises(ValueError, match="requires the checkpoint-compatible legacy Ego chart"):
        SmolVLAConfig(
            worldflow_enable=True,
            worldflow_noise_coupling="independent",
            worldflow_action_fusion="endpoint_geodesic_consensus",
        )


def test_pose9_endpoint_residual_zero_is_exact_ego_and_nonzero_is_conjugate():
    torch.manual_seed(1143)
    model = VLAFlowMatching.__new__(VLAFlowMatching)
    nn.Module.__init__(model)
    model.world_twist_residual_out_proj = nn.Linear(8, 6)
    nn.init.zeros_(model.world_twist_residual_out_proj.weight)
    nn.init.zeros_(model.world_twist_residual_out_proj.bias)

    batch, steps = 2, 4
    time = torch.tensor([0.2, 0.7])
    remaining = (1.0 - time)[:, None, None]
    carrier = se3_exp(torch.randn(batch, 1, 6) * 0.15)
    carrier_inv = torch.linalg.inv(carrier)
    ego_state = se3_exp(torch.randn(batch, steps, 6) * 0.1)
    ego_x_t = torch.cat(
        [matrix_to_pose9(ego_state), torch.randn(batch, steps, 1)],
        dim=-1,
    )
    ego_velocity = torch.randn(batch, steps, 10, requires_grad=True)
    world_x_t = matrix_to_pose9(carrier @ ego_state @ carrier_inv)
    world_features = torch.randn(batch, steps, 8, requires_grad=True)

    exact_ego, zero_world_velocity, zero_residual = model._compose_endpoint_residual_boosting(
        ego_x_t,
        ego_velocity,
        world_x_t,
        world_features,
        time,
        carrier,
        carrier_inv,
    )
    assert torch.equal(exact_ego, ego_velocity)
    assert torch.count_nonzero(zero_residual) == 0
    ego_endpoint = pose9_to_matrix(ego_x_t[..., :9] + remaining * ego_velocity[..., :9])
    zero_world_endpoint = pose9_to_matrix(world_x_t + remaining * zero_world_velocity)
    assert torch.allclose(
        zero_world_endpoint,
        carrier @ ego_endpoint @ carrier_inv,
        atol=5e-5,
        rtol=5e-5,
    )

    nn.init.normal_(model.world_twist_residual_out_proj.weight, std=0.03)
    corrected_ego, corrected_world_velocity, residual = (
        model._compose_endpoint_residual_boosting(
            ego_x_t,
            ego_velocity,
            world_x_t,
            world_features,
            time,
            carrier,
            carrier_inv,
        )
    )
    corrected_ego_endpoint = pose9_to_matrix(
        ego_x_t[..., :9] + remaining * corrected_ego[..., :9]
    )
    corrected_world_endpoint = pose9_to_matrix(
        world_x_t + remaining * corrected_world_velocity
    )
    assert torch.allclose(
        corrected_world_endpoint,
        carrier @ corrected_ego_endpoint @ carrier_inv,
        atol=6e-5,
        rtol=6e-5,
    )
    assert torch.equal(corrected_ego[..., 9:], ego_velocity[..., 9:])

    corrected_ego.square().mean().backward()
    assert ego_velocity.grad is not None and ego_velocity.grad.abs().sum() > 0
    assert world_features.grad is not None and world_features.grad.abs().sum() > 0
    assert model.world_twist_residual_out_proj.weight.grad.abs().sum() > 0


def test_pose9_endpoint_residual_rate_has_terminal_boundary_and_bounded_velocity():
    model = VLAFlowMatching.__new__(VLAFlowMatching)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(worldflow_endpoint_residual_rate_parameterization=True)
    model.world_twist_residual_out_proj = nn.Linear(8, 6)
    nn.init.zeros_(model.world_twist_residual_out_proj.weight)
    with torch.no_grad():
        model.world_twist_residual_out_proj.bias.copy_(
            torch.tensor([0.02, -0.01, 0.03, 0.0, 0.0, 0.0])
        )

    batch, steps = 2, 3
    time = torch.tensor([0.2, 0.8])
    remaining = (1.0 - time)[:, None, None]
    identity = torch.eye(4).expand(batch, steps, -1, -1).clone()
    pose9 = matrix_to_pose9(identity)
    ego_x_t = torch.cat([pose9, torch.zeros(batch, steps, 1)], dim=-1)
    ego_velocity = torch.zeros_like(ego_x_t)
    world_x_t = pose9.clone()
    world_features = torch.zeros(batch, steps, 8)
    carrier = torch.eye(4).expand(batch, 1, -1, -1).clone()

    corrected, world_velocity, effective_residual = model._compose_endpoint_residual_boosting(
        ego_x_t,
        ego_velocity,
        world_x_t,
        world_features,
        time,
        carrier,
        carrier,
    )

    rate = model.world_twist_residual_out_proj.bias[:3]
    assert torch.allclose(
        effective_residual[..., :3],
        remaining * rate,
        atol=1e-7,
        rtol=1e-7,
    )
    assert torch.allclose(corrected[..., :3], rate.expand(batch, steps, -1), atol=2e-6, rtol=2e-6)
    assert torch.allclose(
        world_velocity[..., :3],
        rate.expand(batch, steps, -1),
        atol=2e-6,
        rtol=2e-6,
    )


def test_endpoint_residual_rate_parameterization_contract():
    cfg = SmolVLAConfig(
        worldflow_enable=True,
        worldflow_noise_coupling="projected_ego_path",
        worldflow_action_fusion="endpoint_residual_boosting",
        worldflow_endpoint_residual_rate_parameterization=True,
    )
    assert cfg.worldflow_endpoint_residual_rate_parameterization is True

    with pytest.raises(ValueError, match="requires worldflow_enable=True"):
        SmolVLAConfig(worldflow_endpoint_residual_rate_parameterization=True)
    with pytest.raises(ValueError, match="requires worldflow_action_fusion"):
        SmolVLAConfig(
            worldflow_enable=True,
            worldflow_endpoint_residual_rate_parameterization=True,
        )


def test_endpoint_residual_ego_frame_is_invariant_to_world_reparameterization():
    model = VLAFlowMatching.__new__(VLAFlowMatching)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(
        worldflow_endpoint_residual_rate_parameterization=True,
        worldflow_endpoint_residual_ego_frame_parameterization=True,
    )
    model.world_twist_residual_out_proj = nn.Linear(8, 6)
    nn.init.zeros_(model.world_twist_residual_out_proj.weight)
    with torch.no_grad():
        model.world_twist_residual_out_proj.bias.copy_(
            torch.tensor([0.012, -0.008, 0.004, 0.03, -0.02, 0.01])
        )

    batch, steps = 2, 3
    time = torch.tensor([0.2, 0.7])
    remaining = (1.0 - time)[:, None, None]
    ego_state = se3_exp(torch.randn(batch, steps, 6) * 0.08)
    ego_x_t = torch.cat(
        [matrix_to_pose9(ego_state), torch.zeros(batch, steps, 1)], dim=-1
    )
    ego_velocity = torch.randn_like(ego_x_t) * 0.03
    ego_endpoint = pose9_to_matrix(
        ego_x_t[..., :9] + remaining * ego_velocity[..., :9]
    )
    carrier = se3_exp(torch.randn(batch, 1, 6) * 0.1)
    coordinate_change = se3_exp(torch.randn(batch, 1, 6) * 0.15)
    changed_carrier = coordinate_change @ carrier
    carrier_inv = torch.linalg.inv(carrier)
    changed_carrier_inv = torch.linalg.inv(changed_carrier)
    world_x_t = matrix_to_pose9(carrier @ ego_state @ carrier_inv)
    changed_world_x_t = matrix_to_pose9(
        changed_carrier @ ego_state @ changed_carrier_inv
    )
    world_features = torch.randn(batch, steps, 8)

    corrected, world_velocity, residual = model._compose_endpoint_residual_boosting(
        ego_x_t,
        ego_velocity,
        world_x_t,
        world_features,
        time,
        carrier,
        carrier_inv,
    )
    changed_corrected, changed_world_velocity, changed_residual = (
        model._compose_endpoint_residual_boosting(
            ego_x_t,
            ego_velocity,
            changed_world_x_t,
            world_features,
            time,
            changed_carrier,
            changed_carrier_inv,
        )
    )
    assert torch.allclose(changed_residual, residual, atol=1e-7, rtol=1e-7)
    assert torch.allclose(changed_corrected, corrected, atol=5e-6, rtol=5e-6)

    corrected_world_endpoint = pose9_to_matrix(
        world_x_t + remaining * world_velocity
    )
    changed_corrected_world_endpoint = pose9_to_matrix(
        changed_world_x_t + remaining * changed_world_velocity
    )
    assert torch.allclose(
        changed_corrected_world_endpoint,
        coordinate_change @ corrected_world_endpoint @ torch.linalg.inv(coordinate_change),
        atol=8e-5,
        rtol=8e-5,
    )
    expected_ego_endpoint = se3_exp(residual.float()) @ ego_endpoint
    actual_ego_endpoint = pose9_to_matrix(
        ego_x_t[..., :9] + remaining * corrected[..., :9]
    )
    assert torch.allclose(actual_ego_endpoint, expected_ego_endpoint, atol=8e-5, rtol=8e-5)


def test_endpoint_residual_ego_frame_parameterization_contract():
    cfg = SmolVLAConfig(
        worldflow_enable=True,
        worldflow_noise_coupling="projected_ego_path",
        worldflow_action_fusion="endpoint_residual_boosting",
        worldflow_endpoint_residual_ego_frame_parameterization=True,
    )
    assert cfg.worldflow_endpoint_residual_ego_frame_parameterization is True

    with pytest.raises(ValueError, match="requires worldflow_enable=True"):
        SmolVLAConfig(worldflow_endpoint_residual_ego_frame_parameterization=True)
    with pytest.raises(ValueError, match="requires worldflow_action_fusion"):
        SmolVLAConfig(
            worldflow_enable=True,
            worldflow_endpoint_residual_ego_frame_parameterization=True,
        )


def test_endpoint_residual_body_frame_is_right_invariant_and_world_equivariant():
    model = VLAFlowMatching.__new__(VLAFlowMatching)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(
        worldflow_endpoint_residual_rate_parameterization=True,
        worldflow_endpoint_residual_ego_frame_parameterization=False,
        worldflow_endpoint_residual_body_frame_parameterization=True,
    )
    model.world_twist_residual_out_proj = nn.Linear(8, 6)
    nn.init.zeros_(model.world_twist_residual_out_proj.weight)
    with torch.no_grad():
        model.world_twist_residual_out_proj.bias.copy_(
            torch.tensor([0.012, -0.008, 0.004, 0.03, -0.02, 0.01])
        )

    batch, steps = 2, 3
    time = torch.tensor([0.2, 0.7])
    remaining = (1.0 - time)[:, None, None]
    ego_state = se3_exp(torch.randn(batch, steps, 6) * 0.08)
    ego_x_t = torch.cat(
        [matrix_to_pose9(ego_state), torch.zeros(batch, steps, 1)], dim=-1
    )
    ego_velocity = torch.randn_like(ego_x_t) * 0.03
    ego_endpoint = pose9_to_matrix(
        ego_x_t[..., :9] + remaining * ego_velocity[..., :9]
    )
    carrier = se3_exp(torch.randn(batch, 1, 6) * 0.1)
    coordinate_change = se3_exp(torch.randn(batch, 1, 6) * 0.15)
    changed_carrier = coordinate_change @ carrier
    carrier_inv = torch.linalg.inv(carrier)
    changed_carrier_inv = torch.linalg.inv(changed_carrier)
    world_x_t = matrix_to_pose9(carrier @ ego_state @ carrier_inv)
    changed_world_x_t = matrix_to_pose9(
        changed_carrier @ ego_state @ changed_carrier_inv
    )
    world_features = torch.randn(batch, steps, 8)

    corrected, world_velocity, residual = model._compose_endpoint_residual_boosting(
        ego_x_t,
        ego_velocity,
        world_x_t,
        world_features,
        time,
        carrier,
        carrier_inv,
    )
    changed_corrected, changed_world_velocity, changed_residual = (
        model._compose_endpoint_residual_boosting(
            ego_x_t,
            ego_velocity,
            changed_world_x_t,
            world_features,
            time,
            changed_carrier,
            changed_carrier_inv,
        )
    )
    assert torch.allclose(changed_residual, residual, atol=1e-7, rtol=1e-7)
    assert torch.allclose(changed_corrected, corrected, atol=5e-6, rtol=5e-6)

    corrected_world_endpoint = pose9_to_matrix(
        world_x_t + remaining * world_velocity
    )
    changed_corrected_world_endpoint = pose9_to_matrix(
        changed_world_x_t + remaining * changed_world_velocity
    )
    assert torch.allclose(
        changed_corrected_world_endpoint,
        coordinate_change @ corrected_world_endpoint @ torch.linalg.inv(coordinate_change),
        atol=8e-5,
        rtol=8e-5,
    )
    expected_ego_endpoint = ego_endpoint @ se3_exp(residual.float())
    actual_ego_endpoint = pose9_to_matrix(
        ego_x_t[..., :9] + remaining * corrected[..., :9]
    )
    assert torch.allclose(actual_ego_endpoint, expected_ego_endpoint, atol=8e-5, rtol=8e-5)


def test_endpoint_residual_body_frame_parameterization_contract():
    cfg = SmolVLAConfig(
        worldflow_enable=True,
        worldflow_noise_coupling="projected_ego_path",
        worldflow_action_fusion="endpoint_residual_boosting",
        worldflow_endpoint_residual_body_frame_parameterization=True,
    )
    assert cfg.worldflow_endpoint_residual_body_frame_parameterization is True

    with pytest.raises(ValueError, match="requires worldflow_enable=True"):
        SmolVLAConfig(worldflow_endpoint_residual_body_frame_parameterization=True)
    with pytest.raises(ValueError, match="requires worldflow_action_fusion"):
        SmolVLAConfig(
            worldflow_enable=True,
            worldflow_endpoint_residual_body_frame_parameterization=True,
        )
    with pytest.raises(ValueError, match="Select only one endpoint residual frame"):
        SmolVLAConfig(
            worldflow_enable=True,
            worldflow_noise_coupling="projected_ego_path",
            worldflow_action_fusion="endpoint_residual_boosting",
            worldflow_endpoint_residual_ego_frame_parameterization=True,
            worldflow_endpoint_residual_body_frame_parameterization=True,
        )


def test_body_twist_pose9_velocity_is_the_right_trivialized_tangent():
    torch.manual_seed(11431)
    batch, steps = 3, 5
    transform = se3_exp(torch.randn(batch, steps, 6) * 0.15)
    canonical = matrix_to_pose9(transform)
    basis_1 = canonical[..., 3:6]
    basis_2 = canonical[..., 6:9]
    scale_1 = torch.rand(batch, steps, 1) + 0.7
    scale_2 = torch.rand(batch, steps, 1) + 0.7
    shear = torch.randn(batch, steps, 1) * 0.25
    raw_pose9 = torch.cat(
        [
            canonical[..., :3],
            scale_1 * basis_1,
            shear * basis_1 + scale_2 * basis_2,
        ],
        dim=-1,
    )
    body_twist = torch.randn(batch, steps, 6) * 0.08
    velocity = body_twist_to_pose9_velocity(raw_pose9, body_twist)

    epsilon = 1e-4
    actual = pose9_to_matrix(raw_pose9 + epsilon * velocity)
    expected = pose9_to_matrix(raw_pose9) @ se3_exp(epsilon * body_twist)
    assert torch.allclose(actual, expected, atol=3e-5, rtol=3e-5)


def test_body_velocity_residual_is_exact_zero_safe_and_world_equivariant():
    torch.manual_seed(11432)
    model = VLAFlowMatching.__new__(VLAFlowMatching)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(
        worldflow_body_velocity_residual_parameterization=True,
        worldflow_endpoint_residual_rate_parameterization=False,
        worldflow_endpoint_residual_ego_frame_parameterization=False,
        worldflow_endpoint_residual_body_frame_parameterization=False,
    )
    model.world_twist_residual_out_proj = nn.Linear(8, 6)
    nn.init.zeros_(model.world_twist_residual_out_proj.weight)
    nn.init.zeros_(model.world_twist_residual_out_proj.bias)

    batch, steps = 2, 4
    time = torch.tensor([0.2, 0.65])
    remaining = (1.0 - time)[:, None, None]
    ego_state = se3_exp(torch.randn(batch, steps, 6) * 0.1)
    ego_x_t = torch.cat(
        [matrix_to_pose9(ego_state), torch.randn(batch, steps, 1)],
        dim=-1,
    )
    ego_velocity = torch.randn(batch, steps, 10, requires_grad=True)
    carrier = se3_exp(torch.randn(batch, 1, 6) * 0.1)
    carrier_inv = torch.linalg.inv(carrier)
    coordinate_change = se3_exp(torch.randn(batch, 1, 6) * 0.12)
    changed_carrier = coordinate_change @ carrier
    changed_carrier_inv = torch.linalg.inv(changed_carrier)
    world_x_t = matrix_to_pose9(carrier @ ego_state @ carrier_inv)
    changed_world_x_t = matrix_to_pose9(
        changed_carrier @ ego_state @ changed_carrier_inv
    )
    world_features = torch.randn(batch, steps, 8, requires_grad=True)

    zero_corrected, _, zero_residual = model._compose_endpoint_residual_boosting(
        ego_x_t,
        ego_velocity,
        world_x_t,
        world_features,
        time,
        carrier,
        carrier_inv,
    )
    assert torch.equal(zero_corrected, ego_velocity)
    assert torch.count_nonzero(zero_residual) == 0

    with torch.no_grad():
        nn.init.normal_(model.world_twist_residual_out_proj.weight, std=0.01)
        model.world_twist_residual_out_proj.bias.copy_(
            torch.tensor([0.012, -0.008, 0.004, 0.03, -0.02, 0.01])
        )
    corrected, world_velocity, residual = model._compose_endpoint_residual_boosting(
        ego_x_t,
        ego_velocity,
        world_x_t,
        world_features,
        time,
        carrier,
        carrier_inv,
    )
    changed_corrected, changed_world_velocity, changed_residual = (
        model._compose_endpoint_residual_boosting(
            ego_x_t,
            ego_velocity,
            changed_world_x_t,
            world_features,
            time,
            changed_carrier,
            changed_carrier_inv,
        )
    )
    expected_delta = body_twist_to_pose9_velocity(ego_x_t[..., :9], residual)
    assert torch.allclose(
        corrected[..., :9],
        ego_velocity[..., :9] + expected_delta,
        atol=2e-6,
        rtol=2e-6,
    )
    assert torch.equal(corrected[..., 9:], ego_velocity[..., 9:])
    assert torch.allclose(changed_residual, residual, atol=1e-7, rtol=1e-7)
    assert torch.allclose(changed_corrected, corrected, atol=3e-6, rtol=3e-6)

    corrected_world_endpoint = pose9_to_matrix(
        world_x_t + remaining * world_velocity
    )
    changed_world_endpoint = pose9_to_matrix(
        changed_world_x_t + remaining * changed_world_velocity
    )
    assert torch.allclose(
        changed_world_endpoint,
        coordinate_change
        @ corrected_world_endpoint
        @ torch.linalg.inv(coordinate_change),
        atol=8e-5,
        rtol=8e-5,
    )
    assert torch.allclose(
        corrected_world_endpoint,
        carrier
        @ pose9_to_matrix(ego_x_t[..., :9] + remaining * corrected[..., :9])
        @ carrier_inv,
        atol=8e-5,
        rtol=8e-5,
    )

    corrected.square().mean().backward()
    assert ego_velocity.grad is not None and ego_velocity.grad.abs().sum() > 0
    assert world_features.grad is not None and world_features.grad.abs().sum() > 0
    assert model.world_twist_residual_out_proj.weight.grad.abs().sum() > 0


def test_body_velocity_residual_keep_mask_and_anchor_gradient_routing():
    torch.manual_seed(11433)
    model = VLAFlowMatching.__new__(VLAFlowMatching)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(
        worldflow_body_velocity_residual_parameterization=True,
        worldflow_endpoint_residual_rate_parameterization=False,
        worldflow_endpoint_residual_ego_frame_parameterization=False,
        worldflow_endpoint_residual_body_frame_parameterization=False,
    )
    model.world_twist_residual_out_proj = nn.Linear(8, 6)

    batch, steps = 4, 3
    ego_state = se3_exp(torch.randn(batch, steps, 6) * 0.1)
    ego_x_t = torch.cat(
        [matrix_to_pose9(ego_state), torch.randn(batch, steps, 1)],
        dim=-1,
    )
    ego_velocity = torch.randn(batch, steps, 10, requires_grad=True)
    carrier = se3_exp(torch.randn(batch, 1, 6) * 0.1)
    carrier_inv = torch.linalg.inv(carrier)
    world_x_t = matrix_to_pose9(carrier @ ego_state @ carrier_inv)
    world_features = torch.randn(batch, steps, 8, requires_grad=True)
    time = torch.tensor([0.1, 0.3, 0.5, 0.7])
    keep = torch.tensor([True, False, True, False])

    ordinary, _, _ = model._compose_endpoint_residual_boosting(
        ego_x_t,
        ego_velocity,
        world_x_t,
        world_features,
        time,
        carrier,
        carrier_inv,
        world_to_ego_keep_mask=keep,
    )
    isolated, _, _ = model._compose_endpoint_residual_boosting(
        ego_x_t,
        ego_velocity,
        world_x_t,
        world_features,
        time,
        carrier,
        carrier_inv,
        world_to_ego_keep_mask=keep,
        detach_retained_ego_anchor=True,
    )
    assert torch.equal(isolated, ordinary)
    assert torch.equal(isolated[~keep], ego_velocity[~keep])

    isolated.square().sum().backward()
    assert torch.count_nonzero(ego_velocity.grad[keep]) == 0
    assert ego_velocity.grad[~keep].abs().sum() > 0
    assert world_features.grad[keep].abs().sum() > 0
    assert torch.count_nonzero(world_features.grad[~keep]) == 0
    assert model.world_twist_residual_out_proj.weight.grad.abs().sum() > 0


def test_body_velocity_residual_ablation_preserves_ego_but_trains_world():
    torch.manual_seed(11434)
    model = VLAFlowMatching.__new__(VLAFlowMatching)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(
        worldflow_body_velocity_residual_parameterization=True,
        worldflow_endpoint_residual_rate_parameterization=False,
        worldflow_endpoint_residual_ego_frame_parameterization=False,
        worldflow_endpoint_residual_body_frame_parameterization=False,
    )
    model.world_twist_residual_out_proj = nn.Linear(8, 6)

    batch, steps = 2, 3
    ego_state = se3_exp(torch.randn(batch, steps, 6) * 0.1)
    ego_x_t = torch.cat(
        [matrix_to_pose9(ego_state), torch.randn(batch, steps, 1)],
        dim=-1,
    )
    ego_velocity = torch.randn(batch, steps, 10, requires_grad=True)
    carrier = se3_exp(torch.randn(batch, 1, 6) * 0.1)
    carrier_inv = torch.linalg.inv(carrier)
    world_x_t = matrix_to_pose9(carrier @ ego_state @ carrier_inv)
    world_features = torch.randn(batch, steps, 8, requires_grad=True)

    ego_after, world_velocity, _ = model._compose_endpoint_residual_boosting(
        ego_x_t,
        ego_velocity,
        world_x_t,
        world_features,
        torch.tensor([0.25, 0.6]),
        carrier,
        carrier_inv,
        apply_world_to_ego=False,
        detach_ego_for_world_supervision=True,
    )
    assert torch.equal(ego_after, ego_velocity)
    assert world_velocity.shape == world_x_t.shape

    world_velocity.square().mean().backward()
    assert ego_velocity.grad is None or torch.count_nonzero(ego_velocity.grad) == 0
    assert world_features.grad is not None and world_features.grad.abs().sum() > 0
    assert model.world_twist_residual_out_proj.weight.grad.abs().sum() > 0


def test_body_velocity_residual_parameterization_contract():
    cfg = SmolVLAConfig(
        worldflow_enable=True,
        worldflow_noise_coupling="projected_ego_path",
        worldflow_action_fusion="endpoint_residual_boosting",
        worldflow_body_velocity_residual_parameterization=True,
    )
    assert cfg.worldflow_body_velocity_residual_parameterization is True

    with pytest.raises(ValueError, match="requires worldflow_enable=True"):
        SmolVLAConfig(worldflow_body_velocity_residual_parameterization=True)
    with pytest.raises(ValueError, match="requires worldflow_action_fusion"):
        SmolVLAConfig(
            worldflow_enable=True,
            worldflow_body_velocity_residual_parameterization=True,
        )
    for incompatible in (
        "worldflow_endpoint_residual_rate_parameterization",
        "worldflow_endpoint_residual_ego_frame_parameterization",
        "worldflow_endpoint_residual_body_frame_parameterization",
    ):
        with pytest.raises(ValueError, match="mutually exclusive"):
            SmolVLAConfig(
                worldflow_enable=True,
                worldflow_noise_coupling="projected_ego_path",
                worldflow_action_fusion="endpoint_residual_boosting",
                worldflow_body_velocity_residual_parameterization=True,
                **{incompatible: True},
            )


def test_pose9_endpoint_residual_ablation_preserves_ego_but_trains_world():
    torch.manual_seed(1144)
    model = VLAFlowMatching.__new__(VLAFlowMatching)
    nn.Module.__init__(model)
    model.world_twist_residual_out_proj = nn.Linear(8, 6)
    ego_x_t = torch.randn(1, 3, 10)
    ego_velocity = torch.randn(1, 3, 10, requires_grad=True)
    world_x_t = torch.randn(1, 3, 9)
    features = torch.randn(1, 3, 8, requires_grad=True)
    carrier = se3_exp(torch.randn(1, 1, 6) * 0.1)
    ego_after, world_velocity, _ = model._compose_endpoint_residual_boosting(
        ego_x_t,
        ego_velocity,
        world_x_t,
        features,
        torch.tensor([0.4]),
        carrier,
        torch.linalg.inv(carrier),
        apply_world_to_ego=False,
        detach_ego_for_world_supervision=True,
    )
    assert torch.equal(ego_after, ego_velocity)
    assert world_velocity.shape == world_x_t.shape
    world_velocity.square().mean().backward()
    assert ego_velocity.grad is None or torch.count_nonzero(ego_velocity.grad) == 0
    assert features.grad is not None and features.grad.abs().sum() > 0
    assert model.world_twist_residual_out_proj.weight.grad.abs().sum() > 0


def test_pose9_endpoint_residual_training_keep_mask_has_no_gate_or_inference_change():
    torch.manual_seed(1145)
    model = VLAFlowMatching.__new__(VLAFlowMatching)
    nn.Module.__init__(model)
    model.world_twist_residual_out_proj = nn.Linear(8, 6)

    batch, steps = 3, 4
    ego_x_t = torch.randn(batch, steps, 10)
    ego_velocity = torch.randn(batch, steps, 10)
    world_x_t = torch.randn(batch, steps, 9)
    features = torch.randn(batch, steps, 8)
    carrier = se3_exp(torch.randn(batch, 1, 6) * 0.1)
    carrier_inv = torch.linalg.inv(carrier)
    time = torch.tensor([0.2, 0.4, 0.6])

    full, _, _ = model._compose_endpoint_residual_boosting(
        ego_x_t,
        ego_velocity,
        world_x_t,
        features,
        time,
        carrier,
        carrier_inv,
    )
    keep = torch.tensor([True, False, True])
    mixed, _, _ = model._compose_endpoint_residual_boosting(
        ego_x_t,
        ego_velocity,
        world_x_t,
        features,
        time,
        carrier,
        carrier_inv,
        world_to_ego_keep_mask=keep,
    )
    assert torch.equal(mixed[keep], full[keep])
    assert torch.equal(mixed[~keep], ego_velocity[~keep])

    with pytest.raises(ValueError, match="one World-to-Ego keep decision"):
        model._compose_endpoint_residual_boosting(
            ego_x_t,
            ego_velocity,
            world_x_t,
            features,
            time,
            carrier,
            carrier_inv,
            world_to_ego_keep_mask=torch.ones(batch, 1, dtype=torch.bool),
        )


def test_pose9_endpoint_residual_stop_gradient_keeps_forward_and_splits_gradients():
    torch.manual_seed(1146)
    model = VLAFlowMatching.__new__(VLAFlowMatching)
    nn.Module.__init__(model)
    model.world_twist_residual_out_proj = nn.Linear(8, 6)

    batch, steps = 4, 3
    ego_x_t = torch.randn(batch, steps, 10)
    ego_velocity = torch.randn(batch, steps, 10, requires_grad=True)
    world_x_t = torch.randn(batch, steps, 9)
    features = torch.randn(batch, steps, 8, requires_grad=True)
    carrier = se3_exp(torch.randn(batch, 1, 6) * 0.1)
    carrier_inv = torch.linalg.inv(carrier)
    time = torch.tensor([0.1, 0.3, 0.5, 0.7])
    keep = torch.tensor([True, False, True, False])

    ordinary, _, _ = model._compose_endpoint_residual_boosting(
        ego_x_t,
        ego_velocity,
        world_x_t,
        features,
        time,
        carrier,
        carrier_inv,
        world_to_ego_keep_mask=keep,
    )
    isolated, _, _ = model._compose_endpoint_residual_boosting(
        ego_x_t,
        ego_velocity,
        world_x_t,
        features,
        time,
        carrier,
        carrier_inv,
        world_to_ego_keep_mask=keep,
        detach_retained_ego_anchor=True,
    )
    assert torch.equal(isolated, ordinary)

    isolated.square().sum().backward()
    assert torch.count_nonzero(ego_velocity.grad[keep]) == 0
    assert ego_velocity.grad[~keep].abs().sum() > 0
    assert features.grad[keep].abs().sum() > 0
    assert torch.count_nonzero(features.grad[~keep]) == 0
    assert model.world_twist_residual_out_proj.weight.grad.abs().sum() > 0

    with pytest.raises(ValueError, match="requires a per-sample World-to-Ego keep mask"):
        model._compose_endpoint_residual_boosting(
            ego_x_t,
            ego_velocity,
            world_x_t,
            features,
            time,
            carrier,
            carrier_inv,
            detach_retained_ego_anchor=True,
        )


def test_pose9_endpoint_residual_requires_projected_ego_path_contract():
    cfg = SmolVLAConfig(
        worldflow_enable=True,
        worldflow_noise_coupling="projected_ego_path",
        worldflow_action_fusion="endpoint_residual_boosting",
    )
    assert not cfg.se3_enable
    with pytest.raises(ValueError, match="requires the checkpoint-compatible legacy Ego chart"):
        SmolVLAConfig(
            worldflow_enable=True,
            worldflow_noise_coupling="independent",
            worldflow_action_fusion="endpoint_residual_boosting",
        )


def test_conjugate_residual_boosting_detaches_only_direct_ego_world_gradient():
    torch.manual_seed(115)
    model = VLAFlowMatching.__new__(VLAFlowMatching)
    nn.Module.__init__(model)
    model.world_twist_residual_out_proj = nn.Linear(8, 6)

    ego_velocity = torch.randn(2, 4, 7, requires_grad=True)
    world_features = torch.randn(2, 4, 8, requires_grad=True)
    carrier = se3_exp(torch.randn(2, 1, 6) * 0.2)
    carrier_inv = torch.linalg.inv(carrier)
    corrected_ego, supervised_world, residual = model._compose_conjugate_residual_consensus(
        ego_velocity,
        world_features,
        carrier,
        carrier_inv,
        detach_ego_for_world_supervision=True,
    )

    # Gradient routing changes, but the forward physical consensus does not.
    assert torch.allclose(
        supervised_world,
        transform_se3_twist(corrected_ego[..., :6], carrier),
        atol=2e-6,
        rtol=2e-6,
    )
    supervised_world.square().mean().backward(retain_graph=True)
    assert ego_velocity.grad is None or torch.count_nonzero(ego_velocity.grad) == 0
    assert world_features.grad is not None and world_features.grad.abs().sum() > 0
    assert model.world_twist_residual_out_proj.weight.grad.abs().sum() > 0

    # The main action objective still jointly trains Ego and the residual path.
    corrected_ego.square().mean().backward()
    assert ego_velocity.grad is not None and ego_velocity.grad.abs().sum() > 0
    assert world_features.grad.abs().sum() > 0


def test_world_to_ego_causal_ablation_removes_cross_attention_and_residual_correction():
    model = VLAFlowMatching.__new__(VLAFlowMatching)
    nn.Module.__init__(model)
    model.inference_ablation_modalities = frozenset({"world_to_ego"})
    model.ego_to_world_cross_norm = nn.LayerNorm(8)
    model.world_to_ego_cross_norm = nn.LayerNorm(8)
    model.ego_to_world_cross_attn = nn.MultiheadAttention(8, 2, batch_first=True)

    class _ForbiddenWorldToEgoAttention(nn.Module):
        def forward(self, *_args, **_kwargs):
            raise AssertionError("World-to-Ego attention must not run during its causal ablation.")

    model.world_to_ego_cross_attn = _ForbiddenWorldToEgoAttention()
    ego_out = torch.randn(2, 4, 8)
    world_out = torch.randn(2, 4, 8)
    ego_after, _world_after = model._bidirectional_world_ego_cross_attention(
        ego_out,
        world_out,
        torch.ones(2, 4, dtype=torch.bool),
        torch.ones(2, 4, dtype=torch.bool),
    )
    assert torch.equal(ego_after, ego_out)

    model.config = SimpleNamespace(se3_enable=True, worldflow_action_fusion="conjugate_residual")
    model.world_twist_residual_out_proj = nn.Linear(8, 6)
    model._inject_point_action_features = lambda tokens: tokens
    ego_tokens = torch.randn(2, 4, 8)
    ego_mask = torch.ones(2, 4, dtype=torch.bool)
    ego_att = torch.zeros(2, 4)
    ego_velocity = torch.randn(2, 4, 7)
    model.embed_suffix = lambda *_args, **_kwargs: (ego_tokens, ego_mask, ego_att)
    model._run_world_ego_joint_expert = lambda *_args, **_kwargs: (ego_tokens, world_out)
    model._predict_ego_se3_velocity = lambda *_args, **_kwargs: ego_velocity

    class _WorldBranch:
        @staticmethod
        def embed_action_tokens(*_args, **_kwargs):
            return torch.randn(2, 4, 8), torch.ones(2, 4, dtype=torch.bool)

    model.worldflow_branch = _WorldBranch()
    carrier = se3_exp(torch.randn(2, 1, 6) * 0.2)
    actual_ego, actual_world = model.denoise_step_world_ego(
        prefix_pad_masks=torch.ones(2, 3, dtype=torch.bool),
        past_key_values=object(),
        ego_x_t=torch.randn(2, 4, 10),
        world_x_t=torch.randn(2, 4, 9),
        timestep=torch.tensor([0.25, 0.75]),
        world_scene={},
        lang_tokens=torch.ones(2, 2, dtype=torch.long),
        lang_masks=torch.ones(2, 2, dtype=torch.bool),
        ego_to_world_transform=carrier,
        world_to_ego_transform=torch.linalg.inv(carrier),
    )
    assert torch.equal(actual_ego, ego_velocity)
    assert torch.allclose(
        actual_world,
        transform_se3_twist(ego_velocity[..., :6], carrier),
        atol=3e-6,
        rtol=3e-6,
    )


def test_world_pose9_bootstrap_copies_ego_modules_without_sharing_or_freezing():
    model = VLAFlowMatching.__new__(VLAFlowMatching)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(
        worldflow_se3_head_enable=False,
        worldflow_target_type="world_eef_trajectory",
        worldflow_world_eef_velocity_mode="base_pose9_euclidean",
    )
    model.action_in_proj = nn.Linear(12, 4)
    model.action_out_proj = nn.Linear(4, 12)
    model.action_time_mlp_in = nn.Linear(8, 4)
    model.action_time_mlp_out = nn.Linear(4, 4)
    model.ego_scene_to_expert = nn.Linear(3, 4)
    model.world_scene_to_expert = nn.Linear(3, 4)
    model.world_action_out_proj = nn.Linear(4, 9)
    model.world_se3_action_out_proj = None
    model.point_action_fusion = nn.Linear(3, 4)
    model.pointseg_conditioner = SimpleNamespace(
        foreground_encoder=nn.Linear(6, 3),
        background_encoder=nn.Linear(6, 3),
        null_background_feat=nn.Parameter(torch.randn(3)),
    )
    model.ego_to_world_cross_attn = nn.MultiheadAttention(4, 1, batch_first=True)
    model.world_to_ego_cross_attn = nn.MultiheadAttention(4, 1, batch_first=True)
    model.world_twist_residual_out_proj = None
    model.world_ego_scene_type_embedding = nn.Parameter(torch.randn(2, 4))
    model.world_ego_scene_role_embedding = nn.Parameter(torch.randn(2, 4))
    model.world_ego_action_type_embedding = nn.Parameter(torch.randn(2, 4))
    world = SimpleNamespace(
        scene_encoder=nn.Linear(6, 3),
        background_encoder=nn.Linear(6, 3),
        null_background_feat=nn.Parameter(torch.randn(3)),
        point_action_adapter=nn.Linear(3, 4),
        action_in_proj=nn.Linear(9, 4),
        action_time_mlp_in=nn.Linear(8, 4),
        action_time_mlp_out=nn.Linear(4, 4),
        scene_context_proj=nn.Linear(3, 4),
        language_embedding=nn.Embedding(11, 4),
        language_norm=nn.LayerNorm(4),
    )
    model.worldflow_branch = world
    model.vlm_with_expert = nn.Module()
    model.vlm_with_expert.lm_expert = nn.Linear(4, 4)
    model.world_lm_expert = nn.Linear(4, 4)

    with torch.no_grad():
        for ordinal, parameter in enumerate(model.parameters(), start=1):
            parameter.fill_(ordinal / 100.0)
    report = model.bootstrap_worldflow_from_ego()

    assert report["status"] == "bootstrapped"
    assert report["world_parameters_shared"] is False
    assert torch.equal(world.action_in_proj.weight, model.action_in_proj.weight[:, :9])
    assert world.action_in_proj.weight.data_ptr() != model.action_in_proj.weight.data_ptr()
    assert torch.equal(world.action_time_mlp_in.weight, model.action_time_mlp_in.weight)
    assert torch.equal(world.scene_encoder.weight, model.pointseg_conditioner.foreground_encoder.weight)
    assert torch.equal(world.point_action_adapter.weight, model.point_action_fusion.weight)
    assert torch.equal(model.world_action_out_proj.weight, model.action_out_proj.weight[:9])
    assert torch.equal(model.world_lm_expert.weight, model.vlm_with_expert.lm_expert.weight)
    assert model.world_lm_expert.weight.data_ptr() != model.vlm_with_expert.lm_expert.weight.data_ptr()
    assert report["world_action_expert"] == "independent_bootstrapped_copy"
    assert report["world_output_head"] == "copied_independent_base_pose9"
    assert torch.count_nonzero(world.language_embedding.weight) == 0
    assert torch.count_nonzero(model.ego_to_world_cross_attn.out_proj.weight) == 0
    assert torch.count_nonzero(model.world_to_ego_cross_attn.out_proj.weight) == 0
    assert all(parameter.requires_grad for parameter in model.parameters())


def test_direct_world_trajectory_bootstrap_keeps_independent_se3_head():
    model = VLAFlowMatching.__new__(VLAFlowMatching)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(
        worldflow_se3_head_enable=True,
        worldflow_target_type="world_eef_trajectory",
    )
    model.action_in_proj = nn.Linear(12, 4)
    model.action_out_proj = nn.Linear(4, 12)
    model.action_time_mlp_in = nn.Linear(8, 4)
    model.action_time_mlp_out = nn.Linear(4, 4)
    model.ego_scene_to_expert = nn.Linear(3, 4)
    model.world_scene_to_expert = nn.Linear(3, 4)
    model.world_action_out_proj = None
    model.world_se3_action_out_proj = nn.Linear(4, 6)
    model.point_action_fusion = nn.Linear(3, 4)
    model.pointseg_conditioner = SimpleNamespace(
        foreground_encoder=nn.Linear(6, 3),
        background_encoder=nn.Linear(6, 3),
        null_background_feat=nn.Parameter(torch.randn(3)),
    )
    model.ego_to_world_cross_attn = nn.MultiheadAttention(4, 1, batch_first=True)
    model.world_to_ego_cross_attn = nn.MultiheadAttention(4, 1, batch_first=True)
    model.world_twist_residual_out_proj = None
    model.world_ego_scene_type_embedding = nn.Parameter(torch.randn(2, 4))
    model.world_ego_scene_role_embedding = nn.Parameter(torch.randn(2, 4))
    model.world_ego_action_type_embedding = nn.Parameter(torch.randn(2, 4))
    world = SimpleNamespace(
        scene_encoder=nn.Linear(6, 3),
        background_encoder=nn.Linear(6, 3),
        null_background_feat=nn.Parameter(torch.randn(3)),
        point_action_adapter=nn.Linear(3, 4),
        action_in_proj=nn.Linear(9, 4),
        action_time_mlp_in=nn.Linear(8, 4),
        action_time_mlp_out=nn.Linear(4, 4),
        scene_context_proj=nn.Linear(3, 4),
        language_embedding=nn.Embedding(11, 4),
        language_norm=nn.LayerNorm(4),
    )
    model.worldflow_branch = world

    direct_head_weight = model.world_se3_action_out_proj.weight.detach().clone()
    direct_head_bias = model.world_se3_action_out_proj.bias.detach().clone()
    report = model.bootstrap_worldflow_from_ego()

    assert report["world_output_head"] == "independently_initialized_direct_se3"
    assert model.world_action_out_proj is None
    assert torch.equal(model.world_se3_action_out_proj.weight, direct_head_weight)
    assert torch.equal(model.world_se3_action_out_proj.bias, direct_head_bias)
    assert torch.equal(world.scene_encoder.weight, model.pointseg_conditioner.foreground_encoder.weight)
    assert world.scene_encoder.weight.data_ptr() != model.pointseg_conditioner.foreground_encoder.weight.data_ptr()
    assert torch.count_nonzero(model.world_to_ego_cross_attn.out_proj.weight) == 0
    assert all(parameter.requires_grad for parameter in model.parameters())


def test_dedicated_world_se3_head_decodes_twist_without_pose9_projection():
    model = VLAFlowMatching.__new__(VLAFlowMatching)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(
        worldflow_se3_head_enable=True,
        se3_twist_head_mode="pose9_chart_endpoint",
    )
    model.world_se3_action_out_proj = nn.Linear(4, 6)
    model.world_action_out_proj = nn.Linear(4, 9)
    features = torch.randn(2, 3, 4)
    expected = model.world_se3_action_out_proj(features)

    actual = model._predict_world_se3_velocity(
        features,
        torch.randn(2, 3, 9),
        torch.tensor([0.25, 0.75]),
    )

    assert torch.equal(actual, expected)


def test_se3_geodesic_flow_stays_on_group_and_exact_velocity_reaches_target():
    torch.manual_seed(12)
    noise = se3_exp(torch.randn(4, 7, 6) * 0.15)
    target = se3_exp(torch.randn(4, 7, 6) * 0.15)
    time = torch.tensor([0.0, 0.2, 0.6, 0.9])

    state, velocity = se3_geodesic_flow_state(noise, target, time)
    remaining = (1.0 - time)[:, None, None]
    recovered = se3_left_apply(remaining * velocity, state)

    rotation = state[..., :3, :3]
    identity = torch.eye(3).expand_as(rotation)
    assert torch.allclose(rotation.transpose(-1, -2) @ rotation, identity, atol=4e-5, rtol=4e-5)
    assert torch.allclose(torch.det(rotation), torch.ones_like(torch.det(rotation)), atol=4e-5, rtol=4e-5)
    assert torch.allclose(recovered, target, atol=5e-4, rtol=5e-4)


def test_pose9_velocity_projection_recovers_exact_spatial_twist():
    torch.manual_seed(121)
    transform = se3_exp(torch.randn(4, 7, 6) * 0.2)
    twist = torch.randn(4, 7, 6) * 0.15
    rotation = transform[..., :3, :3]
    position = transform[..., :3, 3]
    omega = twist[..., 3:6]
    x, y, z = omega.unbind(dim=-1)
    zeros = torch.zeros_like(x)
    omega_hat = torch.stack(
        [
            torch.stack([zeros, -z, y], dim=-1),
            torch.stack([z, zeros, -x], dim=-1),
            torch.stack([-y, x, zeros], dim=-1),
        ],
        dim=-2,
    )
    rotation_dot = omega_hat @ rotation
    position_dot = twist[..., :3] + torch.cross(omega, position, dim=-1)
    pose9_velocity = torch.cat(
        [position_dot, rotation_dot[..., :, 0], rotation_dot[..., :, 1]],
        dim=-1,
    )

    recovered = pose9_velocity_to_spatial_twist(
        matrix_to_pose9(transform),
        pose9_velocity,
    )

    assert torch.allclose(recovered, twist, atol=4e-5, rtol=4e-5)


def test_pose9_endpoint_velocity_recovers_projected_clean_endpoint():
    torch.manual_seed(122)
    current = matrix_to_pose9(se3_exp(torch.randn(4, 7, 6) * 0.15))
    legacy_velocity = torch.randn(4, 7, 9) * 0.08
    time = torch.tensor([0.0, 0.2, 0.6, 0.9])
    remaining = (1.0 - time)[:, None, None]
    expected_endpoint = pose9_to_matrix(current + remaining * legacy_velocity)

    twist = pose9_endpoint_velocity_to_spatial_twist(current, legacy_velocity, time)
    recovered_endpoint = se3_left_apply(remaining * twist, pose9_to_matrix(current))

    assert torch.allclose(recovered_endpoint, expected_endpoint, atol=6e-4, rtol=6e-4)


def test_pose9_chart_endpoint_preserves_legacy_endpoint_on_se3():
    torch.manual_seed(123)
    batch, chunk = 4, 7
    raw_chart = torch.randn(batch, chunk, 9) * 0.1
    clean_pose9 = matrix_to_pose9(se3_exp(torch.randn(batch, chunk, 6) * 0.15))
    time = torch.tensor([0.0, 0.2, 0.6, 0.9])
    remaining = (1.0 - time)[:, None, None]
    chart_state = remaining * raw_chart + (1.0 - remaining) * clean_pose9
    legacy_velocity = clean_pose9 - raw_chart
    group_state, _ = se3_geodesic_flow_state(
        pose9_to_matrix(raw_chart),
        pose9_to_matrix(clean_pose9),
        time,
    )

    twist = pose9_endpoint_velocity_to_spatial_twist(
        matrix_to_pose9(group_state),
        legacy_velocity,
        time,
        endpoint_base_pose9=chart_state,
    )
    recovered_endpoint = se3_left_apply(remaining * twist, group_state)

    assert torch.allclose(
        recovered_endpoint,
        pose9_to_matrix(clean_pose9),
        atol=6e-4,
        rtol=6e-4,
    )


def test_pose9_chart_se3_noise_matches_legacy_checkpoint_input_distribution():
    cfg = SmolVLAConfig(
        chunk_size=8,
        n_action_steps=8,
        se3_enable=True,
        se3_twist_head_mode="pose9_chart_endpoint",
        se3_noise_trans_scale=0.1,
        se3_noise_rot_scale=0.1,
        se3_noise_gripper_scale=0.1,
    )
    model = VLAFlowMatching.__new__(VLAFlowMatching)
    nn.Module.__init__(model)
    model.config = cfg
    actions = torch.zeros(3, cfg.chunk_size, 10)

    torch.manual_seed(124)
    expected_chart = torch.randn_like(actions) * 0.1
    torch.manual_seed(124)
    physical_noise, gripper_noise, chart_noise = model.sample_se3_action_noise(actions)

    assert torch.equal(chart_noise, expected_chart)
    assert torch.equal(gripper_noise, expected_chart[..., 9:10])
    assert torch.equal(physical_noise, pose9_to_matrix(expected_chart[..., :9]))


def test_conjugate_se3_geodesic_paths_match_at_every_time():
    torch.manual_seed(14)
    current = se3_exp(torch.randn(4, 6) * 0.15)
    ego_noise = se3_exp(torch.randn(4, 9, 6) * 0.12)
    ego_target = se3_exp(torch.randn(4, 9, 6) * 0.12)
    current_inv = torch.linalg.inv(current)
    world_noise = current.unsqueeze(1) @ ego_noise @ current_inv.unsqueeze(1)
    world_target = current.unsqueeze(1) @ ego_target @ current_inv.unsqueeze(1)
    time = torch.tensor([0.0, 0.25, 0.55, 0.9])

    ego_state, _ = se3_geodesic_flow_state(ego_noise, ego_target, time)
    world_state, _ = se3_geodesic_flow_state(world_noise, world_target, time)
    expected_world_state = current.unsqueeze(1) @ ego_state @ current_inv.unsqueeze(1)

    assert torch.allclose(world_state, expected_world_state, atol=6e-4, rtol=6e-4)


def test_worldflow_noise_respects_physical_translation_and_rotation_scales():
    torch.manual_seed(13)
    cfg = SmolVLAConfig(
        chunk_size=256,
        n_action_steps=16,
        worldflow_noise_trans_scale=0.07,
        worldflow_noise_rot_scale=0.11,
    )
    model = VLAFlowMatching.__new__(VLAFlowMatching)
    nn.Module.__init__(model)
    model.config = cfg

    pose9_noise = model.sample_worldflow_noise(batch_size=128, device=torch.device("cpu"))
    sampled_twist = se3_log(pose9_to_matrix(pose9_noise))

    assert torch.allclose(
        sampled_twist[..., :3].std(),
        torch.tensor(cfg.worldflow_noise_trans_scale),
        atol=3e-3,
        rtol=0.04,
    )
    assert torch.allclose(
        sampled_twist[..., 3:6].std(),
        torch.tensor(cfg.worldflow_noise_rot_scale),
        atol=3e-3,
        rtol=0.04,
    )


def test_ego_pose9_noise_is_valid_se3_and_respects_physical_scales():
    torch.manual_seed(17)
    cfg = SmolVLAConfig(
        chunk_size=256,
        n_action_steps=16,
        pose9_action_noise_enable=True,
        pose9_action_noise_trans_scale=0.08,
        pose9_action_noise_rot_scale=0.14,
        pose9_action_noise_gripper_scale=0.03,
    )
    model = VLAFlowMatching.__new__(VLAFlowMatching)
    nn.Module.__init__(model)
    model.config = cfg

    noise = model.sample_pose9_action_noise((128, cfg.chunk_size, 10), torch.device("cpu"))
    transform = pose9_to_matrix(noise[..., :9])
    sampled_twist = se3_log(transform)

    assert torch.allclose(
        sampled_twist[..., :3].std(),
        torch.tensor(cfg.pose9_action_noise_trans_scale),
        atol=3e-3,
        rtol=0.04,
    )
    assert torch.allclose(
        sampled_twist[..., 3:6].std(),
        torch.tensor(cfg.pose9_action_noise_rot_scale),
        atol=3e-3,
        rtol=0.04,
    )
    assert torch.allclose(
        noise[..., 9].std(),
        torch.tensor(cfg.pose9_action_noise_gripper_scale),
        atol=2e-3,
        rtol=0.04,
    )
    rotation = transform[..., :3, :3]
    identity = torch.eye(3).expand_as(rotation)
    assert torch.allclose(rotation.transpose(-1, -2) @ rotation, identity, atol=2e-5, rtol=2e-5)
    assert torch.allclose(torch.linalg.det(rotation), torch.ones_like(rotation[..., 0, 0]), atol=2e-5)


def test_zero_pose9_and_worldflow_noise_scales_produce_identity_priors():
    cfg = SmolVLAConfig(
        chunk_size=8,
        n_action_steps=8,
        pose9_action_noise_enable=True,
        pose9_action_noise_trans_scale=0.0,
        pose9_action_noise_rot_scale=0.0,
        pose9_action_noise_gripper_scale=0.0,
        worldflow_enable=True,
        worldflow_noise_trans_scale=0.0,
        worldflow_noise_rot_scale=0.0,
    )
    model = VLAFlowMatching.__new__(VLAFlowMatching)
    nn.Module.__init__(model)
    model.config = cfg

    ego_noise = model.sample_pose9_action_noise((3, cfg.chunk_size, 10), torch.device("cpu"))
    world_noise = model.sample_worldflow_noise(batch_size=3, device=torch.device("cpu"))
    identity = torch.eye(4).expand(3, cfg.chunk_size, 4, 4)

    assert torch.equal(pose9_to_matrix(ego_noise[..., :9]), identity)
    assert torch.count_nonzero(ego_noise[..., 9]) == 0
    assert torch.equal(pose9_to_matrix(world_noise), identity)


def test_conjugate_worldflow_noise_matches_ego_prior_exactly():
    torch.manual_seed(23)
    cfg = SmolVLAConfig(
        chunk_size=16,
        n_action_steps=8,
        pose9_action_noise_enable=True,
        worldflow_enable=True,
        worldflow_noise_coupling="conjugate_ego",
    )
    model = VLAFlowMatching.__new__(VLAFlowMatching)
    nn.Module.__init__(model)
    model.config = cfg

    ego_noise = model.sample_pose9_action_noise((5, cfg.chunk_size, 10), torch.device("cpu"))
    current = matrix_to_pose9(se3_exp(torch.randn(5, 6) * 0.2))
    world_noise = model.conjugate_ego_noise_to_world(ego_noise, current)

    current_matrix = _worldflow_carrier_matrix(current, cfg.worldflow_frame_origin)
    expected = (
        current_matrix.unsqueeze(1)
        @ pose9_to_matrix(ego_noise[..., :9])
        @ torch.linalg.inv(current_matrix).unsqueeze(1)
    )
    actual = pose9_to_matrix(world_noise)
    assert torch.allclose(actual, expected, atol=3e-5, rtol=3e-5)

    recovered_ego = (
        torch.linalg.inv(current_matrix).unsqueeze(1)
        @ actual
        @ current_matrix.unsqueeze(1)
    )
    assert torch.allclose(
        recovered_ego,
        pose9_to_matrix(ego_noise[..., :9]),
        atol=4e-5,
        rtol=4e-5,
    )


def test_projected_ego_chart_couples_world_without_changing_legacy_ego_noise():
    torch.manual_seed(230)
    cfg = SmolVLAConfig(
        chunk_size=16,
        n_action_steps=8,
        worldflow_enable=True,
        worldflow_noise_coupling="projected_ego_chart",
    )
    model = VLAFlowMatching.__new__(VLAFlowMatching)
    nn.Module.__init__(model)
    model.config = cfg

    rng_state = torch.random.get_rng_state()
    ego_noise = model.sample_noise((5, cfg.chunk_size, 10), torch.device("cpu"))
    torch.random.set_rng_state(rng_state)
    legacy_reference = model.sample_noise((5, cfg.chunk_size, 10), torch.device("cpu"))
    assert torch.equal(ego_noise, legacy_reference)

    current = matrix_to_pose9(se3_exp(torch.randn(5, 6) * 0.2))
    world_noise = model.conjugate_ego_noise_to_world(ego_noise, current)
    current_matrix = pose9_to_matrix(current)
    expected = (
        current_matrix.unsqueeze(1)
        @ pose9_to_matrix(ego_noise[..., :9])
        @ torch.linalg.inv(current_matrix).unsqueeze(1)
    )
    assert torch.allclose(pose9_to_matrix(world_noise), expected, atol=3e-5, rtol=3e-5)


def test_projected_ego_path_is_conjugate_at_every_time_and_reaches_endpoint():
    torch.manual_seed(2301)
    cfg = SmolVLAConfig(
        chunk_size=8,
        n_action_steps=8,
        worldflow_enable=True,
        worldflow_noise_coupling="projected_ego_path",
        worldflow_frame_origin="current_ee",
    )
    model = VLAFlowMatching.__new__(VLAFlowMatching)
    nn.Module.__init__(model)
    model.config = cfg

    ego_x_t = torch.randn(4, cfg.chunk_size, 10) * 0.1
    current = matrix_to_pose9(se3_exp(torch.randn(4, 6) * 0.2))
    spatial_target = matrix_to_pose9(se3_exp(torch.randn(4, cfg.chunk_size, 6) * 0.2))
    time = torch.tensor([0.0, 0.25, 0.5, 0.75])
    world_x_t, world_u_t = model.project_ego_chart_path_to_world(
        ego_x_t,
        current,
        spatial_target,
        time,
    )

    current_matrix = _worldflow_carrier_matrix(current, cfg.worldflow_frame_origin)
    expected_state = (
        current_matrix.unsqueeze(1)
        @ pose9_to_matrix(ego_x_t[..., :9])
        @ torch.linalg.inv(current_matrix).unsqueeze(1)
    )
    assert torch.allclose(pose9_to_matrix(world_x_t), expected_state, atol=4e-5, rtol=4e-5)
    remaining = (1.0 - time)[:, None, None]
    assert torch.allclose(
        world_x_t + remaining * world_u_t,
        spatial_target,
        atol=3e-6,
        rtol=3e-6,
    )


def test_projected_ego_path_augmentation_transforms_current_state_not_old_chord(monkeypatch):
    torch.manual_seed(2302)
    cfg = SmolVLAConfig(
        chunk_size=4,
        n_action_steps=4,
        worldflow_enable=True,
        worldflow_noise_coupling="projected_ego_path",
        worldflow_augmentation_trans_scale=0.2,
        worldflow_augmentation_rot_scale=0.3,
    )
    model = VLAFlowMatching.__new__(VLAFlowMatching)
    nn.Module.__init__(model)
    model.config = cfg

    class _WorldBranch:
        def __call__(self, *_args, **_kwargs):
            return {"action_tokens": torch.zeros(2, cfg.chunk_size, 8)}

    model.worldflow_branch = _WorldBranch()
    transform = se3_exp(torch.randn(2, 6) * 0.15)
    monkeypatch.setattr(
        "lerobot.policies.smolvla.modeling_smolvla._sample_random_se3",
        lambda *_args, **_kwargs: transform,
    )
    current = se3_exp(torch.randn(2, cfg.chunk_size, 6) * 0.12)
    target = se3_exp(torch.randn(2, cfg.chunk_size, 6) * 0.12)
    noise = se3_exp(torch.randn(2, cfg.chunk_size, 6) * 0.12)
    state = {
        "point_cloud_world": torch.randn(2, 20, 6),
        "point_is_pad": torch.zeros(2, 20, dtype=torch.bool),
        "spatial_gt": target,
        "noise_spatial": noise,
        "x_t": matrix_to_pose9(current),
        "valid": torch.ones(2, cfg.chunk_size, dtype=torch.bool),
    }
    time = torch.tensor([0.25, 0.6])
    augmented, actual_transform = model._augment_worldflow_training_state(
        state,
        torch.ones(2, 3, dtype=torch.long),
        torch.ones(2, 3, dtype=torch.bool),
        time,
    )
    transform_inv = torch.linalg.inv(transform)
    expected_current = transform[:, None] @ current @ transform_inv[:, None]
    expected_target = transform[:, None] @ target @ transform_inv[:, None]
    assert torch.equal(actual_transform, transform)
    assert torch.allclose(
        pose9_to_matrix(augmented["x_t"]),
        expected_current,
        atol=4e-5,
        rtol=4e-5,
    )
    remaining = (1.0 - time)[:, None, None]
    assert torch.allclose(
        augmented["x_t"] + remaining * augmented["u_t"],
        matrix_to_pose9(expected_target),
        atol=3e-6,
        rtol=3e-6,
    )


def test_training_coordinate_frame_reparameterization_preserves_physical_ego_path(monkeypatch):
    torch.manual_seed(2303)
    cfg = SmolVLAConfig(
        chunk_size=4,
        n_action_steps=4,
        pointseg_enable=True,
        worldflow_enable=True,
        worldflow_noise_coupling="projected_ego_path",
        worldflow_frame_origin="current_ee",
        worldflow_feature_dim=16,
        point_action_fusion_heads=4,
        worldflow_equiv_loss_weight=0.0,
        worldflow_max_points=8,
        worldflow_training_coordinate_frame_augmentation=True,
        worldflow_augmentation_trans_scale=0.05,
        worldflow_augmentation_rot_scale=0.2,
    )
    model = VLAFlowMatching.__new__(VLAFlowMatching)
    nn.Module.__init__(model)
    model.config = cfg
    model.worldflow_branch = _make_tiny_worldflow_branch(cfg)
    model.last_worldflow_metrics = {}
    model.train()

    bsize = 2
    foreground = torch.randn(bsize, 12, 6)
    foreground[..., 3:6] = torch.rand(bsize, 12, 3) * 255.0
    model.last_worldflow_payload = {"foreground_pc_ego": foreground}
    absolute_current = se3_exp(torch.randn(bsize, 6) * 0.1)
    body_target = se3_exp(torch.randn(bsize, cfg.chunk_size, 6) * 0.1)
    absolute_target = absolute_current.unsqueeze(1) @ body_target
    ego_noise_h = se3_exp(torch.randn(bsize, cfg.chunk_size, 6) * 0.1)
    ego_x_t_h = se3_exp(torch.randn(bsize, cfg.chunk_size, 6) * 0.1)
    ego_noise = torch.cat(
        [matrix_to_pose9(ego_noise_h), torch.zeros(bsize, cfg.chunk_size, 1)], dim=-1
    )
    ego_x_t = torch.cat(
        [matrix_to_pose9(ego_x_t_h), torch.zeros(bsize, cfg.chunk_size, 1)], dim=-1
    )
    current_pose = matrix_to_pose9(absolute_current)
    context = {
        "current_ee_pose": current_pose,
        "ee_poses": matrix_to_pose9(absolute_target),
        "step_is_pad": torch.zeros(bsize, cfg.chunk_size, dtype=torch.bool),
    }
    lang_tokens = torch.randint(0, 64, (bsize, 5))
    lang_masks = torch.ones(bsize, 5, dtype=torch.bool)
    time = torch.tensor([0.25, 0.6])
    coordinate_change = se3_exp(torch.randn(bsize, 6) * 0.15)
    calls = {"count": 0}

    def fixed_coordinate_change(*_args, **_kwargs):
        calls["count"] += 1
        return coordinate_change

    monkeypatch.setattr(
        "lerobot.policies.smolvla.modeling_smolvla._sample_random_se3",
        fixed_coordinate_change,
    )
    state = model._prepare_worldflow_training_state(
        context,
        lang_tokens,
        lang_masks,
        time,
        actions_is_pad=None,
        ego_noise=ego_noise,
        ego_x_t=ego_x_t,
    )

    canonical_carrier = _worldflow_carrier_matrix(current_pose, cfg.worldflow_frame_origin)
    expected_carrier = coordinate_change @ canonical_carrier
    expected_carrier_inv = torch.linalg.inv(expected_carrier)
    expected_spatial = expected_carrier[:, None] @ body_target @ expected_carrier_inv[:, None]
    expected_noise = expected_carrier[:, None] @ ego_noise_h @ expected_carrier_inv[:, None]
    expected_x_t = expected_carrier[:, None] @ ego_x_t_h @ expected_carrier_inv[:, None]
    canonical_points = _ego_point_cloud_to_world(
        foreground[:, : cfg.worldflow_max_points],
        current_pose,
        frame_origin=cfg.worldflow_frame_origin,
    )
    expected_xyz = (
        canonical_points[..., :3]
        @ coordinate_change[..., :3, :3].transpose(-1, -2)
        + coordinate_change[..., None, :3, 3]
    )

    assert calls["count"] == 1
    assert torch.equal(state["coordinate_frame_transform"], coordinate_change)
    assert torch.allclose(state["current"], expected_carrier, atol=3e-6, rtol=3e-6)
    assert torch.allclose(state["point_cloud_world"][..., :3], expected_xyz, atol=3e-6, rtol=3e-6)
    assert torch.equal(state["point_cloud_world"][..., 3:6], canonical_points[..., 3:6])
    assert torch.allclose(state["spatial_gt"], expected_spatial, atol=4e-5, rtol=4e-5)
    assert torch.allclose(state["noise_spatial"], expected_noise, atol=4e-5, rtol=4e-5)
    assert torch.allclose(pose9_to_matrix(state["x_t"]), expected_x_t, atol=4e-5, rtol=4e-5)
    recovered_body = state["current_inv"][:, None] @ state["spatial_gt"] @ state["current"][:, None]
    assert torch.allclose(recovered_body, body_target, atol=4e-5, rtol=4e-5)
    remaining = (1.0 - time)[:, None, None]
    assert torch.allclose(
        state["x_t"] + remaining * state["u_t"],
        matrix_to_pose9(expected_spatial),
        atol=3e-6,
        rtol=3e-6,
    )

    output = model._finalize_worldflow_training_loss(
        state,
        state["u_t"],
        matrix_to_pose9(body_target),
        time,
    )
    assert output["loss_flow"].item() < 1e-8
    assert model.last_worldflow_metrics["worldflow_coordinate_frame_augmentation_active"].item() == 1.0

    model.eval()
    canonical_state = model._prepare_worldflow_training_state(
        context,
        lang_tokens,
        lang_masks,
        time,
        actions_is_pad=None,
        ego_noise=ego_noise,
        ego_x_t=ego_x_t,
    )
    assert calls["count"] == 1
    assert "coordinate_frame_transform" not in canonical_state
    assert torch.equal(canonical_state["current"], canonical_carrier)


def test_projected_ego_chart_rejects_nonlegacy_ego_prior():
    with pytest.raises(ValueError, match="only for the legacy Euclidean Ego chart"):
        SmolVLAConfig(
            worldflow_enable=True,
            worldflow_noise_coupling="projected_ego_chart",
            pose9_action_noise_enable=True,
        )


def test_current_ee_worldflow_frame_removes_global_origin_lever_arm():
    torch.manual_seed(231)
    cfg = SmolVLAConfig(
        chunk_size=4,
        n_action_steps=4,
        pose9_action_noise_enable=True,
        worldflow_enable=True,
        worldflow_noise_coupling="conjugate_ego",
        worldflow_frame_origin="current_ee",
    )
    model = VLAFlowMatching.__new__(VLAFlowMatching)
    nn.Module.__init__(model)
    model.config = cfg

    body = se3_exp(torch.randn(2, cfg.chunk_size, 6) * 0.03)
    ego_noise = torch.cat(
        [matrix_to_pose9(body), torch.zeros(2, cfg.chunk_size, 1)],
        dim=-1,
    )
    current = se3_exp(torch.randn(2, 6) * 0.2)
    current[..., :3, 3] += torch.tensor([4.0, -3.0, 2.0])
    current_pose9 = matrix_to_pose9(current)

    local_world = pose9_to_matrix(model.conjugate_ego_noise_to_world(ego_noise, current_pose9))
    carrier = current.clone()
    carrier[..., :3, 3] = 0.0
    expected = carrier.unsqueeze(1) @ body @ torch.linalg.inv(carrier).unsqueeze(1)
    recovered = torch.linalg.inv(carrier).unsqueeze(1) @ local_world @ carrier.unsqueeze(1)

    assert torch.allclose(local_world, expected, atol=4e-5, rtol=4e-5)
    assert torch.allclose(recovered, body, atol=4e-5, rtol=4e-5)
    assert torch.allclose(
        torch.linalg.norm(local_world[..., :3, 3], dim=-1),
        torch.linalg.norm(body[..., :3, 3], dim=-1),
        atol=4e-5,
        rtol=4e-5,
    )

    points = torch.tensor([[[0.1, 0.0, 0.0, 1.0, 0.0, 0.0]]]).expand(2, -1, -1)
    centered = _ego_point_cloud_to_world(points, current_pose9, frame_origin="current_ee")
    global_points = _ego_point_cloud_to_world(points, current_pose9, frame_origin="global")
    assert torch.allclose(
        centered[..., :3],
        global_points[..., :3] - current[..., :3, 3].unsqueeze(1),
        atol=4e-5,
        rtol=4e-5,
    )


def test_worldflow_and_complete_se3_flow_can_be_enabled_together():
    cfg = SmolVLAConfig(
        se3_enable=True,
        worldflow_enable=True,
        worldflow_noise_coupling="conjugate_ego",
    )

    assert cfg.se3_enable
    assert cfg.worldflow_enable
    assert "ego_flow=se3_geodesic_left_trivialized" in cfg.flow_contract_summary()
    assert "worldflow=conjugate_ego" in cfg.flow_contract_summary()


def test_worldflow_robot_base_reference_contract():
    cfg = SmolVLAConfig(
        worldflow_enable=True,
        worldflow_reference_frame="robot_base",
        point_action_fusion_enable=True,
    )
    assert cfg.worldflow_reference_frame == "robot_base"

    with pytest.raises(ValueError, match="worldflow_reference_frame"):
        SmolVLAConfig(
            worldflow_enable=True,
            worldflow_reference_frame="current_camera_guess",
            point_action_fusion_enable=True,
        )


def test_world_eef_trajectory_requires_independent_token_only_contract():
    cfg = SmolVLAConfig(
        worldflow_enable=True,
        worldflow_target_type="world_eef_trajectory",
        worldflow_reference_frame="robot_base",
        worldflow_frame_origin="global",
        worldflow_scene_frame_origin="global",
        worldflow_noise_coupling="independent",
        worldflow_action_fusion="cross_attention",
        worldflow_bridge_loss_weight=0.0,
        worldflow_equiv_loss_weight=0.0,
    )
    assert "worldflow_targets=world_eef_trajectory" in cfg.flow_contract_summary()
    assert "worldflow_reference=robot_base" in cfg.flow_contract_summary()

    shared_expert_cfg = SmolVLAConfig(
        worldflow_enable=True,
        worldflow_target_type="world_eef_trajectory",
        worldflow_reference_frame="robot_base",
        worldflow_frame_origin="global",
        worldflow_scene_frame_origin="global",
        worldflow_noise_coupling="left_compose_ego",
        worldflow_world_eef_velocity_mode="base_pose9_euclidean",
        worldflow_action_fusion="point_action_expert_conjugate_bridge",
        worldflow_action_expert_mode="shared",
        worldflow_current_ee_pose_token=False,
        worldflow_bridge_loss_weight=0.0,
        worldflow_equiv_loss_weight=0.0,
    )
    assert shared_expert_cfg.worldflow_action_expert_mode == "shared"

    with pytest.raises(ValueError, match="one shared Action Expert"):
        SmolVLAConfig(
            worldflow_enable=True,
            worldflow_target_type="world_eef_trajectory",
            worldflow_reference_frame="robot_base",
            worldflow_frame_origin="global",
            worldflow_scene_frame_origin="global",
            worldflow_noise_coupling="left_compose_ego",
            worldflow_world_eef_velocity_mode="base_pose9_euclidean",
            worldflow_action_fusion="point_action_expert_conjugate_bridge",
            worldflow_action_expert_mode="independent",
            worldflow_bridge_loss_weight=0.0,
            worldflow_equiv_loss_weight=0.0,
        )

    physically_coupled_cfg = SmolVLAConfig(
        worldflow_enable=True,
        worldflow_target_type="world_eef_trajectory",
        worldflow_reference_frame="robot_base",
        worldflow_frame_origin="global",
        worldflow_scene_frame_origin="global",
        worldflow_noise_coupling="left_compose_ego",
        worldflow_world_eef_velocity_mode="base_decoupled",
        worldflow_action_fusion="physical_trajectory_cross_attention",
        worldflow_action_expert_mode="independent",
        worldflow_current_ee_pose_token=True,
        worldflow_bridge_loss_weight=0.0,
        worldflow_equiv_loss_weight=0.0,
    )
    assert physically_coupled_cfg.worldflow_noise_coupling == "left_compose_ego"
    assert physically_coupled_cfg.worldflow_world_eef_velocity_mode == "base_decoupled"

    calibration_cfg = SmolVLAConfig(
        worldflow_enable=True,
        worldflow_target_type="world_eef_trajectory",
        worldflow_reference_frame="robot_base",
        worldflow_frame_origin="global",
        worldflow_scene_frame_origin="global",
        worldflow_noise_coupling="left_compose_ego",
        worldflow_world_eef_velocity_mode="base_decoupled",
        worldflow_action_fusion="independent_parallel",
        worldflow_action_expert_mode="independent",
        worldflow_current_ee_pose_token=True,
        worldflow_freeze_pretrained_ego=False,
        worldflow_bridge_loss_weight=0.0,
        worldflow_equiv_loss_weight=0.0,
    )
    assert calibration_cfg.worldflow_action_fusion == "independent_parallel"

    with pytest.raises(ValueError, match="fully trainable action flows"):
        SmolVLAConfig(
            worldflow_enable=True,
            worldflow_target_type="world_eef_trajectory",
            worldflow_reference_frame="robot_base",
            worldflow_frame_origin="global",
            worldflow_scene_frame_origin="global",
            worldflow_noise_coupling="left_compose_ego",
            worldflow_world_eef_velocity_mode="base_decoupled",
            worldflow_action_fusion="independent_parallel",
            worldflow_action_expert_mode="independent",
            worldflow_current_ee_pose_token=True,
            worldflow_freeze_pretrained_ego=True,
            worldflow_bridge_loss_weight=0.0,
            worldflow_equiv_loss_weight=0.0,
        )

    independent_cfg = SmolVLAConfig(
        worldflow_enable=True,
        worldflow_target_type="world_eef_trajectory",
        worldflow_reference_frame="robot_base",
        worldflow_frame_origin="global",
        worldflow_scene_frame_origin="global",
        worldflow_noise_coupling="independent",
        worldflow_action_fusion="cross_attention",
        worldflow_action_expert_mode="independent",
        worldflow_bridge_loss_weight=0.0,
        worldflow_equiv_loss_weight=0.0,
    )
    assert "worldflow_action_expert=independent" in independent_cfg.flow_contract_summary()

    execution_cfg = SmolVLAConfig(
        worldflow_enable=True,
        worldflow_target_type="world_eef_trajectory",
        worldflow_reference_frame="robot_base",
        worldflow_frame_origin="global",
        worldflow_scene_frame_origin="global",
        worldflow_noise_coupling="independent",
        worldflow_action_fusion="world_trajectory_arm_ego_gripper",
        worldflow_action_expert_mode="independent",
        worldflow_current_ee_pose_token=True,
        worldflow_bridge_loss_weight=0.0,
        worldflow_equiv_loss_weight=0.0,
    )
    assert execution_cfg.worldflow_current_ee_pose_token

    physical_interaction_cfg = SmolVLAConfig(
        worldflow_enable=True,
        worldflow_target_type="world_eef_trajectory",
        worldflow_reference_frame="robot_base",
        worldflow_frame_origin="global",
        worldflow_scene_frame_origin="global",
        worldflow_noise_coupling="independent",
        worldflow_action_fusion="physical_trajectory_cross_attention",
        worldflow_action_expert_mode="independent",
        worldflow_current_ee_pose_token=True,
        worldflow_bridge_loss_weight=0.0,
        worldflow_equiv_loss_weight=0.0,
    )
    assert (
        physical_interaction_cfg.worldflow_action_fusion
        == "physical_trajectory_cross_attention"
    )

    with pytest.raises(ValueError, match="aligned physical frames"):
        SmolVLAConfig(
            worldflow_enable=True,
            worldflow_target_type="world_eef_trajectory",
            worldflow_reference_frame="robot_base",
            worldflow_frame_origin="global",
            worldflow_scene_frame_origin="global",
            worldflow_noise_coupling="independent",
            worldflow_action_fusion="physical_trajectory_cross_attention",
            worldflow_action_expert_mode="independent",
            worldflow_current_ee_pose_token=False,
            worldflow_bridge_loss_weight=0.0,
            worldflow_equiv_loss_weight=0.0,
        )

    with pytest.raises(ValueError, match="physical execution contract"):
        SmolVLAConfig(
            worldflow_enable=True,
            worldflow_target_type="world_eef_trajectory",
            worldflow_reference_frame="robot_base",
            worldflow_frame_origin="global",
            worldflow_scene_frame_origin="global",
            worldflow_noise_coupling="independent",
            worldflow_action_fusion="world_trajectory_arm_ego_gripper",
            worldflow_action_expert_mode="independent",
            worldflow_current_ee_pose_token=False,
            worldflow_bridge_loss_weight=0.0,
            worldflow_equiv_loss_weight=0.0,
        )

    with pytest.raises(ValueError, match="worldflow_action_expert_mode"):
        SmolVLAConfig(
            worldflow_enable=True,
            worldflow_action_expert_mode="partially_shared",
        )

    with pytest.raises(ValueError, match="strict independent-branch contract"):
        SmolVLAConfig(
            worldflow_enable=True,
            worldflow_target_type="world_eef_trajectory",
            worldflow_reference_frame="robot_base",
            worldflow_frame_origin="global",
            worldflow_scene_frame_origin="global",
            worldflow_noise_coupling="independent",
            worldflow_action_fusion="endpoint_residual_boosting",
            worldflow_bridge_loss_weight=0.0,
            worldflow_equiv_loss_weight=0.0,
        )


def test_worldflow_global_scene_keeps_absolute_translation_with_local_action_carrier():
    cfg = SmolVLAConfig(
        pointseg_enable=True,
        worldflow_enable=True,
        worldflow_frame_origin="current_ee",
        worldflow_scene_frame_origin="global",
    )
    model = VLAFlowMatching.__new__(VLAFlowMatching)
    nn.Module.__init__(model)
    model.config = cfg
    model.worldflow_branch = nn.Identity()

    points_ego = torch.tensor(
        [[[0.10, -0.02, 0.03, 12.0, 34.0, 56.0]]],
        dtype=torch.float32,
    )
    model.last_worldflow_payload = {"foreground_pc_ego": points_ego}
    current = se3_exp(torch.tensor([[0.42, -0.31, 0.27, 0.20, -0.10, 0.15]]))
    current_pose9 = matrix_to_pose9(current)

    scene_points, point_is_pad, _ = model._prepare_worldflow_foreground(current_pose9)
    expected_scene = _ego_point_cloud_to_world(
        points_ego,
        current_pose9,
        frame_origin="global",
    )
    local_action_carrier = _worldflow_carrier_matrix(
        current_pose9,
        cfg.worldflow_frame_origin,
    )

    assert torch.equal(scene_points, expected_scene)
    assert torch.count_nonzero(point_is_pad) == 0
    assert torch.count_nonzero(local_action_carrier[..., :3, 3]) == 0
    assert torch.allclose(
        scene_points[..., :3]
        - _ego_point_cloud_to_world(
            points_ego,
            current_pose9,
            frame_origin="current_ee",
        )[..., :3],
        current[..., :3, 3].unsqueeze(1),
        atol=2e-6,
        rtol=2e-6,
    )


def test_worldflow_background_uses_the_same_robot_base_transform_and_validity_mask():
    cfg = SmolVLAConfig(
        pointseg_enable=True,
        worldflow_enable=True,
        worldflow_frame_origin="global",
        worldflow_scene_frame_origin="global",
        worldflow_max_points=8,
    )
    model = VLAFlowMatching.__new__(VLAFlowMatching)
    nn.Module.__init__(model)
    model.config = cfg

    foreground = torch.tensor(
        [[[0.10, 0.00, 0.02, 1.0, 2.0, 3.0]]],
        dtype=torch.float32,
    )
    background = torch.tensor(
        [
            [
                [-0.12, 0.04, 0.01, 4.0, 5.0, 6.0],
                [0.50, 0.50, 0.50, 7.0, 8.0, 9.0],
            ]
        ],
        dtype=torch.float32,
    )
    background_pad = torch.tensor([[False, True]])
    model.last_worldflow_payload = {
        "foreground_pc_ego": foreground,
        "background_pc_ego": background,
        "background_point_is_pad_ego": background_pad,
    }
    current = se3_exp(torch.tensor([[0.31, -0.22, 0.18, 0.17, -0.08, 0.11]]))
    current_pose9 = matrix_to_pose9(current)

    transformed, transformed_pad = model._prepare_worldflow_background(current_pose9)
    expected_input = background.clone()
    expected_input[:, 1] = 0
    expected = _ego_point_cloud_to_world(
        expected_input,
        current_pose9,
        frame_origin="global",
    )

    assert torch.equal(transformed_pad, background_pad)
    assert torch.allclose(transformed, expected, atol=2e-6, rtol=2e-6)
    assert torch.equal(transformed[..., 3:], expected_input[..., 3:])


def test_worldflow_global_scene_rejects_random_coordinate_reparameterization():
    with pytest.raises(ValueError, match="fixed global scene frame"):
        SmolVLAConfig(
            pointseg_enable=True,
            worldflow_enable=True,
            worldflow_frame_origin="current_ee",
            worldflow_scene_frame_origin="global",
            worldflow_training_coordinate_frame_augmentation=True,
        )


def test_projected_pose9_is_a_valid_complete_se3_head_mode():
    cfg = SmolVLAConfig(
        se3_enable=True,
        se3_twist_head_mode="projected_pose9",
        worldflow_enable=True,
        worldflow_noise_coupling="conjugate_ego",
    )

    assert "se3_geodesic_left_trivialized(head=projected_pose9)" in cfg.flow_contract_summary()

    endpoint_cfg = SmolVLAConfig(
        se3_enable=True,
        se3_twist_head_mode="pose9_endpoint",
    )
    assert "se3_geodesic_left_trivialized(head=pose9_endpoint)" in endpoint_cfg.flow_contract_summary()

    chart_cfg = SmolVLAConfig(
        se3_enable=True,
        se3_twist_head_mode="pose9_chart_endpoint",
        se3_noise_trans_scale=0.1,
        se3_noise_rot_scale=0.1,
        se3_noise_gripper_scale=0.1,
    )
    assert "v0.4.2_pose9_chart_gaussian_projected_to_se3" in chart_cfg.flow_contract_summary()
    assert "head=pose9_chart_endpoint" in chart_cfg.flow_contract_summary()

    with pytest.raises(ValueError, match="se3_twist_head_mode"):
        SmolVLAConfig(se3_enable=True, se3_twist_head_mode="unknown")


def test_conjugate_worldflow_noise_rejects_legacy_invalid_rotation_prior():
    with pytest.raises(ValueError, match="pose9_action_noise_enable=True"):
        SmolVLAConfig(
            worldflow_enable=True,
            pose9_action_noise_enable=False,
            worldflow_noise_coupling="conjugate_ego",
        )


def test_integration_grid_time_sampling_matches_inference_euler_steps():
    cfg = SmolVLAConfig(
        num_steps=10,
        flow_time_sampling="integration_grid",
    )
    model = VLAFlowMatching.__new__(VLAFlowMatching)
    nn.Module.__init__(model)
    model.config = cfg

    torch.manual_seed(7)
    sampled = model.sample_time(2048, torch.device("cpu"))

    scaled = sampled * cfg.num_steps
    assert torch.allclose(scaled, scaled.round())
    assert sampled.min().item() == 0.0
    assert torch.allclose(sampled.max(), torch.tensor(0.9))
    assert set(scaled.to(dtype=torch.int64).tolist()) == set(range(cfg.num_steps))


def test_integration_grid_time_sampling_can_emphasize_exact_zero():
    cfg = SmolVLAConfig(
        num_steps=10,
        flow_time_sampling="integration_grid",
        flow_time_zero_probability=0.5,
    )
    model = VLAFlowMatching.__new__(VLAFlowMatching)
    nn.Module.__init__(model)
    model.config = cfg

    torch.manual_seed(7)
    sampled = model.sample_time(20_000, torch.device("cpu"))

    scaled = sampled * cfg.num_steps
    zero_ratio = (sampled == 0).to(dtype=torch.float32).mean()
    assert torch.allclose(scaled, scaled.round())
    assert torch.allclose(sampled.max(), torch.tensor(0.9))
    assert torch.allclose(zero_ratio, torch.tensor(0.5), atol=0.015)
    assert set(scaled.to(dtype=torch.int64).tolist()) == set(range(cfg.num_steps))


def test_pose9_action_loss_weights_preserve_scale_and_prioritize_physical_groups():
    policy = SimpleNamespace(
        config=SimpleNamespace(
            action_loss_translation_weight=3.0,
            action_loss_rotation_weight=1.0,
            action_loss_gripper_weight=3.0,
        )
    )
    weights = SmolVLAPolicy._action_loss_dimension_weights(
        policy,
        10,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    assert torch.allclose(weights.mean(), torch.tensor(1.0))
    assert weights[:3].min() > weights[3:9].max()
    assert torch.allclose(weights[9], weights[0])


def test_standard_action_metrics_use_physical_endpoint_error_and_padding_mask():
    model = VLAFlowMatching.__new__(VLAFlowMatching)
    nn.Module.__init__(model)
    identity = torch.eye(4).expand(1, 2, 4, 4).clone()
    actions = torch.cat(
        [matrix_to_pose9(identity), torch.zeros(1, 2, 1)],
        dim=-1,
    )
    x_t = actions.clone()
    u_t = torch.zeros_like(actions)
    pred_velocity = torch.zeros_like(actions)
    pred_velocity[0, 0, 0] = 0.01
    pred_velocity[0, 0, 9] = 0.02
    # This much larger error must be ignored because the second step is padded.
    pred_velocity[0, 1, 0] = 1.0
    pred_velocity[0, 1, 9] = 1.0

    model._record_standard_action_metrics(
        actions=actions,
        x_t=x_t,
        u_t=u_t,
        pred_velocity=pred_velocity,
        time=torch.zeros(1),
        actions_is_pad=torch.tensor([[False, True]]),
    )

    assert torch.allclose(model.last_action_metrics["action_endpoint_trans_err"], torch.tensor(0.01))
    assert torch.allclose(
        model.last_action_metrics["action_endpoint_gripper_err"],
        torch.tensor(0.02),
    )
    assert torch.allclose(
        model.last_action_metrics["action_endpoint_rot_err_deg"],
        torch.tensor(0.0),
    )


class _TinySceneEncoder(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.proj = nn.Linear(6, dim)

    def forward(self, point_cloud, point_is_pad=None, *, return_tokens=False):
        tokens = self.proj(point_cloud)
        if point_is_pad is None:
            mask = torch.ones(tokens.shape[:2], dtype=torch.bool, device=tokens.device)
        else:
            mask = ~point_is_pad
        weights = mask.unsqueeze(-1).to(dtype=tokens.dtype)
        global_feat = (tokens * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        assert return_tokens
        return {"global_feat": global_feat, "scene_tok1": tokens, "scene_mask1": mask}


class _RecordingPointAction(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_point_tokens = None

    def forward(
        self,
        action_tokens,
        point_tokens,
        point_mask=None,
        actions_is_pad=None,
    ):
        del point_mask, actions_is_pad
        self.num_point_tokens = point_tokens.shape[1]
        return action_tokens


class _TinyLanguageVLM(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(64, hidden_dim)

    def embed_language_tokens(self, tokens):
        return self.embedding(tokens)


class _TinyPointsegConditioner(nn.Module):
    def forward(self, payload):
        point_cloud = payload["point_cloud"]
        scene_tokens = point_cloud[..., :4]
        global_feat = scene_tokens.mean(dim=1)
        point_mask = torch.ones(scene_tokens.shape[:2], dtype=torch.bool, device=scene_tokens.device)
        scores = torch.full(scene_tokens.shape[:2], 0.75, device=scene_tokens.device)
        return {
            "object_feat": global_feat,
            "background_feat": torch.zeros_like(global_feat),
            "foreground_pc": point_cloud,
            "foreground_scene_tok1": scene_tokens,
            "foreground_scene_mask1": point_mask,
            "operation_prob": scores,
            "pointseg_selection_scores": scores,
        }


def _make_tiny_worldflow_branch(cfg: SmolVLAConfig) -> WorldFlowActionBranch:
    branch = WorldFlowActionBranch(
        cfg,
        action_hidden_dim=32,
        language_vocab_size=64,
    )
    branch.scene_encoder = _TinySceneEncoder(cfg.worldflow_feature_dim)
    return branch


def test_worldflow_branch_consumes_xyzrgb_without_role_or_probability_inputs():
    torch.manual_seed(17)
    cfg = SmolVLAConfig(
        chunk_size=4,
        n_action_steps=4,
        worldflow_feature_dim=16,
        point_action_fusion_heads=4,
    )
    branch = _make_tiny_worldflow_branch(cfg)
    point_cloud = torch.randn(2, 12, 6)
    point_cloud[..., 3:6] = torch.rand(2, 12, 3) * 255.0
    lang_tokens = torch.randint(0, 64, (2, 5))
    lang_masks = torch.ones(2, 5, dtype=torch.bool)
    noisy_actions = torch.randn(2, cfg.chunk_size, 9)
    time = torch.rand(2)
    actions_is_pad = torch.tensor(
        [[False, False, True, True], [False, False, False, False]]
    )

    output = branch(
        point_cloud,
        lang_tokens,
        lang_masks,
        noisy_actions,
        time,
        actions_is_pad=actions_is_pad,
    )

    assert output["scene_tokens"].shape == (2, 12, cfg.worldflow_feature_dim)
    assert output["scene_mask"].shape == (2, 12)
    assert output["global_feat"].shape == (2, cfg.worldflow_feature_dim)
    assert output["action_tokens"].shape == (2, cfg.chunk_size, 32)
    assert torch.isfinite(output["action_tokens"]).all()
    assert torch.equal(
        output["action_tokens"][0, 2:],
        torch.zeros_like(output["action_tokens"][0, 2:]),
    )
    assert not hasattr(branch, "action_expert")


def test_shared_expert_world_frontend_matches_ego_litept_point_action_path():
    cfg = SmolVLAConfig(
        chunk_size=4,
        n_action_steps=4,
        worldflow_enable=True,
        worldflow_target_type="world_eef_trajectory",
        worldflow_reference_frame="robot_base",
        worldflow_frame_origin="global",
        worldflow_scene_frame_origin="global",
        worldflow_noise_coupling="left_compose_ego",
        worldflow_world_eef_velocity_mode="base_pose9_euclidean",
        worldflow_action_fusion="point_action_expert_conjugate_bridge",
        worldflow_action_expert_mode="shared",
        worldflow_current_ee_pose_token=False,
        worldflow_bridge_loss_weight=0.0,
        worldflow_equiv_loss_weight=0.0,
        worldflow_feature_dim=16,
        point_action_fusion_heads=4,
    )
    branch = _make_tiny_worldflow_branch(cfg)
    assert branch.action_dim == 10
    assert branch.action_in_proj.in_features == 10
    assert isinstance(branch.point_action_adapter, PointActionSelfAttention)
    assert branch.scene_context_proj is None
    assert branch.language_embedding is None
    assert branch.language_norm is None

    scene = {
        "scene_tokens": torch.randn(2, 7, 16),
        "scene_mask": torch.ones(2, 7, dtype=torch.bool),
        "global_feat": torch.randn(2, 16),
    }
    actions = torch.randn(2, cfg.chunk_size, 10)
    time = torch.rand(2)
    first, _ = branch.embed_action_tokens(
        scene,
        torch.randint(0, 64, (2, 3)),
        torch.ones(2, 3, dtype=torch.bool),
        actions,
        time,
    )
    scene["global_feat"] = torch.randn_like(scene["global_feat"]) * 1000
    second, _ = branch.embed_action_tokens(
        scene,
        torch.randint(0, 64, (2, 9)),
        torch.zeros(2, 9, dtype=torch.bool),
        actions,
        time,
    )
    assert torch.equal(first, second)


def test_worldflow_point_action_receives_every_litept_token():
    cfg = SmolVLAConfig(
        chunk_size=4,
        n_action_steps=4,
        worldflow_feature_dim=16,
        point_action_fusion_heads=4,
    )
    branch = _make_tiny_worldflow_branch(cfg)
    recorder = _RecordingPointAction()
    branch.point_action_adapter = recorder
    num_points = 37
    output = branch(
        torch.randn(2, num_points, 6),
        torch.randint(0, 64, (2, 5)),
        torch.ones(2, 5, dtype=torch.bool),
        torch.randn(2, cfg.chunk_size, 9),
        torch.rand(2),
    )

    assert recorder.num_point_tokens == num_points
    assert output["global_feat"].shape == (2, cfg.worldflow_feature_dim)


def test_ego_prefix_keeps_all_litept_tokens_but_one_global_scene_feature():
    model = VLAFlowMatching.__new__(VLAFlowMatching)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(vla_adapter_enable=False, encode_robot_state=False)
    model.vlm_with_expert = _TinyLanguageVLM(hidden_dim=8)
    model.pointseg_conditioner = _TinyPointsegConditioner()
    model.pointseg_object_proj = nn.Linear(4, 8)
    model.pointseg_background_proj = nn.Linear(4, 8)
    model.point_action_fusion = nn.Identity()
    model.worldflow_branch = None
    model.capture_pointseg_visualization = False
    model.prefix_length = 0

    num_points = 41
    point_cloud = torch.randn(2, num_points, 6)
    expected_global = point_cloud[..., :4].mean(dim=1)
    model.embed_prefix(
        [point_cloud],
        [torch.ones(2, dtype=torch.bool)],
        torch.randint(0, 64, (2, 5)),
        torch.ones(2, 5, dtype=torch.bool),
    )

    assert model.last_point_action_tokens.shape == (2, num_points, 4)
    assert torch.equal(model.last_point_action_tokens, point_cloud[..., :4])
    assert model.last_ego_scene_global_feat.shape == (2, 4)
    assert torch.allclose(model.last_ego_scene_global_feat, expected_global)
    assert torch.equal(model.last_ego_scene_global_mask, torch.ones(2, dtype=torch.bool))
    recorder = _RecordingPointAction()
    model.point_action_fusion = recorder
    model._inject_point_action_features(torch.randn(2, 4, 8))
    assert recorder.num_point_tokens == num_points


def test_world_ego_joint_attention_preserves_ego_block_and_lets_world_read_ego():
    cfg = SmolVLAConfig(
        chunk_size=2,
        n_action_steps=2,
        pointseg_feature_dim=4,
        worldflow_feature_dim=16,
    )
    model = VLAFlowMatching.__new__(VLAFlowMatching)
    nn.Module.__init__(model)
    model.config = cfg
    model.ego_scene_to_expert = nn.Linear(4, 8)
    model.world_scene_to_expert = nn.Linear(16, 8)
    model.world_ego_scene_type_embedding = nn.Parameter(torch.randn(2, 8))
    model.world_ego_action_type_embedding = nn.Parameter(torch.randn(2, 8))
    model.last_ego_scene_global_feat = torch.randn(1, 4)
    model.last_ego_scene_global_mask = torch.ones(1, dtype=torch.bool)

    ego_actions = torch.randn(1, 2, 8)
    world = {
        "scene_tokens": torch.randn(1, 4, 16),
        "scene_mask": torch.ones(1, 4, dtype=torch.bool),
        "global_feat": torch.randn(1, 16),
        "action_tokens": torch.randn(1, 2, 8),
        "action_mask": torch.ones(1, 2, dtype=torch.bool),
    }
    suffix, pad, blocks, layout = model._build_world_ego_joint_suffix(
        ego_actions,
        torch.ones(1, 2, dtype=torch.bool),
        world,
    )
    mask = make_att_2d_masks(pad, blocks)[0]

    ego_action = torch.arange(layout["ego_action"].start, layout["ego_action"].stop)
    world_action = torch.arange(layout["world_action"].start, layout["world_action"].stop)
    all_scenes = torch.cat(
        [
            torch.arange(layout["ego_scene"].start, layout["ego_scene"].stop),
            torch.arange(layout["world_scene"].start, layout["world_scene"].stop),
        ]
    )
    assert layout["ego_scene"].stop - layout["ego_scene"].start == 1
    assert layout["world_scene"].stop - layout["world_scene"].start == 1
    assert suffix.shape == (1, 2 + 2 * cfg.chunk_size, 8)
    assert torch.equal(suffix[:, layout["ego_action"]], ego_actions)
    assert mask[ego_action[:, None], ego_action].all()
    assert not mask[ego_action[:, None], all_scenes].any()
    assert not mask[ego_action[:, None], world_action].any()
    assert mask[world_action[:, None], ego_action].all()
    assert mask[world_action[:, None], all_scenes].all()
    assert mask[world_action[:, None], world_action].all()


def test_joint_point_action_expert_prefix_preserves_pretrained_ego_scene_tokens():
    model = VLAFlowMatching.__new__(VLAFlowMatching)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(
        vla_adapter_enable=False,
        worldflow_action_fusion="point_action_expert_conjugate_bridge",
        encode_robot_state=False,
    )
    model.prefix_length = 0
    model.pointseg_conditioner = None
    model.point_action_fusion = None
    model.worldflow_branch = nn.Identity()
    model.inference_ablation_modalities = frozenset()

    class MeanExtractor(nn.Module):
        def forward(self, points, _point_is_pad=None):
            return points.mean(dim=1)

    class LanguageEmbedding(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(16, 4)

        def embed_language_tokens(self, tokens):
            return self.embedding(tokens)

    model.extractor = MeanExtractor()
    model.pointcloud_proj = nn.Identity()
    model.vlm_with_expert = LanguageEmbedding()
    point_cloud = torch.randn(1, 5, 4)
    language = torch.tensor([[1, 2, 3]])

    prefix, pad_mask, _ = model.embed_prefix(
        [point_cloud],
        [torch.ones(1, dtype=torch.bool)],
        language,
        torch.ones_like(language, dtype=torch.bool),
    )

    assert prefix.shape[1] == language.shape[1] + 1
    assert pad_mask.shape[1] == language.shape[1] + 1
    assert model.last_ego_point_prefix_slices == (slice(0, 1),)


def test_shared_expert_world_scene_prefix_keeps_ego_values_and_pairs_positions():
    model = VLAFlowMatching.__new__(VLAFlowMatching)
    nn.Module.__init__(model)
    model.world_pointseg_object_proj = nn.Identity()
    model.world_pointseg_background_proj = nn.Identity()
    model.world_point_prefix_type_embedding = nn.Parameter(torch.zeros(3))
    model.last_ego_point_prefix_slices = (slice(1, 3),)
    model.inference_ablation_modalities = frozenset()

    prefix = torch.randn(1, 5, 3)
    original = prefix.clone()
    prefix_mask = torch.tensor([[True, True, True, True, False]])
    prefix_blocks = torch.tensor([[False, False, False, False, True]])
    world_features = torch.tensor(
        [[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]]
    )

    combined, valid, blocks, positions = model._append_shared_expert_world_scene_prefix(
        prefix,
        prefix_mask,
        prefix_blocks,
        {
            "scene_global_tokens": world_features,
            "scene_global_mask": torch.tensor([[True, False]]),
        },
    )

    assert torch.equal(combined[:, :5], original)
    assert torch.allclose(combined[:, 5:], world_features * (3.0**0.5))
    assert valid.tolist() == [[True, True, True, True, False, True, False]]
    assert not blocks.any()
    # Appended World fore/back reuse the pretrained Ego fore/back rotary ids;
    # original image/Ego/language positions do not move.
    assert positions.tolist() == [[0, 1, 2, 3, 3, 1, 2]]


def test_shared_point_action_expert_uses_two_bidirectional_domains_with_block_causality():
    cfg = SimpleNamespace(
        chunk_size=2,
        worldflow_action_fusion="point_action_expert_conjugate_bridge",
        use_cache=True,
    )
    model = VLAFlowMatching.__new__(VLAFlowMatching)
    nn.Module.__init__(model)
    model.config = cfg
    model.conjugate_motion_in_proj = nn.Linear(9, 9)
    nn.init.zeros_(model.conjugate_motion_in_proj.weight)
    nn.init.zeros_(model.conjugate_motion_in_proj.bias)
    model.ego_scene_to_expert = nn.Identity()
    model.world_scene_to_expert = nn.Identity()
    model.world_ego_scene_type_embedding = nn.Parameter(torch.zeros(2, 9))
    model.world_ego_scene_role_embedding = nn.Parameter(
        torch.stack([torch.ones(9), torch.full((9,), 2.0)])
    )
    model.world_ego_action_type_embedding = nn.Parameter(torch.zeros(2, 9))
    model.inference_ablation_modalities = frozenset()
    model.last_ego_scene_global_tokens = torch.zeros(1, 2, 9)
    model.last_ego_scene_global_token_mask = torch.ones(1, 2, dtype=torch.bool)

    ego_actions = torch.randn(1, 2, 9)
    world_actions = torch.randn(1, 2, 9)
    world = {
        "scene_global_tokens": torch.zeros(1, 2, 9),
        "scene_global_mask": torch.ones(1, 2, dtype=torch.bool),
        "action_tokens": world_actions,
        "action_mask": torch.ones(1, 2, dtype=torch.bool),
        "ego_conjugate_motion_pose9": torch.randn(1, 2, 9),
        "world_conjugate_motion_pose9": torch.randn(1, 2, 9),
        "conjugate_motion_mask": torch.ones(1, 2, dtype=torch.bool),
    }
    calls = []

    class RecordingSharedExpert(nn.Module):
        def forward(self, **kwargs):
            calls.append(
                {
                    "attention_mask": kwargs["attention_mask"].clone(),
                    "position_ids": kwargs["position_ids"].clone(),
                    "suffix": kwargs["inputs_embeds"][1].clone(),
                }
            )
            return [None, kwargs["inputs_embeds"][1]], None

    model.vlm_with_expert = RecordingSharedExpert()
    model._bidirectional_world_ego_cross_attention = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        AssertionError("shared PointActionExpert mode must not use post-Expert cross-attention")
    )
    actual_ego, actual_world = model._run_world_ego_joint_expert(
        torch.randn(1, 5, 9),
        torch.ones(1, 5, dtype=torch.bool),
        torch.zeros(1, 5, dtype=torch.bool),
        ego_actions,
        torch.ones(1, 2, dtype=torch.bool),
        world,
        past_key_values=object(),
    )

    assert len(calls) == 1
    assert torch.equal(actual_ego, ego_actions)
    assert torch.equal(actual_world, world_actions)
    # Fore/back now remain in the VLM prefix; the suffix contains only paired
    # Ego actions followed by World actions.
    assert calls[0]["suffix"].shape == (1, 4, 9)
    assert calls[0]["position_ids"].tolist() == [[0, 1, 2, 3, 4, 5, 6, 5, 6]]
    full_attention = calls[0]["attention_mask"]
    suffix_attention = full_attention[:, 5:, 5:]
    action = torch.arange(4)
    # All non-action prefix tokens are bidirectional and cannot read actions.
    assert full_attention[0, :5, :5].all()
    assert not full_attention[0, :5, 5 + action].any()
    # All 2T action tokens read every non-action and action token.
    assert full_attention[0, 5 + action[:, None], :5].all()
    assert suffix_attention[0, action[:, None], action].all()
    # Ego and World action outputs can attend each other in the same Expert call.
    assert suffix_attention[0, :2, 2:4].all()
    assert suffix_attention[0, 2:4, :2].all()

    model._run_world_ego_joint_expert(
        None,
        torch.ones(1, 5, dtype=torch.bool),
        None,
        ego_actions,
        torch.ones(1, 2, dtype=torch.bool),
        world,
        past_key_values=object(),
    )
    cached_attention = calls[1]["attention_mask"]
    assert cached_attention.shape == (1, 4, 9)
    assert cached_attention[0, action].all()


def test_shared_point_action_expert_ego_output_has_direct_world_action_gradient():
    cfg = SimpleNamespace(
        chunk_size=2,
        worldflow_action_fusion="point_action_expert_conjugate_bridge",
        use_cache=True,
    )
    model = VLAFlowMatching.__new__(VLAFlowMatching)
    nn.Module.__init__(model)
    model.config = cfg
    model.conjugate_motion_in_proj = nn.Linear(9, 9)
    model.ego_scene_to_expert = nn.Identity()
    model.world_scene_to_expert = nn.Identity()
    model.world_ego_scene_type_embedding = nn.Parameter(torch.zeros(2, 9))
    model.world_ego_scene_role_embedding = nn.Parameter(torch.zeros(2, 9))
    model.world_ego_action_type_embedding = nn.Parameter(torch.zeros(2, 9))
    model.inference_ablation_modalities = frozenset()
    model.last_ego_scene_global_tokens = torch.randn(1, 2, 9)
    model.last_ego_scene_global_token_mask = torch.ones(1, 2, dtype=torch.bool)

    class MixingSharedExpert(nn.Module):
        def forward(self, **kwargs):
            suffix = kwargs["inputs_embeds"][1]
            # A minimal stand-in for a fully connected self-attention layer.
            mixed = suffix + suffix.mean(dim=1, keepdim=True)
            return [None, mixed], None

    model.vlm_with_expert = MixingSharedExpert()
    ego_actions = torch.randn(1, 2, 9, requires_grad=True)
    world_actions = torch.randn(1, 2, 9, requires_grad=True)
    world = {
        "scene_global_tokens": torch.randn(1, 2, 9),
        "scene_global_mask": torch.ones(1, 2, dtype=torch.bool),
        "action_tokens": world_actions,
        "action_mask": torch.ones(1, 2, dtype=torch.bool),
        "ego_conjugate_motion_pose9": torch.randn(1, 2, 9),
        "world_conjugate_motion_pose9": torch.randn(1, 2, 9),
        "conjugate_motion_mask": torch.ones(1, 2, dtype=torch.bool),
    }

    ego_out, _ = model._run_world_ego_joint_expert(
        None,
        torch.ones(1, 5, dtype=torch.bool),
        None,
        ego_actions,
        torch.ones(1, 2, dtype=torch.bool),
        world,
        past_key_values=object(),
    )
    ego_out.sum().backward()

    assert world_actions.grad is not None
    assert torch.count_nonzero(world_actions.grad) == world_actions.numel()


def test_world_stream_ablation_preserves_layout_but_masks_world_tokens():
    cfg = SmolVLAConfig(
        chunk_size=3,
        n_action_steps=3,
        pointseg_feature_dim=4,
        worldflow_feature_dim=4,
    )
    model = VLAFlowMatching.__new__(VLAFlowMatching)
    nn.Module.__init__(model)
    model.config = cfg
    model.ego_scene_to_expert = nn.Linear(4, 8)
    model.world_scene_to_expert = nn.Linear(4, 8)
    model.world_ego_scene_type_embedding = nn.Parameter(torch.randn(2, 8))
    model.world_ego_action_type_embedding = nn.Parameter(torch.randn(2, 8))
    model.last_ego_scene_global_feat = torch.randn(1, 4)
    model.last_ego_scene_global_mask = torch.ones(1, dtype=torch.bool)
    model.inference_ablation_modalities = frozenset({"world"})
    world = {
        "scene_tokens": torch.randn(1, 5, 4),
        "scene_mask": torch.ones(1, 5, dtype=torch.bool),
        "global_feat": torch.randn(1, 4),
        "action_tokens": torch.randn(1, cfg.chunk_size, 8),
        "action_mask": torch.ones(1, cfg.chunk_size, dtype=torch.bool),
    }

    suffix, pad, _blocks, layout = model._build_world_ego_joint_suffix(
        torch.randn(1, cfg.chunk_size, 8),
        torch.ones(1, cfg.chunk_size, dtype=torch.bool),
        world,
    )

    assert suffix.shape[1] == 2 + 2 * cfg.chunk_size
    assert pad[:, layout["ego_scene"]].all()
    assert pad[:, layout["ego_action"]].all()
    assert not pad[:, layout["world_scene"]].any()
    assert not pad[:, layout["world_action"]].any()


def test_joint_attention_is_ego_safe_but_world_loss_trains_shared_ego_context():
    torch.manual_seed(18)
    cfg = SmolVLAConfig(
        chunk_size=2,
        n_action_steps=2,
        pointseg_feature_dim=4,
        worldflow_feature_dim=4,
    )
    model = VLAFlowMatching.__new__(VLAFlowMatching)
    nn.Module.__init__(model)
    model.config = cfg
    model.ego_scene_to_expert = nn.Linear(4, 8)
    model.world_scene_to_expert = nn.Linear(4, 8)
    model.world_ego_scene_type_embedding = nn.Parameter(torch.randn(2, 8))
    model.world_ego_action_type_embedding = nn.Parameter(torch.randn(2, 8))

    ego_scene = torch.randn(1, 4, requires_grad=True)
    world_scene = torch.randn(1, 4, requires_grad=True)
    ego_actions = torch.randn(1, cfg.chunk_size, 8, requires_grad=True)
    world_actions = torch.randn(1, cfg.chunk_size, 8, requires_grad=True)
    model.last_ego_scene_global_feat = ego_scene
    model.last_ego_scene_global_mask = torch.ones(1, dtype=torch.bool)
    world = {
        "scene_tokens": torch.randn(1, 7, 4),
        "scene_mask": torch.ones(1, 7, dtype=torch.bool),
        "global_feat": world_scene,
        "action_tokens": world_actions,
        "action_mask": torch.ones(1, cfg.chunk_size, dtype=torch.bool),
    }
    suffix, pad, blocks, layout = model._build_world_ego_joint_suffix(
        ego_actions,
        torch.ones(1, cfg.chunk_size, dtype=torch.bool),
        world,
    )
    allowed = make_att_2d_masks(pad, blocks)
    scores = suffix @ suffix.transpose(-1, -2) / suffix.shape[-1] ** 0.5
    weights = torch.softmax(scores.masked_fill(~allowed, torch.finfo(scores.dtype).min), dim=-1)
    attended = weights @ suffix

    ego_loss = attended[:, layout["ego_action"]].square().mean()
    ego_to_world = torch.autograd.grad(
        ego_loss,
        (world_scene, world_actions),
        retain_graph=True,
        allow_unused=True,
    )
    assert all(gradient is None or gradient.abs().sum() == 0 for gradient in ego_to_world)

    world_loss = attended[:, layout["world_action"]].square().mean()
    world_to_ego = torch.autograd.grad(world_loss, (ego_scene, ego_actions))
    assert all(gradient.abs().sum() > 0 for gradient in world_to_ego)


def test_bidirectional_cross_attention_is_function_preserving_without_a_gate():
    torch.manual_seed(181)
    model = VLAFlowMatching.__new__(VLAFlowMatching)
    nn.Module.__init__(model)
    dim = 8
    model.ego_to_world_cross_norm = nn.LayerNorm(dim)
    model.world_to_ego_cross_norm = nn.LayerNorm(dim)
    model.ego_to_world_cross_attn = nn.MultiheadAttention(dim, 2, batch_first=True)
    model.world_to_ego_cross_attn = nn.MultiheadAttention(dim, 2, batch_first=True)
    nn.init.zeros_(model.world_to_ego_cross_attn.out_proj.weight)
    nn.init.zeros_(model.world_to_ego_cross_attn.out_proj.bias)
    ego = torch.randn(2, 4, dim, requires_grad=True)
    world = torch.randn(2, 4, dim, requires_grad=True)
    valid = torch.ones(2, 4, dtype=torch.bool)

    ego_out, world_out = model._bidirectional_world_ego_cross_attention(
        ego,
        world,
        valid,
        valid,
    )

    assert torch.equal(ego_out, ego)
    assert not torch.equal(world_out, world)
    (ego_out.square().mean() + world_out.square().mean()).backward()
    assert model.world_to_ego_cross_attn.out_proj.weight.grad.abs().sum() > 0
    assert model.ego_to_world_cross_attn.out_proj.weight.grad.abs().sum() > 0


def test_worldflow_protected_ego_freezes_only_pretrained_roles():
    model = VLAFlowMatching.__new__(VLAFlowMatching)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(worldflow_freeze_pretrained_ego=True)
    model.action_out_proj = nn.Linear(4, 4)
    model.pointseg_conditioner = nn.Linear(4, 4)
    model.vlm_with_expert = nn.Linear(4, 4)
    model.worldflow_branch = nn.Linear(4, 4)
    model.world_lm_expert = nn.Linear(4, 4)
    model.ego_scene_to_expert = nn.Linear(4, 4)
    model.world_scene_to_expert = nn.Linear(4, 4)
    model.ego_to_world_cross_attn = nn.MultiheadAttention(4, 1, batch_first=True)
    model.world_to_ego_cross_attn = nn.MultiheadAttention(4, 1, batch_first=True)
    model.world_ego_scene_type_embedding = nn.Parameter(torch.randn(2, 4))
    model.world_ego_scene_role_embedding = nn.Parameter(torch.randn(2, 4))
    model.world_ego_action_type_embedding = nn.Parameter(torch.randn(2, 4))

    model._apply_worldflow_pretrained_ego_freeze()

    for module in (model.action_out_proj, model.pointseg_conditioner, model.vlm_with_expert):
        assert not any(parameter.requires_grad for parameter in module.parameters())
    for module in (
        model.worldflow_branch,
        model.world_lm_expert,
        model.ego_scene_to_expert,
        model.world_scene_to_expert,
        model.ego_to_world_cross_attn,
        model.world_to_ego_cross_attn,
    ):
        assert all(parameter.requires_grad for parameter in module.parameters())
    assert model.world_ego_scene_type_embedding.requires_grad
    assert model.world_ego_scene_role_embedding.requires_grad
    assert model.world_ego_action_type_embedding.requires_grad


def test_independent_world_suffix_contains_only_world_scene_and_actions():
    model = VLAFlowMatching.__new__(VLAFlowMatching)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(chunk_size=3)
    model.ego_scene_to_expert = nn.Linear(2, 4, bias=False)
    model.world_scene_to_expert = nn.Linear(2, 4, bias=False)
    model.world_ego_scene_type_embedding = nn.Parameter(torch.zeros(2, 4))
    model.world_ego_action_type_embedding = nn.Parameter(torch.zeros(2, 4))
    model.last_ego_scene_global_feat = torch.tensor([[1.0, 2.0]])
    model.last_ego_scene_global_mask = torch.tensor([True])
    model.inference_ablation_modalities = frozenset()
    with torch.no_grad():
        model.ego_scene_to_expert.weight.fill_(1.0)
        model.world_scene_to_expert.weight.fill_(2.0)
    world_actions = torch.arange(12, dtype=torch.float32).reshape(1, 3, 4)
    suffix, valid, blocks = model._build_independent_world_suffix(
        {
            "global_feat": torch.tensor([[3.0, 4.0]]),
            "scene_mask": torch.tensor([[True, True]]),
            "action_tokens": world_actions,
            "action_mask": torch.ones(1, 3, dtype=torch.bool),
        }
    )

    assert suffix.shape == (1, 4, 4)
    assert torch.equal(suffix[:, -3:], world_actions)
    assert valid.tolist() == [[True, True, True, True]]
    assert blocks.tolist() == [[1, 1, 0, 0]]


def test_independent_parallel_runs_both_experts_without_cross_interaction():
    model = VLAFlowMatching.__new__(VLAFlowMatching)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(
        worldflow_action_expert_mode="independent",
        worldflow_action_fusion="independent_parallel",
    )
    ego_expected = torch.randn(2, 3, 4)
    world_expected = torch.randn(2, 3, 4)
    model._run_ego_suffix_expert = lambda *_args, **_kwargs: ego_expected
    model._run_independent_world_suffix_expert = lambda *_args, **_kwargs: world_expected

    def fail_if_interacted(*_args, **_kwargs):
        raise AssertionError("independent calibration must not invoke cross-attention")

    model._bidirectional_world_ego_cross_attention = fail_if_interacted
    actual_ego, actual_world = model._run_world_ego_joint_expert(
        None,
        torch.ones(2, 5, dtype=torch.bool),
        None,
        torch.randn(2, 3, 4),
        torch.ones(2, 3, dtype=torch.bool),
        {"action_mask": torch.ones(2, 3, dtype=torch.bool)},
        past_key_values=object(),
    )

    assert actual_ego is ego_expected
    assert actual_world is world_expected


def test_independent_world_suffix_prepends_current_robot_base_eef_token():
    model = VLAFlowMatching.__new__(VLAFlowMatching)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(chunk_size=2, worldflow_current_ee_pose_token=True)
    model.ego_scene_to_expert = nn.Linear(2, 4, bias=False)
    model.world_scene_to_expert = nn.Linear(2, 4, bias=False)
    model.world_ego_scene_type_embedding = nn.Parameter(torch.zeros(2, 4))
    model.world_ego_action_type_embedding = nn.Parameter(torch.zeros(2, 4))
    model.last_ego_scene_global_feat = torch.tensor([[1.0, 2.0]])
    model.last_ego_scene_global_mask = torch.tensor([True])
    model.inference_ablation_modalities = frozenset()
    current_token = torch.tensor([[[9.0, 8.0, 7.0, 6.0]]])
    world_actions = torch.arange(8, dtype=torch.float32).reshape(1, 2, 4)

    suffix, valid, blocks = model._build_independent_world_suffix(
        {
            "current_ee_pose_token": current_token,
            "current_ee_pose_mask": torch.ones(1, 1, dtype=torch.bool),
            "global_feat": torch.tensor([[3.0, 4.0]]),
            "scene_mask": torch.tensor([[True, True]]),
            "action_tokens": world_actions,
            "action_mask": torch.ones(1, 2, dtype=torch.bool),
        }
    )

    assert suffix.shape == (1, 4, 4)
    assert torch.equal(suffix[:, :1], current_token)
    assert torch.equal(suffix[:, -2:], world_actions)
    assert valid.tolist() == [[True, True, True, True]]
    assert blocks.tolist() == [[1, 0, 1, 0]]


def test_independent_world_prefix_masks_ego_point_tokens_only():
    model = VLAFlowMatching.__new__(VLAFlowMatching)
    nn.Module.__init__(model)
    model.last_ego_point_prefix_slices = (slice(3, 5), slice(8, 10))
    prefix_mask = torch.ones(2, 12, dtype=torch.bool)

    world_mask = model._independent_world_prefix_mask(prefix_mask)

    assert world_mask[:, :3].all()
    assert not world_mask[:, 3:5].any()
    assert world_mask[:, 5:8].all()
    assert not world_mask[:, 8:10].any()
    assert world_mask[:, 10:].all()
    assert prefix_mask.all()


def test_world_trajectory_execution_is_left_coordinate_change_not_conjugation():
    current = se3_exp(torch.tensor([[0.20, -0.10, 0.30, 0.0, 0.0, 0.35]]))
    relative = se3_exp(
        torch.tensor(
            [[[0.04, 0.01, -0.02, 0.03, -0.04, 0.02], [0.08, -0.03, 0.01, 0.0, 0.02, -0.05]]]
        )
    )
    absolute = current.unsqueeze(1) @ relative

    actual = world_trajectory_to_current_ego_pose9(
        matrix_to_pose9(absolute),
        matrix_to_pose9(current),
    )

    assert torch.allclose(pose9_to_matrix(actual), relative, atol=3e-5, rtol=3e-5)


def test_shared_expert_bridge_conjugates_unchanged_world_and_ego_flow_states():
    current_base = se3_exp(torch.tensor([[0.20, -0.10, 0.30, 0.0, 0.0, 0.35]]))
    ego_state_matrix = se3_exp(
        torch.tensor(
            [[[0.04, 0.01, -0.02, 0.03, -0.04, 0.02], [0.08, -0.03, 0.01, 0.0, 0.02, -0.05]]]
        )
    )
    world_state_matrix = current_base.unsqueeze(1) @ ego_state_matrix

    ego_motion, world_motion = conjugate_world_ego_flow_states_for_shared_expert(
        matrix_to_pose9(ego_state_matrix),
        matrix_to_pose9(world_state_matrix),
        matrix_to_pose9(current_base),
    )
    expected = current_base.unsqueeze(1) @ ego_state_matrix @ torch.linalg.inv(
        current_base
    ).unsqueeze(1)

    assert torch.allclose(pose9_to_matrix(ego_motion), expected, atol=4e-5, rtol=4e-5)
    assert torch.allclose(pose9_to_matrix(world_motion), expected, atol=4e-5, rtol=4e-5)
    # The bridge is additional conditioning only: its inputs retain their
    # original current-EEF and robot-base absolute pose semantics.
    assert torch.allclose(pose9_to_matrix(matrix_to_pose9(ego_state_matrix)), ego_state_matrix)
    assert torch.allclose(pose9_to_matrix(matrix_to_pose9(world_state_matrix)), world_state_matrix)


def test_physical_trajectory_proposals_bridge_complete_predictions_by_conjugation():
    current_base = se3_exp(torch.tensor([[0.20, -0.10, 0.30, 0.0, 0.0, 0.35]]))
    target_current = se3_exp(
        torch.tensor(
            [[[0.04, 0.01, -0.02, 0.03, -0.04, 0.02], [0.08, -0.03, 0.01, 0.0, 0.02, -0.05]]]
        )
    )
    target_base = current_base.unsqueeze(1) @ target_current
    timestep = torch.tensor([0.4])
    remaining = 1.0 - timestep[:, None, None]

    ego_target_pose9 = matrix_to_pose9(target_current)
    ego_state = torch.zeros(1, 2, 10)
    ego_state[..., :9] = ego_target_pose9 - 0.2
    ego_velocity = torch.zeros(1, 2, 10)
    ego_velocity[..., :9] = (ego_target_pose9 - ego_state[..., :9]) / remaining
    world_state_matrix = se3_exp(
        torch.tensor(
            [[[0.01, -0.02, 0.03, 0.01, 0.02, -0.01], [0.03, 0.01, 0.02, -0.02, 0.01, 0.03]]]
        )
    )
    world_state = matrix_to_pose9(world_state_matrix)
    world_velocity = torch.cat(
        [
            (target_base[..., :3, 3] - world_state_matrix[..., :3, 3]) / remaining,
            so3_log(
                target_base[..., :3, :3]
                @ world_state_matrix[..., :3, :3].transpose(-1, -2)
            )
            / remaining,
        ],
        dim=-1,
    )

    ego_motion_world, world_motion_world = (
        physically_aligned_world_ego_trajectory_proposals(
            ego_state,
            ego_velocity,
            world_state,
            world_velocity,
            timestep,
            matrix_to_pose9(current_base),
            "base_decoupled",
        )
    )

    target_motion_world = target_base @ torch.linalg.inv(current_base).unsqueeze(1)
    assert torch.allclose(
        pose9_to_matrix(ego_motion_world),
        target_motion_world,
        atol=4e-5,
        rtol=4e-5,
    )
    assert torch.allclose(
        pose9_to_matrix(world_motion_world),
        target_motion_world,
        atol=4e-5,
        rtol=4e-5,
    )


def test_physical_trajectory_cross_attention_zero_output_preserves_both_experts():
    model = VLAFlowMatching.__new__(VLAFlowMatching)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(
        worldflow_target_type="world_eef_trajectory",
        worldflow_world_eef_velocity_mode="base_decoupled",
    )
    hidden_dim = 8
    model.action_out_proj = nn.Linear(hidden_dim, 10)
    model.world_se3_action_out_proj = nn.Linear(hidden_dim, 6)
    model.ego_physical_trajectory_in_proj = nn.Linear(9, hidden_dim)
    model.world_physical_trajectory_in_proj = nn.Linear(9, hidden_dim)
    model.physical_trajectory_position_embedding = nn.Parameter(torch.zeros(3, hidden_dim))
    model.world_ego_action_type_embedding = nn.Parameter(torch.zeros(2, hidden_dim))
    model.ego_to_world_cross_norm = nn.LayerNorm(hidden_dim)
    model.world_to_ego_cross_norm = nn.LayerNorm(hidden_dim)
    model.ego_to_world_cross_attn = nn.MultiheadAttention(hidden_dim, 2, batch_first=True)
    model.world_to_ego_cross_attn = nn.MultiheadAttention(hidden_dim, 2, batch_first=True)
    for attention in (model.ego_to_world_cross_attn, model.world_to_ego_cross_attn):
        nn.init.zeros_(attention.out_proj.weight)
        nn.init.zeros_(attention.out_proj.bias)
    model.inference_ablation_modalities = frozenset()

    ego_out = torch.randn(2, 3, hidden_dim)
    world_out = torch.randn(2, 3, hidden_dim)
    ego_state = torch.randn(2, 3, 10) * 0.1
    ego_state[..., 3] = 1.0
    ego_state[..., 7] = 1.0
    world_state = matrix_to_pose9(se3_exp(torch.randn(2, 3, 6) * 0.1))
    valid = torch.ones(2, 3, dtype=torch.bool)

    actual_ego, actual_world = model._physical_trajectory_cross_attention(
        ego_out,
        world_out,
        ego_state,
        world_state,
        torch.tensor([0.2, 0.7]),
        matrix_to_pose9(se3_exp(torch.randn(2, 6) * 0.1)),
        valid,
        valid,
    )

    assert torch.equal(actual_ego, ego_out)
    assert torch.equal(actual_world, world_out)


def test_worldflow_protected_ego_config_requires_worldflow_and_no_nominal_ego_lr():
    with pytest.raises(ValueError, match="requires worldflow_enable=True"):
        SmolVLAConfig(worldflow_freeze_pretrained_ego=True)

    with pytest.raises(ValueError, match="worldflow_pretrained_lr_multiplier=1.0"):
        SmolVLAConfig(
            worldflow_enable=True,
            worldflow_freeze_pretrained_ego=True,
            worldflow_pretrained_lr_multiplier=0.2,
        )


def test_worldflow_two_timescale_optimizer_keeps_both_branches_trainable():
    policy = SmolVLAPolicy.__new__(SmolVLAPolicy)
    nn.Module.__init__(policy)
    policy.config = SimpleNamespace(
        vla_adapter_enable=True,
        worldflow_enable=True,
        worldflow_pretrained_lr_multiplier=0.2,
        worldflow_new_lr_multiplier=1.0,
        optimizer_lr=2.5e-5,
    )
    policy.model = nn.Module()
    policy.model.action_out_proj = nn.Linear(4, 4)
    policy.model.worldflow_branch = nn.Linear(4, 4)
    policy.model.world_to_ego_cross_attn = nn.MultiheadAttention(4, 1, batch_first=True)

    groups = policy.get_optim_params()

    assert [group["group_name"] for group in groups] == [
        "pretrained_ego_shared",
        "new_world_bidirectional",
    ]
    assert groups[0]["lr"] == pytest.approx(5e-6)
    assert groups[1]["lr"] == pytest.approx(2.5e-5)
    assert groups[0]["params"]
    assert groups[1]["params"]
    assert all(parameter.requires_grad for group in groups for parameter in group["params"])
    grouped = [id(parameter) for group in groups for parameter in group["params"]]
    expected = [id(parameter) for parameter in policy.parameters() if parameter.requires_grad]
    assert len(grouped) == len(set(grouped)) == len(expected)
    assert set(grouped) == set(expected)


def test_worldflow_three_timescale_optimizer_keeps_all_paths_trainable():
    policy = SmolVLAPolicy.__new__(SmolVLAPolicy)
    nn.Module.__init__(policy)
    policy.config = SimpleNamespace(
        vla_adapter_enable=True,
        worldflow_enable=True,
        worldflow_pretrained_lr_multiplier=0.05,
        worldflow_new_lr_multiplier=0.2,
        worldflow_residual_lr_multiplier=4.0,
        optimizer_lr=2.5e-6,
    )
    policy.model = nn.Module()
    policy.model.action_out_proj = nn.Linear(4, 4)
    policy.model.worldflow_branch = nn.Linear(4, 4)
    policy.model.world_to_ego_cross_attn = nn.MultiheadAttention(4, 1, batch_first=True)
    policy.model.world_twist_residual_out_proj = nn.Linear(4, 4)

    groups = policy.get_optim_params()

    assert [group["group_name"] for group in groups] == [
        "pretrained_ego_shared",
        "new_world_bidirectional",
        "world_physical_residual_head",
    ]
    assert groups[0]["lr"] == pytest.approx(1.25e-7)
    assert groups[1]["lr"] == pytest.approx(5e-7)
    assert groups[2]["lr"] == pytest.approx(1e-5)
    assert all(parameter.requires_grad for group in groups for parameter in group["params"])
    grouped = [id(parameter) for group in groups for parameter in group["params"]]
    expected = [id(parameter) for parameter in policy.parameters() if parameter.requires_grad]
    assert len(grouped) == len(set(grouped)) == len(expected)
    assert set(grouped) == set(expected)


def test_worldflow_residual_lr_multiplier_must_remain_positive():
    config = SmolVLAConfig(
        worldflow_enable=True,
        worldflow_residual_lr_multiplier=4.0,
    )
    assert config.worldflow_residual_lr_multiplier == pytest.approx(4.0)

    with pytest.raises(ValueError, match="worldflow_residual_lr_multiplier must be positive"):
        SmolVLAConfig(
            worldflow_enable=True,
            worldflow_residual_lr_multiplier=0.0,
        )


def test_worldflow_training_world_to_ego_dropout_is_training_only_and_bounded():
    config = SmolVLAConfig(
        worldflow_enable=True,
        worldflow_noise_coupling="projected_ego_path",
        worldflow_action_fusion="endpoint_residual_boosting",
        worldflow_training_world_to_ego_dropout_probability=0.5,
    )
    assert config.worldflow_training_world_to_ego_dropout_probability == pytest.approx(0.5)

    with pytest.raises(ValueError, match=r"must be in \[0,1\)"):
        SmolVLAConfig(
            worldflow_enable=True,
            worldflow_training_world_to_ego_dropout_probability=1.0,
        )
    with pytest.raises(ValueError, match="supported only"):
        SmolVLAConfig(
            worldflow_enable=True,
            worldflow_action_fusion="cross_attention",
            worldflow_training_world_to_ego_dropout_probability=0.5,
        )

    isolated = SmolVLAConfig(
        worldflow_enable=True,
        worldflow_noise_coupling="projected_ego_path",
        worldflow_action_fusion="endpoint_residual_boosting",
        worldflow_training_world_to_ego_dropout_probability=0.5,
        worldflow_training_residual_anchor_stop_gradient=True,
    )
    assert isolated.worldflow_training_residual_anchor_stop_gradient is True
    with pytest.raises(ValueError, match="requires training-time World-to-Ego dropout"):
        SmolVLAConfig(
            worldflow_enable=True,
            worldflow_noise_coupling="projected_ego_path",
            worldflow_action_fusion="endpoint_residual_boosting",
            worldflow_training_residual_anchor_stop_gradient=True,
        )

    projected = SmolVLAConfig(
        worldflow_enable=True,
        worldflow_noise_coupling="projected_ego_path",
        worldflow_action_fusion="endpoint_residual_boosting",
        worldflow_training_world_to_ego_dropout_probability=0.75,
        worldflow_training_residual_anchor_stop_gradient=True,
        worldflow_training_ego_priority_gradient_projection=True,
    )
    assert projected.worldflow_training_ego_priority_gradient_projection is True
    with pytest.raises(ValueError, match="requires endpoint_residual_boosting"):
        SmolVLAConfig(
            worldflow_enable=True,
            worldflow_noise_coupling="projected_ego_path",
            worldflow_action_fusion="endpoint_residual_boosting",
            worldflow_training_world_to_ego_dropout_probability=0.75,
            worldflow_training_ego_priority_gradient_projection=True,
        )

    tangent = SmolVLAConfig(
        worldflow_enable=True,
        worldflow_noise_coupling="projected_ego_path",
        worldflow_action_fusion="endpoint_residual_boosting",
        worldflow_training_world_to_ego_dropout_probability=0.75,
        worldflow_training_residual_anchor_stop_gradient=True,
        worldflow_training_shared_gradient_ego_tangent_projection=True,
    )
    assert tangent.worldflow_training_shared_gradient_ego_tangent_projection is True
    with pytest.raises(ValueError, match="Select only one"):
        SmolVLAConfig(
            worldflow_enable=True,
            worldflow_noise_coupling="projected_ego_path",
            worldflow_action_fusion="endpoint_residual_boosting",
            worldflow_training_world_to_ego_dropout_probability=0.75,
            worldflow_training_residual_anchor_stop_gradient=True,
            worldflow_training_ego_priority_gradient_projection=True,
            worldflow_training_shared_gradient_ego_tangent_projection=True,
        )


def test_worldflow_training_coordinate_frame_augmentation_requires_worldflow_and_valid_scales():
    with pytest.raises(ValueError, match="requires worldflow_enable=True"):
        SmolVLAConfig(worldflow_training_coordinate_frame_augmentation=True)

    with pytest.raises(ValueError, match="worldflow_augmentation_trans_scale must be non-negative"):
        SmolVLAConfig(worldflow_enable=True, worldflow_augmentation_trans_scale=-0.01)

    with pytest.raises(ValueError, match="worldflow_augmentation_rot_scale must be non-negative"):
        SmolVLAConfig(worldflow_enable=True, worldflow_augmentation_rot_scale=-0.01)

    with pytest.raises(ValueError, match="gradient_projection=True requires worldflow_enable=True"):
        SmolVLAConfig(worldflow_training_ego_priority_gradient_projection=True)

    with pytest.raises(ValueError, match="ego_tangent_projection=True requires worldflow_enable=True"):
        SmolVLAConfig(worldflow_training_shared_gradient_ego_tangent_projection=True)


@pytest.mark.parametrize(
    "fusion", ["voxel_cover_fps", "transport_novelty_union", "full_union"]
)
def test_input_multiview_discriminative_optimizer_keeps_all_paths_trainable(fusion):
    policy = SmolVLAPolicy.__new__(SmolVLAPolicy)
    nn.Module.__init__(policy)
    policy.config = SimpleNamespace(
        vla_adapter_enable=True,
        camera_view_fusion=fusion,
        worldflow_enable=False,
        multiview_input_pretrained_lr_multiplier=0.1,
        multiview_input_point_lr_multiplier=1.0,
        optimizer_lr=5e-6,
    )
    policy.model = nn.Module()
    policy.model.action_out_proj = nn.Linear(4, 4)
    policy.model.pointseg_conditioner = nn.Linear(4, 4)
    policy.model.pointseg_object_proj = nn.Linear(4, 4)
    policy.model.pointseg_background_proj = nn.Linear(4, 4)
    policy.model.point_action_fusion = nn.Linear(4, 4)

    groups = policy.get_optim_params()

    assert [group["group_name"] for group in groups] == [
        "pretrained_action_path_jointly_trainable",
        "point_input_adaptation_path",
    ]
    assert groups[0]["lr"] == pytest.approx(5e-7)
    assert groups[1]["lr"] == pytest.approx(5e-6)
    assert all(parameter.requires_grad for group in groups for parameter in group["params"])
    grouped = [id(parameter) for group in groups for parameter in group["params"]]
    expected = [id(parameter) for parameter in policy.parameters() if parameter.requires_grad]
    assert len(grouped) == len(set(grouped)) == len(expected)
    assert set(grouped) == set(expected)


def test_joint_input_multiview_worldflow_optimizer_keeps_all_four_paths_trainable():
    policy = SmolVLAPolicy.__new__(SmolVLAPolicy)
    nn.Module.__init__(policy)
    policy.config = SimpleNamespace(
        vla_adapter_enable=True,
        camera_view_fusion="novelty_union",
        worldflow_enable=True,
        multiview_input_pretrained_lr_multiplier=0.05,
        multiview_input_point_lr_multiplier=1.0,
        worldflow_pretrained_lr_multiplier=0.05,
        worldflow_new_lr_multiplier=0.2,
        worldflow_residual_lr_multiplier=1.0,
        optimizer_lr=1e-6,
    )
    policy.model = nn.Module()
    policy.model.action_out_proj = nn.Linear(4, 4)
    policy.model.pointseg_conditioner = nn.Linear(4, 4)
    policy.model.pointseg_object_proj = nn.Linear(4, 4)
    policy.model.pointseg_background_proj = nn.Linear(4, 4)
    policy.model.point_action_fusion = nn.Linear(4, 4)
    policy.model.worldflow_branch = nn.Linear(4, 4)
    policy.model.world_to_ego_cross_attn = nn.MultiheadAttention(4, 1, batch_first=True)
    policy.model.world_twist_residual_out_proj = nn.Linear(4, 4)

    groups = policy.get_optim_params()

    assert [group["group_name"] for group in groups] == [
        "pretrained_ego_shared_nonpoint",
        "point_input_adaptation_path",
        "new_world_bidirectional",
        "world_physical_residual_head",
    ]
    assert [group["lr"] for group in groups] == pytest.approx(
        [5e-8, 1e-6, 2e-7, 1e-6]
    )
    assert all(parameter.requires_grad for group in groups for parameter in group["params"])
    grouped = [id(parameter) for group in groups for parameter in group["params"]]
    expected = [id(parameter) for parameter in policy.parameters() if parameter.requires_grad]
    assert len(grouped) == len(set(grouped)) == len(expected)
    assert set(grouped) == set(expected)


def test_joint_multiview_worldflow_symmetric_point_adaptation_groups_both_input_paths():
    policy = SmolVLAPolicy.__new__(SmolVLAPolicy)
    nn.Module.__init__(policy)
    policy.config = SimpleNamespace(
        vla_adapter_enable=True,
        camera_view_fusion="multiscale_novelty_union",
        worldflow_enable=True,
        multiview_input_pretrained_lr_multiplier=0.005,
        multiview_input_point_lr_multiplier=0.05,
        multiview_input_symmetric_point_path_adaptation=True,
        worldflow_pretrained_lr_multiplier=0.005,
        worldflow_new_lr_multiplier=0.005,
        worldflow_residual_lr_multiplier=0.005,
        optimizer_lr=1e-6,
    )
    policy.model = nn.Module()
    policy.model.action_out_proj = nn.Linear(4, 4)
    policy.model.pointseg_conditioner = nn.Linear(4, 4)
    policy.model.pointseg_object_proj = nn.Linear(4, 4)
    policy.model.pointseg_background_proj = nn.Linear(4, 4)
    policy.model.point_action_fusion = nn.Linear(4, 4)
    policy.model.worldflow_branch = nn.Module()
    policy.model.worldflow_branch.scene_encoder = nn.Linear(4, 4)
    policy.model.worldflow_branch.scene_context_proj = nn.Linear(4, 4)
    policy.model.worldflow_branch.point_action_adapter = nn.Linear(4, 4)
    policy.model.worldflow_branch.action_in_proj = nn.Linear(4, 4)
    policy.model.ego_scene_to_expert = nn.Linear(4, 4)
    policy.model.world_scene_to_expert = nn.Linear(4, 4)
    policy.model.world_to_ego_cross_attn = nn.MultiheadAttention(4, 1, batch_first=True)
    policy.model.world_twist_residual_out_proj = nn.Linear(4, 4)

    groups = policy.get_optim_params()

    assert [group["group_name"] for group in groups] == [
        "pretrained_ego_shared_nonpoint",
        "point_input_adaptation_path",
        "new_world_bidirectional",
        "world_physical_residual_head",
    ]
    assert [group["lr"] for group in groups] == pytest.approx(
        [5e-9, 5e-8, 5e-9, 5e-9]
    )
    point_ids = {id(parameter) for parameter in groups[1]["params"]}
    for module in (
        policy.model.point_action_fusion,
        policy.model.worldflow_branch.scene_encoder,
        policy.model.worldflow_branch.scene_context_proj,
        policy.model.worldflow_branch.point_action_adapter,
        policy.model.ego_scene_to_expert,
        policy.model.world_scene_to_expert,
    ):
        assert {id(parameter) for parameter in module.parameters()} <= point_ids
    world_action_input_ids = {
        id(parameter) for parameter in policy.model.worldflow_branch.action_in_proj.parameters()
    }
    assert not (world_action_input_ids & point_ids)
    world_only_role_ids = policy.get_worldflow_ego_tangent_world_only_parameter_ids()
    for module in (
        policy.model.worldflow_branch.scene_encoder,
        policy.model.worldflow_branch.scene_context_proj,
        policy.model.worldflow_branch.point_action_adapter,
        policy.model.ego_scene_to_expert,
        policy.model.world_scene_to_expert,
    ):
        assert {id(parameter) for parameter in module.parameters()} <= world_only_role_ids
    for module in (
        policy.model.pointseg_conditioner,
        policy.model.pointseg_object_proj,
        policy.model.pointseg_background_proj,
        policy.model.point_action_fusion,
    ):
        assert not ({id(parameter) for parameter in module.parameters()} & world_only_role_ids)
    assert world_only_role_ids < point_ids
    grouped = [id(parameter) for group in groups for parameter in group["params"]]
    expected = [id(parameter) for parameter in policy.parameters() if parameter.requires_grad]
    assert len(grouped) == len(set(grouped)) == len(expected)
    assert set(grouped) == set(expected)


def test_symmetric_point_path_adaptation_requires_worldflow():
    with pytest.raises(ValueError, match="requires worldflow_enable=True"):
        SmolVLAConfig(multiview_input_symmetric_point_path_adaptation=True)

    with pytest.raises(ValueError, match="requires a supported input-layer multi-view fusion"):
        SmolVLAConfig(
            worldflow_enable=True,
            multiview_input_symmetric_point_path_adaptation=True,
        )


def test_symmetric_world_point_gradient_roles_are_disabled_by_default():
    policy = SmolVLAPolicy.__new__(SmolVLAPolicy)
    nn.Module.__init__(policy)
    policy.config = SimpleNamespace(
        worldflow_enable=True,
        multiview_input_symmetric_point_path_adaptation=False,
    )
    policy.model = nn.Linear(4, 4)

    assert policy.get_worldflow_ego_tangent_world_only_parameter_ids() == set()


def test_joint_input_multiview_worldflow_requires_one_shared_ego_learning_rate():
    policy = SmolVLAPolicy.__new__(SmolVLAPolicy)
    nn.Module.__init__(policy)
    policy.config = SimpleNamespace(
        vla_adapter_enable=True,
        camera_view_fusion="novelty_union",
        worldflow_enable=True,
        multiview_input_pretrained_lr_multiplier=0.1,
        multiview_input_point_lr_multiplier=1.0,
        worldflow_pretrained_lr_multiplier=0.05,
        worldflow_new_lr_multiplier=0.2,
        worldflow_residual_lr_multiplier=1.0,
        optimizer_lr=1e-6,
    )
    policy.model = nn.Module()
    policy.model.action_out_proj = nn.Linear(4, 4)
    policy.model.pointseg_conditioner = nn.Linear(4, 4)

    with pytest.raises(ValueError, match="one unambiguous learning rate"):
        policy.get_optim_params()


def test_input_view_dropout_requires_supported_fusion_and_multiple_views():
    cfg = SmolVLAConfig(
        camera_views="agentview,robot0_eye_in_hand",
        camera_view_fusion="fps",
        multiview_input_view_dropout_enable=True,
    )
    assert cfg.multiview_input_view_dropout_enable is True

    full_union_cfg = SmolVLAConfig(
        camera_views="agentview,robot0_eye_in_hand",
        camera_view_fusion="full_union",
        multiview_input_view_dropout_enable=True,
    )
    assert full_union_cfg.multiview_input_view_dropout_enable is True

    novelty_union_cfg = SmolVLAConfig(
        camera_views="agentview,robot0_eye_in_hand",
        camera_view_fusion="novelty_union",
        multiview_input_view_dropout_enable=True,
    )
    assert novelty_union_cfg.multiview_input_view_dropout_enable is True

    multiscale_novelty_union_cfg = SmolVLAConfig(
        camera_views="agentview,robot0_eye_in_hand",
        camera_view_fusion="multiscale_novelty_union",
        multiview_input_view_dropout_enable=True,
    )
    assert multiscale_novelty_union_cfg.multiview_input_view_dropout_enable is True
    assert multiscale_novelty_union_cfg.camera_view_coarse_novelty_scale == 3.0

    conservative_multiscale_cfg = SmolVLAConfig(
        camera_views="agentview,robot0_eye_in_hand",
        camera_view_fusion="multiscale_novelty_union",
        camera_view_coarse_novelty_scale=4.0,
    )
    assert conservative_multiscale_cfg.camera_view_coarse_novelty_scale == 4.0
    consensus_multiscale_cfg = SmolVLAConfig(
        camera_views="agentview,robot0_eye_in_hand",
        camera_view_fusion="consensus_multiscale_novelty_union",
        camera_view_coarse_novelty_scale=4.0,
        multiview_input_view_dropout_enable=True,
    )
    assert consensus_multiscale_cfg.camera_view_coarse_novelty_scale == 4.0
    assert consensus_multiscale_cfg.multiview_input_view_dropout_enable is True
    with pytest.raises(ValueError, match="coarse_novelty_scale"):
        SmolVLAConfig(camera_view_coarse_novelty_scale=1.0)

    transport_novelty_cfg = SmolVLAConfig(
        camera_views="agentview,robot0_eye_in_hand",
        camera_view_fusion="transport_novelty_union",
        multiview_input_view_dropout_enable=True,
    )
    assert transport_novelty_cfg.multiview_input_view_dropout_enable is True

    uniform_union_cfg = SmolVLAConfig(
        camera_views="agentview,robot0_eye_in_hand",
        camera_view_fusion="uniform_union",
        multiview_input_view_dropout_enable=True,
    )
    assert uniform_union_cfg.multiview_input_view_dropout_enable is True

    with pytest.raises(ValueError, match="camera_view_fusion='fps', 'novelty_union'"):
        SmolVLAConfig(
            camera_views="agentview,robot0_eye_in_hand",
            camera_view_fusion="legacy_budget",
            multiview_input_view_dropout_enable=True,
        )
    with pytest.raises(ValueError, match="requires at least two camera_views"):
        SmolVLAConfig(
            camera_views="agentview",
            camera_view_fusion="fps",
            multiview_input_view_dropout_enable=True,
        )
    with pytest.raises(ValueError, match="paired_coverage requires"):
        SmolVLAConfig(multiview_input_view_dropout_paired_coverage=True)

def test_full_union_prepare_point_clouds_accepts_padded_variable_length_input():
    policy = SmolVLAPolicy.__new__(SmolVLAPolicy)
    nn.Module.__init__(policy)
    policy.config = SimpleNamespace(
        camera_view_fusion="full_union",
        camera_view_fps_target_points=8,
    )
    policy.model = SimpleNamespace(pointseg_conditioner=None)
    point_cloud = torch.zeros(2, 14, 6)
    point_is_pad = torch.zeros(2, 14, dtype=torch.bool)
    point_is_pad[0, 8:] = True

    payloads, masks = policy.prepare_point_clouds(
        {
            "observation.point_cloud": point_cloud,
            "observation.point_cloud_is_pad": point_is_pad,
        }
    )

    assert payloads[0]["point_cloud"].shape == (2, 14, 6)
    assert torch.equal(payloads[0]["point_is_pad"], point_is_pad)
    assert masks[0].tolist() == [True, True]


@pytest.mark.parametrize("num_points", [10_000, 50_000])
def test_prepare_point_clouds_accepts_dataset_defined_point_count(num_points):
    policy = SmolVLAPolicy.__new__(SmolVLAPolicy)
    nn.Module.__init__(policy)
    policy.config = SimpleNamespace(
        camera_view_fusion="legacy_budget",
        camera_view_fps_target_points=10_000,
    )
    policy.model = SimpleNamespace(pointseg_conditioner=None)
    point_cloud = torch.zeros(2, num_points, 6)

    payloads, masks = policy.prepare_point_clouds(
        {"observation.point_cloud": point_cloud}
    )

    assert payloads[0].shape == (2, num_points, 6)
    assert payloads[0].dtype == torch.float32
    assert masks[0].tolist() == [True, True]


def test_prepare_point_clouds_rejects_misaligned_point_padding_mask():
    policy = SmolVLAPolicy.__new__(SmolVLAPolicy)
    nn.Module.__init__(policy)
    policy.config = SimpleNamespace()
    policy.model = SimpleNamespace(pointseg_conditioner=None)

    with pytest.raises(ValueError, match="point_cloud_is_pad must match"):
        policy.prepare_point_clouds(
            {
                "observation.point_cloud": torch.zeros(2, 50_000, 6),
                "observation.point_cloud_is_pad": torch.zeros(2, 10_000, dtype=torch.bool),
            }
        )


def test_pointseg_batchnorm_stats_can_be_frozen_without_freezing_parameters():
    policy = SmolVLAPolicy.__new__(SmolVLAPolicy)
    nn.Module.__init__(policy)
    policy.config = SimpleNamespace(pointseg_freeze_batchnorm_stats=True)
    policy.model = nn.Module()
    policy.model.pointseg_conditioner = nn.Sequential(
        nn.BatchNorm1d(4),
        nn.Linear(4, 4),
    )

    policy.train(True)

    batchnorm = policy.model.pointseg_conditioner[0]
    linear = policy.model.pointseg_conditioner[1]
    assert not batchnorm.training
    assert linear.training
    assert batchnorm.weight.requires_grad
    assert batchnorm.bias.requires_grad


def test_worldflow_protected_ego_also_freezes_ego_pointseg_batchnorm_stats():
    policy = SmolVLAPolicy.__new__(SmolVLAPolicy)
    nn.Module.__init__(policy)
    policy.config = SimpleNamespace(
        pointseg_freeze_batchnorm_stats=False,
        worldflow_freeze_pretrained_ego=True,
        worldflow_freeze_batchnorm_stats=False,
    )
    policy.model = nn.Module()
    policy.model.pointseg_conditioner = nn.Sequential(
        nn.BatchNorm1d(4),
        nn.Linear(4, 4),
    )

    policy.train(True)

    assert not policy.model.pointseg_conditioner[0].training
    assert policy.model.pointseg_conditioner[1].training


def test_worldflow_batchnorm_stats_can_be_frozen_without_freezing_parameters():
    policy = SmolVLAPolicy.__new__(SmolVLAPolicy)
    nn.Module.__init__(policy)
    policy.config = SimpleNamespace(worldflow_freeze_batchnorm_stats=True)
    policy.model = nn.Module()
    policy.model.worldflow_branch = nn.Module()
    policy.model.worldflow_branch.scene_encoder = nn.Sequential(
        nn.BatchNorm1d(4),
        nn.Linear(4, 4),
    )

    policy.train(True)

    batchnorm = policy.model.worldflow_branch.scene_encoder[0]
    linear = policy.model.worldflow_branch.scene_encoder[1]
    assert not batchnorm.training
    assert linear.training
    assert batchnorm.weight.requires_grad
    assert batchnorm.bias.requires_grad


def test_worldflow_joint_loss_uses_foreground_only_and_backpropagates_bridge():
    torch.manual_seed(19)
    cfg = SmolVLAConfig(
        chunk_size=4,
        n_action_steps=4,
        pointseg_enable=True,
        worldflow_enable=True,
        worldflow_feature_dim=16,
        point_action_fusion_heads=4,
        worldflow_equiv_loss_weight=0.0,
        worldflow_max_points=8,
    )
    model = VLAFlowMatching.__new__(VLAFlowMatching)
    nn.Module.__init__(model)
    model.config = cfg
    model.worldflow_branch = _make_tiny_worldflow_branch(cfg)
    model.last_worldflow_metrics = {}

    bsize, chunk_size = 2, cfg.chunk_size
    foreground = torch.randn(bsize, 12, 6)
    foreground[..., 3:6] = torch.rand(bsize, 12, 3) * 255.0
    model.last_worldflow_payload = {"foreground_pc_ego": foreground}
    current = se3_exp(torch.randn(bsize, 6) * 0.1)
    target = se3_exp(torch.randn(bsize, chunk_size, 6) * 0.1)
    body_prediction = matrix_to_pose9(
        torch.linalg.inv(current).unsqueeze(1) @ target
    ).detach().requires_grad_(True)
    context = {
        "current_ee_pose": matrix_to_pose9(current),
        "ee_poses": matrix_to_pose9(target),
        "step_is_pad": torch.zeros(bsize, chunk_size, dtype=torch.bool),
    }
    lang_tokens = torch.randint(0, 64, (bsize, 5))
    lang_masks = torch.ones(bsize, 5, dtype=torch.bool)
    time = torch.rand(bsize)

    state = model._prepare_worldflow_training_state(
        context,
        lang_tokens,
        lang_masks,
        time,
        actions_is_pad=None,
    )
    pred_velocity = torch.zeros_like(state["u_t"], requires_grad=True)
    output = model._finalize_worldflow_training_loss(
        state,
        pred_velocity,
        body_prediction,
        time,
    )
    output["per_sample_loss"].mean().backward()

    assert output["pred_spatial"].shape == (bsize, chunk_size, 4, 4)
    assert body_prediction.grad is not None
    assert "loss_worldflow_geo" in model.last_worldflow_metrics
    assert model.last_worldflow_metrics["worldflow_foreground_points"].item() == cfg.worldflow_max_points


def test_world_eef_trajectory_is_not_conjugated_or_residualized():
    torch.manual_seed(23)
    cfg = SmolVLAConfig(
        chunk_size=3,
        n_action_steps=3,
        pointseg_enable=True,
        worldflow_enable=True,
        worldflow_target_type="world_eef_trajectory",
        worldflow_reference_frame="robot_base",
        worldflow_frame_origin="global",
        worldflow_scene_frame_origin="global",
        worldflow_noise_coupling="independent",
        worldflow_action_fusion="cross_attention",
        worldflow_bridge_loss_weight=0.0,
        worldflow_equiv_loss_weight=0.0,
        worldflow_feature_dim=16,
        point_action_fusion_heads=4,
    )
    model = VLAFlowMatching.__new__(VLAFlowMatching)
    nn.Module.__init__(model)
    model.config = cfg
    model.worldflow_branch = _make_tiny_worldflow_branch(cfg)
    model.last_worldflow_metrics = {}
    foreground_pc_ego = torch.randn(2, 8, 6)
    model.last_worldflow_payload = {"foreground_pc_ego": foreground_pc_ego}

    current_eef = matrix_to_pose9(se3_exp(torch.randn(2, 6) * 0.4))
    world_eef_target = matrix_to_pose9(se3_exp(torch.randn(2, cfg.chunk_size, 6) * 0.1))
    state = model._prepare_worldflow_training_state(
        {
            "current_ee_pose": current_eef,
            "eef_trajectory": world_eef_target,
            "step_is_pad": torch.zeros(2, cfg.chunk_size, dtype=torch.bool),
        },
        torch.randint(0, 64, (2, 5)),
        torch.ones(2, 5, dtype=torch.bool),
        torch.rand(2),
        actions_is_pad=None,
    )

    assert torch.allclose(state["spatial_gt"], pose9_to_matrix(world_eef_target), atol=1e-6)
    current_eef_matrix = pose9_to_matrix(current_eef)
    expected_xyz_base = (
        current_eef_matrix[:, None, :3, :3]
        @ foreground_pc_ego[..., :3, None]
    ).squeeze(-1) + current_eef_matrix[:, None, :3, 3]
    assert torch.allclose(
        state["point_cloud_world"][..., :3],
        expected_xyz_base,
        atol=1e-5,
    )
    assert state["u_t"].shape == (2, cfg.chunk_size, 6)
    identity = torch.eye(4).expand(2, -1, -1)
    assert torch.equal(state["current"], identity)
    output = model._finalize_worldflow_training_loss(
        state,
        torch.zeros_like(state["u_t"], requires_grad=True),
        matrix_to_pose9(se3_exp(torch.randn(2, cfg.chunk_size, 6))),
        torch.rand(2),
    )
    assert output["loss_bridge"].item() == 0.0
    assert model.last_worldflow_metrics["loss_worldflow_bridge"].item() == 0.0


def test_world_eef_trajectory_left_composes_ego_prior_in_robot_base():
    torch.manual_seed(29)
    cfg = SmolVLAConfig(
        chunk_size=3,
        n_action_steps=3,
        pointseg_enable=True,
        worldflow_enable=True,
        worldflow_target_type="world_eef_trajectory",
        worldflow_reference_frame="robot_base",
        worldflow_frame_origin="global",
        worldflow_scene_frame_origin="global",
        worldflow_noise_coupling="left_compose_ego",
        worldflow_world_eef_velocity_mode="base_decoupled",
        worldflow_action_fusion="physical_trajectory_cross_attention",
        worldflow_action_expert_mode="independent",
        worldflow_current_ee_pose_token=True,
        worldflow_bridge_loss_weight=0.0,
        worldflow_equiv_loss_weight=0.0,
        worldflow_feature_dim=16,
        point_action_fusion_heads=4,
    )
    model = VLAFlowMatching.__new__(VLAFlowMatching)
    nn.Module.__init__(model)
    model.config = cfg
    model.worldflow_branch = _make_tiny_worldflow_branch(cfg)
    model.last_worldflow_metrics = {}
    model.last_worldflow_payload = {"foreground_pc_ego": torch.randn(2, 8, 6)}

    current = se3_exp(torch.randn(2, 6) * 0.2)
    ego_noise_h = se3_exp(torch.randn(2, cfg.chunk_size, 6) * 0.1)
    ego_noise = torch.cat(
        [matrix_to_pose9(ego_noise_h), torch.zeros(2, cfg.chunk_size, 1)],
        dim=-1,
    )
    target = current.unsqueeze(1) @ se3_exp(
        torch.randn(2, cfg.chunk_size, 6) * 0.1
    )
    time = torch.tensor([0.2, 0.7])
    state = model._prepare_worldflow_training_state(
        {
            "current_ee_pose": matrix_to_pose9(current),
            "eef_trajectory": matrix_to_pose9(target),
            "step_is_pad": torch.zeros(2, cfg.chunk_size, dtype=torch.bool),
        },
        torch.randint(0, 64, (2, 5)),
        torch.ones(2, 5, dtype=torch.bool),
        time,
        actions_is_pad=None,
        ego_noise=ego_noise,
    )

    expected_noise = current.unsqueeze(1) @ ego_noise_h
    assert torch.allclose(state["noise_spatial"], expected_noise, atol=1e-5)
    wrong_conjugate = current.unsqueeze(1) @ ego_noise_h @ torch.linalg.inv(current).unsqueeze(1)
    assert not torch.allclose(state["noise_spatial"], wrong_conjugate, atol=1e-4)
    assert torch.allclose(
        state["noise_conjugacy_error"],
        torch.zeros_like(state["noise_conjugacy_error"]),
        atol=1e-6,
    )
    output = model._finalize_worldflow_training_loss(
        state,
        state["u_t"].clone().requires_grad_(True),
        matrix_to_pose9(target),
        time,
    )
    assert output["per_sample_loss"].abs().max().item() < 1e-5
    assert model.last_worldflow_metrics["worldflow_flow_translation_err_m"].item() < 1e-6
    assert model.last_worldflow_metrics["worldflow_flow_rotation_probe_err_m"].item() < 1e-6
    assert model.last_worldflow_metrics["worldflow_endpoint_translation_err_m"].item() < 1e-6
    assert model.last_worldflow_metrics["worldflow_endpoint_rotation_probe_err_m"].item() < 1e-6


def test_world_eef_pose9_path_is_exactly_left_equivariant_to_ego_chart():
    torch.manual_seed(31)
    cfg = SmolVLAConfig(
        chunk_size=3,
        n_action_steps=3,
        pointseg_enable=True,
        worldflow_enable=True,
        worldflow_target_type="world_eef_trajectory",
        worldflow_reference_frame="robot_base",
        worldflow_frame_origin="global",
        worldflow_scene_frame_origin="global",
        worldflow_noise_coupling="left_compose_ego",
        worldflow_world_eef_velocity_mode="base_pose9_euclidean",
        worldflow_action_fusion="independent_parallel",
        worldflow_action_expert_mode="independent",
        worldflow_current_ee_pose_token=True,
        worldflow_freeze_pretrained_ego=False,
        worldflow_bridge_loss_weight=0.0,
        worldflow_equiv_loss_weight=0.0,
        worldflow_feature_dim=16,
        point_action_fusion_heads=4,
    )
    model = VLAFlowMatching.__new__(VLAFlowMatching)
    nn.Module.__init__(model)
    model.config = cfg
    model.worldflow_branch = _make_tiny_worldflow_branch(cfg)
    model.last_worldflow_metrics = {}
    model.last_worldflow_payload = {"foreground_pc_ego": torch.randn(2, 8, 6)}

    current = se3_exp(torch.randn(2, 6) * 0.2)
    # The formal checkpoint uses the standard raw Gaussian action prior, not
    # a valid SE(3) pose prior.  This regression deliberately keeps both 6D
    # rotation columns unconstrained: projecting them before the frame change
    # would make the World path non-affine and fail the assertions below.
    ego_noise_pose9 = torch.randn(2, cfg.chunk_size, 9) * 0.1
    ego_target_h = se3_exp(torch.randn(2, cfg.chunk_size, 6) * 0.2)
    ego_noise = torch.cat(
        [ego_noise_pose9, torch.randn(2, cfg.chunk_size, 1) * 0.1], dim=-1
    )
    ego_target = matrix_to_pose9(ego_target_h)
    time = torch.tensor([0.25, 0.75])
    ego_gripper_target = torch.randn(2, cfg.chunk_size, 1) * 0.1
    ego_action_target = torch.cat([ego_target, ego_gripper_target], dim=-1)
    ego_action_x_t = (
        (1.0 - time[:, None, None]) * ego_noise
        + time[:, None, None] * ego_action_target
    )
    world_target_h = current.unsqueeze(1) @ ego_target_h
    state = model._prepare_worldflow_training_state(
        {
            "current_ee_pose": matrix_to_pose9(current),
            "eef_trajectory": matrix_to_pose9(world_target_h),
            "step_is_pad": torch.zeros(2, cfg.chunk_size, dtype=torch.bool),
        },
        torch.randint(0, 64, (2, 5)),
        torch.ones(2, 5, dtype=torch.bool),
        time,
        actions_is_pad=None,
        ego_noise=ego_noise,
        ego_x_t=ego_action_x_t,
        ego_target=ego_action_target,
    )

    ego_x_t = (1.0 - time[:, None, None]) * ego_noise[..., :9] + time[
        :, None, None
    ] * ego_target
    ego_u_t = ego_target - ego_noise[..., :9]
    current_rotation = current[:, None, :3, :3]
    expected_x_t = ego_x_t.clone()
    expected_x_t[..., :3] = (
        current_rotation @ ego_x_t[..., :3, None]
    ).squeeze(-1) + current[:, None, :3, 3]
    expected_x_t[..., 3:6] = (
        current_rotation @ ego_x_t[..., 3:6, None]
    ).squeeze(-1)
    expected_x_t[..., 6:9] = (
        current_rotation @ ego_x_t[..., 6:9, None]
    ).squeeze(-1)
    expected_u_t = ego_u_t.clone()
    for component in (slice(0, 3), slice(3, 6), slice(6, 9)):
        expected_u_t[..., component] = (
            current_rotation @ ego_u_t[..., component, None]
        ).squeeze(-1)

    raw_world_noise = model.left_compose_ego_pose_to_world(ego_noise, matrix_to_pose9(current))
    projected_world_noise = matrix_to_pose9(
        current.unsqueeze(1) @ pose9_to_matrix(ego_noise[..., :9])
    )

    assert torch.allclose(state["x_t"][..., :9], expected_x_t, atol=2e-5, rtol=2e-5)
    assert torch.equal(state["x_t"][..., 9:10], ego_action_x_t[..., 9:10])
    assert torch.allclose(raw_world_noise, expected_x_t - time[:, None, None] * expected_u_t, atol=2e-5, rtol=2e-5)
    assert not torch.allclose(raw_world_noise, projected_world_noise, atol=1e-4, rtol=1e-4)
    assert torch.allclose(state["u_t"][..., :9], expected_u_t, atol=2e-5, rtol=2e-5)
    assert torch.equal(
        state["u_t"][..., 9:10],
        ego_action_target[..., 9:10] - ego_noise[..., 9:10],
    )
    output = model._finalize_worldflow_training_loss(
        state,
        state["u_t"].clone().requires_grad_(True),
        ego_target,
        time,
    )
    assert output["per_sample_loss"].abs().max().item() < 2e-5
    assert model.last_worldflow_metrics["worldflow_flow_rotation6d_rmse"].item() < 1e-7

    gripper_offset_prediction = state["u_t"].clone()
    gripper_offset_prediction[..., 9] += 0.2
    gripper_output = model._finalize_worldflow_training_loss(
        state,
        gripper_offset_prediction,
        ego_target,
        time,
    )
    # One erroneous channel among ten: 0.2^2 / 10.
    assert gripper_output["loss_flow"].item() == pytest.approx(0.004, abs=1e-7)
    assert model.last_worldflow_metrics["loss_worldflow_gripper"].item() == pytest.approx(
        0.04,
        abs=1e-7,
    )


def test_joint_se3_worldflow_training_path_is_exactly_conjugate():
    torch.manual_seed(21)
    cfg = SmolVLAConfig(
        chunk_size=4,
        n_action_steps=4,
        pointseg_enable=True,
        worldflow_enable=True,
        se3_enable=True,
        worldflow_noise_coupling="conjugate_ego",
        worldflow_feature_dim=16,
        point_action_fusion_heads=4,
        worldflow_equiv_loss_weight=0.0,
        worldflow_max_points=8,
    )
    model = VLAFlowMatching.__new__(VLAFlowMatching)
    nn.Module.__init__(model)
    model.config = cfg
    model.worldflow_branch = _make_tiny_worldflow_branch(cfg)
    model.last_worldflow_metrics = {}

    bsize, chunk_size = 2, cfg.chunk_size
    foreground = torch.randn(bsize, 12, 6)
    foreground[..., 3:6] = torch.rand(bsize, 12, 3) * 255.0
    model.last_worldflow_payload = {"foreground_pc_ego": foreground}
    current = se3_exp(torch.randn(bsize, 6) * 0.1)
    body_target = se3_exp(torch.randn(bsize, chunk_size, 6) * 0.1)
    target = current.unsqueeze(1) @ body_target
    ego_noise_h = se3_exp(torch.randn(bsize, chunk_size, 6) * 0.1)
    ego_noise = torch.cat(
        [matrix_to_pose9(ego_noise_h), torch.zeros(bsize, chunk_size, 1)],
        dim=-1,
    )
    time = torch.tensor([0.2, 0.7])
    ego_state_h, _ = se3_geodesic_flow_state(ego_noise_h, body_target, time)
    ego_x_t = torch.cat(
        [matrix_to_pose9(ego_state_h), torch.zeros(bsize, chunk_size, 1)],
        dim=-1,
    )
    context = {
        "current_ee_pose": matrix_to_pose9(current),
        "ee_poses": matrix_to_pose9(target),
        "step_is_pad": torch.zeros(bsize, chunk_size, dtype=torch.bool),
    }

    state = model._prepare_worldflow_training_state(
        context,
        torch.randint(0, 64, (bsize, 5)),
        torch.ones(bsize, 5, dtype=torch.bool),
        time,
        actions_is_pad=None,
        ego_noise=ego_noise,
        ego_x_t=ego_x_t,
    )
    output = model._finalize_worldflow_training_loss(
        state,
        state["u_t"],
        matrix_to_pose9(body_target),
        time,
    )

    assert torch.allclose(state["path_conjugacy_error"], torch.zeros_like(state["path_conjugacy_error"]), atol=2e-4)
    assert torch.allclose(output["pred_spatial"], state["spatial_gt"], atol=5e-4, rtol=5e-4)
    assert output["loss_flow"].item() < 1e-7
    assert output["loss_geo"].item() < 1e-4
    assert output["loss_bridge"].item() < 1e-4


def test_world_ego_bridge_formula_matches_body_transform():
    torch.manual_seed(23)
    current = se3_exp(torch.randn(3, 6) * 0.2)
    body = se3_exp(torch.randn(3, 6) * 0.2)
    target = current @ body
    spatial = target @ torch.linalg.inv(current)
    recovered_body = torch.linalg.inv(current) @ spatial @ current

    assert torch.allclose(recovered_body, body, atol=3e-4, rtol=3e-4)


def test_worldflow_coordinate_change_uses_se3_conjugation():
    torch.manual_seed(29)
    current = se3_exp(torch.randn(3, 6) * 0.2)
    target = se3_exp(torch.randn(3, 6) * 0.2)
    coordinate_change = se3_exp(torch.randn(3, 6) * 0.2)

    spatial = target @ torch.linalg.inv(current)
    spatial_aug = coordinate_change @ spatial @ torch.linalg.inv(coordinate_change)
    current_aug = coordinate_change @ current
    recovered_body = torch.linalg.inv(current_aug) @ spatial_aug @ current_aug

    expected_body = torch.linalg.inv(current) @ target
    assert torch.allclose(recovered_body, expected_body, atol=3e-4, rtol=3e-4)


def test_worldflow_branch_is_part_of_policy_inference_call_path():
    assert hasattr(VLAFlowMatching, "denoise_step_world_ego")


def test_worldflow_scalar_residual_gates_are_rejected():
    with pytest.raises(ValueError, match="residual gates are unsupported"):
        SmolVLAConfig(
            worldflow_enable=True,
            worldflow_ego_residual_gate_init=0.0,
        )
