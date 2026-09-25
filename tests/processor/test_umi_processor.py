import torch

from lerobot.processor.core import TransitionKey
from lerobot.processor.umi_processor import UMIProcessor


def _pose10(x: float, gripper: float = 0.08) -> list[float]:
    return [x, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, gripper]


def _pose10_from_transform(transform: torch.Tensor, gripper: float = 0.08) -> torch.Tensor:
    return torch.cat(
        [
            transform[:3, 3],
            transform[:3, 0],
            transform[:3, 1],
            transform.new_tensor([gripper]),
        ]
    )


def _transform(x: float, y: float, yaw: float) -> torch.Tensor:
    cosine = torch.cos(torch.tensor(yaw))
    sine = torch.sin(torch.tensor(yaw))
    return torch.tensor(
        [
            [cosine, -sine, 0.0, x],
            [sine, cosine, 0.0, y],
            [0.0, 0.0, 1.0, 0.2],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )


def test_umi_processor_uses_observed_state_as_origin_for_absolute_targets():
    state = torch.tensor([[_pose10(0.1)]], dtype=torch.float32)
    action = torch.tensor(
        [[_pose10(0.3), _pose10(0.4)]],
        dtype=torch.float32,
    )

    output = UMIProcessor()(
        {
            TransitionKey.OBSERVATION: {"observation.state": state},
            TransitionKey.ACTION: action,
        }
    )

    output_state = output[TransitionKey.OBSERVATION]["observation.state"]
    output_action = output[TransitionKey.ACTION]
    assert torch.allclose(output_state[0, 0, :3], torch.zeros(3), atol=1e-6)
    assert torch.allclose(
        output_action[0, :, :3],
        torch.tensor([[0.2, 0.0, 0.0], [0.3, 0.0, 0.0]]),
        atol=1e-6,
    )
    assert torch.allclose(output_action[..., 9], action[..., 9])


def test_umi_processor_supports_observation_only_inference():
    state = torch.tensor([_pose10(0.1)], dtype=torch.float32)

    output = UMIProcessor()(
        {TransitionKey.OBSERVATION: {"observation.state": state}}
    )

    output_state = output[TransitionKey.OBSERVATION]["observation.state"]
    assert output_state.shape == state.shape
    assert torch.allclose(output_state[0, :3], torch.zeros(3), atol=1e-6)
    assert torch.allclose(
        output_state[0, 3:9],
        torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]),
        atol=1e-6,
    )


def test_umi_processor_matches_current_eef_body_transform_with_rotation():
    processor = UMIProcessor()
    episode_origin = _transform(0.4, -0.2, 0.3)
    current = _transform(0.6, 0.1, -0.5)
    targets = torch.stack(
        [_transform(0.7, 0.0, 0.2), _transform(0.5, 0.4, 1.0)]
    )
    episode_origin_inv = torch.linalg.inv(episode_origin)
    stored_state = episode_origin_inv @ current
    stored_actions = episode_origin_inv.unsqueeze(0) @ targets

    state = _pose10_from_transform(stored_state).view(1, 1, 10)
    action = torch.stack([_pose10_from_transform(target) for target in stored_actions]).unsqueeze(0)
    output = processor(
        {
            TransitionKey.OBSERVATION: {"observation.state": state},
            TransitionKey.ACTION: action,
        }
    )

    actual = processor.pose9_to_homo(output[TransitionKey.ACTION][..., :9]).squeeze(0)
    expected = torch.linalg.inv(current).unsqueeze(0) @ targets
    assert torch.allclose(actual, expected, atol=2e-6)
    assert output[TransitionKey.ACTION].device == action.device
    assert output[TransitionKey.ACTION].dtype == action.dtype


@torch.no_grad()
def test_umi_tensor_path_preserves_low_precision_dtype():
    processor = UMIProcessor()
    poses = torch.tensor(
        [[_pose10(0.1)[:9], _pose10(0.2)[:9]]],
        dtype=torch.bfloat16,
    )
    result = processor.from_world_to_umi_tra_pose9_tensor(poses)
    assert result.device == poses.device
    assert result.dtype == poses.dtype
    assert torch.allclose(result[0, 0, :3].float(), torch.zeros(3), atol=1e-3)
