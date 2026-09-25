#!/usr/bin/env python

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy, VLAFlowMatching


class _DummyFrozenVLM(nn.Module):
    def __init__(self, hidden_dim: int = 8):
        super().__init__()
        self.language_embedding = nn.Embedding(16, hidden_dim)
        self.language_embedding.requires_grad_(False)

    def embed_image(self, image: torch.Tensor) -> torch.Tensor:
        value = image.mean(dim=(1, 2, 3), keepdim=False)[:, None, None]
        return value.expand(-1, 2, self.language_embedding.embedding_dim)

    def embed_language_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.language_embedding(tokens)


class _MeanPointExtractor(nn.Module):
    def forward(self, point_cloud: torch.Tensor, point_is_pad=None) -> torch.Tensor:
        del point_is_pad
        return point_cloud.mean(dim=1)


def _make_prefix_model() -> VLAFlowMatching:
    model = VLAFlowMatching.__new__(VLAFlowMatching)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(vla_adapter_enable=True, encode_robot_state=False)
    model.vlm_with_expert = _DummyFrozenVLM(hidden_dim=8)
    model.extractor = _MeanPointExtractor()
    model.pointcloud_proj = nn.Linear(6, 8)
    model.pointseg_conditioner = None
    model.pointseg_object_proj = None
    model.pointseg_background_proj = None
    model.point_action_fusion = None
    model.worldflow_head = None
    model.add_image_special_tokens = False
    model.prefix_length = -1
    return model


def test_adapter_config_enforces_pretrained_frozen_vlm():
    with pytest.warns(UserWarning):
        config = SmolVLAConfig(vla_adapter_enable=True)

    assert config.load_vlm_weights
    assert config.train_expert_only
    assert config.freeze_vision_encoder


def test_prepare_images_uses_last_static_frame_and_official_range():
    image_key = "observation.images.overhead"
    policy = SimpleNamespace(
        config=SimpleNamespace(
            image_features={
                image_key: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 4, 5)),
            },
            resize_imgs_with_padding=None,
            empty_cameras=0,
        )
    )
    image_sequence = torch.stack(
        [torch.zeros(2, 3, 4, 5), torch.ones(2, 3, 4, 5)],
        dim=1,
    )

    images, masks = SmolVLAPolicy.prepare_images(policy, {image_key: image_sequence})

    assert len(images) == len(masks) == 1
    assert images[0].shape == (2, 3, 4, 5)
    assert torch.equal(images[0], torch.ones_like(images[0]))
    assert torch.equal(masks[0], torch.ones(2, dtype=torch.bool))


def test_image_point_language_share_prefix_while_only_point_path_gets_gradient():
    torch.manual_seed(3)
    model = _make_prefix_model()
    point_cloud = torch.randn(2, 7, 6)
    point_mask = torch.ones(2, dtype=torch.bool)
    image = torch.rand(2, 3, 4, 4)
    image_mask = torch.ones(2, dtype=torch.bool)
    language = torch.tensor([[1, 2, 3], [4, 5, 0]])
    language_mask = torch.tensor([[True, True, True], [True, True, False]])

    prefix, pad_mask, block_mask = model.embed_prefix(
        [point_cloud],
        [point_mask],
        language,
        language_mask,
        images=[image],
        image_masks=[image_mask],
    )

    # Two image tokens, one point token, then three language tokens.
    assert prefix.shape == (2, 6, 8)
    assert torch.equal(pad_mask[:, :3], torch.ones(2, 3, dtype=torch.bool))
    assert torch.equal(pad_mask[:, 3:], language_mask)
    assert not block_mask.any()

    prefix.sum().backward()
    assert model.pointcloud_proj.weight.grad is not None
    assert model.vlm_with_expert.language_embedding.weight.grad is None


def test_adapter_prefix_rejects_missing_rgb():
    model = _make_prefix_model()
    with pytest.raises(ValueError, match="requires static RGB"):
        model.embed_prefix(
            [torch.randn(1, 4, 6)],
            [torch.ones(1, dtype=torch.bool)],
            torch.ones(1, 2, dtype=torch.long),
            torch.ones(1, 2, dtype=torch.bool),
        )
