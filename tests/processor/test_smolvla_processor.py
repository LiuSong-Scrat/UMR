#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for SmolVLA policy processor."""

from unittest.mock import patch

import pytest
import torch

from lerobot.configs.types import FeatureType, NormalizationMode, PipelineFeatureType, PolicyFeature
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.processor_smolvla import (
    SmolVLANewLineProcessor,
    make_smolvla_pre_post_processors,
    validate_smolvla_worldflow_preprocessor,
)
from lerobot.processor import (
    AddBatchDimensionProcessorStep,
    DeviceProcessorStep,
    EnvTransition,
    NormalizerProcessorStep,
    PolicyProcessorPipeline,
    ProcessorStep,
    RenameObservationsProcessorStep,
    TransitionKey,
    UMIProcessor,
    UnnormalizerProcessorStep,
)
from lerobot.processor.converters import create_transition, transition_to_batch
from lerobot.utils.constants import ACTION, OBS_IMAGE, OBS_STATE


class MockTokenizerProcessorStep(ProcessorStep):
    """Mock tokenizer processor step for testing."""

    def __init__(self, *args, **kwargs):
        # Accept any arguments to mimic the real TokenizerProcessorStep interface
        pass

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        # Pass through transition unchanged
        return transition

    def transform_features(self, features):
        # Pass through features unchanged
        return features


def create_default_config():
    """Create a default SmolVLA configuration for testing."""
    config = SmolVLAConfig()
    config.input_features = {
        OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(10,)),
        OBS_IMAGE: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224)),
    }
    config.output_features = {
        ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(10,)),
    }
    config.normalization_mapping = {
        FeatureType.STATE: NormalizationMode.MEAN_STD,
        FeatureType.VISUAL: NormalizationMode.IDENTITY,
        FeatureType.ACTION: NormalizationMode.MIN_MAX,
    }
    config.device = "cpu"
    config.vlm_model_name = "HuggingFaceTB/SmolVLM-Instruct"
    config.pad_language_to = "max_length"
    config.tokenizer_max_length = 100
    return config


def create_default_stats():
    """Create default dataset statistics for testing."""
    return {
        OBS_STATE: {"mean": torch.zeros(10), "std": torch.ones(10)},
        OBS_IMAGE: {},  # No normalization for images
        ACTION: {"min": torch.full((10,), -1.0), "max": torch.ones(10)},
    }


def test_make_smolvla_processor_basic():
    """Test basic creation of SmolVLA processor."""
    config = create_default_config()
    stats = create_default_stats()

    with patch(
        "lerobot.policies.smolvla.processor_smolvla.TokenizerProcessorStep", MockTokenizerProcessorStep
    ):
        preprocessor, postprocessor = make_smolvla_pre_post_processors(
            config,
            stats,
        )

    # Check processor names
    assert preprocessor.name == "policy_preprocessor"
    assert postprocessor.name == "policy_postprocessor"

    # Check steps in preprocessor
    assert len(preprocessor.steps) == 7
    assert isinstance(preprocessor.steps[0], RenameObservationsProcessorStep)
    assert isinstance(preprocessor.steps[1], AddBatchDimensionProcessorStep)
    assert isinstance(preprocessor.steps[2], SmolVLANewLineProcessor)
    # Step 3 would be TokenizerProcessorStep but it's mocked
    assert isinstance(preprocessor.steps[4], UMIProcessor)
    assert isinstance(preprocessor.steps[5], DeviceProcessorStep)
    assert isinstance(preprocessor.steps[6], NormalizerProcessorStep)
    validate_smolvla_worldflow_preprocessor(preprocessor)

    # Check steps in postprocessor
    assert len(postprocessor.steps) == 2
    assert isinstance(postprocessor.steps[0], UnnormalizerProcessorStep)
    assert isinstance(postprocessor.steps[1], DeviceProcessorStep)


def test_smolvla_newline_processor_single_task():
    """Test SmolVLANewLineProcessor with single task string."""
    processor = SmolVLANewLineProcessor()

    # Test with task that doesn't have newline
    transition = create_transition(complementary_data={"task": "test task"})
    result = processor(transition)
    assert result[TransitionKey.COMPLEMENTARY_DATA]["task"] == "test task\n"

    # Test with task that already has newline
    transition = create_transition(complementary_data={"task": "test task\n"})
    result = processor(transition)
    assert result[TransitionKey.COMPLEMENTARY_DATA]["task"] == "test task\n"


def test_smolvla_newline_processor_list_of_tasks():
    """Test SmolVLANewLineProcessor with list of task strings."""
    processor = SmolVLANewLineProcessor()

    # Test with list of tasks
    tasks = ["task1", "task2\n", "task3"]
    transition = create_transition(complementary_data={"task": tasks})
    result = processor(transition)
    expected = ["task1\n", "task2\n", "task3\n"]
    assert result[TransitionKey.COMPLEMENTARY_DATA]["task"] == expected


def test_smolvla_newline_processor_empty_transition():
    """Test SmolVLANewLineProcessor with empty transition."""
    processor = SmolVLANewLineProcessor()

    # Test with no complementary_data
    transition = create_transition()
    result = processor(transition)
    assert result == transition

    # Test with complementary_data but no task
    transition = create_transition(complementary_data={"other": "data"})
    result = processor(transition)
    assert result == transition

    # Test with None task
    transition = create_transition(complementary_data={"task": None})
    result = processor(transition)
    assert result == transition


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_smolvla_processor_cuda():
    """Test SmolVLA processor with CUDA device."""
    config = create_default_config()
    config.device = "cuda"
    stats = create_default_stats()

    # Mock the tokenizer processor to act as pass-through
    class MockTokenizerProcessorStep(ProcessorStep):
        def __init__(self, *args, **kwargs):
            pass

        def __call__(self, transition):
            return transition

        def state_dict(self):
            return {}

        def load_state_dict(self, state):
            pass

        def reset(self):
            pass

        def get_config(self):
            return {"tokenizer_name": "HuggingFaceTB/SmolVLM-Instruct"}

        def transform_features(self, features):
            return features

    with patch(
        "lerobot.policies.smolvla.processor_smolvla.TokenizerProcessorStep", MockTokenizerProcessorStep
    ):
        preprocessor, postprocessor = make_smolvla_pre_post_processors(
            config,
            stats,
        )

    # Create CPU data
    observation = {
        OBS_STATE: torch.randn(10),
        OBS_IMAGE: torch.randn(3, 224, 224),
    }
    action = torch.randn(10)
    transition = create_transition(observation, action, complementary_data={"task": "test task"})

    batch = transition_to_batch(transition)

    # Process through preprocessor

    processed = preprocessor(batch)

    # Check that data is on CUDA
    assert processed[OBS_STATE].device.type == "cuda"
    assert processed[OBS_IMAGE].device.type == "cuda"
    assert processed[TransitionKey.ACTION.value].device.type == "cuda"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_smolvla_processor_accelerate_scenario():
    """Test SmolVLA processor in simulated Accelerate scenario."""
    config = create_default_config()
    config.device = "cuda:0"
    stats = create_default_stats()

    # Mock the tokenizer processor to act as pass-through
    class MockTokenizerProcessorStep(ProcessorStep):
        def __init__(self, *args, **kwargs):
            pass

        def __call__(self, transition):
            return transition

        def state_dict(self):
            return {}

        def load_state_dict(self, state):
            pass

        def reset(self):
            pass

        def get_config(self):
            return {"tokenizer_name": "HuggingFaceTB/SmolVLM-Instruct"}

        def transform_features(self, features):
            return features

    with patch(
        "lerobot.policies.smolvla.processor_smolvla.TokenizerProcessorStep", MockTokenizerProcessorStep
    ):
        preprocessor, postprocessor = make_smolvla_pre_post_processors(
            config,
            stats,
        )

    # Simulate Accelerate: data already on GPU and batched
    device = torch.device("cuda:0")
    observation = {
        OBS_STATE: torch.randn(1, 10).to(device),
        OBS_IMAGE: torch.randn(1, 3, 224, 224).to(device),
    }
    action = torch.randn(1, 10).to(device)
    transition = create_transition(observation, action, complementary_data={"task": ["test task"]})

    batch = transition_to_batch(transition)

    # Process through preprocessor

    processed = preprocessor(batch)

    # Check that data stays on same GPU
    assert processed[OBS_STATE].device == device
    assert processed[OBS_IMAGE].device == device
    assert processed[TransitionKey.ACTION.value].device == device


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Requires at least 2 GPUs")
def test_smolvla_processor_multi_gpu():
    """Test SmolVLA processor with multi-GPU setup."""
    config = create_default_config()
    config.device = "cuda:0"
    stats = create_default_stats()

    # Mock the tokenizer processor to act as pass-through
    class MockTokenizerProcessorStep(ProcessorStep):
        def __init__(self, *args, **kwargs):
            pass

        def __call__(self, transition):
            return transition

        def state_dict(self):
            return {}

        def load_state_dict(self, state):
            pass

        def reset(self):
            pass

        def get_config(self):
            return {"tokenizer_name": "HuggingFaceTB/SmolVLM-Instruct"}

        def transform_features(self, features):
            return features

    with patch(
        "lerobot.policies.smolvla.processor_smolvla.TokenizerProcessorStep", MockTokenizerProcessorStep
    ):
        preprocessor, postprocessor = make_smolvla_pre_post_processors(
            config,
            stats,
        )

    # Simulate data on different GPU
    device = torch.device("cuda:1")
    observation = {
        OBS_STATE: torch.randn(1, 10).to(device),
        OBS_IMAGE: torch.randn(1, 3, 224, 224).to(device),
    }
    action = torch.randn(1, 10).to(device)
    transition = create_transition(observation, action, complementary_data={"task": ["test task"]})

    batch = transition_to_batch(transition)

    # Process through preprocessor

    processed = preprocessor(batch)

    # Check that data stays on cuda:1
    assert processed[OBS_STATE].device == device
    assert processed[OBS_IMAGE].device == device
    assert processed[TransitionKey.ACTION.value].device == device


def test_smolvla_processor_without_stats():
    """Test SmolVLA processor creation without dataset statistics."""
    config = create_default_config()

    # Mock the tokenizer processor
    with patch(
        "lerobot.policies.smolvla.processor_smolvla.TokenizerProcessorStep", MockTokenizerProcessorStep
    ):
        preprocessor, postprocessor = make_smolvla_pre_post_processors(
            config,
            dataset_stats=None,
        )

    # Should still create processors
    assert preprocessor is not None
    assert postprocessor is not None


def test_worldflow_preprocessor_validation_rejects_missing_or_duplicate_umi():
    missing = PolicyProcessorPipeline(steps=[], name="missing_umi")
    with pytest.raises(ValueError, match="exactly one UMIProcessor"):
        validate_smolvla_worldflow_preprocessor(missing)

    duplicate = PolicyProcessorPipeline(steps=[UMIProcessor(), UMIProcessor()], name="duplicate_umi")
    with pytest.raises(ValueError, match="exactly one UMIProcessor"):
        validate_smolvla_worldflow_preprocessor(duplicate)

    wrong_order = PolicyProcessorPipeline(
        steps=[
            NormalizerProcessorStep(features={}, norm_map={}, stats={}),
            UMIProcessor(),
        ],
        name="wrong_order",
    )
    with pytest.raises(ValueError, match="before NormalizerProcessorStep"):
        validate_smolvla_worldflow_preprocessor(wrong_order)


def test_worldflow_umi_contract_survives_processor_serialization(tmp_path):
    pipeline = PolicyProcessorPipeline(steps=[UMIProcessor()], name="worldflow_preprocessor")
    pipeline.save_pretrained(tmp_path, config_filename="worldflow_preprocessor.json")
    loaded = PolicyProcessorPipeline.from_pretrained(
        tmp_path,
        config_filename="worldflow_preprocessor.json",
    )
    validate_smolvla_worldflow_preprocessor(loaded)
    assert isinstance(loaded.steps[0], UMIProcessor)


def test_smolvla_newline_processor_state_dict():
    """Test SmolVLANewLineProcessor state dict methods."""
    processor = SmolVLANewLineProcessor()

    # Test state_dict (should be empty)
    state = processor.state_dict()
    assert state == {}

    # Test load_state_dict (should do nothing)
    processor.load_state_dict({})

    # Test reset (should do nothing)
    processor.reset()

    # Test get_config
    config = processor.get_config()
    assert config == {}


def test_smolvla_newline_processor_transform_features():
    """Test SmolVLANewLineProcessor transform_features method."""
    processor = SmolVLANewLineProcessor()

    # Test transform_features
    features = {
        PipelineFeatureType.OBSERVATION: {OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(10,))},
    }
    result = processor.transform_features(features)
    assert result == features  # Should return unchanged


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_smolvla_processor_bfloat16_device_float32_normalizer():
    """Test: DeviceProcessor(bfloat16) + NormalizerProcessor(float32) → output bfloat16 via automatic adaptation"""
    config = create_default_config()
    config.device = "cuda"
    stats = create_default_stats()

    with patch(
        "lerobot.policies.smolvla.processor_smolvla.TokenizerProcessorStep", MockTokenizerProcessorStep
    ):
        preprocessor, _ = make_smolvla_pre_post_processors(
            config,
            stats,
        )

    # Modify the pipeline to use bfloat16 device processor with float32 normalizer
    modified_steps = []
    for step in preprocessor.steps:
        if isinstance(step, DeviceProcessorStep):
            # Device processor converts to bfloat16
            modified_steps.append(DeviceProcessorStep(device=config.device, float_dtype="bfloat16"))
        elif isinstance(step, NormalizerProcessorStep):
            # Normalizer stays configured as float32 (will auto-adapt to bfloat16)
            modified_steps.append(
                NormalizerProcessorStep(
                    features=step.features,
                    norm_map=step.norm_map,
                    stats=step.stats,
                    device=config.device,
                    dtype=torch.float32,  # Deliberately configured as float32
                )
            )
        else:
            modified_steps.append(step)
    preprocessor.steps = modified_steps

    # Verify initial normalizer configuration.
    normalizer_step = preprocessor.steps[6]
    assert normalizer_step.dtype == torch.float32

    # Create test data with both state and visual observations
    observation = {
        OBS_STATE: torch.randn(10, dtype=torch.float32),
        OBS_IMAGE: torch.randn(3, 224, 224, dtype=torch.float32),
    }
    action = torch.randn(10, dtype=torch.float32)
    transition = create_transition(
        observation, action, complementary_data={"task": "test bfloat16 adaptation"}
    )

    batch = transition_to_batch(transition)

    # Process through full pipeline
    processed = preprocessor(batch)

    # Verify: DeviceProcessor → bfloat16, NormalizerProcessor adapts → final output is bfloat16
    assert processed[OBS_STATE].dtype == torch.bfloat16
    assert processed[OBS_IMAGE].dtype == torch.bfloat16  # IDENTITY normalization still gets dtype conversion
    assert processed[TransitionKey.ACTION.value].dtype == torch.bfloat16

    # Verify normalizer automatically adapted its internal state
    assert normalizer_step.dtype == torch.bfloat16
    # Check state stats (has normalization)
    for stat_tensor in normalizer_step._tensor_stats[OBS_STATE].values():
        assert stat_tensor.dtype == torch.bfloat16
    # OBS_IMAGE uses IDENTITY normalization, so no stats to check
