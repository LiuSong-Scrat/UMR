#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
from datasets import Dataset

from lerobot.datasets.push_dataset_to_hub.utils import calculate_episode_data_index
from lerobot.datasets.sampler import EpisodeAwareSampler, TaskBalancedFrameSampler
from lerobot.datasets.utils import (
    hf_transform_to_torch,
)


def test_drop_n_first_frames():
    dataset = Dataset.from_dict(
        {
            "timestamp": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            "index": [0, 1, 2, 3, 4, 5],
            "episode_index": [0, 0, 1, 2, 2, 2],
        },
    )
    dataset.set_transform(hf_transform_to_torch)
    episode_data_index = calculate_episode_data_index(dataset)
    sampler = EpisodeAwareSampler(episode_data_index["from"], episode_data_index["to"], drop_n_first_frames=1)
    assert sampler.indices == [1, 4, 5]
    assert len(sampler) == 3
    assert list(sampler) == [1, 4, 5]


def test_drop_n_last_frames():
    dataset = Dataset.from_dict(
        {
            "timestamp": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            "index": [0, 1, 2, 3, 4, 5],
            "episode_index": [0, 0, 1, 2, 2, 2],
        },
    )
    dataset.set_transform(hf_transform_to_torch)
    episode_data_index = calculate_episode_data_index(dataset)
    sampler = EpisodeAwareSampler(episode_data_index["from"], episode_data_index["to"], drop_n_last_frames=1)
    assert sampler.indices == [0, 3, 4]
    assert len(sampler) == 3
    assert list(sampler) == [0, 3, 4]


def test_episode_indices_to_use():
    dataset = Dataset.from_dict(
        {
            "timestamp": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            "index": [0, 1, 2, 3, 4, 5],
            "episode_index": [0, 0, 1, 2, 2, 2],
        },
    )
    dataset.set_transform(hf_transform_to_torch)
    episode_data_index = calculate_episode_data_index(dataset)
    sampler = EpisodeAwareSampler(
        episode_data_index["from"], episode_data_index["to"], episode_indices_to_use=[0, 2]
    )
    assert sampler.indices == [0, 1, 3, 4, 5]
    assert len(sampler) == 5
    assert list(sampler) == [0, 1, 3, 4, 5]


def test_shuffle():
    dataset = Dataset.from_dict(
        {
            "timestamp": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            "index": [0, 1, 2, 3, 4, 5],
            "episode_index": [0, 0, 1, 2, 2, 2],
        },
    )
    dataset.set_transform(hf_transform_to_torch)
    episode_data_index = calculate_episode_data_index(dataset)
    sampler = EpisodeAwareSampler(episode_data_index["from"], episode_data_index["to"], shuffle=False)
    assert sampler.indices == [0, 1, 2, 3, 4, 5]
    assert len(sampler) == 6
    assert list(sampler) == [0, 1, 2, 3, 4, 5]
    sampler = EpisodeAwareSampler(episode_data_index["from"], episode_data_index["to"], shuffle=True)
    assert sampler.indices == [0, 1, 2, 3, 4, 5]
    assert len(sampler) == 6
    assert set(sampler) == {0, 1, 2, 3, 4, 5}


def test_task_balanced_frame_sampler_has_exact_equal_group_mass():
    sampler = TaskBalancedFrameSampler(
        dataset_from_indices=[0, 2, 6],
        dataset_to_indices=[2, 6, 9],
        episode_group_ids=["short", "long", "long"],
        shuffle=False,
        num_samples=8,
    )
    sampled = list(sampler)
    assert len(sampled) == 8
    assert sum(index < 2 for index in sampled) == 4
    assert sum(index >= 2 for index in sampled) == 4
    assert set(sampled).issubset(set(range(9)))


def test_task_balanced_frame_sampler_respects_episode_subset_and_paired_offset():
    sampler = TaskBalancedFrameSampler(
        dataset_from_indices=[0, 2, 5],
        dataset_to_indices=[2, 5, 9],
        episode_group_ids=[0, 1, 1],
        episode_indices_to_use=[0, 2],
        shuffle=False,
        num_samples=4,
    )
    sampler.add_index_offset(9)
    sampled = list(sampler)
    assert len(sampled) == 8
    assert sampled[:4] == [0, 1, 5, 6]
    assert sampled[4:] == [9, 10, 14, 15]


def test_episode_aware_sampler_rebases_filtered_dataset_rows():
    sampler = EpisodeAwareSampler(
        dataset_from_indices=[0, 2, 5, 9],
        dataset_to_indices=[2, 5, 9, 11],
        episode_indices_to_use=[1, 3],
        rebase_selected_episodes=True,
    )
    assert sampler.indices == [0, 1, 2, 3, 4]


def test_task_balanced_sampler_rebases_filtered_dataset_rows_and_paired_offset():
    sampler = TaskBalancedFrameSampler(
        dataset_from_indices=[0, 2, 5, 9],
        dataset_to_indices=[2, 5, 9, 11],
        episode_group_ids=["unused", "task_a", "unused", "task_b"],
        episode_indices_to_use=[1, 3],
        rebase_selected_episodes=True,
        shuffle=False,
        num_samples=4,
    )
    sampler.add_index_offset(5)
    sampled = list(sampler)
    assert sampled[:4] == [0, 1, 3, 4]
    assert sampled[4:] == [5, 6, 8, 9]
