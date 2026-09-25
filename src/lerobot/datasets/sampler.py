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
from collections import defaultdict
from collections.abc import Hashable, Iterator, Sequence

import torch


class EpisodeAwareSampler:
    def __init__(
        self,
        dataset_from_indices: list[int],
        dataset_to_indices: list[int],
        episode_indices_to_use: list | None = None,
        drop_n_first_frames: int = 0,
        drop_n_last_frames: int = 0,
        shuffle: bool = False,
        rebase_selected_episodes: bool = False,
    ):
        """Sampler that optionally incorporates episode boundary information.

        Args:
            dataset_from_indices: List of indices containing the start of each episode in the dataset.
            dataset_to_indices: List of indices containing the end of each episode in the dataset.
            episode_indices_to_use: List of episode indices to use. If None, all episodes are used.
                                    Assumes that episodes are indexed from 0 to N-1.
            drop_n_first_frames: Number of frames to drop from the start of each episode.
            drop_n_last_frames: Number of frames to drop from the end of each episode.
            shuffle: Whether to shuffle the indices.
        """
        indices = []
        selected = None if episode_indices_to_use is None else set(episode_indices_to_use)
        relative_start = 0
        for episode_idx, (start_index, end_index) in enumerate(
            zip(dataset_from_indices, dataset_to_indices, strict=True)
        ):
            if selected is not None and episode_idx not in selected:
                continue
            if rebase_selected_episodes:
                episode_length = int(end_index) - int(start_index)
                start_index = relative_start
                end_index = relative_start + episode_length
                relative_start = end_index
            indices.extend(range(start_index + drop_n_first_frames, end_index - drop_n_last_frames))

        self.indices = indices
        self.shuffle = shuffle

    def __iter__(self) -> Iterator[int]:
        if self.shuffle:
            for i in torch.randperm(len(self.indices)):
                yield self.indices[i]
        else:
            for i in self.indices:
                yield i

    def __len__(self) -> int:
        return len(self.indices)


class TaskBalancedFrameSampler:
    """Sample valid frames with equal mass per episode-level task.

    The sampler keeps the epoch length equal to the number of eligible source
    frames. Shorter tasks are cycled with fresh permutations and longer tasks
    are subsampled, so task balance does not require changing the dataset,
    objective, or model. Each episode must have exactly one externally resolved
    task/group identifier.
    """

    def __init__(
        self,
        dataset_from_indices: Sequence[int],
        dataset_to_indices: Sequence[int],
        episode_group_ids: Sequence[Hashable],
        episode_indices_to_use: Sequence[int] | None = None,
        drop_n_first_frames: int = 0,
        drop_n_last_frames: int = 0,
        shuffle: bool = True,
        num_samples: int | None = None,
        rebase_selected_episodes: bool = False,
    ):
        if not (
            len(dataset_from_indices) == len(dataset_to_indices) == len(episode_group_ids)
        ):
            raise ValueError(
                "Episode boundaries and episode_group_ids must have identical lengths, got "
                f"{len(dataset_from_indices)}, {len(dataset_to_indices)}, "
                f"and {len(episode_group_ids)}."
            )
        selected = None if episode_indices_to_use is None else set(episode_indices_to_use)
        grouped: dict[Hashable, list[int]] = defaultdict(list)
        relative_start = 0
        for episode_idx, (start_index, end_index, group_id) in enumerate(
            zip(dataset_from_indices, dataset_to_indices, episode_group_ids, strict=True)
        ):
            if selected is not None and episode_idx not in selected:
                continue
            if rebase_selected_episodes:
                episode_length = int(end_index) - int(start_index)
                start_index = relative_start
                end_index = relative_start + episode_length
                relative_start = end_index
            start = int(start_index) + int(drop_n_first_frames)
            end = int(end_index) - int(drop_n_last_frames)
            if end > start:
                grouped[group_id].extend(range(start, end))
        if not grouped:
            raise ValueError("TaskBalancedFrameSampler found no eligible frames.")

        self.grouped_indices = dict(grouped)
        self.group_ids = list(grouped)
        eligible_samples = sum(len(indices) for indices in self.grouped_indices.values())
        self.num_samples = eligible_samples if num_samples is None else int(num_samples)
        if self.num_samples <= 0:
            raise ValueError(f"num_samples must be positive, got {self.num_samples}.")
        self.shuffle = bool(shuffle)
        self.index_offsets = [0]

    def add_index_offset(self, offset: int) -> None:
        offset = int(offset)
        if offset <= 0:
            raise ValueError(f"A paired dataset index offset must be positive, got {offset}.")
        if offset not in self.index_offsets:
            self.index_offsets.append(offset)

    def _draw_group(self, indices: list[int], count: int) -> list[int]:
        if not self.shuffle:
            return [indices[i % len(indices)] for i in range(count)]
        drawn: list[int] = []
        while len(drawn) < count:
            order = torch.randperm(len(indices)).tolist()
            take = min(count - len(drawn), len(order))
            drawn.extend(indices[i] for i in order[:take])
        return drawn

    def __iter__(self) -> Iterator[int]:
        group_count = len(self.group_ids)
        per_group, remainder = divmod(self.num_samples, group_count)
        base_indices: list[int] = []
        for position, group_id in enumerate(self.group_ids):
            count = per_group + int(position < remainder)
            base_indices.extend(self._draw_group(self.grouped_indices[group_id], count))
        if self.shuffle:
            base_indices = [base_indices[i] for i in torch.randperm(len(base_indices)).tolist()]

        expanded = [index + offset for offset in self.index_offsets for index in base_indices]
        if self.shuffle and len(self.index_offsets) > 1:
            expanded = [expanded[i] for i in torch.randperm(len(expanded)).tolist()]
        yield from expanded

    def __len__(self) -> int:
        return self.num_samples * len(self.index_offsets)
