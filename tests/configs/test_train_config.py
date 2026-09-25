#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

from pathlib import Path

from lerobot.configs.train import _output_dir_has_training_artifacts


def test_empty_output_dir_has_no_training_artifacts(tmp_path: Path):
    assert not _output_dir_has_training_artifacts(tmp_path)


def test_wandb_only_output_dir_has_no_training_artifacts(tmp_path: Path):
    wandb_dir = tmp_path / "wandb"
    wandb_dir.mkdir()
    (wandb_dir / "latest-run").touch()

    assert not _output_dir_has_training_artifacts(tmp_path)


def test_checkpoint_output_dir_has_training_artifacts(tmp_path: Path):
    (tmp_path / "wandb").mkdir()
    (tmp_path / "checkpoints").mkdir()

    assert _output_dir_has_training_artifacts(tmp_path)


def test_train_config_output_file_counts_as_training_artifact(tmp_path: Path):
    (tmp_path / "train_config.json").touch()

    assert _output_dir_has_training_artifacts(tmp_path)
