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

"""Evaluate Song/SmolVLA checkpoints without changing model parameters.

This entry point deliberately reuses the dataset, PointSeg cache, collate, policy,
and preprocessing path from ``train_song.py``.  Unlike the training entry point it:

* keeps the policy in evaluation mode;
* runs every forward pass under ``torch.inference_mode()``;
* never creates an optimizer or learning-rate scheduler;
* never calls backward or saves a model checkpoint;
* traverses the evaluation data at most once.

For CLI compatibility, ``--steps`` is interpreted as the maximum number of
evaluation batches.  If it exceeds the DataLoader length, only one complete pass
over the dataset is evaluated.
"""

import json
import logging
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from pprint import pformat
from typing import Any

import torch
from accelerate import Accelerator
from termcolor import colored
from tqdm import tqdm

from lerobot.configs import parser
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.factory import make_dataset
from lerobot.datasets.sampler import EpisodeAwareSampler
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.rl.wandb_utils import WandBLogger
from lerobot.scripts.train_song import (
    OnlinePointSegBatchCollator,
    ensure_ddp_parameters_initialized,
    make_song_train_collate_fn,
    maybe_wrap_point_cloud_memmap_dataset,
    maybe_wrap_pointseg_cache_dataset,
    maybe_wrap_worldflow_dataset,
)
from lerobot.utils.constants import ACTION
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.random_utils import set_seed
from lerobot.utils.utils import init_logging, inside_slurm

EVAL_METRIC_KEYS = (
    "loss",
    "loss_action",
    "loss_action_translation",
    "loss_action_rotation6d",
    "loss_action_gripper",
    "action_endpoint_trans_err",
    "action_endpoint_rot_err_deg",
    "action_endpoint_gripper_err",
    "loss_pointseg_aux",
    "loss_se3_twist",
    "loss_se3_endpoint",
    "loss_se3_gripper",
    "loss_se3_equivariance",
    "se3_action_trans_err",
    "se3_action_rot_err_deg",
    "loss_worldflow_flow",
    "loss_worldflow_geo",
    "loss_worldflow_bridge",
    "loss_worldflow_equiv",
    "worldflow_trans_err",
    "worldflow_rot_err_deg",
    "worldflow_valid_ratio",
    "worldflow_foreground_points",
    "worldflow_noise_conjugacy_error",
    "worldflow_path_conjugacy_error",
    "pointseg_foreground_ratio",
    "pointseg_operation_prob_mean",
    "pointseg_selection_score_mean",
    "pred_operation_prob",
    "loss_soft_bce",
    "loss_smoothness",
    "pseudo_valid_ratio",
    "pseudo_foreground_ratio",
    "pseudo_soft_foreground_mean",
    "pred_foreground_ratio",
    "sample_action_mse",
    "sample_action_translation_l2_m",
    "sample_action_rot6d_mse",
    "sample_action_gripper_mae_m",
)

FLOW_TIME_SWEEP_VALUES = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.999)
FLOW_TIME_SWEEP_METRICS = (
    "loss_action",
    "action_endpoint_trans_err",
    "action_endpoint_rot_err_deg",
    "action_endpoint_gripper_err",
)
EVAL_METRIC_KEYS = EVAL_METRIC_KEYS + tuple(
    f"flow_t{round(time_value * 1000):03d}_{metric_name}"
    for time_value in FLOW_TIME_SWEEP_VALUES
    for metric_name in FLOW_TIME_SWEEP_METRICS
)


@dataclass
class SongEvalPipelineConfig(TrainPipelineConfig):
    """Read-only evaluation options that do not belong to the training config."""

    libero_dataset_domain_action_mse: bool = False
    libero_suite: str | None = None
    libero_task_id: int | None = None
    sample_action_mse: bool = False
    sample_action_noise_mode: str = "policy"
    flow_time_sweep: bool = False


class EvalFrameSubset(torch.utils.data.Dataset):
    """Frame subset that preserves access to LeRobot metadata and wrapper attributes."""

    def __init__(self, dataset: torch.utils.data.Dataset, indices: list[int]) -> None:
        self.dataset = dataset
        self.indices = [int(index) for index in indices]

    def __getattr__(self, name: str) -> Any:
        return getattr(self.dataset, name)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> Any:
        return self.dataset[self.indices[index]]


def _resolve_libero_dataset_domain_selection(
    cfg: SongEvalPipelineConfig,
) -> dict[str, Any] | None:
    """Resolve one LIBERO suite/task to the exact episodes stored in the training dataset."""

    if not cfg.libero_dataset_domain_action_mse:
        if cfg.libero_suite is not None or cfg.libero_task_id is not None:
            raise ValueError(
                "--libero_suite/--libero_task_id require "
                "--libero_dataset_domain_action_mse=true."
            )
        return None
    if cfg.libero_suite is None or cfg.libero_task_id is None:
        raise ValueError(
            "--libero_dataset_domain_action_mse=true requires both "
            "--libero_suite and --libero_task_id."
        )
    if cfg.dataset.streaming:
        raise ValueError("LIBERO dataset-domain action MSE does not support streaming datasets.")

    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

    try:
        from libero.libero import benchmark
    except Exception as exc:
        raise RuntimeError(
            "LIBERO must be importable to resolve --libero_suite and --libero_task_id."
        ) from exc

    benchmark_dict = benchmark.get_benchmark_dict()
    suite_name = str(cfg.libero_suite)
    if suite_name not in benchmark_dict:
        raise ValueError(
            f"Unknown LIBERO suite {suite_name!r}; available suites are "
            f"{sorted(benchmark_dict)}."
        )
    suite = benchmark_dict[suite_name]()
    task_id = int(cfg.libero_task_id)
    if task_id < 0 or task_id >= len(suite.tasks):
        raise ValueError(
            f"Invalid task id {task_id} for {suite_name}; expected 0..{len(suite.tasks) - 1}."
        )
    task = suite.get_task(task_id)
    task_language = str(task.language)

    metadata = LeRobotDatasetMetadata(
        cfg.dataset.repo_id,
        root=cfg.dataset.root,
        revision=cfg.dataset.revision,
    )
    matching_episode_indices = [
        int(episode["episode_index"])
        for episode in metadata.episodes
        if task_language in list(episode["tasks"])
    ]
    if cfg.dataset.episodes is not None:
        requested = {int(index) for index in cfg.dataset.episodes}
        matching_episode_indices = [
            index for index in matching_episode_indices if index in requested
        ]
    if not matching_episode_indices:
        available_tasks = sorted(str(task_name) for task_name in metadata.tasks.index)
        preview = available_tasks[:20]
        raise ValueError(
            f"No episodes for {suite_name} task {task_id} ({task_language!r}) were found "
            f"in dataset {cfg.dataset.repo_id!r}. Available task labels include: {preview}"
        )

    frame_indices: list[int] = []
    for episode_index in matching_episode_indices:
        episode = metadata.episodes[episode_index]
        frame_indices.extend(
            range(
                int(episode["dataset_from_index"]),
                int(episode["dataset_to_index"]),
            )
        )
    return {
        "mode": "libero_training_dataset_action_mse",
        "suite": suite_name,
        "task_id": task_id,
        "task_name": str(task.name),
        "task_language": task_language,
        "episode_indices": matching_episode_indices,
        "frame_indices": frame_indices,
        "episode_count": len(matching_episode_indices),
        "frame_count": len(frame_indices),
        "environment_source": "converted_training_dataset",
        "benchmark_rollout": False,
    }


def _make_eval_dataset(
    cfg: SongEvalPipelineConfig,
    dataset_domain_selection: dict[str, Any] | None,
):
    # Build the complete dataset first. Applying the frame subset after the
    # PointSeg/cache wrappers preserves their global dataset-index alignment.
    requested_episodes = cfg.dataset.episodes
    if dataset_domain_selection is not None:
        cfg.dataset.episodes = None
    dataset = make_dataset(cfg)
    dataset = maybe_wrap_pointseg_cache_dataset(dataset, cfg.pointseg_sample_cache_dir, cfg.policy)
    dataset = maybe_wrap_point_cloud_memmap_dataset(dataset)
    dataset = maybe_wrap_worldflow_dataset(dataset, cfg.policy)
    if dataset_domain_selection is not None:
        dataset = EvalFrameSubset(dataset, dataset_domain_selection["frame_indices"])
        cfg.dataset.episodes = requested_episodes
    return dataset


def _make_eval_preprocessor(cfg: TrainPipelineConfig, policy, dataset, device: torch.device):
    processor_kwargs: dict[str, Any] = {}
    if cfg.policy.pretrained_path is not None:
        processor_kwargs["dataset_stats"] = dataset.meta.stats
        processor_kwargs["preprocessor_overrides"] = {
            "device_processor": {"device": device.type},
            "normalizer_processor": {
                "stats": dataset.meta.stats,
                "features": {**policy.config.input_features, **policy.config.output_features},
                "norm_map": policy.config.normalization_mapping,
            },
            "rename_observations_processor": {"rename_map": cfg.rename_map},
        }

    preprocessor, _postprocessor = make_pre_post_processors(
        policy_cfg=cfg.policy,
        pretrained_path=cfg.policy.pretrained_path,
        **processor_kwargs,
    )
    return preprocessor


def _make_eval_dataloader(cfg: SongEvalPipelineConfig, dataset, device: torch.device):
    dataset_domain_action_mse = bool(cfg.libero_dataset_domain_action_mse)
    if hasattr(cfg.policy, "drop_n_last_frames") and not dataset_domain_action_mse:
        sampler = EpisodeAwareSampler(
            dataset.meta.episodes["dataset_from_index"],
            dataset.meta.episodes["dataset_to_index"],
            episode_indices_to_use=dataset.episodes,
            drop_n_last_frames=cfg.policy.drop_n_last_frames,
            shuffle=True,
        )
    else:
        sampler = None

    collate_fn = make_song_train_collate_fn(dataset)
    dataloader_num_workers = int(cfg.num_workers)
    if isinstance(collate_fn, OnlinePointSegBatchCollator) and collate_fn.device.type == "cuda":
        if dataloader_num_workers > 0:
            logging.warning(
                "Online PointSeg pseudo labels use CUDA; setting num_workers=0 for safe evaluation."
            )
        dataloader_num_workers = 0

    return torch.utils.data.DataLoader(
        dataset,
        num_workers=dataloader_num_workers,
        batch_size=cfg.batch_size,
        shuffle=sampler is None and not cfg.dataset.streaming and not dataset_domain_action_mse,
        sampler=sampler,
        pin_memory=device.type == "cuda",
        drop_last=False,
        prefetch_factor=2 if dataloader_num_workers > 0 else None,
        persistent_workers=dataloader_num_workers > 0,
        collate_fn=collate_fn,
    )


def _to_scalar(value: Any) -> float | None:
    if torch.is_tensor(value):
        if value.numel() != 1:
            return None
        value = value.detach().item()
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _gather_metric_sums(
    accelerator: Accelerator,
    metrics: dict[str, Any],
    local_batch_size: int,
) -> tuple[dict[str, float], dict[str, float]]:
    """Gather weighted metric sums/counts from every process."""

    payload: list[float] = []
    for key in EVAL_METRIC_KEYS:
        value = _to_scalar(metrics.get(key))
        if value is None:
            payload.extend((0.0, 0.0))
        else:
            payload.extend((value * local_batch_size, float(local_batch_size)))

    local = torch.tensor(payload, device=accelerator.device, dtype=torch.float64).unsqueeze(0)
    gathered = accelerator.gather(local)
    if gathered.ndim == 1:
        gathered = gathered.unsqueeze(0)

    sums: dict[str, float] = {}
    counts: dict[str, float] = {}
    for index, key in enumerate(EVAL_METRIC_KEYS):
        sums[key] = float(gathered[:, 2 * index].sum().item())
        counts[key] = float(gathered[:, 2 * index + 1].sum().item())
    return sums, counts


def _format_metrics(metrics: dict[str, float]) -> str:
    parts = []
    for key in EVAL_METRIC_KEYS:
        if key in metrics:
            parts.append(f"{key}:{metrics[key]:.4g}")
    return " ".join(parts)


def _sampled_action_metrics(
    policy,
    batch: dict[str, torch.Tensor],
    noise_mode: str = "policy",
) -> dict[str, torch.Tensor]:
    """Compare one complete flow-matching sample with its ground-truth chunk.

    The normal ``loss_action`` is a velocity-field objective at one random
    diffusion time.  It is useful for optimization, but it does not directly
    measure the trajectory returned by all denoising steps.  These optional
    metrics expose that distinction without changing parameters or gradients.
    """

    policy.reset()
    if noise_mode not in {"policy", "zero", "pose9_identity"}:
        raise ValueError(
            "sample_action_noise_mode must be one of policy, zero, pose9_identity; "
            f"got {noise_mode!r}."
        )
    predict_kwargs: dict[str, torch.Tensor] = {}
    if noise_mode != "policy":
        batch_size = int(batch[ACTION].shape[0])
        chunk_size = int(policy.config.chunk_size)
        action_dim = int(policy.config.max_action_dim)
        noise = torch.zeros(
            batch_size,
            chunk_size,
            action_dim,
            device=batch[ACTION].device,
            dtype=torch.float32,
        )
        if noise_mode == "pose9_identity":
            if action_dim < 10:
                raise ValueError("pose9_identity noise requires max_action_dim >= 10.")
            noise[..., 3] = 1.0
            noise[..., 7] = 1.0
        predict_kwargs["noise"] = noise
        if policy.config.worldflow_enable:
            world_noise = torch.zeros(
                batch_size,
                chunk_size,
                9,
                device=batch[ACTION].device,
                dtype=torch.float32,
            )
            world_noise[..., 3] = 1.0
            world_noise[..., 7] = 1.0
            predict_kwargs["worldflow_noise"] = world_noise
    predicted = policy.predict_action_chunk(batch, **predict_kwargs)
    target = batch[ACTION][..., : predicted.shape[-1]]
    if predicted.shape != target.shape:
        raise ValueError(
            f"Sampled/target action chunk shapes differ: {predicted.shape} != {target.shape}."
        )
    actions_is_pad = batch.get(f"{ACTION}_is_pad")
    if torch.is_tensor(actions_is_pad):
        valid = ~actions_is_pad.to(device=predicted.device, dtype=torch.bool)
    else:
        valid = torch.ones(predicted.shape[:2], device=predicted.device, dtype=torch.bool)
    valid_f = valid.to(dtype=predicted.dtype)
    valid_count = valid_f.sum().clamp_min(1.0)
    error = predicted - target
    metrics = {
        "sample_action_mse": (
            error.square().mean(dim=-1) * valid_f
        ).sum()
        / valid_count,
        "sample_action_translation_l2_m": (
            torch.linalg.norm(error[..., :3], dim=-1) * valid_f
        ).sum()
        / valid_count,
        "sample_action_rot6d_mse": (
            error[..., 3:9].square().mean(dim=-1) * valid_f
        ).sum()
        / valid_count,
    }
    if predicted.shape[-1] >= 10:
        metrics["sample_action_gripper_mae_m"] = (
            error[..., 9].abs() * valid_f
        ).sum() / valid_count
    return metrics


def _flow_time_sweep_metrics(
    policy,
    batch: dict[str, torch.Tensor],
    *,
    seed: int,
) -> dict[str, torch.Tensor]:
    """Measure velocity/endpoint error across the complete integration interval.

    Standard SmolVLA samples training time from ``Beta(1.5, 1)``. On very small
    datasets this can leave the beginning of the inference ODE under-trained.
    Reusing the same Ego/World noise and point-operation RNG state at every
    requested time isolates that coverage issue from ordinary sampling noise.
    """

    action = batch[ACTION]
    if not torch.is_tensor(action):
        raise TypeError("Flow-time sweep requires a tensor action chunk.")
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Use the exact training/inference prior saved in the checkpoint.  A
    # hard-coded Gaussian makes the sweep an out-of-distribution diagnostic
    # whenever pose9_action_noise_enable is active, and even the legacy prior
    # is unit Gaussian rather than N(0, 0.1).
    model = getattr(policy, "model", None)
    if model is None:
        raise AttributeError("Flow-time sweep requires policy.model.")
    if bool(getattr(policy.config, "se3_enable", False)):
        _noise_transform, _gripper_noise, fixed_noise = model.sample_se3_action_noise(action)
    elif bool(getattr(policy.config, "pose9_action_noise_enable", False)):
        fixed_noise = model.sample_pose9_action_noise(tuple(action.shape), action.device)
    else:
        fixed_noise = model.sample_noise(action.shape, action.device)
    fixed_noise = fixed_noise.to(dtype=torch.float32)
    metrics: dict[str, torch.Tensor] = {}
    for time_value in FLOW_TIME_SWEEP_VALUES:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        time = torch.full(
            (action.shape[0],),
            float(time_value),
            device=action.device,
            dtype=torch.float32,
        )
        loss, output = policy(batch, noise=fixed_noise, time=time)
        output = dict(output or {})
        output.setdefault("loss_action", loss)
        prefix = f"flow_t{round(time_value * 1000):03d}_"
        for metric_name in FLOW_TIME_SWEEP_METRICS:
            value = output.get(metric_name)
            if torch.is_tensor(value):
                metrics[prefix + metric_name] = value.detach()
            else:
                scalar = _to_scalar(value)
                if scalar is not None:
                    metrics[prefix + metric_name] = action.new_tensor(scalar)
    return metrics


@parser.wrap()
def evaluate(cfg: SongEvalPipelineConfig, accelerator: Accelerator | None = None) -> dict[str, Any]:
    # TrainPipelineConfig normally rejects an output directory containing prior
    # artifacts. Evaluation only replaces eval_metrics.json and never writes a
    # checkpoint, so reusing an evaluation directory is safe and convenient.
    requested_output_dir = cfg.output_dir
    if requested_output_dir is not None and Path(requested_output_dir).is_dir():
        cfg.output_dir = None
    cfg.validate()
    if requested_output_dir is not None:
        cfg.output_dir = Path(requested_output_dir)
    if cfg.resume:
        raise ValueError(
            "eval_song.py does not resume optimizer state; use --policy.path instead of --resume."
        )
    if cfg.policy.pretrained_path is None:
        raise ValueError("eval_song.py requires a trained checkpoint provided through --policy.path.")

    if accelerator is None:
        from accelerate.utils import DistributedDataParallelKwargs

        accelerator = Accelerator(
            step_scheduler_with_optimizer=False,
            kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)],
            cpu=cfg.policy.device == "cpu",
        )

    init_logging(accelerator=accelerator)
    is_main_process = accelerator.is_main_process
    if is_main_process:
        logging.info(pformat(cfg.to_dict()))

    if cfg.seed is not None:
        set_seed(cfg.seed, accelerator=accelerator)

    device = accelerator.device
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    dataset_domain_selection = _resolve_libero_dataset_domain_selection(cfg)

    if is_main_process:
        logging.info("Creating evaluation dataset")
        if dataset_domain_selection is not None:
            logging.info(
                "LIBERO dataset-domain action MSE: suite=%s task=%s episodes=%s frames=%s",
                dataset_domain_selection["suite"],
                dataset_domain_selection["task_id"],
                dataset_domain_selection["episode_count"],
                dataset_domain_selection["frame_count"],
            )
        dataset = _make_eval_dataset(cfg, dataset_domain_selection)
    accelerator.wait_for_everyone()
    if not is_main_process:
        dataset = _make_eval_dataset(cfg, dataset_domain_selection)

    if is_main_process:
        logging.info("Loading evaluation policy")
    policy = make_policy(cfg=cfg.policy, ds_meta=dataset.meta, rename_map=cfg.rename_map)
    if is_main_process and hasattr(policy.config, "flow_contract_summary"):
        logging.info("Resolved flow contract: %s", policy.config.flow_contract_summary())
    ensure_ddp_parameters_initialized(policy, accelerator)
    preprocessor = _make_eval_preprocessor(cfg, policy, dataset, device)
    dataloader = _make_eval_dataloader(cfg, dataset, device)

    policy, dataloader = accelerator.prepare(policy, dataloader)
    policy.eval()
    raw_policy = accelerator.unwrap_model(policy)
    parameter_versions_before = {
        name: int(parameter._version) for name, parameter in raw_policy.named_parameters()
    }
    available_batches = len(dataloader)
    max_batches = min(int(cfg.steps), available_batches)
    if max_batches <= 0:
        raise ValueError(
            f"No evaluation batches requested: steps={cfg.steps}, available={available_batches}."
        )

    output_dir = Path(cfg.output_dir)
    if is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
        logging.info(colored("Evaluation output dir:", "yellow", attrs=["bold"]) + f" {output_dir}")
        logging.info(
            "Evaluation is read-only: batches=%s/%s, batch_size=%s, processes=%s, gradients=disabled",
            max_batches,
            available_batches,
            cfg.batch_size,
            accelerator.num_processes,
        )
    accelerator.wait_for_everyone()

    wandb_logger = WandBLogger(cfg) if cfg.wandb.enable and cfg.wandb.project and is_main_process else None

    running_sums = dict.fromkeys(EVAL_METRIC_KEYS, 0.0)
    running_counts = dict.fromkeys(EVAL_METRIC_KEYS, 0.0)

    progress = None
    if is_main_process:
        progress = tqdm(
            total=max_batches,
            desc="Evaluating",
            unit="batch",
            disable=inside_slurm(),
            position=0,
            leave=True,
        )

    for eval_step, raw_batch in enumerate(dataloader, start=1):
        if eval_step > max_batches:
            break

        batch = preprocessor(raw_batch)
        local_batch_size = int(batch[ACTION].shape[0])

        # Make the flow-matching noise/time sampling reproducible for a given
        # evaluation step while preserving the model's native sampling logic.
        eval_seed = int(cfg.seed or 0) + eval_step
        torch.manual_seed(eval_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(eval_seed)

        with torch.inference_mode(), accelerator.autocast():
            loss, output_dict = policy(batch)

        local_metrics = dict(output_dict or {})
        local_metrics["loss"] = loss.detach()
        if cfg.sample_action_mse:
            # Use a separate deterministic seed so this diagnostic neither
            # depends on nor perturbs the native loss sampling above.
            sample_seed = int(cfg.seed or 0) + 1_000_000 + eval_step
            torch.manual_seed(sample_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(sample_seed)
            with torch.inference_mode(), accelerator.autocast():
                local_metrics.update(
                    _sampled_action_metrics(
                        accelerator.unwrap_model(policy),
                        batch,
                        noise_mode=cfg.sample_action_noise_mode,
                    )
                )
        if cfg.flow_time_sweep:
            sweep_seed = int(cfg.seed or 0) + 2_000_000 + eval_step
            with torch.inference_mode(), accelerator.autocast():
                local_metrics.update(
                    _flow_time_sweep_metrics(
                        policy,
                        batch,
                        seed=sweep_seed,
                    )
                )
        batch_sums, batch_counts = _gather_metric_sums(
            accelerator,
            local_metrics,
            local_batch_size,
        )
        for key in EVAL_METRIC_KEYS:
            running_sums[key] += batch_sums[key]
            running_counts[key] += batch_counts[key]

        batch_metrics = {
            key: batch_sums[key] / batch_counts[key] for key in EVAL_METRIC_KEYS if batch_counts[key] > 0
        }

        if is_main_process and progress is not None:
            progress.update(1)
        if is_main_process and cfg.log_freq > 0 and eval_step % cfg.log_freq == 0:
            logging.info("eval_step:%s %s", eval_step, _format_metrics(batch_metrics))
            if wandb_logger is not None:
                wandb_logger.log_dict(
                    {f"eval/{key}": value for key, value in batch_metrics.items()},
                    eval_step,
                )

    if progress is not None:
        progress.close()

    summary_metrics = {
        key: running_sums[key] / running_counts[key] for key in EVAL_METRIC_KEYS if running_counts[key] > 0
    }
    evaluated_samples = int(running_counts.get("loss_action", 0.0))
    dataset_domain_report = None
    if dataset_domain_selection is not None:
        dataset_domain_report = {
            key: value
            for key, value in dataset_domain_selection.items()
            if key != "frame_indices"
        }
    summary = {
        "checkpoint": str(cfg.policy.pretrained_path),
        "dataset": str(cfg.dataset.repo_id),
        "pointseg_sample_cache_dir": str(cfg.pointseg_sample_cache_dir),
        "evaluated_batches": max_batches,
        "evaluated_samples": evaluated_samples,
        "batch_size_per_process": int(cfg.batch_size),
        "num_processes": int(accelerator.num_processes),
        "seed": cfg.seed,
        "metrics": summary_metrics,
        "dataset_domain_selection": dataset_domain_report,
        "metric_semantics": {
            "loss_action": (
                "The checkpoint's native flow-matching action velocity MSE, "
                "computed by policy(batch) with the same preprocessing, action chunks, "
                "padding masks, and PointSeg cache path as training."
            ),
            "sample_action_mse": (
                "Optional MSE between the fully denoised action chunk returned by "
                "predict_action_chunk and the ground-truth action chunk. Noise mode: "
                f"{cfg.sample_action_noise_mode}."
            ),
            "flow_time_sweep": (
                "Optional fixed-noise velocity and one-step endpoint diagnostics at "
                f"times {FLOW_TIME_SWEEP_VALUES}; this does not update parameters."
            ),
        },
    }

    gradients_created = [
        name for name, parameter in raw_policy.named_parameters() if parameter.grad is not None
    ]
    if gradients_created:
        raise RuntimeError(
            "Evaluation unexpectedly created parameter gradients: " + ", ".join(gradients_created[:10])
        )
    parameters_modified = [
        name
        for name, parameter in raw_policy.named_parameters()
        if int(parameter._version) != parameter_versions_before[name]
    ]
    if parameters_modified:
        raise RuntimeError(
            "Evaluation unexpectedly modified model parameters in place: "
            + ", ".join(parameters_modified[:10])
        )
    summary["gradients_created"] = False
    summary["model_parameters_unchanged"] = True

    if is_main_process:
        logging.info("Evaluation summary: %s", _format_metrics(summary_metrics))
        summary_path = output_dir / "eval_metrics.json"
        with summary_path.open("w", encoding="utf-8") as file:
            json.dump(summary, file, indent=2, ensure_ascii=False)
        logging.info("Saved evaluation metrics to %s", summary_path)

    accelerator.wait_for_everyone()
    accelerator.end_training()
    return summary


def main() -> None:
    register_third_party_plugins()
    evaluate()


def _apply_song_eval_debug_defaults() -> None:
    if len(sys.argv) > 1 and os.environ.get("SONG_EVAL_DEBUG_DEFAULTS") != "1":
        return
    sys.argv = [
        "eval_song.py",
        "--policy.path=/home/liusong/ProgramFiles/Huggingface/lerobot/benchmarks/song_real_libero/outputs/real_setting/train/ep_vla/checkpoints/last/pretrained_model",
        "--policy.push_to_hub=false",
        "--dataset.repo_id=/home/liusong/ProgramFiles/Huggingface/lerobot/benchmarks/song_real_libero/data/real_setting/real_lerobot_dataset",
        "--pointseg_sample_cache_dir=/opt/data/private/liusong/benchmarks/song_real_libero/data/real_setting/real_priorseg_cache",
        "--policy.vla_adapter_enable=true",
        "--policy.vla_adapter_freeze_vlm=true",
        "--policy.vlm_model_name=/home/liusong/hf_models/SmolVLM2-500M-Video-Instruct",
        "--policy.vlm_weights_path=/home/liusong/hf_models/smolvla_base",
        "--policy.load_vlm_weights=true",
        "--batch_size=8",
        # In eval_song, --steps is a maximum batch count. Evaluation still
        # traverses the finite DataLoader at most once, even when this matches
        # the much larger training update count.
        "--steps=80000",
        "--log_freq=1",
        "--output_dir=/home/liusong/temp/eval_song_test",
        "--job_name=temp_eval",
        "--policy.device=cuda",
        "--wandb.enable=false",
        "--wandb.disable_artifact=true",
        "--num_workers=8",
        "--policy.pointseg_enable=true",
        "--policy.pointseg_backbone_type=litept",
        "--policy.pointseg_grid_size=0.01",
        "--policy.pointseg_feature_dim=64",
        "--policy.pointseg_aux_loss_weight=0.001",
        "--policy.pointseg_foreground_ratio=0.08",
        "--policy.pointseg_background_ratio=0.08",
        "--policy.pointseg_min_foreground_points=4000",
        "--policy.pointseg_min_background_points=0",
        "--policy.pointseg_use_temporal_priors_as_input=false",
        "--policy.pointseg_use_pseudo_selection=false",
        "--policy.worldflow_enable=false",
        "--policy.worldflow_se3_head_enable=false",
        "--policy.se3_enable=false",
        "--policy.se3_final_correction_enable=false",
    ]


if __name__ == "__main__":
    _apply_song_eval_debug_defaults()
    main()
