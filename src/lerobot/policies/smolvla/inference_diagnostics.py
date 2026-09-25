"""Inference-time diagnostics for SmolVLA modality influence and PointSeg scores.

The modality analysis deliberately uses fixed-noise token-group ablations.  This
avoids confusing stochastic flow-matching variation with modality influence.
Attention weights alone are not treated as causal importance scores.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from collections.abc import Iterable
from contextlib import suppress
from multiprocessing import shared_memory
from pathlib import Path
from typing import Any

import numpy as np
import torch

from lerobot.utils.constants import (
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
)

MODALITY_ABLATIONS = ("language", "rgb", "point", "world", "action_context")
MODALITY_LABELS = {
    "language": "Language tokens",
    "rgb": "RGB tokens",
    "point": "Point tokens",
    "world": "World-flow tokens",
    "action_context": "Action-token context",
}


def _to_numpy(value: Any) -> np.ndarray:
    if torch.is_tensor(value):
        return value.detach().float().cpu().numpy()
    return np.asarray(value, dtype=np.float32)


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_safe(item) for item in value]
    return value


def _rot6d_to_matrix(rot6d: np.ndarray) -> np.ndarray:
    rot6d = np.asarray(rot6d, dtype=np.float64)
    a1 = rot6d[..., :3]
    a2 = rot6d[..., 3:6]
    b1 = a1 / np.clip(np.linalg.norm(a1, axis=-1, keepdims=True), 1e-8, None)
    b2 = a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = b2 / np.clip(np.linalg.norm(b2, axis=-1, keepdims=True), 1e-8, None)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=-1)


def _rotation_difference_deg(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    lhs_rot = _rot6d_to_matrix(lhs[..., 3:9])
    rhs_rot = _rot6d_to_matrix(rhs[..., 3:9])
    relative = np.swapaxes(lhs_rot, -1, -2) @ rhs_rot
    cosine = np.clip((np.trace(relative, axis1=-2, axis2=-1) - 1.0) * 0.5, -1.0, 1.0)
    return np.rad2deg(np.arccos(cosine)).astype(np.float32)


def action_difference_metrics(reference: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    """Compare two action chunks in the same (preferably postprocessed) space."""
    reference = np.asarray(reference, dtype=np.float32)
    candidate = np.asarray(candidate, dtype=np.float32)
    if reference.ndim == 3:
        reference = reference[0]
    if candidate.ndim == 3:
        candidate = candidate[0]
    steps = min(reference.shape[0], candidate.shape[0])
    dims = min(reference.shape[-1], candidate.shape[-1])
    reference = reference[:steps, :dims]
    candidate = candidate[:steps, :dims]
    delta = candidate - reference
    per_step_l2 = np.linalg.norm(delta, axis=-1)
    metrics: dict[str, Any] = {
        "action_mse": float(np.mean(delta**2)),
        "action_l2_mean": float(per_step_l2.mean()),
        "action_l2_max": float(per_step_l2.max(initial=0.0)),
        "per_step_action_l2": per_step_l2.astype(np.float32),
    }
    if dims >= 3:
        translation_l2 = np.linalg.norm(delta[:, :3], axis=-1)
        metrics.update(
            translation_l2_mean=float(translation_l2.mean()),
            translation_l2_max=float(translation_l2.max(initial=0.0)),
            per_step_translation_l2=translation_l2.astype(np.float32),
        )
    if dims >= 9:
        rotation_deg = _rotation_difference_deg(reference, candidate)
        metrics.update(
            rotation_deg_mean=float(rotation_deg.mean()),
            rotation_deg_max=float(rotation_deg.max(initial=0.0)),
            per_step_rotation_deg=rotation_deg,
        )
    if dims >= 1:
        gripper_abs = np.abs(delta[:, -1])
        metrics.update(
            gripper_abs_mean=float(gripper_abs.mean()),
            gripper_abs_max=float(gripper_abs.max(initial=0.0)),
            per_step_gripper_abs=gripper_abs.astype(np.float32),
        )
    return metrics


class SmolVLAModalityAnalyzer:
    """Run fixed-noise, stateless token-group ablations on a SmolVLA policy."""

    def __init__(self, policy: Any, postprocessor: Any | None = None) -> None:
        self.policy = policy
        self.postprocessor = postprocessor

    @staticmethod
    def _clone_batch(batch: dict[str, Any]) -> dict[str, Any]:
        return {key: value.clone() if torch.is_tensor(value) else value for key, value in batch.items()}

    def _make_fixed_noise(
        self,
        model_batch: dict[str, Any],
        seed: int,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        state = self.policy.prepare_state(model_batch)
        model = self.policy.model
        device = state.device
        cuda_devices: list[int] = []
        if device.type == "cuda":
            cuda_devices = [device.index if device.index is not None else torch.cuda.current_device()]
        with torch.random.fork_rng(devices=cuda_devices):
            torch.manual_seed(int(seed))
            if device.type == "cuda":
                torch.cuda.manual_seed_all(int(seed))
            shape = (state.shape[0], model.config.chunk_size, model.config.max_action_dim)
            if model.config.se3_enable:
                dummy = torch.zeros(shape, dtype=torch.float32, device=device)
                ego_noise = model.sample_se3_action_noise(dummy)[2]
            elif model.config.pose9_action_noise_enable:
                ego_noise = model.sample_pose9_action_noise(shape, device)
            else:
                ego_noise = model.sample_noise(shape, device)

            world_noise = None
            if model.config.worldflow_enable:
                coupling = getattr(model.config, "worldflow_noise_coupling", "independent")
                if coupling == "independent":
                    world_noise = model.sample_worldflow_noise(state.shape[0], device)
                elif coupling != "conjugate_ego":
                    raise ValueError(f"Unknown worldflow_noise_coupling={coupling!r}.")
            return ego_noise, world_noise

    def _predict_normalized(
        self,
        model_batch: dict[str, Any],
        noise: torch.Tensor,
        worldflow_noise: torch.Tensor | None,
        ablations: Iterable[str],
    ) -> torch.Tensor:
        batch = self._clone_batch(model_batch)
        batch = self.policy._prepare_batch(batch)
        pc_feats, pc_masks = self.policy.prepare_point_clouds(batch)
        if self.policy.config.vla_adapter_enable:
            images, image_masks = self.policy.prepare_images(batch)
        else:
            images, image_masks = None, None
        state = self.policy.prepare_state(batch)
        lang_tokens = batch[OBS_LANGUAGE_TOKENS]
        lang_masks = batch[OBS_LANGUAGE_ATTENTION_MASK]

        model = self.policy.model
        current_ee_pose = batch.get("worldflow.current_ee_pose")
        old_ablations = getattr(model, "inference_ablation_modalities", frozenset())
        model.inference_ablation_modalities = frozenset(str(name) for name in ablations)
        try:
            actions = model.sample_actions(
                pc_feats,
                pc_masks,
                lang_tokens,
                lang_masks,
                state,
                noise=noise,
                current_ee_pose=current_ee_pose,
                worldflow_noise=worldflow_noise,
                images=images,
                image_masks=image_masks,
            )
        finally:
            model.inference_ablation_modalities = old_ablations

        original_action_dim = self.policy.config.action_feature.shape[0]
        actions = actions[:, :, :original_action_dim]
        if self.policy.config.adapt_to_pi_aloha:
            actions = self.policy._pi_aloha_encode_actions(actions)
        return actions

    def _postprocess(self, action: torch.Tensor) -> np.ndarray:
        if self.postprocessor is not None:
            action = self.postprocessor(action)
        return _to_numpy(action)

    @torch.inference_mode()
    def analyze(
        self,
        model_batch: dict[str, Any],
        *,
        seed: int = 0,
        ablations: Iterable[str] = MODALITY_ABLATIONS,
        reference_action_chunk: torch.Tensor | np.ndarray | None = None,
    ) -> dict[str, Any]:
        requested = tuple(dict.fromkeys(str(name) for name in ablations))
        unknown = sorted(set(requested) - set(MODALITY_ABLATIONS))
        if unknown:
            raise ValueError(f"Unknown SmolVLA modality ablations: {unknown}")

        noise, worldflow_noise = self._make_fixed_noise(model_batch, seed)
        baseline_normalized = self._predict_normalized(model_batch, noise, worldflow_noise, ())
        baseline = self._postprocess(baseline_normalized)
        variants: dict[str, np.ndarray] = {}
        influence: dict[str, dict[str, Any]] = {}
        for name in requested:
            variant = self._postprocess(
                self._predict_normalized(model_batch, noise, worldflow_noise, (name,))
            )
            variants[name] = variant
            influence[name] = action_difference_metrics(baseline, variant)

        reference: np.ndarray | None = None
        quality: dict[str, Any] | None = None
        if reference_action_chunk is not None:
            reference_tensor = torch.as_tensor(reference_action_chunk)
            if reference_tensor.ndim == 2:
                reference_tensor = reference_tensor.unsqueeze(0)
            reference_tensor = reference_tensor.to(
                device=baseline_normalized.device,
                dtype=baseline_normalized.dtype,
            )
            reference_tensor = reference_tensor[..., : baseline_normalized.shape[-1]]
            reference = self._postprocess(reference_tensor)
            baseline_error = action_difference_metrics(reference, baseline)
            variant_errors: dict[str, Any] = {}
            for name, variant in variants.items():
                error = action_difference_metrics(reference, variant)
                error["action_mse_delta_vs_baseline"] = float(
                    error["action_mse"] - baseline_error["action_mse"]
                )
                variant_errors[name] = error
            quality = {
                "baseline_reference_error": baseline_error,
                "ablated_reference_errors": variant_errors,
                "interpretation": (
                    "Positive action_mse_delta_vs_baseline means removing that token group made the "
                    "prediction worse, so the group helped on this sample; negative means the ablated "
                    "prediction was closer to the supplied reference."
                ),
            }

        scalar_strength = np.asarray(
            [float(influence[name]["action_l2_mean"]) for name in requested], dtype=np.float64
        )
        total_strength = float(scalar_strength.sum())
        normalized_strength = {
            name: float(value / total_strength) if total_strength > 0 else 0.0
            for name, value in zip(requested, scalar_strength, strict=True)
        }
        return {
            "method": "fixed_noise_token_group_ablation",
            "seed": int(seed),
            "created_unix_s": time.time(),
            "caveat": (
                "Without a ground-truth action or rollout reward, these values measure causal sensitivity, "
                "not whether a modality made the action better."
            ),
            "ablation_definitions": {
                "language": "Mask language prefix tokens while preserving all other inputs.",
                "rgb": "Mask image and image-special prefix tokens after vision encoding.",
                "point": "Mask global point prefix tokens and disable local point-to-action fusion.",
                "world": (
                    "Mask World scene and World action tokens while preserving the joint suffix layout, "
                    "Ego tokens, fixed Ego noise and fixed World noise."
                ),
                "action_context": "Keep each action token and prefix cross-attention, but remove action-to-action attention.",
            },
            "baseline_action": baseline,
            "variant_actions": variants,
            "influence_vs_baseline": influence,
            "normalized_action_l2_influence": normalized_strength,
            "reference_action": reference,
            "quality_with_reference": quality,
        }


def save_modality_influence_report(report: dict[str, Any], output_dir: str | Path) -> dict[str, str]:
    """Save JSON plus trajectory, bar, and per-step plots."""
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "modality_influence.json"
    with open(report_path, "w", encoding="utf-8") as stream:
        json.dump(_json_safe(report), stream, indent=2, ensure_ascii=False)

    paths = {"json": str(report_path)}
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[warn] matplotlib unavailable; saved modality JSON only: {exc!r}", flush=True)
        return paths

    influence = report["influence_vs_baseline"]
    names = list(influence)
    labels = [MODALITY_LABELS.get(name, name) for name in names]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    values_by_axis = (
        ("translation_l2_mean", "Mean translation deviation"),
        ("rotation_deg_mean", "Mean rotation deviation (deg)"),
        ("gripper_abs_mean", "Mean gripper deviation"),
    )
    for axis, (metric, title) in zip(axes, values_by_axis, strict=True):
        values = [float(influence[name].get(metric, 0.0)) for name in names]
        axis.bar(labels, values, color=["#4C78A8", "#F58518", "#54A24B", "#B279A2"][: len(names)])
        axis.set_title(title)
        axis.tick_params(axis="x", rotation=22)
        axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    bar_path = output_dir / "modality_influence_bars.png"
    fig.savefig(bar_path, dpi=180)
    plt.close(fig)
    paths["bars"] = str(bar_path)

    baseline = np.asarray(report["baseline_action"], dtype=np.float32)[0]
    variants = {
        name: np.asarray(value, dtype=np.float32)[0] for name, value in report["variant_actions"].items()
    }
    fig = plt.figure(figsize=(8, 7))
    axis = fig.add_subplot(111, projection="3d")
    axis.plot(*baseline[:, :3].T, color="black", linewidth=2.5, label="baseline")
    colors = ["#4C78A8", "#F58518", "#54A24B", "#B279A2"]
    for color, (name, action) in zip(colors, variants.items(), strict=False):
        axis.plot(*action[:, :3].T, color=color, linewidth=1.5, label=f"without {name}")
    if report.get("reference_action") is not None:
        reference = np.asarray(report["reference_action"], dtype=np.float32)[0]
        axis.plot(*reference[:, :3].T, color="#E45756", linestyle="--", linewidth=2.0, label="reference")
    axis.set_xlabel("x")
    axis.set_ylabel("y")
    axis.set_zlabel("z")
    axis.set_title("Fixed-noise action trajectories")
    axis.legend(fontsize=8)
    fig.tight_layout()
    trajectory_path = output_dir / "modality_trajectory_comparison.png"
    fig.savefig(trajectory_path, dpi=180)
    plt.close(fig)
    paths["trajectory"] = str(trajectory_path)

    fig, axis = plt.subplots(figsize=(9, 4.5))
    for color, name in zip(colors, names, strict=False):
        values = np.asarray(influence[name]["per_step_action_l2"], dtype=np.float32)
        axis.plot(np.arange(len(values)), values, color=color, label=MODALITY_LABELS.get(name, name))
    axis.set_xlabel("Action step")
    axis.set_ylabel("L2 deviation from baseline")
    axis.set_title("Where each token group changes the action chunk")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)
    fig.tight_layout()
    step_path = output_dir / "modality_per_step_deviation.png"
    fig.savefig(step_path, dpi=180)
    plt.close(fig)
    paths["per_step"] = str(step_path)
    return paths


def foreground_score_colors(scores: np.ndarray, base_rgb: np.ndarray | None = None) -> np.ndarray:
    """Blue -> cyan -> yellow -> red foreground-probability heat map."""
    scores = np.clip(np.asarray(scores, dtype=np.float32).reshape(-1), 0.0, 1.0)
    anchors_x = np.asarray([0.0, 0.33, 0.66, 1.0], dtype=np.float32)
    anchors_rgb = np.asarray(
        [[0.05, 0.10, 0.95], [0.00, 0.90, 0.95], [1.00, 0.90, 0.05], [0.95, 0.05, 0.02]],
        dtype=np.float32,
    )
    colors = np.stack(
        [np.interp(scores, anchors_x, anchors_rgb[:, channel]) for channel in range(3)], axis=-1
    ).astype(np.float32)
    if base_rgb is not None:
        base = np.asarray(base_rgb, dtype=np.float32)
        if base.max(initial=0.0) > 1.0:
            base = base / 255.0
        colors = 0.12 * np.clip(base, 0.0, 1.0) + 0.88 * colors
    return np.clip(colors, 0.0, 1.0)


class ForegroundScoreVisualizer:
    """Persistent PointSeg window isolated from the inference OpenGL context."""

    def __init__(
        self,
        *,
        enabled: bool = False,
        max_points: int = 50000,
        window_name: str = "PointSeg foreground score: blue=0, red=1",
        width: int = 960,
        height: int = 720,
        print_every: int = 10,
    ) -> None:
        self.enabled = bool(enabled)
        self.max_points = max(1, int(max_points))
        self.window_name = str(window_name)
        self.width = int(width)
        self.height = int(height)
        self.print_every = max(1, int(print_every))
        self._data_shm: shared_memory.SharedMemory | None = None
        self._meta_shm: shared_memory.SharedMemory | None = None
        self._data_array: np.ndarray | None = None
        self._meta_array: np.ndarray | None = None
        self._process: subprocess.Popen[Any] | None = None
        self._started_once = False
        self._updates = 0
        self._failed = False
        self.last_stats: dict[str, float] = {}

    def enable(self) -> None:
        self.enabled = True

    def _ensure_process(self) -> bool:
        if not self.enabled or self._failed:
            return False
        if self._process is not None:
            if self._process.poll() is None:
                return True
            self._failed = True
            self.enabled = False
            print("[warn] foreground score window process exited; visualization disabled", flush=True)
            return False
        try:
            data_nbytes = self.max_points * 6 * np.dtype(np.float32).itemsize
            meta_nbytes = 2 * np.dtype(np.int64).itemsize
            self._data_shm = shared_memory.SharedMemory(create=True, size=data_nbytes)
            self._meta_shm = shared_memory.SharedMemory(create=True, size=meta_nbytes)
            self._data_array = np.ndarray((self.max_points, 6), dtype=np.float32, buffer=self._data_shm.buf)
            self._meta_array = np.ndarray((2,), dtype=np.int64, buffer=self._meta_shm.buf)
            self._meta_array[:] = 0
            viewer_script = Path(__file__).with_name("foreground_score_viewer.py")
            self._process = subprocess.Popen(
                [
                    sys.executable,
                    str(viewer_script),
                    "--data-shm",
                    self._data_shm.name,
                    "--meta-shm",
                    self._meta_shm.name,
                    "--max-points",
                    str(self.max_points),
                    "--window-name",
                    self.window_name,
                    "--width",
                    str(self.width),
                    "--height",
                    str(self.height),
                    "--parent-pid",
                    str(os.getpid()),
                ],
                close_fds=True,
            )
            self._started_once = True
            return True
        except Exception as exc:
            self._failed = True
            self._release_shared_memory()
            print(f"[warn] foreground score window disabled: {exc!r}", flush=True)
            return False

    def _release_shared_memory(self) -> None:
        self._data_array = None
        self._meta_array = None
        for memory in (self._data_shm, self._meta_shm):
            if memory is None:
                continue
            with suppress(Exception):
                memory.close()
            with suppress(FileNotFoundError):
                memory.unlink()
        self._data_shm = None
        self._meta_shm = None

    def update_colored(self, xyzrgb: torch.Tensor | np.ndarray, *, batch_index: int = 0) -> bool:
        """Publish an explicitly colored point cloud through the isolated viewer.

        This is also used by inference-time trajectory debugging.  Keeping the
        Open3D window in the same standalone viewer process as foreground-score
        visualization prevents MuJoCo EGL and Open3D GLX contexts from being
        created in the policy / environment process.
        """
        if not self.enabled:
            return False
        points = _to_numpy(xyzrgb)
        if points.ndim == 3:
            points = points[batch_index]
        if points.ndim != 2 or points.shape[-1] < 6:
            raise ValueError(
                "Colored point-cloud visualization expects an (N, >=6) xyzrgb array, "
                f"got {points.shape}."
            )

        points = np.asarray(points[:, :6], dtype=np.float32)
        finite = np.isfinite(points).all(axis=1)
        points = points[finite]
        if len(points) == 0:
            return False
        if len(points) > self.max_points:
            indices = np.linspace(0, len(points) - 1, self.max_points, dtype=np.int64)
            points = points[indices]
        colors = points[:, 3:6]
        if colors.max(initial=0.0) > 1.0:
            colors = colors / 255.0
        colors = np.clip(colors, 0.0, 1.0)
        if not self._ensure_process():
            return False

        assert self._data_array is not None and self._meta_array is not None
        count = len(points)
        self._meta_array[0] += 1  # odd: writer owns the shared frame
        self._data_array[:count, :3] = points[:, :3]
        self._data_array[:count, 3:6] = colors
        self._meta_array[1] = count
        self._meta_array[0] += 1  # even: complete frame is available
        if self._process is not None and self._process.poll() is not None:
            self._failed = True
            self.enabled = False
            return False
        self._updates += 1
        return True

    def update(
        self,
        xyzrgb: torch.Tensor | np.ndarray,
        scores: torch.Tensor | np.ndarray,
        point_is_pad: torch.Tensor | np.ndarray | None = None,
        *,
        batch_index: int = 0,
    ) -> bool:
        if not self.enabled:
            return False
        points = _to_numpy(xyzrgb)
        probabilities = _to_numpy(scores)
        if points.ndim == 3:
            points = points[batch_index]
        if probabilities.ndim == 2:
            probabilities = probabilities[batch_index]
        probabilities = probabilities.reshape(-1)
        if point_is_pad is not None:
            padding = _to_numpy(point_is_pad)
            if padding.ndim == 2:
                padding = padding[batch_index]
            valid = ~padding.astype(bool).reshape(-1)
            points = points[valid]
            probabilities = probabilities[valid]
        if points.ndim != 2 or points.shape[-1] < 3 or len(points) != len(probabilities):
            raise ValueError(
                f"Foreground visualization expects aligned (N,6)/(N,) arrays, got {points.shape}/{probabilities.shape}."
            )
        finite = np.isfinite(points[:, :3]).all(axis=1) & np.isfinite(probabilities)
        points = points[finite]
        probabilities = np.clip(probabilities[finite], 0.0, 1.0)
        if len(points) == 0:
            return False
        if len(points) > self.max_points:
            indices = np.linspace(0, len(points) - 1, self.max_points, dtype=np.int64)
            points = points[indices]
            probabilities = probabilities[indices]
        if not self._ensure_process():
            return False

        base_rgb = points[:, 3:6] if points.shape[-1] >= 6 else None
        assert self._data_array is not None and self._meta_array is not None
        count = len(points)
        self._meta_array[0] += 1  # odd: writer owns the shared frame
        self._data_array[:count, :3] = points[:, :3]
        self._data_array[:count, 3:6] = foreground_score_colors(probabilities, base_rgb)
        self._meta_array[1] = count
        self._meta_array[0] += 1  # even: complete frame is available
        if self._process is not None and self._process.poll() is not None:
            self._failed = True
            self.enabled = False
            return False

        self._updates += 1
        self.last_stats = {
            "mean": float(probabilities.mean()),
            "p50": float(np.quantile(probabilities, 0.50)),
            "p90": float(np.quantile(probabilities, 0.90)),
            "foreground_ratio_0.5": float(np.mean(probabilities >= 0.5)),
            "num_points": float(len(probabilities)),
        }
        if self._updates == 1 or self._updates % self.print_every == 0:
            print(
                "[pointseg-vis] "
                f"points={len(probabilities)} mean={self.last_stats['mean']:.3f} "
                f"p50={self.last_stats['p50']:.3f} p90={self.last_stats['p90']:.3f} "
                f"fg@0.5={self.last_stats['foreground_ratio_0.5']:.3f}",
                flush=True,
            )
        return True

    def update_from_model(self, model: Any, *, batch_index: int = 0) -> bool:
        snapshot = getattr(model, "last_pointseg_visualization", None)
        if not isinstance(snapshot, dict):
            return self.refresh()
        return self.update(
            snapshot["point_cloud"],
            snapshot["operation_prob"],
            snapshot.get("point_is_pad"),
            batch_index=batch_index,
        )

    def refresh(self) -> bool:
        if self._process is None:
            return False
        if self._process.poll() is not None:
            self.enabled = False
            self._failed = self._started_once
            return False
        return True

    def close(self) -> None:
        if self._meta_array is not None:
            self._meta_array[1] = -1
            self._meta_array[0] += 1
        if self._process is not None:
            with suppress(subprocess.TimeoutExpired):
                self._process.wait(timeout=2.0)
            if self._process.poll() is None:
                self._process.terminate()
                with suppress(subprocess.TimeoutExpired):
                    self._process.wait(timeout=1.0)
        self._process = None
        self._release_shared_memory()
