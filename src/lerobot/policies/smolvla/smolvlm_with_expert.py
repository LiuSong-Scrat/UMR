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

import copy
import json
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from huggingface_hub.constants import SAFETENSORS_SINGLE_FILE
from safetensors import safe_open
from safetensors.torch import load_file
from torch import nn
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForImageTextToText,
    AutoProcessor,
    SmolVLMForConditionalGeneration,
)

_CONFIG_NAME = "config.json"
_SMOLVLA_POLICY_VLM_PREFIXES = (
    "module.model.vlm_with_expert.vlm.",
    "model.vlm_with_expert.vlm.",
    "vlm_with_expert.vlm.",
)


def _is_disabled_source(value: str | None) -> bool:
    return value is None or str(value).strip().lower() in {"", "0", "false", "none", "off"}


def _read_json_file(path: Path) -> dict | None:
    try:
        with open(path, encoding="utf-8") as file:
            return json.load(file)
    except FileNotFoundError:
        return None


def _is_smolvla_policy_config(config: dict | None) -> bool:
    return isinstance(config, dict) and (config.get("type") == "smolvla" or "vlm_model_name" in config)


def _resolve_smolvla_policy_checkpoint(source: str | None) -> tuple[dict | None, str | None]:
    """Resolve a local/Hub SmolVLA policy checkpoint without mistaking raw SmolVLM for one."""
    if _is_disabled_source(source):
        return None, None
    source_str = str(source)
    local_path = Path(source_str).expanduser()
    if local_path.is_file() and local_path.suffix == ".safetensors":
        return None, str(local_path)
    if local_path.is_dir():
        policy_config = _read_json_file(local_path / _CONFIG_NAME)
        if not _is_smolvla_policy_config(policy_config):
            return None, None
        model_file = local_path / SAFETENSORS_SINGLE_FILE
        if not model_file.is_file():
            raise FileNotFoundError(
                f"SmolVLA policy checkpoint is missing {SAFETENSORS_SINGLE_FILE}: {local_path}"
            )
        return policy_config, str(model_file)

    try:
        config_file = hf_hub_download(repo_id=source_str, filename=_CONFIG_NAME)
    except Exception:
        return None, None
    policy_config = _read_json_file(Path(config_file))
    if not _is_smolvla_policy_config(policy_config):
        return None, None
    model_file = hf_hub_download(repo_id=source_str, filename=SAFETENSORS_SINGLE_FILE)
    return policy_config, model_file


def _extract_vlm_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    for prefix in _SMOLVLA_POLICY_VLM_PREFIXES:
        vlm_state = {key[len(prefix) :]: value for key, value in state_dict.items() if key.startswith(prefix)}
        if vlm_state:
            return vlm_state
    if any(key.startswith(("model.", "lm_head.")) for key in state_dict):
        return state_dict
    raise KeyError(
        "Could not find VLM weights. Expected a raw SmolVLM state dict or keys prefixed by "
        f"one of {_SMOLVLA_POLICY_VLM_PREFIXES}."
    )


def _missing_key_is_unused_after_truncation(key: str, num_vlm_layers: int) -> bool:
    marker = ".text_model.layers."
    if marker in key:
        suffix = key.split(marker, maxsplit=1)[1]
        layer_str = suffix.split(".", maxsplit=1)[0]
        if layer_str.isdigit() and int(layer_str) >= num_vlm_layers:
            return True
    # The policy never uses the language-model output head.
    return key.startswith("lm_head.")


def _load_vlm_policy_weights(vlm: nn.Module, model_file: str, num_vlm_layers: int) -> None:
    state_dict = load_file(model_file, device="cpu")
    vlm_state = _extract_vlm_state_dict(state_dict)
    missing_keys, unexpected_keys = vlm.load_state_dict(vlm_state, strict=False)
    required_missing = [
        key for key in missing_keys if not _missing_key_is_unused_after_truncation(key, num_vlm_layers)
    ]
    if required_missing:
        preview = ", ".join(required_missing[:8])
        raise RuntimeError(
            "The supplied checkpoint does not contain all VLM parameters used by this model. "
            f"Missing {len(required_missing)} required keys, including: {preview}"
        )
    print(
        "Loaded frozen VLM weights from SmolVLA/raw checkpoint "
        f"{model_file} (ignored_missing={len(missing_keys) - len(required_missing)}, "
        f"unexpected={len(unexpected_keys)})."
    )


def _policy_tensor_key(
    target_key: str,
    source_keys: set[str],
) -> str | None:
    for prefix in ("module.model.", "model.", ""):
        candidate = f"{prefix}{target_key}"
        if candidate in source_keys:
            return candidate
    return None


def load_pretrained_action_expert_weights(
    target: nn.Module,
    source: str,
    *,
    load_action_projections: bool = False,
) -> dict[str, int | str]:
    """Initialize a fresh Song SmolVLA Action Expert from a full SmolVLA policy.

    `target` is the inner VLAFlowMatching module, whose state-dict keys do not
    include the outer ``model.`` prefix. Transformer and timestep-conditioning
    tensors must match exactly. The official action head has 32 channels whose
    semantics need not match Song pose9 policies, so its projections remain
    task-specific by default. If explicitly requested, they are copied over the
    shared leading channels.

    This function validates every tensor before mutating the target, preventing
    a malformed or raw SmolVLM checkpoint from partially initializing a model.
    """

    _policy_config, model_file = _resolve_smolvla_policy_checkpoint(source)
    if model_file is None:
        raise ValueError(
            "Action Expert initialization requires a complete SmolVLA policy checkpoint "
            f"containing {SAFETENSORS_SINGLE_FILE}; got {source!r}."
        )

    target_state = target.state_dict()
    exact_keys = [
        key
        for key in target_state
        if key.startswith("vlm_with_expert.lm_expert.")
        or key.startswith("action_time_mlp_in.")
        or key.startswith("action_time_mlp_out.")
    ]
    projection_keys = (
        [
            key
            for key in (
                "action_in_proj.weight",
                "action_in_proj.bias",
                "action_out_proj.weight",
                "action_out_proj.bias",
            )
            if key in target_state
        ]
        if load_action_projections
        else []
    )
    if not exact_keys:
        raise RuntimeError("Target model has no Action Expert parameters to initialize.")

    assignments: list[tuple[str, torch.Tensor]] = []
    missing_keys: list[str] = []
    shape_errors: list[str] = []
    with safe_open(model_file, framework="pt", device="cpu") as checkpoint:
        source_keys = set(checkpoint.keys())

        for target_key in exact_keys:
            source_key = _policy_tensor_key(target_key, source_keys)
            if source_key is None:
                missing_keys.append(target_key)
                continue
            source_tensor = checkpoint.get_tensor(source_key)
            if source_tensor.shape != target_state[target_key].shape:
                shape_errors.append(
                    f"{target_key}: source={tuple(source_tensor.shape)} "
                    f"target={tuple(target_state[target_key].shape)}"
                )
                continue
            assignments.append((target_key, source_tensor))

        for target_key in projection_keys:
            source_key = _policy_tensor_key(target_key, source_keys)
            if source_key is None:
                missing_keys.append(target_key)
                continue
            source_tensor = checkpoint.get_tensor(source_key)
            target_tensor = target_state[target_key]

            if target_key == "action_in_proj.weight":
                compatible = (
                    source_tensor.ndim == 2
                    and source_tensor.shape[0] == target_tensor.shape[0]
                    and source_tensor.shape[1] >= target_tensor.shape[1]
                )
                adapted = source_tensor[:, : target_tensor.shape[1]] if compatible else source_tensor
            elif target_key == "action_out_proj.weight":
                compatible = (
                    source_tensor.ndim == 2
                    and source_tensor.shape[0] >= target_tensor.shape[0]
                    and source_tensor.shape[1] == target_tensor.shape[1]
                )
                adapted = source_tensor[: target_tensor.shape[0], :] if compatible else source_tensor
            elif target_key == "action_out_proj.bias":
                compatible = source_tensor.ndim == 1 and source_tensor.shape[0] >= target_tensor.shape[0]
                adapted = source_tensor[: target_tensor.shape[0]] if compatible else source_tensor
            else:
                compatible = source_tensor.shape == target_tensor.shape
                adapted = source_tensor

            if not compatible or adapted.shape != target_tensor.shape:
                shape_errors.append(
                    f"{target_key}: source={tuple(source_tensor.shape)} "
                    f"target={tuple(target_tensor.shape)}"
                )
                continue
            assignments.append((target_key, adapted))

    if missing_keys or shape_errors:
        details = []
        if missing_keys:
            details.append(f"missing={missing_keys[:8]}")
        if shape_errors:
            details.append(f"shape_mismatch={shape_errors[:8]}")
        raise RuntimeError(
            "The supplied SmolVLA checkpoint is incompatible with the current Action Expert: "
            + "; ".join(details)
        )

    with torch.no_grad():
        for target_key, source_tensor in assignments:
            target_state[target_key].copy_(
                source_tensor.to(
                    device=target_state[target_key].device,
                    dtype=target_state[target_key].dtype,
                )
            )

    report: dict[str, int | str] = {
        "source": model_file,
        "expert_and_time_tensors": len(exact_keys),
        "projection_tensors": len(projection_keys),
        "total_tensors": len(assignments),
    }
    print(
        "Initialized Action Expert from SmolVLA policy "
        f"{model_file} (expert/time={len(exact_keys)}, projections={len(projection_keys)})."
    )
    return report


def apply_rope(x, positions, max_wavelength=10_000):
    """
    Applies RoPE positions [B, L] to x [B, L, H, D].
    """
    d_half = x.shape[-1] // 2
    device = x.device
    dtype = x.dtype
    x = x.to(torch.float32)

    freq_exponents = (2.0 / x.shape[-1]) * torch.arange(d_half, dtype=torch.float32, device=device)
    timescale = max_wavelength**freq_exponents
    radians = positions[..., None].to(torch.float32) / timescale[None, None, :].to(torch.float32)

    radians = radians[..., None, :]

    sin = torch.sin(radians)  # .to(dtype=dtype)
    cos = torch.cos(radians)  # .to(dtype=dtype)

    x1, x2 = x.split(d_half, dim=-1)
    res = torch.empty_like(x)
    res[..., :d_half] = x1 * cos - x2 * sin
    res[..., d_half:] = x2 * cos + x1 * sin

    return res.to(dtype)


def get_intermediate_size(hidden_dim, ffn_dim_multiplier=4, multiple_of=256):
    hidden_dim = int(2 * hidden_dim / 3)
    hidden_dim = int(ffn_dim_multiplier * hidden_dim)
    hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
    return hidden_dim


class SmolVLMWithExpertModel(nn.Module):
    def __init__(
        self,
        model_id: str = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
        vlm_weights_path: str | None = None,
        load_vlm_weights: bool = True,
        train_expert_only: bool = True,
        freeze_vision_encoder: bool = False,
        attention_mode: str = "self_attn",
        num_expert_layers: int = -1,
        num_vlm_layers: int = -1,
        self_attn_every_n_layers: int = -1,
        expert_width_multiplier: float = 0.5,
        device: str = "auto",
    ):
        super().__init__()
        policy_config, policy_vlm_file = _resolve_smolvla_policy_checkpoint(model_id)
        architecture_model_id = policy_config.get("vlm_model_name", model_id) if policy_config else model_id

        weights_override_requested = not _is_disabled_source(vlm_weights_path)
        if weights_override_requested:
            _weights_policy_config, weights_policy_vlm_file = _resolve_smolvla_policy_checkpoint(
                vlm_weights_path
            )
            if weights_policy_vlm_file is not None:
                policy_vlm_file = weights_policy_vlm_file
            else:
                # A raw Hugging Face SmolVLM directory/repository is both the
                # architecture/processor and pretrained-weight source.
                architecture_model_id = str(vlm_weights_path)

        if (weights_override_requested or policy_vlm_file is not None) and not load_vlm_weights:
            raise ValueError("A pretrained VLM weight source was provided but load_vlm_weights=False.")

        if load_vlm_weights and policy_vlm_file is None:
            print(f"Loading VLM architecture/weights from {architecture_model_id} ...")
            self.vlm = AutoModelForImageTextToText.from_pretrained(
                architecture_model_id,
                torch_dtype="bfloat16",
                low_cpu_mem_usage=True,
            )
            config = self.vlm.config
        else:
            config = AutoConfig.from_pretrained(architecture_model_id)
            self.vlm = SmolVLMForConditionalGeneration(config=config)
            if policy_vlm_file is not None:
                retained_layers = num_vlm_layers if num_vlm_layers > 0 else config.text_config.num_hidden_layers
                _load_vlm_policy_weights(self.vlm, policy_vlm_file, retained_layers)

        self.processor = AutoProcessor.from_pretrained(architecture_model_id)
        if num_vlm_layers > 0:
            print(f"Reducing the number of VLM layers to {num_vlm_layers} ...")
            self.get_vlm_model().text_model.layers = self.get_vlm_model().text_model.layers[:num_vlm_layers]
        self.num_vlm_layers = len(self.get_vlm_model().text_model.layers)
        self.config = config
        # Smaller lm expert
        lm_expert_config = copy.deepcopy(config.text_config)
        hidden_size = lm_expert_config.hidden_size
        lm_expert_config.hidden_size = int(hidden_size * expert_width_multiplier)  # hidden_size // 2
        lm_expert_config.intermediate_size = get_intermediate_size(int(hidden_size * expert_width_multiplier))
        lm_expert_config.num_hidden_layers = self.num_vlm_layers
        if num_expert_layers > 0:
            assert len(self.get_vlm_model().text_model.layers) % num_expert_layers == 0, (
                f"Number of layers in the VLM {len(self.get_vlm_model().text_model.layers)} are not multiple of num_expert_layers {num_expert_layers}"
            )
            lm_expert_config.num_hidden_layers = num_expert_layers
        self.lm_expert = AutoModel.from_config(lm_expert_config)

        self.num_expert_layers = len(self.lm_expert.layers)
        self.self_attn_every_n_layers = self_attn_every_n_layers
        if "cross" in attention_mode:
            # Reshape qkv projections to have the same input dimension as the vlm
            for layer_idx in range(len(self.lm_expert.layers)):
                if self.self_attn_every_n_layers > 0 and layer_idx % self.self_attn_every_n_layers == 0:
                    continue
                self.lm_expert.layers[layer_idx].self_attn.k_proj = nn.Linear(
                    config.text_config.num_key_value_heads * config.text_config.head_dim,
                    lm_expert_config.num_key_value_heads * lm_expert_config.head_dim,
                    bias=lm_expert_config.attention_bias,
                )
                self.lm_expert.layers[layer_idx].self_attn.v_proj = nn.Linear(
                    config.text_config.num_key_value_heads * config.text_config.head_dim,
                    lm_expert_config.num_key_value_heads * lm_expert_config.head_dim,
                    bias=lm_expert_config.attention_bias,
                )
        # Remove unused embed_tokens
        self.lm_expert.embed_tokens = None

        self.num_attention_heads = self.config.text_config.num_attention_heads
        self.num_key_value_heads = self.config.text_config.num_key_value_heads

        self.freeze_vision_encoder = freeze_vision_encoder
        self.train_expert_only = train_expert_only
        self.attention_mode = attention_mode
        self.expert_hidden_size = lm_expert_config.hidden_size
        self.set_requires_grad()

    def get_vlm_model(self):
        return self.vlm.model

    def set_requires_grad(self):
        if self.freeze_vision_encoder:
            self.get_vlm_model().vision_model.eval()
            for params in self.get_vlm_model().vision_model.parameters():
                params.requires_grad = False
        if self.train_expert_only:
            self.vlm.eval()
            for params in self.vlm.parameters():
                params.requires_grad = False
        else:
            # To avoid unused params issue with distributed training
            last_layers = [self.num_vlm_layers - 1]
            if (
                self.num_vlm_layers != self.num_expert_layers
                and self.num_vlm_layers % self.num_expert_layers == 0
            ):
                last_layers.append(self.num_vlm_layers - 2)
            frozen_layers = [
                "lm_head",
                "text_model.model.norm.weight",
            ]
            for layer in last_layers:
                frozen_layers.append(f"text_model.model.layers.{layer}.")

            for name, params in self.vlm.named_parameters():
                if any(k in name for k in frozen_layers):
                    params.requires_grad = False
        # To avoid unused params issue with distributed training
        for name, params in self.lm_expert.named_parameters():
            if "lm_head" in name:
                params.requires_grad = False
    def train(self, mode: bool = True):
        super().train(mode)

        if self.freeze_vision_encoder:
            self.get_vlm_model().vision_model.eval()

        if self.train_expert_only:
            self.vlm.eval()

    def embed_image(self, image: torch.Tensor):
        patch_attention_mask = None
        # Get sequence from the vision encoder
        image_hidden_states = (
            self.get_vlm_model()
            .vision_model(
                pixel_values=image.to(dtype=self.get_vlm_model().vision_model.dtype),
                patch_attention_mask=patch_attention_mask,
            )
            .last_hidden_state
        )
        # Modality projection & resampling
        image_hidden_states = self.get_vlm_model().connector(image_hidden_states)
        return image_hidden_states

    def embed_language_tokens(self, tokens: torch.Tensor):
        return self.get_vlm_model().text_model.get_input_embeddings()(tokens)

    def forward_attn_layer(
        self,
        model_layers,
        inputs_embeds,
        layer_idx,
        position_ids,
        attention_mask,
        batch_size,
        head_dim,
        use_cache: bool = True,
        fill_kv_cache: bool = True,
        past_key_values=None,
    ) -> list[torch.Tensor]:
        query_states = []
        key_states = []
        value_states = []
        for i, hidden_states in enumerate(inputs_embeds):
            layer = model_layers[i][layer_idx]
            if hidden_states is None or layer is None:
                continue
            hidden_states = layer.input_layernorm(hidden_states)

            input_shape = hidden_states.shape[:-1]
            hidden_shape = (*input_shape, -1, layer.self_attn.head_dim)

            hidden_states = hidden_states.to(dtype=layer.self_attn.q_proj.weight.dtype)
            query_state = layer.self_attn.q_proj(hidden_states).view(hidden_shape)
            key_state = layer.self_attn.k_proj(hidden_states).view(hidden_shape)
            value_state = layer.self_attn.v_proj(hidden_states).view(hidden_shape)

            query_states.append(query_state)
            key_states.append(key_state)
            value_states.append(value_state)

        # B,L,H,D with L sequence length, H number of heads, D head dim
        # concatenate on the number of embeddings/tokens
        query_states = torch.cat(query_states, dim=1)
        key_states = torch.cat(key_states, dim=1)
        value_states = torch.cat(value_states, dim=1)
        seq_len = query_states.shape[1]
        if seq_len < position_ids.shape[1]:
            _position_ids = position_ids[:, :seq_len]
            _attention_mask = attention_mask[:, :seq_len, :seq_len]
        else:
            _position_ids = position_ids
            _attention_mask = attention_mask

        attention_mask_ = _attention_mask
        position_ids_ = _position_ids

        query_states = apply_rope(query_states, position_ids_)
        key_states = apply_rope(key_states, position_ids_)

        if use_cache and past_key_values is None:
            past_key_values = {}

        if use_cache:
            if fill_kv_cache:
                past_key_values[layer_idx] = {
                    "key_states": key_states,
                    "value_states": value_states,
                }
            else:
                # TODO here, some optimization can be done - similar to a `StaticCache` we can declare the `max_len` before.
                # so we create an empty cache, with just one cuda malloc, and if (in autoregressive case) we reach
                # the max len, then we (for instance) double the cache size. This implementation already exists
                # in `transformers`. (molbap)
                key_states = torch.cat([past_key_values[layer_idx]["key_states"], key_states], dim=1)
                value_states = torch.cat([past_key_values[layer_idx]["value_states"], value_states], dim=1)

        attention_interface = self.get_attention_interface()

        att_output = attention_interface(
            attention_mask_, batch_size, head_dim, query_states, key_states, value_states
        )
        return [att_output], past_key_values

    def forward_cross_attn_layer(
        self,
        model_layers,
        inputs_embeds,
        layer_idx,
        position_ids,
        attention_mask,
        batch_size,
        head_dim,
        use_cache: bool = True,
        fill_kv_cache: bool = True,
        past_key_values=None,
    ) -> list[torch.Tensor]:
        attention_interface = self.get_attention_interface()

        att_outputs = []
        assert len(inputs_embeds) == 2 or (use_cache and past_key_values is not None and not fill_kv_cache), (
            f"Both len(inputs_embeds) == {len(inputs_embeds)} and past_key_values is {past_key_values}"
        )

        if len(inputs_embeds) == 2 and not past_key_values:
            # Prefix attention
            seq_len = inputs_embeds[0].shape[1]
            position_id, expert_position_id = position_ids[:, :seq_len], position_ids[:, seq_len:]
            prefix_attention_mask = attention_mask[:, :seq_len, :seq_len]

            layer = model_layers[0][layer_idx]

            hidden_states = layer.input_layernorm(inputs_embeds[0])

            input_shape = hidden_states.shape[:-1]
            hidden_shape = (*input_shape, -1, layer.self_attn.head_dim)

            hidden_states = hidden_states.to(dtype=layer.self_attn.q_proj.weight.dtype)
            query_state = layer.self_attn.q_proj(hidden_states).view(hidden_shape)
            key_state = layer.self_attn.k_proj(hidden_states).view(hidden_shape)
            value_states = layer.self_attn.v_proj(hidden_states).view(hidden_shape)

            # B,L,H,D with L sequence length, H number of heads, D head dim
            query_states = apply_rope(query_state, position_id)
            key_states = apply_rope(key_state, position_id)

            att_output = attention_interface(
                prefix_attention_mask, batch_size, head_dim, query_states, key_states, value_states
            )
            att_outputs.append(att_output)
        else:
            expert_position_id = position_ids

        if use_cache and past_key_values is None:
            past_key_values = {}

        if use_cache:
            if fill_kv_cache:
                past_key_values[layer_idx] = {
                    "key_states": key_states,
                    "value_states": value_states,
                }
            else:
                # TODO here, some optimization can be done - similar to a `StaticCache` we can declare the `max_len` before.
                # so we create an empty cache, with just one cuda malloc, and if (in autoregressive case) we reach
                # the max len, then we (for instance) double the cache size. This implementation already exists
                # in `transformers`. (molbap)
                key_states = past_key_values[layer_idx]["key_states"]
                value_states = past_key_values[layer_idx]["value_states"]

        # Expert
        expert_layer = model_layers[1][layer_idx]
        if expert_layer is not None:
            expert_hidden_states = expert_layer.input_layernorm(inputs_embeds[1])

            expert_input_shape = expert_hidden_states.shape[:-1]
            expert_hidden_shape = (*expert_input_shape, -1, expert_layer.self_attn.head_dim)

            expert_hidden_states = expert_hidden_states.to(dtype=expert_layer.self_attn.q_proj.weight.dtype)
            expert_query_state = expert_layer.self_attn.q_proj(expert_hidden_states).view(expert_hidden_shape)

            _key_states = key_states.to(dtype=expert_layer.self_attn.k_proj.weight.dtype).view(
                *key_states.shape[:2], -1
            )
            expert_key_states = expert_layer.self_attn.k_proj(_key_states).view(
                *_key_states.shape[:-1], -1, expert_layer.self_attn.head_dim
            )  # k_proj should have same dim as kv

            _value_states = value_states.to(dtype=expert_layer.self_attn.v_proj.weight.dtype).view(
                *value_states.shape[:2], -1
            )
            expert_value_states = expert_layer.self_attn.v_proj(_value_states).view(
                *_value_states.shape[:-1], -1, expert_layer.self_attn.head_dim
            )

            expert_position_id = (
                expert_position_id - torch.min(expert_position_id, dim=1, keepdim=True).values
            )  # start from 0
            expert_attention_mask = attention_mask[
                :, -inputs_embeds[1].shape[1] :, : expert_key_states.shape[1] :
            ]  # take into account kv

            expert_query_states = apply_rope(expert_query_state, expert_position_id)

            att_output = attention_interface(
                expert_attention_mask,
                batch_size,
                head_dim,
                expert_query_states,
                expert_key_states,
                expert_value_states,
            )
            att_outputs.append(att_output)
        else:
            att_outputs.append(None)

        # att_output = att_output.to(dtype=models[i].dtype)
        return att_outputs, past_key_values

    def get_model_layers(self, models: list) -> list:
        vlm_layers = []
        expert_layers = []
        multiple_of = self.num_vlm_layers // self.num_expert_layers
        for i in range(self.num_vlm_layers):
            if multiple_of > 0 and i > 0 and i % multiple_of != 0:
                expert_layer = None
            else:
                expert_layer_index = i // multiple_of if multiple_of > 0 else i
                expert_layer = models[1].layers[expert_layer_index]
            vlm_layers.append(models[0].layers[i])
            expert_layers.append(expert_layer)
        return [vlm_layers, expert_layers]

    def forward(
        self,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: list[torch.FloatTensor] | None = None,
        inputs_embeds: list[torch.FloatTensor] = None,
        use_cache: bool | None = None,
        fill_kv_cache: bool | None = None,
        expert_model: nn.Module | None = None,
    ):
        models = [
            self.get_vlm_model().text_model,
            self.lm_expert if expert_model is None else expert_model,
        ]
        model_layers = self.get_model_layers(models)
        for hidden_states in inputs_embeds:
            # TODO this is very inefficient
            # dtype is always the same, batch size too (if > 1 len)
            # device could be trickier in multi gpu edge cases but that's it
            if hidden_states is None:
                continue
            batch_size = hidden_states.shape[0]

        # RMSNorm
        num_layers = self.num_vlm_layers
        head_dim = self.vlm.config.text_config.head_dim
        for layer_idx in range(num_layers):
            if (
                fill_kv_cache
                or "cross" not in self.attention_mode
                or (self.self_attn_every_n_layers > 0 and layer_idx % self.self_attn_every_n_layers == 0)
            ):
                att_outputs, past_key_values = self.forward_attn_layer(
                    model_layers,
                    inputs_embeds,
                    layer_idx,
                    position_ids,
                    attention_mask,
                    batch_size,
                    head_dim,
                    use_cache=use_cache,
                    fill_kv_cache=fill_kv_cache,
                    past_key_values=past_key_values,
                )
            else:
                att_outputs, past_key_values = self.forward_cross_attn_layer(
                    model_layers,
                    inputs_embeds,
                    layer_idx,
                    position_ids,
                    attention_mask,
                    batch_size,
                    head_dim,
                    use_cache=use_cache,
                    fill_kv_cache=fill_kv_cache,
                    past_key_values=past_key_values,
                )
            outputs_embeds = []
            start = 0
            for i, hidden_states in enumerate(inputs_embeds):
                layer = model_layers[i][layer_idx]
                att_output = (
                    att_outputs[i] if i < len(att_outputs) else att_outputs[0]
                )  # in case of self_attn
                if hidden_states is not None:
                    if layer is None:
                        outputs_embeds.append(hidden_states)
                        continue
                    end = start + hidden_states.shape[1]

                    if att_output.dtype != layer.self_attn.o_proj.weight.dtype:
                        att_output = att_output.to(layer.self_attn.o_proj.weight.dtype)
                    att_out = att_output[:, start:end]
                    out_emb = layer.self_attn.o_proj(att_out)

                    out_emb += hidden_states
                    after_first_residual = out_emb.clone()

                    out_emb = layer.post_attention_layernorm(out_emb)
                    out_emb = layer.mlp(out_emb)

                    out_emb += after_first_residual

                    outputs_embeds.append(out_emb)

                    start = end if len(att_outputs) == 1 else 0
                else:
                    outputs_embeds.append(None)

            inputs_embeds = outputs_embeds

        # final norm
        outputs_embeds = []
        for i, hidden_states in enumerate(inputs_embeds):
            if hidden_states is not None:
                out_emb = models[i].norm(hidden_states)
                outputs_embeds.append(out_emb)
            else:
                outputs_embeds.append(None)
        return outputs_embeds, past_key_values

    def get_attention_interface(self):
        attention_interface = self.eager_attention_forward
        return attention_interface

    def eager_attention_forward(
        self, attention_mask, batch_size, head_dim, query_states, key_states, value_states
    ):
        num_att_heads = self.num_attention_heads
        num_key_value_heads = self.num_key_value_heads
        num_key_value_groups = num_att_heads // num_key_value_heads

        sequence_length = key_states.shape[1]

        key_states = key_states[:, :, :, None, :].expand(
            batch_size, sequence_length, num_key_value_heads, num_key_value_groups, head_dim
        )
        key_states = key_states.reshape(
            batch_size, sequence_length, num_key_value_heads * num_key_value_groups, head_dim
        )

        value_states = value_states[:, :, :, None, :].expand(
            batch_size, sequence_length, num_key_value_heads, num_key_value_groups, head_dim
        )
        value_states = value_states.reshape(
            batch_size, sequence_length, num_key_value_heads * num_key_value_groups, head_dim
        )

        # Attention here is upcasted to float32 to match the original eager implementation.
        query_states = query_states.to(dtype=torch.float32)
        key_states = key_states.to(dtype=torch.float32)

        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)

        att_weights = torch.matmul(query_states, key_states.transpose(2, 3))
        att_weights *= head_dim**-0.5

        att_weights = att_weights.to(dtype=torch.float32)
        big_neg = torch.finfo(att_weights.dtype).min  # -2.3819763e38  # See gemma/modules.py
        masked_att_weights = torch.where(attention_mask[:, None, :, :], att_weights, big_neg)
        probs = nn.functional.softmax(masked_att_weights, dim=-1)
        probs = probs.to(dtype=value_states.dtype)

        att_output = torch.matmul(probs, value_states.permute(0, 2, 1, 3))

        att_output = att_output.permute(0, 2, 1, 3)
        # we use -1 because sequence length can change
        att_output = att_output.reshape(batch_size, -1, num_key_value_heads * num_key_value_groups * head_dim)

        return att_output
