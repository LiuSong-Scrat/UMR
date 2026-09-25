#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RLBENCH_ROOT="${RLBENCH_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
TOOLS_DIR="${SCRIPT_DIR}/tools"
REPO_ROOT="${LEROBOT_ROOT:-$(cd "${RLBENCH_ROOT}/../.." && pwd)}"
PYTHON="${PYTHON:-python}"
SONG_SCRIPT_ROOT="${SONG_SCRIPTS:-${REPO_ROOT}/benchmarks/song_real_libero/scripts}"
TRAIN_SCRIPT="${SONG_TRAIN_SCRIPT:-${SONG_SCRIPT_ROOT}/train_song_benchmark.py}"
ACCELERATE_BIN="${ACCELERATE_BIN:-}"
PRETRAINED_POLICY="${PRETRAINED_POLICY-${RLBENCH_ROOT}/outputs/wep_vla_v041_rlbench_10tasks_0808/checkpoints/022000/pretrained_model}"
TRAIN_CONFIG="${TRAIN_CONFIG-${PRETRAINED_POLICY}/train_config.json}"
RESUME="${RESUME:-true}"
OUTPUT_DIR="${OUTPUT_DIR:-${RLBENCH_ROOT}/outputs/wep_vla_v041_rlbench_10tasks_0808/checkpoints}"
VLM_MODEL_ROOT="${VLM_MODEL_ROOT:-${RLBENCH_ROOT}/../vlm_model}"
VLM_MODEL_NAME="${VLM_MODEL_NAME:-${HF_VLM_MODEL_NAME:-${VLM_MODEL_ROOT}/SmolVLM2-500M-Video-Instruct}}"
VLM_WEIGHTS_PATH="${VLM_WEIGHTS_PATH:-${HF_VLM_WEIGHTS_PATH:-${VLM_MODEL_ROOT}/smolvla_base}}"
POLICY_DEVICE="${POLICY_DEVICE:-cuda}"
BATCH_SIZE="${BATCH_SIZE:-200}"
TRAIN_STEPS="${TRAIN_STEPS:-35000}"
NUM_WORKERS="${NUM_WORKERS:-8}"
SAVE_FREQ="${SAVE_FREQ:-1000}"
EVAL_FREQ="${EVAL_FREQ:-2000}"
WANDB_ENABLE="${WANDB_ENABLE:-false}"
WANDB_DISABLE_ARTIFACT="${WANDB_DISABLE_ARTIFACT:-true}"
POLICY_USE_AMP="${POLICY_USE_AMP:-false}"
WORLDFLOW_ENABLE="${WORLDFLOW_ENABLE:-false}"
CAMERA_VIEWS="${CAMERA_VIEWS:-front}"
RGB_CAMERA_VIEWS="${RGB_CAMERA_VIEWS:-front}"
DATASET_ROOT="${DATASET_ROOT:-${RLBENCH_ROOT}/datasets/rlbench_box_tasks_100traj_lerobot_raw_expert_target_20260810_173629}"
POINTSEG_CACHE_DIR="${POINTSEG_CACHE_DIR:-${DATASET_ROOT}_pointseg_cache_new}"
export LEROBOT_ROOT="${REPO_ROOT}"
export LEROBOT_SRC="${LEROBOT_SRC:-${REPO_ROOT}/src}"
export SONG_SCRIPTS="${SONG_SCRIPT_ROOT}"
export PYTHONPATH="${LEROBOT_SRC}${PYTHONPATH:+:${PYTHONPATH}}"

if [[ -n "${ACCELERATE_BIN}" ]]; then
    ACCELERATE_COMMAND=("${ACCELERATE_BIN}" launch)
else
    ACCELERATE_COMMAND=("${PYTHON}" -m accelerate.commands.launch)
fi
CONFIG_ARGS=()
if [[ -n "${TRAIN_CONFIG}" ]]; then
    CONFIG_ARGS+=(--config_path="${TRAIN_CONFIG}")
fi
POLICY_ARGS=()
if [[ -n "${PRETRAINED_POLICY}" ]]; then
    POLICY_ARGS+=(--policy.path="${PRETRAINED_POLICY}")
fi

"${PYTHON}" "${TOOLS_DIR}/validate_reap_dataset.py" \
    "${DATASET_ROOT}" --cache-dir "${POINTSEG_CACHE_DIR}"

[[ -n "${VLM_MODEL_NAME}" && -n "${VLM_WEIGHTS_PATH}" ]] || {
    echo "Set VLM_MODEL_NAME and VLM_WEIGHTS_PATH (or HF_VLM_MODEL_NAME/HF_VLM_WEIGHTS_PATH)." >&2
    exit 2
}

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" "${ACCELERATE_COMMAND[@]}" --num_processes=1 "${TRAIN_SCRIPT}" \
    "${POLICY_ARGS[@]}" \
    --policy.type=smolvla \
    --policy.push_to_hub=false \
    --resume="${RESUME}" \
    "${CONFIG_ARGS[@]}" \
    --dataset.repo_id="${DATASET_ROOT}" \
    --pointseg_sample_cache_dir="${POINTSEG_CACHE_DIR}" \
    --output_dir="${OUTPUT_DIR}" \
    --policy.vla_adapter_enable=true \
    --policy.vla_adapter_freeze_vlm=true \
    --policy.vlm_model_name="${VLM_MODEL_NAME}" \
    --policy.vlm_weights_path="${VLM_WEIGHTS_PATH}" \
    --policy.load_vlm_weights=true \
    --batch_size="${BATCH_SIZE}" \
    --steps="${TRAIN_STEPS}" \
    --log_freq=1 \
    --job_name=wep_vla_v041_rlbench \
    --policy.device="${POLICY_DEVICE}" \
    --wandb.enable="${WANDB_ENABLE}" \
    --wandb.disable_artifact="${WANDB_DISABLE_ARTIFACT}" \
    --save_freq="${SAVE_FREQ}" \
    --eval_freq="${EVAL_FREQ}" \
    --num_workers="${NUM_WORKERS}" \
    --policy.use_amp="${POLICY_USE_AMP}" \
    --policy.camera_views="${CAMERA_VIEWS}" \
    --policy.rgb_camera_views="${RGB_CAMERA_VIEWS}" \
    --policy.pointseg_enable=true \
    --policy.pointseg_backbone_type=litept \
    --policy.pointseg_grid_size=0.01 \
    --policy.pointseg_feature_dim=64 \
    --policy.pointseg_aux_loss_weight=0.0005 \
    --policy.pointseg_foreground_ratio=0.025 \
    --policy.pointseg_background_ratio=0.025 \
    --policy.pointseg_min_foreground_points=2500 \
    --policy.pointseg_min_background_points=0 \
    --policy.pointseg_use_temporal_priors_as_input=false \
    --policy.pointseg_use_pseudo_selection=false \
    --policy.worldflow_enable="${WORLDFLOW_ENABLE}" \
    --policy.worldflow_se3_head_enable=false \
    --policy.se3_enable=false \
    --policy.se3_final_correction_enable=false
