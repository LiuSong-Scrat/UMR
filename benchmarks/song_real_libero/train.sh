#!/usr/bin/env bash
set -euo pipefail

# Reproduce section 3 of WEPVLA_V043_DoubleFLow.md.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${HOME}/miniconda3/envs/reap/bin/python}"
DATASET_ROOT="${DATASET_ROOT:-${REPO_ROOT}/benchmarks/song_real_libero/data/libero_setting/v043_doubleflow_task6_task8_100ep}"
CACHE_ROOT="${CACHE_ROOT:-${REPO_ROOT}/benchmarks/song_real_libero/data/libero_setting/v043_doubleflow_task6_task8_100ep_pointseg_cache}"
BASE_POLICY="${BASE_POLICY:-}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/benchmarks/song_real_libero/outputs/libero_setting/v043_doubleflow_task6_task8_3gpu_b32_1300steps}"
JOB_NAME="${JOB_NAME:-v043_doubleflow_task6_task8_3gpu_b32_1300steps}"
VLM_MODEL="${VLM_MODEL:-${REPO_ROOT}/benchmarks/vlm_model/SmolVLM2-500M-Video-Instruct}"
VLM_WEIGHTS="${VLM_WEIGHTS:-${REPO_ROOT}/benchmarks/vlm_model/smolvla_base}"
GPU_IDS="${GPU_IDS:-3,4,5}"

[[ -x "${PYTHON_BIN}" ]] || { echo "Python not executable: ${PYTHON_BIN}" >&2; exit 1; }
[[ -n "${BASE_POLICY}" ]] || { echo "Set BASE_POLICY to a pretrained policy directory." >&2; exit 1; }
[[ -f "${BASE_POLICY}/config.json" ]] || { echo "Base policy not found: ${BASE_POLICY}" >&2; exit 1; }
[[ -f "${CACHE_ROOT}/manifest.json" ]] || { echo "Cache manifest not found: ${CACHE_ROOT}/manifest.json" >&2; exit 1; }
[[ -d "${DATASET_ROOT}" ]] || { echo "Dataset root not found: ${DATASET_ROOT}" >&2; exit 1; }
[[ -d "${VLM_MODEL}" ]] || { echo "VLM model not found: ${VLM_MODEL}" >&2; exit 1; }
[[ -d "${VLM_WEIGHTS}" ]] || { echo "VLM weights not found: ${VLM_WEIGHTS}" >&2; exit 1; }
if [[ -e "${OUTPUT_ROOT}" ]]; then
  echo "Refusing to overwrite training output: ${OUTPUT_ROOT}" >&2
  echo "Set OUTPUT_ROOT to a new directory." >&2
  exit 1
fi

cd "${REPO_ROOT}"
export PYTHONHASHSEED=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export MALLOC_ARENA_MAX=2 CUDA_VISIBLE_DEVICES="${GPU_IDS}"
export PYTHONPATH="${REPO_ROOT}/src" SONG_POINTSEG_REQUIRE_POINTOPS=1

exec "${PYTHON_BIN}" -m accelerate.commands.launch --multi_gpu --num_processes=3 \
  --num_machines=1 --mixed_precision=no --dynamo_backend=no --main_process_port=0 \
  "${SCRIPT_DIR}/scripts/train_song_benchmark.py" \
  --policy.path="${BASE_POLICY}" --policy.push_to_hub=false \
  --dataset.repo_id="${DATASET_ROOT}" --pointseg_sample_cache_dir="${CACHE_ROOT}" \
  --task_balanced_sampling=true --batch_size=32 --gradient_accumulation_steps=1 \
  --steps=1300 --save_freq=1300 --save_steps='[100,260,520,780,1040,1300]' \
  --log_freq=1 --eval_freq=1300 --num_workers=8 --output_dir="${OUTPUT_ROOT}" \
  --job_name="${JOB_NAME}" --policy.device=cuda --wandb.enable=true \
  --wandb.disable_artifact=true --policy.optimizer_lr=0.0001 \
  --policy.scheduler_warmup_steps=50 --policy.scheduler_decay_steps=1300 \
  --policy.scheduler_decay_lr=0.00001 --policy.camera_views=agentview \
  --policy.rgb_camera_views=agentview --policy.vla_adapter_enable=true \
  --policy.vla_adapter_freeze_vlm=true --policy.vlm_model_name="${VLM_MODEL}" \
  --policy.vlm_weights_path="${VLM_WEIGHTS}" --policy.load_vlm_weights=true \
  --policy.pointseg_enable=true --policy.pointseg_backbone_type=litept \
  --policy.pointseg_grid_size=0.01 --policy.pointseg_feature_dim=64 \
  --policy.pointseg_aux_loss_weight=0.0005 --policy.pointseg_foreground_ratio=0.025 \
  --policy.pointseg_background_ratio=0.025 --policy.pointseg_min_foreground_points=2500 \
  --policy.pointseg_min_background_points=0 --policy.pointseg_use_temporal_priors_as_input=false \
  --policy.pointseg_use_pseudo_selection=false --policy.point_action_fusion_enable=true \
  --policy.worldflow_enable=true --policy.worldflow_target_type=world_eef_trajectory \
  --policy.worldflow_world_eef_velocity_mode=base_pose9_euclidean \
  --policy.worldflow_reference_frame=robot_base --policy.worldflow_frame_origin=global \
  --policy.worldflow_scene_frame_origin=global --policy.worldflow_noise_coupling=left_compose_ego \
  --policy.worldflow_action_fusion=point_action_expert_conjugate_bridge \
  --policy.worldflow_action_expert_mode=shared --policy.worldflow_current_ee_pose_token=false \
  --policy.worldflow_bootstrap_from_ego=true --policy.worldflow_freeze_pretrained_ego=false \
  --policy.worldflow_feature_dim=64 --policy.worldflow_grid_size=0.01 \
  --policy.worldflow_max_points=2048 --policy.worldflow_loss_weight=1.0 \
  --policy.worldflow_geo_loss_weight=0.0 --policy.worldflow_bridge_loss_weight=0.0 \
  --policy.worldflow_equiv_loss_weight=0.0 --policy.worldflow_training_coordinate_frame_augmentation=false \
  --policy.worldflow_pretrained_lr_multiplier=1.0 --policy.worldflow_new_lr_multiplier=1.0 \
  --policy.worldflow_trans_weight=1.0 --policy.worldflow_rot_weight=1.0 \
  --policy.worldflow_eef_probe_radius_m=0.10 --policy.worldflow_require_action_target_sidecar=true \
  --policy.worldflow_se3_head_enable=false --policy.se3_enable=false \
  --policy.se3_final_correction_enable=false
