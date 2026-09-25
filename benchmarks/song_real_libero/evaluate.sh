#!/usr/bin/env bash
set -euo pipefail

# Reproduce section 4 of WEPVLA_V043_DoubleFLow.md.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${HOME}/miniconda3/envs/reap/bin/python}"
CONFIG_PATH="${CONFIG_PATH:-${REPO_ROOT}/benchmarks/song_real_libero/configs/libero.json}"
POLICY_PATH="${POLICY_PATH:-${REPO_ROOT}/benchmarks/song_real_libero/test_checkpoints/checkpoints/pretrained_model}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/benchmarks/song_real_libero/outputs/eval_$(date +%Y%m%d_%H%M%S)}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
MUJOCO_EGL_DEVICE_ID="${MUJOCO_EGL_DEVICE_ID:-0}"
EPISODES="${EPISODES:-50}"
TASK_WORKERS="${TASK_WORKERS:-2}"
EPISODE_WORKERS_PER_TASK="${EPISODE_WORKERS_PER_TASK:-25}"
TASK_WORKER_BACKEND="${TASK_WORKER_BACKEND:-process}"
INFERENCE_BATCH_SIZE="${INFERENCE_BATCH_SIZE:-50}"
INFERENCE_BATCHING_MODE="${INFERENCE_BATCHING_MODE:-fixed_barrier}"
SUITES="${SUITES:-libero_spatial libero_object libero_goal libero_10}"
ALL_TASKS="${ALL_TASKS:-1}"
VLM_MODEL_NAME="${VLM_MODEL_NAME:-${REPO_ROOT}/benchmarks/vlm_model/SmolVLM2-500M-Video-Instruct}"
VLM_WEIGHTS_PATH="${VLM_WEIGHTS_PATH:-${REPO_ROOT}/benchmarks/vlm_model/smolvla_base}"

SUITE_ARGS=()
read -r -a SUITE_LIST <<< "${SUITES//,/ }"
for suite in "${SUITE_LIST[@]}"; do
  SUITE_ARGS+=(--suite "${suite}")
done

if [[ "${ALL_TASKS}" == "1" ]]; then
  TASK_ARGS=(--all-tasks)
elif [[ -n "${TASK_IDS:-}" ]]; then
  TASK_ARGS=()
  read -r -a TASK_ID_LIST <<< "${TASK_IDS//,/ }"
  for task_id in "${TASK_ID_LIST[@]}"; do
    TASK_ARGS+=(--task-id "${task_id}")
  done
else
  TASK_ARGS=(--task-id 6 --task-id 8)
fi

[[ -x "${PYTHON_BIN}" ]] || { echo "Python not executable: ${PYTHON_BIN}" >&2; exit 1; }
[[ -f "${CONFIG_PATH}" ]] || { echo "Config not found: ${CONFIG_PATH}" >&2; exit 1; }
[[ -f "${POLICY_PATH}/config.json" ]] || { echo "Policy not found: ${POLICY_PATH}" >&2; exit 1; }
if [[ -e "${OUTPUT_DIR}" ]]; then
  echo "Refusing to overwrite evaluation output: ${OUTPUT_DIR}" >&2
  echo "Set OUTPUT_DIR to a new directory." >&2
  exit 1
fi

cd "${REPO_ROOT}"
export PYTHONHASHSEED=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export MALLOC_ARENA_MAX=2 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" MUJOCO_EGL_DEVICE_ID
export PYTHONPATH="${REPO_ROOT}/src"
export VLM_MODEL_NAME VLM_WEIGHTS_PATH

exec "${PYTHON_BIN}" "${SCRIPT_DIR}/scripts/libero_setting/libero_pointcloud_eval.py" \
  --config "${CONFIG_PATH}" --policy.path "${POLICY_PATH}" --device cuda \
  "${SUITE_ARGS[@]}" "${TASK_ARGS[@]}" --episodes "${EPISODES}" \
  --isolated-policy-workers 1 --task-workers "${TASK_WORKERS}" --episode-workers-per-task "${EPISODE_WORKERS_PER_TASK}" \
  --task-worker-backend "${TASK_WORKER_BACKEND}" \
  --inference-batch-size "${INFERENCE_BATCH_SIZE}" --inference-batching-mode "${INFERENCE_BATCHING_MODE}" \
  --policy-noise-seed 0 --env-seed 7 \
  --strict-official-init \
  --action-index 0 --exec-action-steps 24 \
  --adaptive-exec-max-steps 24 --grasp-exec-steps 24 \
  --waypoint-max-hold-steps 1 \
  --gripper-control-mode delta_width_initial_sync --gripper-delta-threshold 0.002 \
  --gripper-delta-alignment current_minus_previous \
  --no-release-event-exec-enable \
  --control-freq 20 --max-steps 1000 --no-use-suite-max-steps \
  --recreate-env-per-episode --render-mode offscreen \
  --no-visualize-foreground --no-save-video \
  --no-world-to-ego-causal-ablation \
  --output-dir "${OUTPUT_DIR}"
