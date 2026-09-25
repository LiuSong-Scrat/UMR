#!/usr/bin/env bash
set -euo pipefail

# Reproduce section 1 of WEPVLA_V043_DoubleFLow.md.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${HOME}/miniconda3/envs/reap/bin/python}"
CONFIG_PATH="${CONFIG_PATH:-${REPO_ROOT}/benchmarks/song_real_libero/configs/libero.json}"
DEMO_ROOT="${DEMO_ROOT:-${REPO_ROOT}/benchmarks/song_real_libero/data/libero_setting/libero_100_demos}"
DATASET_ROOT="${DATASET_ROOT:-${REPO_ROOT}/benchmarks/song_real_libero/data/libero_setting/v043_doubleflow_task6_task8_100ep}"
VIS_ROOT="${VIS_ROOT:-${DATASET_ROOT}/visualizations}"
CUDA_DEVICE="${CUDA_DEVICE:-5}"
REPO_ID="${REPO_ID:-song_libero10_task6_task8_world_eef_doubleflow}"

[[ -x "${PYTHON_BIN}" ]] || { echo "Python not executable: ${PYTHON_BIN}" >&2; exit 1; }
[[ -f "${CONFIG_PATH}" ]] || { echo "Config not found: ${CONFIG_PATH}" >&2; exit 1; }
[[ -d "${DEMO_ROOT}" ]] || { echo "Demo root not found: ${DEMO_ROOT}" >&2; exit 1; }
if [[ -e "${DATASET_ROOT}" && "${OVERWRITE:-0}" != "1" ]]; then
  echo "Dataset already exists: ${DATASET_ROOT}" >&2
  echo "Set OVERWRITE=1 to pass --overwrite explicitly." >&2
  exit 1
fi

cd "${REPO_ROOT}"
export PYTHONHASHSEED=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}"
export MUJOCO_EGL_DEVICE_ID="${CUDA_DEVICE}" PYTHONPATH="${REPO_ROOT}/src"

args=(
  --config "${CONFIG_PATH}"
  --demo-root "${DEMO_ROOT}"
  --suite libero_10 --task-id 6 --task-id 8 --episodes 50
  --num-workers 10 --worker-scope episode --num-points 10000
  --point-cloud-storage zarr --fps 20 --replay-mode states
  --state-observation-offset 0 --restore-demo-model
  --require-source-fps-match --save-rgb-images --image-camera agentview
  --download-demos --download-use-huggingface --no-save-video --vis-count 2
  --resume-temp-artifacts --vis-dir "${VIS_ROOT}"
  --output-root "${DATASET_ROOT}" --repo-id "${REPO_ID}" --no-overwrite
)
if [[ "${OVERWRITE:-0}" == "1" ]]; then
  args[-1]=--overwrite
fi

exec "${PYTHON_BIN}" "${SCRIPT_DIR}/scripts/libero_setting/libero_hdf5_to_dataset.py" "${args[@]}"
