#!/usr/bin/env bash
set -euo pipefail

# Reproduce section 2 of WEPVLA_V043_DoubleFLow.md.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${HOME}/miniconda3/envs/reap/bin/python}"
DATASET_ROOT="${DATASET_ROOT:-${REPO_ROOT}/benchmarks/song_real_libero/data/libero_setting/v043_doubleflow_task6_task8_100ep}"
CACHE_ROOT="${CACHE_ROOT:-${REPO_ROOT}/benchmarks/song_real_libero/data/libero_setting/v043_doubleflow_task6_task8_100ep_pointseg_cache}"
GPU_IDS="${GPU_IDS:-3,4,5}"
NPROC="${NPROC:-3}"

[[ -x "${PYTHON_BIN}" ]] || { echo "Python not executable: ${PYTHON_BIN}" >&2; exit 1; }
[[ -d "${DATASET_ROOT}" ]] || { echo "Dataset root not found: ${DATASET_ROOT}" >&2; exit 1; }
if [[ -e "${CACHE_ROOT}" && "${OVERWRITE:-0}" != "1" ]]; then
  echo "Cache already exists: ${CACHE_ROOT}" >&2
  echo "Set OVERWRITE=1 to pass --overwrite explicitly." >&2
  exit 1
fi

cd "${REPO_ROOT}"
export PYTHONHASHSEED=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 MALLOC_ARENA_MAX=2
export CUDA_VISIBLE_DEVICES="${GPU_IDS}" PYTHONPATH="${REPO_ROOT}/src"
export SONG_POINTSEG_REQUIRE_POINTOPS=1 SONG_POINTCLOUD_GRIPPER_POINTS=500

args=(
  --dataset.repo_id="${DATASET_ROOT}" --camera-views=agentview
  --camera-view-fusion=legacy_budget --output-dir="${CACHE_ROOT}"
  --current-points=10000 --future-points=10000 --batch-size=24
  --num-workers=8 --shard-size=2048 --storage-dtype=float16
  --nn-chunk-size=1024 --vis-count=4
)
if [[ "${OVERWRITE:-0}" == "1" ]]; then args+=(--overwrite); fi

exec "${PYTHON_BIN}" -m torch.distributed.run --standalone \
  --nproc_per_node="${NPROC}" \
  "${SCRIPT_DIR}/scripts/song_cache_pointseg_samples.py" "${args[@]}"
