#!/usr/bin/env bash
set -euo pipefail

# Edit this list to choose which RLBench tasks are collected by default.
# Task names must match RLBench task module names.
TASK_LIST=(
  close_box
  close_fridge
  close_laptop_lid
  phone_on_base
  stack_wine
  sweep_to_dustpan
  take_frame_off_hanger
  take_umbrella_out_of_umbrella_stand
  toilet_seat_down
  water_plants
)

# Parallel collection uses one independent CoppeliaSim process per worker.
# Start one reachable X display per worker and set COLLECTION_WORKERS>1.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RLBENCH_ROOT="${RLBENCH_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
TOOLS_DIR="${SCRIPT_DIR}/tools"
REPO_ROOT="${LEROBOT_ROOT:-$(cd "${RLBENCH_ROOT}/../.." && pwd)}"

# RLBench camera rendering uses the Xorg server prepared for CoppeliaSim.
export DISPLAY="${DISPLAY:-}"
export QT_QPA_PLATFORM="${QT_QPA_PLATFORM:-xcb}"
export COPPELIASIM_ROOT="${COPPELIASIM_ROOT:-${RLBENCH_ROOT}/../CoppeliaSim}"
export LD_LIBRARY_PATH="${COPPELIASIM_ROOT}:${LD_LIBRARY_PATH:-}"
export QT_QPA_PLATFORM_PLUGIN_PATH="${COPPELIASIM_ROOT}"
PYTHON="${PYTHON:-python}"
if [[ "${PYTHON}" != */* ]]; then
  PYTHON="$(command -v "${PYTHON}" || true)"
fi
if [[ -z "${PYTHON}" || ! -x "${PYTHON}" ]]; then
  echo "Python executable not found; set PYTHON to an executable path." >&2
  exit 2
fi
SONG_SCRIPT_ROOT="${SONG_SCRIPTS:-${REPO_ROOT}/benchmarks/song_real_libero/scripts}"
export LEROBOT_ROOT="${REPO_ROOT}"
export LEROBOT_SRC="${LEROBOT_SRC:-${REPO_ROOT}/src}"
export SONG_SCRIPTS="${SONG_SCRIPT_ROOT}"

# Foreground parameter records. PARAM_SET=27e is the tuned 27e_contact_sharp
# set; raw matches the original PointSeg/cache parser defaults and is the default.
PARAM_SET="${PARAM_SET:-raw}"


ACTION_LABEL_MODE="${ACTION_LABEL_MODE:-expert_target}"
# Raw point count is used while collecting each frame. Cache point counts are
# independent and can be smaller than the raw cloud.
RAW_POINT_COUNT="${RAW_POINT_COUNT:-20000}"
CACHE_CURRENT_POINTS="${CACHE_CURRENT_POINTS:-20000}"
CACHE_FUTURE_POINTS="${CACHE_FUTURE_POINTS:-16384}"
# Virtual gripper is merged into every raw point cloud before PointSeg cache
# sampling and pseudo-label generation. reap is canonical RLBench-aligned v4.
GRIPPER_TEMPLATE="${GRIPPER_TEMPLATE:-reap}" 
POST_SUCCESS_FRAMES="${POST_SUCCESS_FRAMES:-10}"
EXPERT_PATH_MODE="${EXPERT_PATH_MODE:-linear_then_rrt}"
COLLECT_WATER_PLANT_COLLISION="${RLBENCH_WATER_PLANT_COLLISION:-enabled}"
COLLECT_WATER_DROP_COLLISION="${RLBENCH_WATER_DROP_COLLISION:-original}"
GENERATE_WORLD_BASE_WORLDFLOW_SIDECARS="${GENERATE_WORLD_BASE_WORLDFLOW_SIDECARS:-true}"
export RLBENCH_WATER_PLANT_COLLISION="${COLLECT_WATER_PLANT_COLLISION}"
export RLBENCH_WATER_DROP_COLLISION="${COLLECT_WATER_DROP_COLLISION}"
FORWARD_ARGS=()

usage() {
  cat <<'EOF'
Usage:
  bash collect_data.sh [options] [collector options]

Options:
  --param-set NAME       Foreground parameter set: 27e (default) or raw.
  --action-label-mode M  Action labels: expert_target (default) or executed.
  --raw-point-count N    Points sampled into each raw point cloud (default: 20000).
  --cache-current-points N  Points retained for current-frame cache samples.
  --cache-future-points N   Points retained for future-frame cache samples.
  --post-success-frames N   Keep N frames after first success; -1 disables (default: 10).
  --dataset-root PATH    Alias for the output dataset directory.
  --[no-]world-base-worldflow-sidecars
                         Enable/disable Panda-link0-base achieved/command-target
                         WorldFlow sidecars, matching LIBERO (default: enabled).
  --help                 Show this help.

The default task list is maintained near the top of this script.
The dataset path can also be changed with OUTPUT_ROOT or DATASET_ROOT.
Individual foreground parameters can be overridden with their environment
variables, for example CONTACT_RADIUS=0.18.

Examples:
  ACTION_LABEL_MODE=executed bash collect_data.sh
  bash collect_data.sh --action-label-mode executed
EOF
}
    
while (($# > 0)); do
  case "$1" in
    --param-set)
      [[ $# -ge 2 ]] || { echo "--param-set requires 27e or raw" >&2; exit 2; }
      PARAM_SET="$2"
      shift 2
      ;;
    --action-label-mode)
      [[ $# -ge 2 ]] || { echo "--action-label-mode requires expert_target or executed" >&2; exit 2; }
      ACTION_LABEL_MODE="$2"
      shift 2
      ;;
    --raw-point-count)
      [[ $# -ge 2 ]] || { echo "--raw-point-count requires a positive integer" >&2; exit 2; }
      RAW_POINT_COUNT="$2"
      shift 2
      ;;
    --cache-current-points)
      [[ $# -ge 2 ]] || { echo "--cache-current-points requires a positive integer" >&2; exit 2; }
      CACHE_CURRENT_POINTS="$2"
      shift 2
      ;;
    --cache-future-points)
      [[ $# -ge 2 ]] || { echo "--cache-future-points requires a positive integer" >&2; exit 2; }
      CACHE_FUTURE_POINTS="$2"
      shift 2
      ;;
    --post-success-frames)
      [[ $# -ge 2 ]] || { echo "--post-success-frames requires -1 or a non-negative integer" >&2; exit 2; }
      POST_SUCCESS_FRAMES="$2"
      shift 2
      ;;
    --dataset-root)
      [[ $# -ge 2 ]] || { echo "--dataset-root requires a path" >&2; exit 2; }
      OUTPUT_ROOT="$2"
      shift 2
      ;;
    --world-base-worldflow-sidecars)
      GENERATE_WORLD_BASE_WORLDFLOW_SIDECARS=true
      shift
      ;;
    --no-world-base-worldflow-sidecars)
      GENERATE_WORLD_BASE_WORLDFLOW_SIDECARS=false
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    --)
      shift
      FORWARD_ARGS+=("$@")
      break
      ;;
    *)
      FORWARD_ARGS+=("$1")
      shift
      ;;
  esac
done

case "${GENERATE_WORLD_BASE_WORLDFLOW_SIDECARS}" in
  true|false) ;;
  *)
    echo "GENERATE_WORLD_BASE_WORLDFLOW_SIDECARS must be true or false" >&2
    exit 2
    ;;
esac

case "${PARAM_SET}" in
  27e)
    MOTION_ROTATION_RADIUS="${MOTION_ROTATION_RADIUS:-0.08}"
    MOTION_BASELINE_THRESHOLD="${MOTION_BASELINE_THRESHOLD:-0.015}"
    MOTION_BASELINE_TEMPERATURE="${MOTION_BASELINE_TEMPERATURE:-0.005}"
    MOTION_RELATIVE_MARGIN="${MOTION_RELATIVE_MARGIN:-0.10}"
    MOTION_RELATIVE_TAU="${MOTION_RELATIVE_TAU:-0.10}"
    TRAJECTORY_SIGMA="${TRAJECTORY_SIGMA:-0.13}"
    CONTACT_RADIUS="${CONTACT_RADIUS:-0.22}"
    CONTACT_TEMPERATURE="${CONTACT_TEMPERATURE:-0.02}"
    APPROACH_MARGIN="${APPROACH_MARGIN:-0.005}"
    APPROACH_TAU="${APPROACH_TAU:-0.025}"
    BACKGROUND_TRAJECTORY_SIGMA="${BACKGROUND_TRAJECTORY_SIGMA:-0.20}"
    ;;
  raw)
    MOTION_ROTATION_RADIUS="${MOTION_ROTATION_RADIUS:-0.18}"
    MOTION_BASELINE_THRESHOLD="${MOTION_BASELINE_THRESHOLD:-0.010}"
    MOTION_BASELINE_TEMPERATURE="${MOTION_BASELINE_TEMPERATURE:-0.006}"
    MOTION_RELATIVE_MARGIN="${MOTION_RELATIVE_MARGIN:-0.05}"
    MOTION_RELATIVE_TAU="${MOTION_RELATIVE_TAU:-0.08}"
    TRAJECTORY_SIGMA="${TRAJECTORY_SIGMA:-0.22}"
    CONTACT_RADIUS="${CONTACT_RADIUS:-0.22}"
    CONTACT_TEMPERATURE="${CONTACT_TEMPERATURE:-0.055}"
    APPROACH_MARGIN="${APPROACH_MARGIN:-0.0}"
    APPROACH_TAU="${APPROACH_TAU:-0.04}"
    BACKGROUND_TRAJECTORY_SIGMA="${BACKGROUND_TRAJECTORY_SIGMA:-0.32}"
    ;;
  *)
    echo "PARAM_SET must be 27e or raw, got: ${PARAM_SET}" >&2
    exit 2
    ;;
esac

case "${ACTION_LABEL_MODE}" in
  expert_target|executed)
    ;;
  *)
    echo "ACTION_LABEL_MODE must be expert_target or executed, got: ${ACTION_LABEL_MODE}" >&2
    exit 2
    ;;
esac

for value_name in RAW_POINT_COUNT CACHE_CURRENT_POINTS CACHE_FUTURE_POINTS; do
  value="${!value_name}"
  if ! [[ "${value}" =~ ^[1-9][0-9]*$ ]]; then
    echo "${value_name} must be a positive integer, got: ${value}" >&2
    exit 2
  fi
done
if ! [[ "${POST_SUCCESS_FRAMES}" =~ ^(-1|[0-9]+)$ ]]; then
  echo "POST_SUCCESS_FRAMES must be -1 or a non-negative integer, got: ${POST_SUCCESS_FRAMES}" >&2
  exit 2
fi

OUTPUT_ROOT="${OUTPUT_ROOT:-${DATASET_ROOT:-}}"
if [[ -z "${OUTPUT_ROOT}" ]]; then
  OUTPUT_ROOT="${RLBENCH_ROOT}/datasets/rlbench_10tasks_100traj_lerobot_${PARAM_SET}_${ACTION_LABEL_MODE}_$(date +%Y%m%d_%H%M%S)"
fi

COLLECTION_WORKERS="${COLLECTION_WORKERS:-1}"
if ! [[ "${COLLECTION_WORKERS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "COLLECTION_WORKERS must be a positive integer" >&2
  exit 2
fi

if [[ -n "${COLLECTION_DISPLAY_BASE:-}" ]]; then
  DISPLAY_BASE="${COLLECTION_DISPLAY_BASE}"
elif [[ "${DISPLAY}" =~ ^:([0-9]+)(\.[
  0-9]+)?$ ]]; then
  DISPLAY_BASE="${BASH_REMATCH[1]}"
else
  # Server-5090 exposes the prepared CoppeliaSim X server as :99. Override
  # this with COLLECTION_DISPLAY_BASE when using another display range.
  DISPLAY_BASE="99"
fi
if ! [[ "${DISPLAY_BASE}" =~ ^[0-9]+$ ]]; then
  echo "COLLECTION_DISPLAY_BASE must be a display number" >&2
  exit 2
fi
export DISPLAY=":${DISPLAY_BASE}"

# Start missing local X servers after one interactive sudo authentication.
# A pre-existing but inaccessible socket is treated as stale/occupied and is
# never removed automatically.
if (( COLLECTION_WORKERS > 1 )); then
  NEED_X_START=false
  for offset in $(seq 0 $((COLLECTION_WORKERS - 1))); do
    display_number=$((DISPLAY_BASE + offset))
    display=":${display_number}"
    if DISPLAY="${display}" xdpyinfo >/dev/null 2>&1; then
      continue
    fi
    socket="/tmp/.X11-unix/X${display_number}"
    if [[ -e "${socket}" ]]; then
      echo "${display} exists but is not reachable: ${socket}" >&2
      echo "Use COLLECTION_DISPLAY_BASE with a free display range or clean up the stale X server." >&2
      exit 2
    fi
    NEED_X_START=true
  done

  if [[ "${NEED_X_START}" == "true" ]]; then
    sudo -v
    for offset in $(seq 0 $((COLLECTION_WORKERS - 1))); do
      display_number=$((DISPLAY_BASE + offset))
      display=":${display_number}"
      if DISPLAY="${display}" xdpyinfo >/dev/null 2>&1; then
        continue
      fi
      sudo nohup X "${display}" -nolisten tcp -noreset -ac \
        </dev/null >"/tmp/rlbench_x${display_number}.log" 2>&1 &
    done
    sleep 3
  fi

  for offset in $(seq 0 $((COLLECTION_WORKERS - 1))); do
    display_number=$((DISPLAY_BASE + offset))
    display=":${display_number}"
    DISPLAY="${display}" xdpyinfo >/dev/null || {
      echo "X display ${display} is not reachable; see /tmp/rlbench_x${display_number}.log" >&2
      exit 2
    }
  done
fi

EXTRA_ARGS=()
if [[ -n "${TASKS:-}" ]]; then
  read -r -a TASK_LIST <<< "${TASKS}"
fi
if [[ "${ALL_TASKS:-false}" == "true" ]]; then
  EXTRA_ARGS+=(--all-tasks)
elif [[ ${#TASK_LIST[@]} -gt 0 ]]; then
  EXTRA_ARGS+=(--tasks "${TASK_LIST[@]}")
fi
if [[ "${OVERWRITE:-false}" == "true" ]]; then
  EXTRA_ARGS+=(--overwrite)
fi
if [[ "${NO_RESUME:-false}" == "true" ]]; then
  EXTRA_ARGS+=(--no-resume)
fi
if [[ "${DELETE_ARTIFACTS_AFTER_PACK:-false}" == "true" ]]; then
  EXTRA_ARGS+=(--delete-artifacts-after-pack)
fi
if [[ "${SKIP_POINTSEG_CACHE:-false}" == "true" ]]; then
  EXTRA_ARGS+=(--skip-pointseg-cache)
fi
if [[ "${OVERWRITE_CACHE:-false}" == "true" ]]; then
  EXTRA_ARGS+=(--overwrite-cache)
fi
if [[ "${GENERATE_WORLD_BASE_WORLDFLOW_SIDECARS}" == "true" ]]; then
  EXTRA_ARGS+=(--generate-world-base-worldflow-sidecars)
else
  EXTRA_ARGS+=(--no-generate-world-base-worldflow-sidecars)
fi
echo "[param-set] ${PARAM_SET}" >&2
echo "[action-label-mode] ${ACTION_LABEL_MODE}" >&2
echo "[raw-point-count] ${RAW_POINT_COUNT}" >&2
echo "[cache-current-points] ${CACHE_CURRENT_POINTS}" >&2
echo "[cache-future-points] ${CACHE_FUTURE_POINTS}" >&2
echo "[post-success-frames] ${POST_SUCCESS_FRAMES}" >&2
echo "[expert-path-mode] ${EXPERT_PATH_MODE}" >&2
echo "[water-plant-collision] ${RLBENCH_WATER_PLANT_COLLISION}" >&2
echo "[water-drop-collision] ${RLBENCH_WATER_DROP_COLLISION}" >&2
echo "[dataset-root] ${OUTPUT_ROOT}" >&2
echo "[delete-artifacts-after-pack] ${DELETE_ARTIFACTS_AFTER_PACK:-false}" >&2
echo "[world-base-worldflow-sidecars] ${GENERATE_WORLD_BASE_WORLDFLOW_SIDECARS}" >&2

exec "${PYTHON}" "${TOOLS_DIR}/collect_lerobot_pointcloud.py" \
  --output-root "${OUTPUT_ROOT}" \
  --repo-id "${REPO_ID:-rlbench_song_pointcloud}" \
  --episodes-per-task "${EPISODES_PER_TASK:-1}" \
  --num-points "${RAW_POINT_COUNT}" \
  --cache-current-points "${CACHE_CURRENT_POINTS}" \
  --cache-future-points "${CACHE_FUTURE_POINTS}" \
  --gripper-points "${GRIPPER_POINTS:-500}" \
  --gripper-template "${GRIPPER_TEMPLATE}" \
  --image-size "${IMAGE_SIZE:-256}" \
  --fps "${FPS:-20}" \
  --max-demo-attempts "${MAX_DEMO_ATTEMPTS:-10}" \
  --post-success-frames "${POST_SUCCESS_FRAMES}" \
  --expert-path-mode "${EXPERT_PATH_MODE}" \
  --collection-workers "${COLLECTION_WORKERS}" \
  --collection-display-base "${DISPLAY_BASE}" \
  --cache-batch-size "${CACHE_BATCH_SIZE:-4}" \
  --cache-num-workers "${CACHE_NUM_WORKERS:-8}" \
  --cache-device "${CACHE_DEVICE:-cuda}" \
  --cache-python "${PYTHON}" \
  --cache-vis-count "${CACHE_VIS_COUNT:-8}" \
  --cache-vis-one-episode-per-task \
  --motion-rotation-radius "${MOTION_ROTATION_RADIUS}" \
  --motion-baseline-threshold "${MOTION_BASELINE_THRESHOLD}" \
  --motion-baseline-temperature "${MOTION_BASELINE_TEMPERATURE}" \
  --motion-relative-margin "${MOTION_RELATIVE_MARGIN}" \
  --motion-relative-tau "${MOTION_RELATIVE_TAU}" \
  --trajectory-sigma "${TRAJECTORY_SIGMA}" \
  --contact-radius "${CONTACT_RADIUS}" \
  --contact-temperature "${CONTACT_TEMPERATURE}" \
  --approach-margin "${APPROACH_MARGIN}" \
  --approach-tau "${APPROACH_TAU}" \
  --background-trajectory-sigma "${BACKGROUND_TRAJECTORY_SIGMA}" \
  --action-alignment "${ACTION_ALIGNMENT:-transition}" \
  --action-label-mode "${ACTION_LABEL_MODE}" \
  "${EXTRA_ARGS[@]}" \
  "${FORWARD_ARGS[@]}"
