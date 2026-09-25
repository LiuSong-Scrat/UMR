#!/usr/bin/env bash
set -euo pipefail

# Canonical user-facing RLBench evaluation interface. Checkpoint and task
# parameters live in rlbench_eval_registry.json; the backend is internal and
# should not be invoked directly for new evaluations.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RLBENCH_ROOT="${RLBENCH_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
export RLBENCH_ROOT
REGISTRY="${RLBENCH_EVAL_REGISTRY:-${SCRIPT_DIR}/rlbench_eval_registry.json}"
RESOLVER="${SCRIPT_DIR}/tools/eval_registry.py"
BACKEND="${SCRIPT_DIR}/internal/evaluate_backend.sh"
PYTHON_BIN="${PYTHON:-python}"

usage() {
    cat <<'EOF'
RLBench unified evaluation interface

Usage:
  bash evaluate_profile.sh --list
  bash evaluate_profile.sh --checkpoint ALIAS --tasks TASK [TASK ...] [options]
  bash evaluate_profile.sh --checkpoint ALIAS --tasks all --show

Selection options:
  --checkpoint ALIAS_OR_PATH   Registered checkpoint/profile alias or matching path.
  --tasks TASK...              One or more registered tasks; use 'all' explicitly.
  --list                       List registered checkpoint profiles and tasks.
  --show, --dry-run            Print resolved per-task parameters without running.
  --registry PATH              Use another registry file.

Runtime options handled by this interface:
  --run-root PATH              Exact output root; it must not already exist.
  --display NUMBER             CoppeliaSim X display number.
  --gpu IDS                    GPU id or comma-separated ids (serial evaluator uses first).

Any unrecognized options are forwarded to the evaluator after registry
parameters, so they are explicit one-run overrides. Example:

  bash evaluate_profile.sh \
    --checkpoint 0808_022000 \
    --tasks close_box water_plants \
    --episodes 5

  bash evaluate_profile.sh \
    --checkpoint phone_action9_obs9_007000 \
    --tasks phone_on_base \
    --display 361
EOF
}

CHECKPOINT=""
TASKS=()
FORWARD_ARGS=()
SHOW_ONLY=0
LIST_ONLY=0
RUN_ROOT=""
DISPLAY_NUMBER=""
GPU_IDS=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            usage
            exit 0
            ;;
        --list)
            LIST_ONLY=1
            shift
            ;;
        --show|--dry-run)
            SHOW_ONLY=1
            shift
            ;;
        --registry)
            [[ $# -ge 2 ]] || { echo "--registry requires a path" >&2; exit 2; }
            REGISTRY="$2"
            shift 2
            ;;
        --registry=*)
            REGISTRY="${1#*=}"
            shift
            ;;
        --checkpoint)
            [[ $# -ge 2 ]] || { echo "--checkpoint requires an alias or path" >&2; exit 2; }
            CHECKPOINT="$2"
            shift 2
            ;;
        --checkpoint=*)
            CHECKPOINT="${1#*=}"
            shift
            ;;
        --tasks|--task)
            shift
            while [[ $# -gt 0 && "$1" != --* ]]; do
                IFS=',' read -r -a task_parts <<< "$1"
                for task in "${task_parts[@]}"; do
                    [[ -n "${task}" ]] && TASKS+=("${task}")
                done
                shift
            done
            ;;
        --tasks=*|--task=*)
            IFS=',' read -r -a task_parts <<< "${1#*=}"
            for task in "${task_parts[@]}"; do
                [[ -n "${task}" ]] && TASKS+=("${task}")
            done
            shift
            ;;
        --run-root)
            [[ $# -ge 2 ]] || { echo "--run-root requires a path" >&2; exit 2; }
            RUN_ROOT="$2"
            shift 2
            ;;
        --run-root=*)
            RUN_ROOT="${1#*=}"
            shift
            ;;
        --display)
            [[ $# -ge 2 ]] || { echo "--display requires a number" >&2; exit 2; }
            DISPLAY_NUMBER="$2"
            shift 2
            ;;
        --display=*)
            DISPLAY_NUMBER="${1#*=}"
            shift
            ;;
        --gpu|--gpus)
            [[ $# -ge 2 ]] || { echo "$1 requires one or more GPU ids" >&2; exit 2; }
            GPU_IDS="$2"
            shift 2
            ;;
        --gpu=*|--gpus=*)
            GPU_IDS="${1#*=}"
            shift
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

REGISTRY="$(realpath -m "${REGISTRY}")"
[[ -f "${REGISTRY}" ]] || { echo "Registry not found: ${REGISTRY}" >&2; exit 2; }
[[ -f "${RESOLVER}" ]] || { echo "Registry resolver not found: ${RESOLVER}" >&2; exit 2; }
[[ -f "${BACKEND}" ]] || { echo "Evaluation backend not found: ${BACKEND}" >&2; exit 2; }

if (( LIST_ONLY == 1 )); then
    printf '%-36s %-70s %s\n' "CHECKPOINT/PROFILE" "TASKS" "DESCRIPTION"
    while IFS=$'\t' read -r checkpoint_id tasks description; do
        printf '%-36s %-70s %s\n' "${checkpoint_id}" "${tasks}" "${description}"
    done < <("${PYTHON_BIN}" "${RESOLVER}" --registry "${REGISTRY}" list)
    exit 0
fi

[[ -n "${CHECKPOINT}" ]] || { echo "--checkpoint is required (use --list)" >&2; exit 2; }
(( ${#TASKS[@]} > 0 )) || { echo "--tasks is required; use --tasks all explicitly" >&2; exit 2; }

CHECKPOINT_ID="$("${PYTHON_BIN}" "${RESOLVER}" --registry "${REGISTRY}" checkpoint-id --checkpoint "${CHECKPOINT}")"
TRAINING_SETTING="$("${PYTHON_BIN}" "${RESOLVER}" --registry "${REGISTRY}" training-setting --checkpoint "${CHECKPOINT_ID}")"
CHECKPOINT_STEP="$("${PYTHON_BIN}" "${RESOLVER}" --registry "${REGISTRY}" checkpoint-step --checkpoint "${CHECKPOINT_ID}")"
POLICY_PATH="$("${PYTHON_BIN}" "${RESOLVER}" --registry "${REGISTRY}" policy --checkpoint "${CHECKPOINT_ID}")"
mapfile -t REGISTERED_TASKS < <("${PYTHON_BIN}" "${RESOLVER}" --registry "${REGISTRY}" tasks --checkpoint "${CHECKPOINT_ID}")

if (( ${#TASKS[@]} == 1 )) && [[ "${TASKS[0]}" == "all" ]]; then
    TASKS=("${REGISTERED_TASKS[@]}")
elif [[ " ${TASKS[*]} " == *" all "* ]]; then
    echo "Use --tasks all by itself, or name tasks explicitly" >&2
    exit 2
fi

for task in "${TASKS[@]}"; do
    "${PYTHON_BIN}" "${RESOLVER}" --registry "${REGISTRY}" show \
        --checkpoint "${CHECKPOINT_ID}" --task "${task}" >/dev/null
done

# Registry environment values are defaults. Explicit caller environment values
# and the dedicated interface options below take precedence.
while IFS='=' read -r env_name env_value; do
    [[ -n "${env_name}" ]] || continue
    if [[ -z "${!env_name+x}" ]]; then
        export "${env_name}=${env_value}"
    fi
done < <("${PYTHON_BIN}" "${RESOLVER}" --registry "${REGISTRY}" env \
    --checkpoint "${CHECKPOINT_ID}" --format lines)

export EVAL_CHECKPOINT_TASK_REGISTRY="${REGISTRY}"
export EVAL_CHECKPOINT_ID="${CHECKPOINT_ID}"
export EVAL_TRAINING_SETTING="${TRAINING_SETTING}"
export EVAL_CHECKPOINT_STEP="${CHECKPOINT_STEP}"
# Runs launched through the unified interface use one readable directory-name
# scheme. Direct calls to historical s4 backends keep their legacy names.
export EVAL_RUN_NAMING_STYLE="${EVAL_RUN_NAMING_STYLE:-canonical_v1}"
if [[ -n "${RUN_ROOT}" ]]; then
    export EVAL_ROOT="${RUN_ROOT}"
fi
if [[ -n "${DISPLAY_NUMBER}" ]]; then
    export EVAL_DISPLAY_BASE="${DISPLAY_NUMBER}"
fi
if [[ -n "${GPU_IDS}" ]]; then
    export EVAL_GPU_IDS="${GPU_IDS}"
fi
if [[ -z "${EVAL_BASE_DIR+x}" ]]; then
    # The registry stores checkpoint/task parameters; evaluation artifacts
    # belong beside the historical checkpoint result directories, not inside
    # eval/registry.  A checkpoint may still provide an explicit EVAL_BASE_DIR
    # through its registered environment when a shorter family name is wanted.
    export EVAL_BASE_DIR="${RLBENCH_ROOT}/eval/${TRAINING_SETTING}_${CHECKPOINT_STEP}"
fi

echo "[eval-selection] checkpoint=${CHECKPOINT_ID}"
echo "[eval-selection] policy=${POLICY_PATH}"
echo "[eval-selection] tasks=${TASKS[*]}"
echo "[eval-selection] registry=${REGISTRY}"
if [[ -n "${RUN_ROOT}" ]]; then
    echo "[eval-output] exact_run_root=${RUN_ROOT}"
else
    echo "[eval-output] base=${EVAL_BASE_DIR} naming=${EVAL_RUN_NAMING_STYLE} training_setting=${TRAINING_SETTING} checkpoint_step=${CHECKPOINT_STEP}"
fi
for task in "${TASKS[@]}"; do
    resolved_args="$("${PYTHON_BIN}" "${RESOLVER}" --registry "${REGISTRY}" args \
        --checkpoint "${CHECKPOINT_ID}" --task "${task}" --format shell)"
    echo "[eval-task-parameters] task=${task} ${resolved_args}"
done
if (( ${#FORWARD_ARGS[@]} > 0 )); then
    printf '[eval-overrides]'
    printf ' %q' "${FORWARD_ARGS[@]}"
    printf '\n'
fi

if (( SHOW_ONLY == 1 )); then
    echo "[eval-dry-run] no evaluation was started"
    exit 0
fi

mkdir -p "${EVAL_BASE_DIR}"
exec bash "${BACKEND}" \
    --tasks "${TASKS[@]}" \
    --policy-path "${POLICY_PATH}" \
    "${FORWARD_ARGS[@]}"
