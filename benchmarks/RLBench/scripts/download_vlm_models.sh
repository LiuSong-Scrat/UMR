#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RLBENCH_ROOT="${RLBENCH_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
MODEL_ROOT="${VLM_MODEL_ROOT:-${RLBENCH_ROOT}/../vlm_model}"
HF_CLI="${HF_CLI:-hf}"

command -v "${HF_CLI}" >/dev/null 2>&1 || {
  echo "Hugging Face CLI not found: ${HF_CLI}. Activate an environment with huggingface_hub installed or set HF_CLI." >&2
  exit 1
}
mkdir -p "${MODEL_ROOT}"

# Keep the raw VLM architecture/processor separate from the SmolVLA policy weights.
"${HF_CLI}" download HuggingFaceTB/SmolVLM2-500M-Video-Instruct \
  --local-dir "${MODEL_ROOT}/SmolVLM2-500M-Video-Instruct" \
  --exclude 'onnx/*' 'openvino/*' 'tflite/*' '*.onnx' '*.xml' '*.bin'
"${HF_CLI}" download lerobot/smolvla_base \
  --local-dir "${MODEL_ROOT}/smolvla_base" \
  --exclude 'onnx/*' 'openvino/*' 'tflite/*' '*.onnx' '*.xml' '*.bin'

test -s "${MODEL_ROOT}/SmolVLM2-500M-Video-Instruct/config.json"
test -s "${MODEL_ROOT}/SmolVLM2-500M-Video-Instruct/processor_config.json"
test -s "${MODEL_ROOT}/smolvla_base/config.json"
test -s "${MODEL_ROOT}/smolvla_base/model.safetensors"
printf 'VLM model and weights are ready under %s\n' "${MODEL_ROOT}"
