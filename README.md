
# UMR: Universal Manipulation Representation

## Installation

### System dependencies

```bash
sudo apt-get update
sudo apt-get install -y git git-lfs tmux xvfb xauth libegl1 libgl1-mesa-glx
git lfs install
```

### Python environment

```bash
cd /path/to/lerobot
conda create -n wepvla python=3.10 -y
conda activate wepvla
pip install -e ".[smolvla,libero]"
```

For the bundled RLBench implementation:

```bash
pip install -e benchmarks/RLBench
python -c "import pyrep, rlbench; print('RLBench import ok')"
```

Verify the runtime before starting an experiment:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())"
nvidia-smi
```

The repository also includes dependency records in `benchmarks/requirements.txt` and
`benchmarks/environment-rlbench.yml`. CUDA extensions must be rebuilt for the target machine's
PyTorch and CUDA versions; exported local paths are not portable installation instructions.

## Model Resources

The experiments use two local resources:

- `SmolVLM2-500M-Video-Instruct`: VLM architecture, tokenizer, and processor;
- `smolvla_base`: pretrained SmolVLA weights.

Download them with:

```bash
bash benchmarks/RLBench/scripts/download_vlm_models.sh
```

The expected layout is:

```text
benchmarks/vlm_model/
├── SmolVLM2-500M-Video-Instruct/
└── smolvla_base/
```

For offline runs, pass the model paths directly to the command that needs them, for example:

```bash
VLM_MODEL_NAME=/path/to/SmolVLM2-500M-Video-Instruct \
VLM_WEIGHTS_PATH=/path/to/smolvla_base \
PYTHON=python \
bash benchmarks/RLBench/scripts/collect_data.sh \
  --dataset-root /path/to/rlbench_dataset
```

## Quick Start

All commands below are run from the repository root.

### RLBench

RLBench requires a compatible CoppeliaSim installation. Download it from
<https://www.coppeliarobotics.com/downloads> and pass its path to each command:

```bash
COPPELIASIM_ROOT=/path/to/CoppeliaSim \
LD_LIBRARY_PATH="/path/to/CoppeliaSim:${LD_LIBRARY_PATH:-}" \
QT_QPA_PLATFORM=xcb \
QT_QPA_PLATFORM_PLUGIN_PATH=/path/to/CoppeliaSim \
QT_PLUGIN_PATH="" \
bash benchmarks/RLBench/scripts/evaluate.sh --help
```

On a headless server:

```bash
Xvfb :99 -screen 0 1280x1024x24 -nolisten tcp >/tmp/rlbench-xvfb.log 2>&1 &
```

#### 1. Collect data and build cache

```bash
PYTHON=python \
DATASET_ROOT=/path/to/rlbench_dataset \
COPPELIASIM_ROOT=/path/to/CoppeliaSim \
LD_LIBRARY_PATH="/path/to/CoppeliaSim:${LD_LIBRARY_PATH:-}" \
QT_QPA_PLATFORM=xcb \
QT_QPA_PLATFORM_PLUGIN_PATH=/path/to/CoppeliaSim \
QT_PLUGIN_PATH="" \
bash benchmarks/RLBench/scripts/collect_data.sh \
  --dataset-root /path/to/rlbench_dataset
```

The collection flow writes the LeRobot dataset and the PointSeg cache. To rebuild only the cache,
use the cache utility under `benchmarks/RLBench/scripts/tools/`.

#### 2. Train

```bash
DATASET_ROOT=/path/to/rlbench_dataset \
OUTPUT_ROOT=/path/to/rlbench_output \
GPU_IDS=0 \
bash benchmarks/RLBench/scripts/train.sh
```

#### 3. Evaluate

```bash
EVAL_POLICY_PATH=/path/to/checkpoint/pretrained_model \
EVAL_ROOT=/path/to/rlbench_eval \
EVAL_SAVE_VIDEO=0 \
EVAL_SAVE_ACTION_RECORDS=0 \
EVAL_SAVE_ACTION_CHUNKS=0 \
DISPLAY=:99 \
bash benchmarks/RLBench/scripts/evaluate.sh \
  --tasks close_box close_fridge close_laptop_lid phone_on_base stack_wine \
  sweep_to_dustpan take_frame_off_hanger \
  take_umbrella_out_of_umbrella_stand toilet_seat_down water_plants \
  --episodes 100
```

Use tmux for long-running evaluations:

```bash
tmux new -d -s rlbench_eval \
  "DISPLAY=:99 bash benchmarks/RLBench/scripts/evaluate.sh --episodes 100"
```

### LIBERO

The four LIBERO entry points are under `benchmarks/song_real_libero/`.

#### 1. Convert demonstrations

```bash
PYTHON_BIN=/path/to/python \
DEMO_ROOT=/path/to/libero_demos \
DATASET_ROOT=/path/to/libero_dataset \
bash benchmarks/song_real_libero/prepare_dataset.sh
```

#### 2. Build PointSeg cache

```bash
PYTHON_BIN=/path/to/python \
DATASET_ROOT=/path/to/libero_dataset \
CACHE_ROOT=/path/to/libero_cache \
GPU_IDS=0 NPROC=1 \
bash benchmarks/song_real_libero/build_cache.sh
```

#### 3. Train

```bash
PYTHON_BIN=/path/to/python \
DATASET_ROOT=/path/to/libero_dataset \
CACHE_ROOT=/path/to/libero_cache \
BASE_POLICY=/path/to/base_policy/pretrained_model \
OUTPUT_ROOT=/path/to/libero_output \
GPU_IDS=0 \
bash benchmarks/song_real_libero/train.sh
```

#### 4. Evaluate all suites

```bash
PYTHON_BIN=/path/to/python \
POLICY_PATH=/path/to/checkpoint/pretrained_model \
OUTPUT_DIR="benchmarks/song_real_libero/outputs/eval_$(date +%Y%m%d_%H%M%S)" \
CUDA_DEVICE=0 EPISODES=50 \
bash benchmarks/song_real_libero/evaluate.sh
```

The default LIBERO evaluator uses two task workers, one episode shard per worker, and
`inference-batch-size=2`. Every run requires a new output directory and refuses to overwrite an
existing result.

## Benchmark Results

The following numbers are completed, reproducible subsets from the current evaluation protocol.
They should not be interpreted as a full-suite score unless every task and episode is complete.

| Benchmark subset           | Episodes | Success rate |
| :------------------------- | -------: | -----------: |
| RLBench, 6 completed tasks |      600 |   **92.33%** |
| LIBERO spatial             |      500 |   **98.20%** |
| LIBERO object              |      500 |   **99.60%** |

The remaining suite and task-level results are written to each run's `summary.json` and
`progress.json`.

## Data and Coordinate Conventions

- Point clouds are stored as XYZRGB with XYZ in meters and RGB in `[0, 255]`.
- Point clouds and actions use the current EEF coordinate frame.
- Actions use `xyz + rotation-6D + gripper`.
- PointSeg caches are tied to dataset point order, sampling, camera views, and coordinate frames.
  Regenerate the cache whenever any of these change.

## Troubleshooting

**GPU index errors**: run `nvidia-smi` and use only GPU IDs visible in the current environment.

**RLBench cannot start**: check `COPPELIASIM_ROOT`, `LD_LIBRARY_PATH`, `DISPLAY`, Xvfb, PyRep,
and the CoppeliaSim version.

**Existing output directory**: choose a new `OUTPUT_DIR`, `OUTPUT_ROOT`, or `EVAL_ROOT`.

**NVIDIA driver/NVML mismatch**: this is a system-level driver problem. On a shared server, do not
reload or uninstall NVIDIA kernel modules while other users may be using the GPU.

## Additional Documentation

- [RLBench script guide](RLBench/scripts/README.md)
- [LIBERO implementation notes](song_real_libero/README.md)
- [LIBERO experiment protocol](song_real_libero/WEPVLA_V043_DoubleFLow.md)

## Citation

If you use this codebase, please cite the underlying LeRobot and SmolVLA work together with your
project-specific WEP-VLA paper or technical report.

```bibtex
@misc{wepvla,
  title  = {WEP-VLA: Geometry-Aware Vision-Language-Action Policies},
  year   = {2026},
  note   = {Anonymous submission}
}
```
