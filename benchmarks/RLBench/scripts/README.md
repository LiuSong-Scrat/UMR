# Script Layout

The root of this directory contains the maintained, reproducible workflow.
Optional setup and maintenance commands live under `setup/` and `maintenance/`.
Historical experiments and one-off diagnostics are kept under local archive
directories and are excluded from Git.

## Evaluation

- `evaluate.sh` is the portable direct evaluator wrapper.
- `evaluate_profile.sh` resolves a named checkpoint/profile from
  `rlbench_eval_registry.json`.
- `tools/official_eval.py` is the evaluator implementation.

## Data and training

- `collect_data.sh` collects RLBench data.
- `maintenance/build_cache.sh` builds the PointSeg cache from an existing
  dataset; collection already runs the cache stage unless explicitly skipped.
- `train.sh` starts training from a dataset.
- `resume_train.sh` resumes or restarts training.
- `setup/download_vlm_models.sh` downloads the optional local VLM assets.

Python implementations live in `tools/`; the evaluation shell backend lives
in `internal/` and is not a public entry point.

Set `RLBENCH_ROOT`, `COPPELIASIM_ROOT`, `PYTHON`, `LEROBOT_ROOT`,
`LEROBOT_SRC`, `SONG_SCRIPTS`, and `EVAL_POLICY_PATH` for a machine-specific
installation. `.env.example` lists the same variables.
