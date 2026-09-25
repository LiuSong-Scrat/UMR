from __future__ import annotations

import json
import os
from collections.abc import Iterable
from pathlib import Path
from typing import Any


SCRIPTS_ROOT = Path(__file__).resolve().parent
BENCHMARK_ROOT = SCRIPTS_ROOT.parent
REPO_ROOT = BENCHMARK_ROOT.parents[1]
CONFIG_ROOT = BENCHMARK_ROOT / "configs"
DATA_ROOT = BENCHMARK_ROOT / "data"
LIBERO_DATA_ROOT = DATA_ROOT / "libero_setting"
REAL_DATA_ROOT = DATA_ROOT / "real_setting"
OUTPUT_ROOT = BENCHMARK_ROOT / "outputs"

DEFAULT_REAL_CONFIG = CONFIG_ROOT / "local.json"
DEFAULT_LIBERO_CONFIG = CONFIG_ROOT / "libero.json"


def resolve_benchmark_path(value: str | Path) -> Path:
    """Resolve config paths relative to this benchmark instead of the shell cwd."""
    path = Path(os.path.expandvars(str(value))).expanduser()
    return path.resolve() if path.is_absolute() else (BENCHMARK_ROOT / path).resolve()


def load_json_config(path: str | Path, *, path_keys: Iterable[str] = ()) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as config_file:
        config = json.load(config_file)
    for key in path_keys:
        value = config.get(key)
        if value not in (None, ""):
            config[key] = str(resolve_benchmark_path(value))
    return config
