"""Report LeRobot task/stat metadata against direct parquet calculations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root

    tables = [pq.read_table(path) for path in sorted((root / "data").rglob("*.parquet"))]
    table = pa.concat_tables(tables)
    stats = json.loads((root / "meta" / "stats.json").read_text())
    summary = json.loads((root / "libero_collect_summary.json").read_text())
    tasks = pq.read_table(root / "meta" / "tasks.parquet").to_pylist()

    differences: dict[str, dict] = {}
    for key in ("observation.state", "action"):
        source = np.asarray(table[key].combine_chunks().to_pylist(), dtype=np.float32)
        entry = stats[key]
        differences[key] = {}
        for name in ("min", "max", "mean", "std"):
            stored = np.asarray(entry[name], dtype=np.float64)
            direct64 = getattr(np, name)(source.astype(np.float64), axis=0)
            direct32 = getattr(np, name)(source, axis=0).astype(np.float64)
            differences[key][name] = {
                "stored": stored.tolist(),
                "direct_float64": direct64.tolist(),
                "direct_float32": direct32.tolist(),
                "max_abs_diff_float64": float(np.max(np.abs(stored - direct64))),
                "max_abs_diff_float32": float(np.max(np.abs(stored - direct32))),
            }

    print(
        json.dumps(
            {
                "task_rows": tasks,
                "summary_task_language": summary.get("episodes", [{}])[0].get("task_language"),
                "stats_differences": differences,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
