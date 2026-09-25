#!/usr/bin/env python3
"""Reject RLBench datasets that do not contain the canonical REAP v4 cloud."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from reap_gripper import (
    LIBERO_GRIPPER_TEMPLATE_VERSION,
    LIBERO_GRIPPER_WIDTH_NORMALIZATION_MAX,
    LIBERO_REAP_GRIPPER_LEN,
    LIBERO_REAP_OPENING_MAX_WIDTH,
    LIBERO_REAP_TEMPLATE_MAX_WIDTH,
)


def require_close(metadata: dict, key: str, expected: float) -> None:
    if key not in metadata or not np.isclose(
        float(metadata[key]), float(expected), rtol=0.0, atol=1e-9
    ):
        raise RuntimeError(
            f"RLBench dataset has {key}={metadata.get(key)!r}; expected {expected!r}."
        )


def validate_dataset(dataset_root: Path) -> dict:
    dataset_root = dataset_root.expanduser().resolve()
    metadata_path = dataset_root / "meta" / "rlbench_conversion.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"Missing RLBench conversion metadata: {metadata_path}. "
            "Refusing to treat an unversioned dataset as canonical REAP v4."
        )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    version = metadata.get("gripper_template_version", metadata.get("gripper_template"))
    if version != LIBERO_GRIPPER_TEMPLATE_VERSION:
        raise RuntimeError(
            f"RLBench dataset gripper version is {version!r}; expected "
            f"{LIBERO_GRIPPER_TEMPLATE_VERSION!r}. Recollect/rebuild the dataset "
            "and regenerate its PointSeg cache."
        )
    require_close(
        metadata,
        "virtual_gripper_width_normalization_max_m",
        LIBERO_GRIPPER_WIDTH_NORMALIZATION_MAX,
    )
    require_close(
        metadata,
        "virtual_gripper_geometry_max_width_m",
        LIBERO_REAP_TEMPLATE_MAX_WIDTH,
    )
    require_close(
        metadata,
        "virtual_gripper_opening_max_width_m",
        LIBERO_REAP_OPENING_MAX_WIDTH,
    )
    offset = np.asarray(metadata.get("virtual_gripper_local_offset_m"), dtype=np.float64)
    expected_offset = np.asarray([0.0, 0.0, -LIBERO_REAP_GRIPPER_LEN], dtype=np.float64)
    if offset.shape != (3,) or not np.allclose(offset, expected_offset, rtol=0.0, atol=1e-9):
        raise RuntimeError(
            f"RLBench dataset virtual_gripper_local_offset_m={offset.tolist()}; "
            f"expected {expected_offset.tolist()}."
        )
    return {
        "dataset_root": str(dataset_root),
        "gripper_template_version": version,
        "status": "canonical_reap_v4",
    }


def validate_cache_source(cache_dir: Path, dataset_root: Path) -> None:
    cache_dir = cache_dir.expanduser().resolve()
    manifest_path = cache_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing PointSeg cache manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    cache_args = manifest.get("args", {})
    dataset_root = dataset_root.expanduser().resolve()
    cached_dataset = cache_args.get("dataset_repo_id") or cache_args.get("dataset_root")
    cached_cloud = cache_args.get("point_cloud_dir")
    source_matches = False
    if cached_dataset:
        source_matches |= Path(cached_dataset).expanduser().resolve() == dataset_root
    if cached_cloud:
        source_matches |= Path(cached_cloud).expanduser().resolve() == dataset_root / "point_clouds"
    if not source_matches:
        raise RuntimeError(
            f"PointSeg cache {cache_dir} was not built from canonical dataset "
            f"{dataset_root}; cached dataset={cached_dataset!r}, "
            f"point_cloud_dir={cached_cloud!r}."
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--cache-dir", type=Path, default=None)
    args = parser.parse_args()
    result = validate_dataset(args.dataset_root)
    if args.cache_dir is not None:
        validate_cache_source(args.cache_dir, args.dataset_root)
        result["pointseg_cache"] = str(args.cache_dir.expanduser().resolve())
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
