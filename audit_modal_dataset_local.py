"""Audit a converted LIBERO dataset from inside its mounted Modal Volume.

This deliberately runs in a Modal container instead of using the public
``Volume.listdir(recursive=True)`` API.  Large recursive listings can terminate
their gRPC stream before all image and Zarr chunk paths have been returned.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import zarr
from PIL import Image


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write(path: Path, data: bytes) -> None:
    temporary = path.with_name(f".{path.name}.audit-tmp-{os.getpid()}")
    temporary.write_bytes(data)
    os.replace(temporary, path)


def regular_files(root: Path) -> dict[str, int]:
    files: dict[str, int] = {}
    for directory, _, names in os.walk(root):
        base = Path(directory)
        for name in names:
            path = base / name
            files[path.relative_to(root).as_posix()] = path.stat().st_size
    return files


def validate_png(path: Path) -> bool:
    try:
        with Image.open(path) as image:
            valid = image.format == "PNG" and image.size == (256, 256) and image.mode == "RGB"
            image.verify()
        return valid
    except Exception:
        return False


def load_data_arrays(root: Path) -> tuple[dict[str, np.ndarray], list[Path]]:
    data_files = sorted((root / "data").rglob("*.parquet"))
    tables = [pq.read_table(path) for path in data_files]
    if not tables:
        return {}, data_files
    import pyarrow as pa

    table = pa.concat_tables(tables)
    arrays = {
        "action": np.asarray(table["action"].combine_chunks().to_pylist(), dtype=np.float32),
        "observation.state": np.asarray(
            table["observation.state"].combine_chunks().to_pylist(), dtype=np.float32
        ),
        "timestamp": np.asarray(table["timestamp"].combine_chunks().to_pylist(), dtype=np.float32).reshape(-1),
        "index": np.asarray(table["index"].combine_chunks().to_pylist(), dtype=np.int64).reshape(-1),
        "episode_index": np.asarray(
            table["episode_index"].combine_chunks().to_pylist(), dtype=np.int64
        ).reshape(-1),
        "frame_index": np.asarray(
            table["frame_index"].combine_chunks().to_pylist(), dtype=np.int64
        ).reshape(-1),
        "task_index": np.asarray(
            table["task_index"].combine_chunks().to_pylist(), dtype=np.int64
        ).reshape(-1),
    }
    return arrays, data_files


def array_sha256(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    return hashlib.sha256(contiguous.tobytes()).hexdigest()


def parquet_contract(
    root: Path,
    expected_frames: int,
    frames_by_episode: dict[int, int],
    reference_root: Path | None,
) -> tuple[dict, dict[str, np.ndarray]]:
    arrays, data_files = load_data_arrays(root)
    episode_files = sorted((root / "meta" / "episodes").rglob("*.parquet"))
    episode_tables = [pq.read_table(path) for path in episode_files]
    episode_rows = sum(table.num_rows for table in episode_tables)
    expected_episode_indices = np.concatenate(
        [np.full(frames, episode, dtype=np.int64) for episode, frames in frames_by_episode.items()]
    )
    expected_frame_indices = np.concatenate(
        [np.arange(frames, dtype=np.int64) for _, frames in frames_by_episode.items()]
    )
    reference_match = True
    reference_hashes: dict[str, str] = {}
    if reference_root is not None:
        reference_arrays, _ = load_data_arrays(reference_root)
        for key in arrays:
            reference_match = reference_match and key in reference_arrays and np.array_equal(
                arrays[key], reference_arrays[key]
            )
            if key in reference_arrays:
                reference_hashes[key] = array_sha256(reference_arrays[key])
    report = {
        "data_files_nonempty": bool(data_files)
        and all(path.stat().st_size > 0 for path in data_files),
        "data_rows": bool(arrays) and len(arrays["index"]) == expected_frames,
        "data_file_count": len(data_files),
        "episode_metadata_nonempty": bool(episode_files)
        and all(path.stat().st_size > 0 for path in episode_files),
        "episode_metadata_file_count": len(episode_files),
        "episode_metadata_rows": episode_rows == len(frames_by_episode),
        "state_action_shape": arrays.get("observation.state", np.empty(0)).shape
        == (expected_frames, 10)
        and arrays.get("action", np.empty(0)).shape == (expected_frames, 10),
        "state_action_finite": bool(arrays)
        and bool(np.isfinite(arrays["observation.state"]).all())
        and bool(np.isfinite(arrays["action"]).all()),
        "global_index_contiguous": bool(arrays)
        and np.array_equal(arrays["index"], np.arange(expected_frames, dtype=np.int64)),
        "episode_frame_indices": bool(arrays)
        and np.array_equal(arrays["episode_index"], expected_episode_indices)
        and np.array_equal(arrays["frame_index"], expected_frame_indices),
        "task_indices_zero": bool(arrays)
        and np.array_equal(arrays["task_index"], np.zeros(expected_frames, dtype=np.int64)),
        "timestamp_20hz": bool(arrays)
        and all(
            np.allclose(
                arrays["timestamp"][arrays["episode_index"] == episode],
                np.arange(frames, dtype=np.float32) / 20.0,
                rtol=0.0,
                atol=1e-6,
            )
            for episode, frames in frames_by_episode.items()
        ),
        "reference_arrays_exact": reference_match,
        "array_sha256": {key: array_sha256(value) for key, value in arrays.items()},
        "reference_sha256": reference_hashes,
    }
    return report, arrays


def stats_contract(root: Path, arrays: dict[str, np.ndarray], expected_frames: int) -> dict:
    stats = read_json(root / "meta" / "stats.json")
    report: dict[str, bool] = {}
    for key in ("observation.state", "action"):
        entry = stats.get(key, {})
        source = arrays.get(key, np.empty(0))
        report[f"{key}_shape_and_count"] = (
            source.shape == (expected_frames, 10)
            and entry.get("count") == [expected_frames]
            and all(len(entry.get(stat, [])) == 10 for stat in ("min", "max", "mean", "std"))
        )
        report[f"{key}_finite"] = all(
            bool(np.isfinite(np.asarray(entry.get(stat, []), dtype=np.float64)).all())
            for stat in ("min", "max", "mean", "std")
        )
        report[f"{key}_matches_data"] = source.shape == (expected_frames, 10) and all(
            np.allclose(
                np.asarray(entry[stat], dtype=np.float64),
                getattr(np, stat)(source.astype(np.float64), axis=0),
                rtol=1e-5,
                atol=1e-5,
            )
            for stat in ("min", "max", "mean", "std")
        )
    return report


def tasks_contract(root: Path, task_language: str) -> dict:
    table = pq.read_table(root / "meta" / "tasks.parquet")
    rows = table.to_pylist()
    return {
        "tasks_one_row": len(rows) == 1,
        "tasks_index_zero": len(rows) == 1 and rows[0].get("task_index") == 0,
        "tasks_language_exact": len(rows) == 1
        and rows[0].get("__index_level_0__") == task_language,
    }


def point_cloud_contract(
    root: Path,
    frames_by_episode: dict[int, int],
    expected_points: int,
) -> tuple[dict[int, dict], bool]:
    def inspect_episode(item: tuple[int, int]) -> tuple[int, dict]:
        episode, frames = item
        base = root / "point_clouds" / f"episode_{episode:06d}.zarr"
        attrs = read_json(base / ".zattrs")
        channel_reports: dict[str, dict] = {}
        all_arrays_valid = True
        for channel, expected_dtype in (("xyz", "<f2"), ("rgb", "|u1")):
            array_dir = base / channel
            metadata = read_json(array_dir / ".zarray")
            expected_chunk_names = {f"{frame}.0.0" for frame in range(frames)}
            actual_chunk_names = {
                path.name
                for path in array_dir.iterdir()
                if path.is_file() and path.name[0:1].isdigit()
            }
            compressor = metadata.get("compressor") or {}
            metadata_valid = (
                metadata.get("shape") == [frames, expected_points, 3]
                and metadata.get("dtype") == expected_dtype
                and metadata.get("chunks") == [1, expected_points, 3]
                and metadata.get("order") == "C"
                and compressor.get("id") == "blosc"
                and compressor.get("cname") == "zstd"
                and actual_chunk_names == expected_chunk_names
                and all((array_dir / name).stat().st_size > 0 for name in actual_chunk_names)
            )
            channel_reports[channel] = {
                "metadata_and_chunks": metadata_valid,
                "chunk_count": len(actual_chunk_names),
            }
            all_arrays_valid = all_arrays_valid and metadata_valid

        sampled_arrays_valid = False
        sampled_finite_xyz = False
        unique_xyz_samples = False
        gripper_suffix_color = False
        gripper_suffix_current_eef_bounds = False
        gripper_suffix_max_norm_m = None
        if all_arrays_valid:
            group = zarr.open_group(str(base), mode="r")
            sample_frames = sorted({0, max(0, frames // 2), max(0, frames - 1)})
            xyz = np.stack([np.asarray(group["xyz"][index]) for index in sample_frames])
            rgb = np.stack([np.asarray(group["rgb"][index]) for index in sample_frames])
            sampled_arrays_valid = (
                xyz.shape == (len(sample_frames), expected_points, 3)
                and rgb.shape == (len(sample_frames), expected_points, 3)
                and xyz.dtype == np.dtype("float16")
                and rgb.dtype == np.dtype("uint8")
            )
            sampled_finite_xyz = bool(np.isfinite(xyz).all())
            unique_xyz_samples = all(
                np.unique(xyz[index], axis=0).shape[0] >= int(expected_points * 0.90)
                for index in range(len(sample_frames))
            )
            # shuffle_points=false guarantees that the virtual REAP cloud is
            # the 500-point suffix.  After reference->current_eef, that suffix
            # must be the red canonical gripper close to the local origin.  A
            # camera/world-frame cloud would be displaced by the robot pose.
            gripper_xyz = xyz[:, -500:, :].astype(np.float32)
            gripper_rgb = rgb[:, -500:, :]
            gripper_suffix_color = bool(
                np.equal(gripper_rgb, np.array([204, 51, 51], dtype=np.uint8)).all()
            )
            gripper_suffix_max_norm_m = float(
                np.linalg.norm(gripper_xyz, axis=-1).max()
            )
            gripper_suffix_current_eef_bounds = gripper_suffix_max_norm_m < 0.20

        report = {
            "attrs_shape": attrs.get("shape") == [frames, expected_points, 6],
            "xyz": channel_reports["xyz"],
            "rgb": channel_reports["rgb"],
            "sampled_frames": sample_frames if all_arrays_valid else [],
            "sampled_arrays_readable": sampled_arrays_valid,
            "sampled_xyz_finite": sampled_finite_xyz,
            "sample_unique_xyz_ge_90pct": unique_xyz_samples,
            "gripper_suffix_color": gripper_suffix_color,
            "gripper_suffix_current_eef_bounds": gripper_suffix_current_eef_bounds,
            "gripper_suffix_max_norm_m": gripper_suffix_max_norm_m,
        }

        return episode, report

    with ThreadPoolExecutor(max_workers=min(16, len(frames_by_episode))) as pool:
        reports = dict(pool.map(inspect_episode, frames_by_episode.items()))

    valid = all(
        report["attrs_shape"]
        and report["xyz"]["metadata_and_chunks"]
        and report["rgb"]["metadata_and_chunks"]
        and report["sampled_arrays_readable"]
        and report["sampled_xyz_finite"]
        and report["sample_unique_xyz_ge_90pct"]
        and report["gripper_suffix_color"]
        and report["gripper_suffix_current_eef_bounds"]
        for report in reports.values()
    )
    return reports, valid


def sidecar_contract(
    root: Path,
    directory: str,
    frames_by_episode: dict[int, int],
) -> tuple[dict[int, dict], bool]:
    reports: dict[int, dict] = {}
    for episode, frames in frames_by_episode.items():
        path = root / directory / f"episode_{episode:06d}.npy"
        if not path.is_file():
            reports[episode] = {"present": False}
            continue
        array = np.load(path, allow_pickle=False)
        reports[episode] = {
            "present": True,
            "shape": list(array.shape),
            "dtype": str(array.dtype),
            "c_contiguous": bool(array.flags.c_contiguous),
            "finite": bool(np.isfinite(array).all()),
        }
    valid = len(reports) == len(frames_by_episode) and all(
        report.get("present") is True
        and report.get("shape") == [frames_by_episode[episode], 9]
        and report.get("dtype") == "float32"
        and report.get("c_contiguous") is True
        and report.get("finite") is True
        for episode, report in reports.items()
    )
    return reports, valid


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--reference-root")
    parser.add_argument("--patch-metadata", action="store_true")
    parser.add_argument("--expected-episodes", type=int, default=50)
    parser.add_argument("--expected-frames", type=int, default=7882)
    parser.add_argument("--expected-points", type=int, default=10000)
    args = parser.parse_args()

    root = Path(args.root)
    if not root.is_dir():
        raise FileNotFoundError(root)

    files = regular_files(root)
    info_path = root / "meta" / "info.json"
    summary_path = root / "libero_collect_summary.json"
    point_meta_path = root / "point_clouds" / "meta.json"
    info = read_json(info_path)
    summary_before_bytes = summary_path.read_bytes()
    summary = json.loads(summary_before_bytes)
    point_meta_before_bytes = point_meta_path.read_bytes()
    point_meta_before = json.loads(point_meta_before_bytes)

    records = summary.get("episodes", [])
    record_ids = [int(record.get("episode_index", -1)) for record in records]
    expected_ids = list(range(args.expected_episodes))
    frames_by_episode = {
        int(record["episode_index"]): int(record["frames"]) for record in records
    }

    image_checks: dict[str, bool] = {}
    image_counts: dict[str, int] = {}
    image_paths_by_camera: dict[str, set[str]] = {}
    for camera in ("agentview", "robot0_eye_in_hand"):
        prefix = f"images/observation.images.{camera}/"
        actual = {
            path for path in files if path.startswith(prefix) and path.endswith(".png")
        }
        expected = {
            f"{prefix}episode-{episode:06d}/frame-{frame:06d}.png"
            for episode, frames in frames_by_episode.items()
            for frame in range(frames)
        }
        image_counts[camera] = len(actual)
        image_paths_by_camera[camera] = actual
        image_checks[f"{camera}_image_paths"] = actual == expected
        image_checks[f"{camera}_images_nonempty"] = all(files[path] > 0 for path in actual)
        with ThreadPoolExecutor(max_workers=32) as pool:
            decoded = all(
                pool.map(validate_png, (root / relative_path for relative_path in sorted(actual)))
            )
        image_checks[f"{camera}_png_decode_256_rgb"] = decoded

    sampled_camera_pairs_distinct = True
    for episode, frames in frames_by_episode.items():
        for frame in sorted({0, max(0, frames // 2), max(0, frames - 1)}):
            suffix = f"episode-{episode:06d}/frame-{frame:06d}.png"
            agent_path = root / "images" / "observation.images.agentview" / suffix
            wrist_path = root / "images" / "observation.images.robot0_eye_in_hand" / suffix
            sampled_camera_pairs_distinct = sampled_camera_pairs_distinct and (
                sha256(agent_path) != sha256(wrist_path)
            )
    image_checks["sampled_camera_pairs_distinct"] = sampled_camera_pairs_distinct

    point_reports, points_valid = point_cloud_contract(
        root, frames_by_episode, args.expected_points
    )
    world_reports, world_valid = sidecar_contract(
        root, "world_ee_poses", frames_by_episode
    )
    target_reports, target_valid = sidecar_contract(
        root, "action_target_ee_poses", frames_by_episode
    )
    reference_root = Path(args.reference_root) if args.reference_root else None
    parquet_report, data_arrays = parquet_contract(
        root,
        args.expected_frames,
        frames_by_episode,
        reference_root,
    )
    parquet_checks = {
        key: value for key, value in parquet_report.items() if isinstance(value, bool)
    }
    stats_checks = stats_contract(root, data_arrays, args.expected_frames)
    task_language = records[0].get("task_language", "") if records else ""
    task_checks = tasks_contract(root, task_language)
    world_meta = read_json(root / "world_ee_poses" / "meta.json")

    expected_visual_features = {
        "observation.images.agentview",
        "observation.images.robot0_eye_in_hand",
    }
    visual_features = {
        key for key in info.get("features", {}) if key.startswith("observation.images.")
    }
    required_nonempty = [
        "meta/info.json",
        "meta/stats.json",
        "meta/tasks.parquet",
        "libero_collect_summary.json",
    ]
    core_checks = {
        "episode_count": info.get("total_episodes") == args.expected_episodes == len(records),
        "episode_indices": record_ids == expected_ids,
        "frame_count": info.get("total_frames") == args.expected_frames == sum(frames_by_episode.values()),
        "task_count": info.get("total_tasks") == 1,
        "fps": info.get("fps") == 20,
        "visual_features": visual_features == expected_visual_features,
        **image_checks,
        "point_contract": points_valid and set(point_reports) == set(expected_ids),
        "world_pose_contract": world_valid,
        "action_target_contract": target_valid,
        "world_pose_frame": world_meta.get("coordinate_frame") == "overview_camera",
        "causal_records": all(
            record.get("task_id") == 8
            and record.get("source_fps") == 20.0
            and record.get("state_observation_offset") == 1
            and record.get("model_restoration_verified") is True
            and record.get("causal_action_alignment") is True
            and record.get("action_source_index_offset") == 1
            and record.get("gripper_action_state_offset") == 2
            for record in records
        ),
        "summary_modalities": summary.get("pointcloud_camera_names") == ["agentview"]
        and summary.get("image_cameras") == ["agentview", "robot0_eye_in_hand"],
        "gripper_contract": summary.get("add_gripper_cloud") is True
        and summary.get("gripper_points") == 500
        and summary.get("gripper_template") == "reap",
        "required_metadata": all(path in files and files[path] > 0 for path in required_nonempty),
        **parquet_checks,
        **stats_checks,
        **task_checks,
    }

    metadata_was_patched = False
    metadata_already_correct = (
        point_meta_before.get("coordinate_frame") == "current_eef"
        and point_meta_before.get("source_reference_frame") == "agentview"
        and point_meta_before.get("source_cameras") == ["agentview"]
        and point_meta_before.get("frame_semantics")
        == "each frame uses its achieved model EEF pose as origin"
        and summary.get("point_cloud_coordinate_frame") == "current_eef"
        and all(
            record.get("point_cloud_coordinate_frame") == "current_eef"
            for record in summary.get("episodes", [])
        )
    )
    if args.patch_metadata and not metadata_already_correct:
        boolean_checks = {
            key: value for key, value in core_checks.items() if isinstance(value, bool)
        }
        if not all(boolean_checks.values()):
            print(json.dumps({"core_checks": core_checks, "metadata_patched": False}, indent=2))
            return 1
        correction_path = root / "metadata_correction_20260807.json"
        if correction_path.exists():
            raise RuntimeError(
                f"Refusing to overwrite existing provenance record: {correction_path}"
            )
        point_meta = dict(point_meta_before)
        point_meta.update(
            {
                "coordinate_frame": "current_eef",
                "source_reference_frame": "agentview",
                "source_cameras": ["agentview"],
                "frame_semantics": "each frame uses its achieved model EEF pose as origin",
            }
        )
        summary["point_cloud_coordinate_frame"] = "current_eef"
        for record in summary.get("episodes", []):
            record["point_cloud_coordinate_frame"] = "current_eef"
        audit_note = {
            "reason": "Correct provenance metadata emitted by the pre-fix converter; point arrays were already transformed to current_eef.",
            "summary_before_sha256": hashlib.sha256(summary_before_bytes).hexdigest(),
            "point_cloud_meta_before_sha256": hashlib.sha256(point_meta_before_bytes).hexdigest(),
            "point_cloud_meta_before": point_meta_before,
            "point_cloud_meta_after": point_meta,
            "data_arrays_modified": False,
        }
        atomic_write(
            point_meta_path,
            (json.dumps(point_meta, indent=2) + "\n").encode(),
        )
        atomic_write(
            summary_path,
            (json.dumps(summary, indent=2) + "\n").encode(),
        )
        atomic_write(
            correction_path,
            (json.dumps(audit_note, indent=2) + "\n").encode(),
        )
        metadata_was_patched = True

    point_meta = read_json(point_meta_path)
    summary_after = read_json(summary_path)
    metadata_checks = {
        "point_coordinate_frame": point_meta.get("coordinate_frame") == "current_eef",
        "point_source_camera": point_meta.get("source_reference_frame") == "agentview"
        and point_meta.get("source_cameras") == ["agentview"],
        "point_storage": point_meta.get("storage_format") == "zarr"
        and point_meta.get("zarr_encoding") == "packed_xyz_float16_rgb_uint8"
        and point_meta.get("path_format") == "point_clouds/episode_{episode_index:06d}.zarr",
        "point_frame_semantics": point_meta.get("frame_semantics")
        == "each frame uses its achieved model EEF pose as origin",
        "summary_point_frame": summary_after.get("point_cloud_coordinate_frame") == "current_eef"
        and all(
            record.get("point_cloud_coordinate_frame") == "current_eef"
            for record in summary_after.get("episodes", [])
        ),
    }
    checks = {**core_checks, **metadata_checks}
    boolean_checks = {key: value for key, value in checks.items() if isinstance(value, bool)}
    report = {
        "dataset_root": str(root),
        "checks": checks,
        "all_checks_pass": all(boolean_checks.values()),
        "metadata_patched": metadata_was_patched,
        "info": {
            "total_episodes": info.get("total_episodes"),
            "total_frames": info.get("total_frames"),
            "total_tasks": info.get("total_tasks"),
            "visual_features": sorted(visual_features),
        },
        "image_counts": image_counts,
        "point_cloud": {
            "episodes": len(point_reports),
            "episode_contracts_pass": points_valid,
        },
        "sidecars": {
            "world_ee_pose_episodes": len(world_reports),
            "action_target_episodes": len(target_reports),
        },
        "parquet": {
            "data_file_count": parquet_report["data_file_count"],
            "episode_metadata_file_count": parquet_report["episode_metadata_file_count"],
            "array_sha256": parquet_report["array_sha256"],
            "reference_sha256": parquet_report["reference_sha256"],
        },
        "summary_sha256": sha256(summary_path),
        "point_meta_sha256": sha256(point_meta_path),
    }
    report_bytes = (json.dumps(report, indent=2) + "\n").encode()
    (root / "full_contract_audit_20260807.json").write_bytes(report_bytes)
    print(report_bytes.decode(), end="")
    print("AUDIT_SHA256", hashlib.sha256(report_bytes).hexdigest())
    return 0 if report["all_checks_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
