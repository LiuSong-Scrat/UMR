#!/usr/bin/env python
"""Standalone Open3D frontend for SmolVLA colored point-cloud shared memory.

This process intentionally does not import torch, LeRobot, or the caller's main
module. Keeping its OpenGL context outside the policy process avoids MuJoCo /
Open3D GLX context collisions and is safe for deploy scripts without a Python
``if __name__ == '__main__'`` guard.  The same transport is used for PointSeg
scores and UMI observation/trajectory debugging.
"""

from __future__ import annotations

import argparse
import os
import time
from contextlib import suppress
from multiprocessing import resource_tracker, shared_memory

import numpy as np


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-shm", required=True)
    parser.add_argument("--meta-shm", required=True)
    parser.add_argument("--max-points", required=True, type=int)
    parser.add_argument("--window-name", required=True)
    parser.add_argument("--width", required=True, type=int)
    parser.add_argument("--height", required=True, type=int)
    parser.add_argument("--parent-pid", required=True, type=int)
    return parser.parse_args()


def _parent_alive(parent_pid: int) -> bool:
    try:
        os.kill(int(parent_pid), 0)
        return True
    except OSError:
        return False


def main() -> None:
    args = _parse_args()
    data_shm = shared_memory.SharedMemory(name=args.data_shm, create=False)
    meta_shm = shared_memory.SharedMemory(name=args.meta_shm, create=False)
    # The parent owns unlinking. Prevent this attaching process's resource
    # tracker from racing the parent during normal shutdown.
    with suppress(Exception):
        resource_tracker.unregister(data_shm._name, "shared_memory")
    with suppress(Exception):
        resource_tracker.unregister(meta_shm._name, "shared_memory")

    data = np.ndarray((int(args.max_points), 6), dtype=np.float32, buffer=data_shm.buf)
    metadata = np.ndarray((2,), dtype=np.int64, buffer=meta_shm.buf)
    visualizer = None
    try:
        import open3d as o3d

        visualizer = o3d.visualization.Visualizer()
        created = visualizer.create_window(
            window_name=str(args.window_name),
            width=int(args.width),
            height=int(args.height),
            visible=True,
        )
        if created is False:
            raise RuntimeError("Open3D create_window returned False")
        point_cloud = o3d.geometry.PointCloud()
        geometry_added = False
        last_version = -1
        while _parent_alive(args.parent_pid):
            version_before = int(metadata[0])
            count = int(metadata[1])
            if count < 0:
                break
            if version_before % 2 == 0 and version_before != last_version and count > 0:
                count = min(count, int(args.max_points))
                # Copy once per model update so the parent can publish the next
                # frame without modifying memory used by Open3D.
                frame = data[:count].copy()
                version_after = int(metadata[0])
                if version_after == version_before:
                    point_cloud.points = o3d.utility.Vector3dVector(frame[:, :3].astype(np.float64))
                    point_cloud.colors = o3d.utility.Vector3dVector(frame[:, 3:6].astype(np.float64))
                    if not geometry_added:
                        visualizer.add_geometry(point_cloud, reset_bounding_box=True)
                        render_option = visualizer.get_render_option()
                        if render_option is not None:
                            render_option.point_size = 3.0
                        geometry_added = True
                    else:
                        visualizer.update_geometry(point_cloud)
                    last_version = version_after
            if visualizer.poll_events() is False:
                break
            visualizer.update_renderer()
            time.sleep(0.01)
    except Exception as exc:
        print(f"[warn] isolated point-cloud window stopped: {exc!r}", flush=True)
    finally:
        if visualizer is not None:
            with suppress(Exception):
                visualizer.destroy_window()
        del data
        del metadata
        data_shm.close()
        meta_shm.close()


if __name__ == "__main__":
    main()
