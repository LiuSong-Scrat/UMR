#!/usr/bin/env python3
"""Canonical REAP virtual-gripper geometry for every RLBench pipeline."""

from __future__ import annotations

import os
from pathlib import Path
import sys

import numpy as np


REPO_ROOT = Path(
    os.environ.get("LEROBOT_ROOT", str(Path(__file__).resolve().parents[3]))
).expanduser().resolve()
SONG_SCRIPT_ROOT = Path(
    os.environ.get(
        "SONG_SCRIPTS",
        str(REPO_ROOT / "benchmarks" / "song_real_libero" / "scripts"),
    )
).expanduser().resolve()
if str(SONG_SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SONG_SCRIPT_ROOT))

from libero_setting.libero_pointcloud_utils import create_gripper_points  # noqa: E402


LIBERO_GRIPPER_TEMPLATE = "reap"
LIBERO_GRIPPER_TEMPLATE_VERSION = (
    "libero_reap_four_box_physical_opening_geom0p06_rlbench_offset0p09_v4"
)

# The normalized control range follows LIBERO, while RLBench only reaches
# 0.08 m of this 0.10 m range.  Decoupling the opening scale from the fixed
# body width keeps the original REAP body but makes the finger gap physical.
LIBERO_GRIPPER_WIDTH_NORMALIZATION_MAX = 0.1
LIBERO_REAP_TEMPLATE_MAX_WIDTH = 0.06
LIBERO_REAP_OPENING_MAX_WIDTH = 0.1

# In the RLBench EEF frame the original LIBERO -0.06 m translation leaves the
# REAP fingertips 0.03 m ahead of the Panda tip.  -0.09 m aligns both tips.
# Keep the calibrated value separate from the selected evaluation geometry:
# evaluation may deliberately vary the virtual tool length, while 0.09 m stays
# the zero-offset contract relating Panda_tip to the model's virtual TCP.
#
# IMPORTANT: do not edit this constant for an evaluation ablation. Pass
# ``--gripper-len`` to official_eval.py instead. If this calibration
# moves together with the selected length, the virtual-TCP transform becomes
# identity and reproduces the old broken "move points only" behavior.
RLBENCH_REAP_ALIGNED_GRIPPER_LEN = 0.09

# Default selected geometry for collection and evaluation. This is deliberately
# not an alias assignment to the calibration constant: the two values have
# different semantics even though their canonical defaults are both 0.09 m.
LIBERO_REAP_GRIPPER_LEN = 0.09
LIBERO_REAP_LOCAL_OFFSET_M = (0.0, 0.0, -LIBERO_REAP_GRIPPER_LEN)
RLBENCH_PANDA_PHYSICAL_MAX_WIDTH = 0.08


def libero_reap_width_percent_from_physical(width_m):
    """Map a physical Panda finger gap in metres to the shared REAP control."""

    return np.clip(
        np.asarray(width_m, dtype=np.float32)
        / LIBERO_GRIPPER_WIDTH_NORMALIZATION_MAX,
        0.0,
        1.0,
    )


def rlbench_physical_width_from_open_fraction(open_fraction):
    """Convert RLBench's [0, 1] gripper state to its physical finger gap."""

    return np.clip(np.asarray(open_fraction, dtype=np.float32), 0.0, 1.0) * (
        RLBENCH_PANDA_PHYSICAL_MAX_WIDTH
    )


def create_rlbench_reap_points_from_physical_width(
    physical_width_m: float,
    pose: np.ndarray,
    count: int,
    rng: np.random.Generator,
    *,
    finger_length: float = 0.08,
) -> np.ndarray:
    """Create the canonical RLBench-aligned REAP cloud at a physical aperture."""

    return create_gripper_points(
        float(libero_reap_width_percent_from_physical(physical_width_m)),
        pose,
        int(count),
        rng,
        gripper_len=LIBERO_REAP_GRIPPER_LEN,
        max_width=LIBERO_REAP_TEMPLATE_MAX_WIDTH,
        opening_max_width=LIBERO_REAP_OPENING_MAX_WIDTH,
        finger_length=float(finger_length),
    )


def canonical_reap_metadata() -> dict:
    """Serializable geometry metadata shared by collection and exporters."""

    return {
        "gripper_template": LIBERO_GRIPPER_TEMPLATE,
        "gripper_template_version": LIBERO_GRIPPER_TEMPLATE_VERSION,
        "virtual_gripper_width_normalization_max_m": (
            LIBERO_GRIPPER_WIDTH_NORMALIZATION_MAX
        ),
        "virtual_gripper_geometry_max_width_m": LIBERO_REAP_TEMPLATE_MAX_WIDTH,
        "virtual_gripper_opening_max_width_m": LIBERO_REAP_OPENING_MAX_WIDTH,
        "virtual_gripper_local_offset_m": list(LIBERO_REAP_LOCAL_OFFSET_M),
        "virtual_tcp_aligned_gripper_len_m": RLBENCH_REAP_ALIGNED_GRIPPER_LEN,
    }
