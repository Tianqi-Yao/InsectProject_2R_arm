"""Read-only consumer of kinematics_fit's kinematics_calib.json -- see
teach/kinematics_read.py for the twin of this module. Unlike teach/, an
inaccurate fallback here directly affects scan accuracy (card_scan drives
IK-computed positions, it doesn't just display them next to hand-taught
joint angles), so callers should heed the warning this raises rather than
silently scanning with CAD-nominal guesses."""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Optional

from .kinematics import ArmParams

KINEMATICS_CALIB_PATH = Path.home() / ".config" / "arm2r" / "kinematics_calib.json"


def load_params(path: Optional[Path] = None) -> ArmParams:
    path = path if path is not None else KINEMATICS_CALIB_PATH
    if not path.exists():
        warnings.warn(
            f"{path} not found -- falling back to CAD-nominal kinematics. "
            f"Run `arm-kinfit run` first for an accurate scan.", stacklevel=2)
        return ArmParams.nominal()
    with open(path) as f:
        d = json.load(f)
    fields = {"L1", "L2", "base_x", "base_y", "servo1_offset_deg", "servo2_offset_deg",
              "servo1_dir", "servo2_dir", "elbow_offset_mm"}
    return ArmParams(**{k: v for k, v in d.items() if k in fields})
