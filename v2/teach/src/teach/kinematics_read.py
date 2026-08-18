"""Optional read-only consumer of kinematics_fit's kinematics_calib.json:
if a fit already exists, teach/ can show/accept workspace-mm coordinates
alongside raw joint angles; if not (e.g. this is a fresh rig and teach/ is
being used specifically because there's no camera yet), teach/ still works
in pure joint-space using CAD nominal values for display only. Deliberately
reimplements the small amount of parsing needed rather than importing
kinematics_fit -- see homography_read.py in kinematics_fit for the twin of
this pattern."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from .kinematics import ArmParams

# Matches kinematics_fit.config.DEFAULT_PATH exactly -- a file-format
# convention shared across packages, not a code dependency.
KINEMATICS_CALIB_PATH = Path.home() / ".config" / "arm2r" / "kinematics_calib.json"


def load_params(path: Optional[Path] = None) -> ArmParams:
    """Returns the fitted ArmParams if kinematics_fit has produced one,
    otherwise CAD-nominal values (fine for teach/'s workspace-mm DISPLAY
    purposes -- taught points are always saved with their raw joint angles
    as the source of truth, so an inaccurate nominal here never corrupts
    what gets replayed, only what's shown alongside it)."""
    path = path if path is not None else KINEMATICS_CALIB_PATH
    if not path.exists():
        return ArmParams.nominal()
    with open(path) as f:
        d = json.load(f)
    fields = {"L1", "L2", "base_x", "base_y", "servo1_offset_deg", "servo2_offset_deg",
              "servo1_dir", "servo2_dir", "elbow_offset_mm"}
    return ArmParams(**{k: v for k, v in d.items() if k in fields})
