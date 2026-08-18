"""Persistence for kinematics_calib.json: the fitted ArmParams, the fixed
hardware constants that are never fit (servo1_dir/servo2_dir/elbow_offset_mm),
the end-effector tag id, motion tuning, and self-check spot-check poses.
"""

from __future__ import annotations

import json
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .kinematics import ArmParams

DEFAULT_PATH = Path.home() / ".config" / "arm2r" / "kinematics_calib.json"
HISTORY_DIRNAME = "kinematics_calib_history"


@dataclass
class FitReportSummary:
    n_points: int
    rms_error_mm: float
    max_error_mm: float


@dataclass
class KinematicsCalib:
    L1: float = 125.0
    L2: float = 95.0
    base_x: float = 100.0
    base_y: float = -45.0
    servo1_offset_deg: float = 23.08
    servo2_offset_deg: float = 0.0
    servo1_dir: int = 1
    servo2_dir: int = 1
    # Fixed, hand-measured (calipers/CAD) -- never written by fit_kinematics.
    elbow_offset_mm: float = 0.0
    fit_report: Optional[FitReportSummary] = None

    # End-effector AprilTag id, mounted on the arm -- used both during
    # calibration sampling and the Tier-2 self-check spot-check.
    ee_tag_id: int = 10

    # Self-check Tier 2: known (servo1_deg, servo2_deg) poses to visit and
    # compare FK-predicted vs. camera-measured end-effector position.
    spotcheck_poses: list[dict] = field(default_factory=list)
    arm_position_halt_mm: float = 3.0

    # Calibration data-collection tuning.
    grid_nx: int = 6
    grid_ny: int = 5
    grid_margin_mm: float = 15.0
    settle_s: float = 0.3

    # Motion tuning for the controller that drives calibration/spot-check moves.
    vmax_deg_s: float = 60.0
    amax_deg_s2: float = 120.0
    control_hz: float = 50.0

    def params(self) -> ArmParams:
        return ArmParams(L1=self.L1, L2=self.L2, base_x=self.base_x, base_y=self.base_y,
                          servo1_offset_deg=self.servo1_offset_deg,
                          servo2_offset_deg=self.servo2_offset_deg,
                          servo1_dir=self.servo1_dir, servo2_dir=self.servo2_dir,
                          elbow_offset_mm=self.elbow_offset_mm)

    def to_dict(self) -> dict:
        d = {
            "L1": self.L1, "L2": self.L2, "base_x": self.base_x, "base_y": self.base_y,
            "servo1_offset_deg": self.servo1_offset_deg, "servo2_offset_deg": self.servo2_offset_deg,
            "servo1_dir": self.servo1_dir, "servo2_dir": self.servo2_dir,
            "elbow_offset_mm": self.elbow_offset_mm,
            "fit_report": (self.fit_report.__dict__ if self.fit_report else None),
            "ee_tag_id": self.ee_tag_id,
            "spotcheck_poses": self.spotcheck_poses,
            "arm_position_halt_mm": self.arm_position_halt_mm,
            "grid_nx": self.grid_nx, "grid_ny": self.grid_ny, "grid_margin_mm": self.grid_margin_mm,
            "settle_s": self.settle_s,
            "vmax_deg_s": self.vmax_deg_s, "amax_deg_s2": self.amax_deg_s2,
            "control_hz": self.control_hz,
        }
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "KinematicsCalib":
        known = {f for f in cls.__dataclass_fields__}
        filtered = {k: v for k, v in d.items() if k in known and k != "fit_report"}
        obj = cls(**filtered)
        fr = d.get("fit_report")
        obj.fit_report = FitReportSummary(**fr) if fr else None
        return obj


def validate(calib: KinematicsCalib) -> None:
    if calib.L1 <= 0 or calib.L2 <= 0:
        raise ValueError("L1/L2 must be positive")
    if calib.servo1_dir not in (1, -1) or calib.servo2_dir not in (1, -1):
        raise ValueError("servo1_dir/servo2_dir must be +1 or -1")
    if not (-60.0 <= calib.elbow_offset_mm <= 60.0):
        raise ValueError(f"elbow_offset_mm={calib.elbow_offset_mm} outside sane bounds "
                          f"[-60, 60] -- re-check the caliper/CAD measurement")
    if calib.arm_position_halt_mm <= 0:
        raise ValueError("arm_position_halt_mm must be positive")
    for pose in calib.spotcheck_poses:
        if set(pose) != {"joint1", "joint2"}:
            raise ValueError(f"spotcheck pose {pose!r} must have exactly joint1/joint2 keys")


def load(path: Optional[Path] = None) -> KinematicsCalib:
    path = path if path is not None else DEFAULT_PATH
    if not path.exists():
        return KinematicsCalib()
    with open(path) as f:
        return KinematicsCalib.from_dict(json.load(f))


def save(calib: KinematicsCalib, path: Optional[Path] = None) -> None:
    path = path if path is not None else DEFAULT_PATH
    validate(calib)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        history_dir = path.parent / HISTORY_DIRNAME
        history_dir.mkdir(exist_ok=True)
        stamp = time.strftime("%Y%m%dT%H%M%S")
        shutil.copy(path, history_dir / f"kinematics_calib_{stamp}.json")
    with open(path, "w") as f:
        json.dump(calib.to_dict(), f, indent=2)
        f.write("\n")
