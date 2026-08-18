"""Vision-fitted kinematic parameter calibration: nonlinear least squares
over (commanded servo angle -> camera-measured end-effector mm position)
samples, solving for L1, L2, base_x, base_y, servo1_offset_deg,
servo2_offset_deg. elbow_offset_mm/servo1_dir/servo2_dir are fixed inputs,
never part of the fit vector -- see kinematics.ArmParams' docstring for the
mathematical reason why."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.optimize import least_squares

from arm_hw_core.limits import JointLimits

from .kinematics import ArmParams, IKResult, fk_from_servo_angles, ik_solve

_PARAM_ORDER = ("L1", "L2", "base_x", "base_y", "servo1_offset_deg", "servo2_offset_deg")

DEFAULT_BOUNDS = dict(
    L1=(80.0, 170.0), L2=(60.0, 130.0),
    base_x=(50.0, 150.0), base_y=(-90.0, 0.0),
    # +-180 (not +-90): the STS3215's raw angle spans a full 0-360deg
    # circle, and depending on how a servo happens to be mounted, its zero
    # offset can land anywhere in that circle (e.g. ~179deg, near a servo's
    # physical center, is a normal offset on real hardware -- not a bug).
    servo1_offset_deg=(-180.0, 180.0), servo2_offset_deg=(-180.0, 180.0),
    # elbow_offset_mm is NOT fit -- this range only sanity-checks a
    # hand-entered config value (see config.validate).
    elbow_offset_mm=(-60.0, 60.0),
)


@dataclass
class CalibSample:
    servo1_deg: float
    servo2_deg: float
    x_mm: float
    y_mm: float


@dataclass
class FitReport:
    n_points: int
    rms_error_mm: float
    max_error_mm: float
    per_point_error_mm: list[float]
    params: ArmParams


def _residuals(vec: np.ndarray, samples: list[CalibSample],
                servo1_dir: int, servo2_dir: int, elbow_offset_mm: float) -> np.ndarray:
    p = ArmParams.from_vector(vec, servo1_dir=servo1_dir, servo2_dir=servo2_dir,
                               elbow_offset_mm=elbow_offset_mm)
    out = np.empty(2 * len(samples))
    for i, s in enumerate(samples):
        wx, wy = fk_from_servo_angles(p, s.servo1_deg, s.servo2_deg)
        out[2 * i] = wx - s.x_mm
        out[2 * i + 1] = wy - s.y_mm
    return out


def fit_kinematics(samples: list[CalibSample], x0: Optional[ArmParams] = None,
                    bounds: Optional[dict] = None) -> FitReport:
    """Jointly solve L1, L2, base position, and servo offsets from a set of
    (measured servo angle pair -> camera-measured mm position) samples.

    6 unknowns => 3 samples is the bare mathematical minimum, but that
    leaves zero residual degrees of freedom (no way to judge fit quality).
    Use >=6, ideally 15-30, spread per generate_calibration_targets's
    placement guidance."""
    if len(samples) < 6:
        raise ValueError(f"need >=6 calibration samples for a well-posed fit, got {len(samples)}")

    x0 = x0 or ArmParams.nominal()
    bounds = bounds or DEFAULT_BOUNDS
    lower = [bounds[k][0] for k in _PARAM_ORDER]
    upper = [bounds[k][1] for k in _PARAM_ORDER]

    result = least_squares(_residuals, x0=x0.as_vector(),
                            args=(samples, x0.servo1_dir, x0.servo2_dir, x0.elbow_offset_mm),
                            bounds=(lower, upper), loss="soft_l1", f_scale=1.0)

    fitted = ArmParams.from_vector(result.x, servo1_dir=x0.servo1_dir, servo2_dir=x0.servo2_dir,
                                    elbow_offset_mm=x0.elbow_offset_mm)
    errs = [math.hypot(result.fun[2 * i], result.fun[2 * i + 1]) for i in range(len(samples))]
    return FitReport(
        n_points=len(samples),
        rms_error_mm=float(np.sqrt(np.mean(np.square(errs)))),
        max_error_mm=float(max(errs)),
        per_point_error_mm=errs,
        params=fitted,
    )


def generate_calibration_targets(params: Optional[ArmParams] = None,
                                  width_mm: float = 200.0, height_mm: float = 150.0,
                                  nx: int = 6, ny: int = 5, margin_mm: float = 15.0,
                                  seed: Optional[int] = None,
                                  joint_limits: Optional[JointLimits] = None) -> list[IKResult]:
    """Generate a grid of workspace targets for automatic calibration data
    collection, filtered to reachable/in-limits servo poses and shuffled.

    Point placement matters more than point count for identifiability:
    - A wide angular spread (targets near both left/right edges) is needed
      to decouple base_x/base_y (translation) from servo1_offset_deg (an
      overall rotation) -- in a narrow angular slice the two look alike.
    - A spread of near/far points is needed to decouple L1 from L2, since
      that requires seeing a range of elbow (theta2) angles.
    - Shuffling avoids a monotonic scan coupling servo backlash/direction-
      dependent error with position."""
    params = params or ArmParams.nominal()
    xs = np.linspace(margin_mm, width_mm - margin_mm, nx)
    ys = np.linspace(margin_mm, height_mm - margin_mm, ny)
    targets = []
    for x in xs:
        for y in ys:
            r = ik_solve(params, float(x), float(y), joint_limits=joint_limits)
            if r.reachable:
                targets.append(r)
    random.Random(seed).shuffle(targets)
    return targets
