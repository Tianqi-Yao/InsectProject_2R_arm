"""Core kinematics, vision-fusion, calibration, and self-check logic for the 2R arm.

This is the one file meant to be read end-to-end: everything here is either
geometry/math or a decision (halt vs. continue, accept vs. reject a fit).
Hardware I/O (servo bus protocol, camera capture, AprilTag detection) lives in
arm_hardware.py and is treated as a black box behind a handful of plain
methods (see the `hw` parameter of run_selfcheck, and how main.py uses it).
"""

from __future__ import annotations

import json
import logging
import math
import random
import shutil
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from scipy.optimize import least_squares

logger = logging.getLogger("arm_core")

THIS_DIR = Path(__file__).parent
DEFAULT_CALIB_PATH = THIS_DIR / "calib.json"
CALIB_HISTORY_DIR = THIS_DIR / "calib_history"
ALARMS_LOG_PATH = THIS_DIR / "alarms.log"


# ── 1. Kinematics ─────────────────────────────────────────────────────────
#
# Same 2R planar-arm geometry as the old firmware/kinematics.h, turned into
# a parameterized model whose constants (L1, L2, base position, servo
# offsets) are no longer hardcoded but fitted from vision data (see
# section 3).
#
# NOTE on a parameter that was deliberately *not* added: a "base mounting
# rotation" (the arm's shoulder axis not being aligned with the paper's
# x-axis) looks like a natural 7th unknown, but it is mathematically
# indistinguishable from servo1_offset_deg when only end-effector *position*
# is measured (no orientation sensing): rotating the whole base by delta
# has the exact same effect on (ex, ey) as shifting theta1 by delta for
# every sample, because both angles feeding the FK (theta1 and theta1+theta2)
# shift by the same delta either way. A synthetic-data test confirmed this
# empirically (least_squares happily converges, but splits the true rotation
# arbitrarily between the two parameters instead of recovering either one).
# So: any base-mounting misalignment is simply absorbed into
# servo1_offset_deg automatically -- no separate parameter needed.

@dataclass
class ArmParams:
    L1: float
    L2: float
    base_x: float
    base_y: float
    servo1_offset_deg: float
    servo2_offset_deg: float
    # +1 or -1: which way a joint's raw servo angle increases relative to
    # our math convention (theta measured CCW). This is a fixed fact about
    # how a servo happens to be mounted/wired -- not something to fit
    # numerically (a sign flip is a reflection, no amount of offset/L1/L2
    # tuning can reproduce it) -- so it's excluded from as_vector/from_vector
    # and never touched by fit_kinematics.
    servo1_dir: int = 1
    servo2_dir: int = 1

    @classmethod
    def nominal(cls) -> "ArmParams":
        """CAD-theoretical starting values, taken from the old firmware/kinematics.h.
        Used as the least-squares initial guess, not as a value to trust."""
        return cls(L1=125.0, L2=95.0, base_x=100.0, base_y=-45.0,
                   servo1_offset_deg=23.08, servo2_offset_deg=0.0)

    def as_vector(self) -> list[float]:
        return [self.L1, self.L2, self.base_x, self.base_y,
                self.servo1_offset_deg, self.servo2_offset_deg]

    @classmethod
    def from_vector(cls, vec, servo1_dir: int = 1, servo2_dir: int = 1) -> "ArmParams":
        return cls(*vec, servo1_dir=servo1_dir, servo2_dir=servo2_dir)


@dataclass
class IKResult:
    theta1_deg: float = 0.0
    theta2_deg: float = 0.0
    servo1_deg: float = 0.0
    servo2_deg: float = 0.0
    reachable: bool = False


def ik_solve(p: ArmParams, x_ws: float, y_ws: float) -> IKResult:
    """Inverse kinematics: workspace (x, y) in mm -> joint/servo angles.
    Direct generalization of firmware/kinematics.h's ik_solve()."""
    ax, ay = x_ws - p.base_x, y_ws - p.base_y

    d2 = ax * ax + ay * ay
    c2 = (d2 - p.L1 ** 2 - p.L2 ** 2) / (2.0 * p.L1 * p.L2)
    if c2 < -1.0 or c2 > 1.0:
        return IKResult(reachable=False)

    s2 = math.sqrt(1.0 - c2 * c2)  # elbow-up: theta2 > 0
    theta2 = math.degrees(math.atan2(s2, c2))
    alpha = math.degrees(math.atan2(ay, ax))
    beta = math.degrees(math.atan2(p.L2 * s2, p.L1 + p.L2 * c2))
    theta1 = alpha - beta

    servo1 = p.servo1_offset_deg + p.servo1_dir * theta1
    servo2 = p.servo2_offset_deg + p.servo2_dir * theta2
    return IKResult(theta1, theta2, servo1, servo2, reachable=True)


def fk_from_servo_angles(p: ArmParams, servo1_deg: float, servo2_deg: float) -> tuple[float, float]:
    """Forward kinematics from *measured* servo angles -> workspace (x, y) mm.

    This is the one function shared by both the kinematic-parameter fit
    (fit_kinematics) and the boot self-check (run_selfcheck): both boil down
    to "take a real encoder angle pair, predict where the end effector
    should be, compare against what the camera measured."
    """
    theta1 = math.radians(p.servo1_dir * (servo1_deg - p.servo1_offset_deg))
    theta2 = math.radians(p.servo2_dir * (servo2_deg - p.servo2_offset_deg))
    ex = p.L1 * math.cos(theta1) + p.L2 * math.cos(theta1 + theta2)
    ey = p.L1 * math.sin(theta1) + p.L2 * math.sin(theta1 + theta2)
    return p.base_x + ex, p.base_y + ey


def fk_joint_positions(p: ArmParams, servo1_deg: float, servo2_deg: float
                        ) -> tuple[tuple[float, float], tuple[float, float]]:
    """Like fk_from_servo_angles, but also returns the elbow position --
    for drawing the two links separately (e.g. manual_test/gui.py), not
    needed by the fit/self-check, which only care about the end effector."""
    theta1 = math.radians(p.servo1_dir * (servo1_deg - p.servo1_offset_deg))
    theta2 = math.radians(p.servo2_dir * (servo2_deg - p.servo2_offset_deg))
    ex1 = p.L1 * math.cos(theta1)
    ey1 = p.L1 * math.sin(theta1)
    ex2 = ex1 + p.L2 * math.cos(theta1 + theta2)
    ey2 = ey1 + p.L2 * math.sin(theta1 + theta2)
    elbow = (p.base_x + ex1, p.base_y + ey1)
    ee = (p.base_x + ex2, p.base_y + ey2)
    return elbow, ee


# ── 2. Homography (pixel <-> workspace mm) ────────────────────────────────
#
# Four AprilTags are stuck at the four corners of the 200x150mm work sheet,
# at known world coordinates. A homography fit from their detected pixel
# centers to those known coordinates gives an exact pixel->mm mapping for
# any point lying on the *same* physical plane -- exact regardless of
# camera tilt, as long as the tilt stays within the fixed-focus lens's
# depth of field. The end effector's tracking tag must be physically held
# at (approximately) that same plane height, or camera tilt will introduce
# a parallax error the homography can't correct for.

def compute_homography(pixel_points: list[tuple[float, float]],
                        world_points: list[tuple[float, float]]) -> tuple[np.ndarray, float]:
    """Fit a pixel->mm homography from >=4 known correspondences.
    Returns (H, reprojection_rms_px) so callers can sanity-check fit quality."""
    if len(pixel_points) < 4:
        raise ValueError(f"homography needs >=4 point pairs, got {len(pixel_points)}")
    img_pts = np.array(pixel_points, dtype=np.float64)
    world_pts = np.array(world_points, dtype=np.float64)
    H, _ = cv2.findHomography(img_pts, world_pts, method=0)
    if H is None:
        raise ValueError("cv2.findHomography failed to converge")
    reproj = cv2.perspectiveTransform(img_pts.reshape(-1, 1, 2), H).reshape(-1, 2)
    rms_px = float(np.sqrt(np.mean(np.sum((reproj - world_pts) ** 2, axis=1))))
    return H, rms_px


def apply_homography(H: np.ndarray, pixel_xy: tuple[float, float]) -> tuple[float, float]:
    """Map one pixel coordinate to a workspace mm coordinate through H."""
    pt = np.array([[pixel_xy]], dtype=np.float64)
    out = cv2.perspectiveTransform(pt, H)[0, 0]
    return float(out[0]), float(out[1])


def homography_drift_mm(H_prev: np.ndarray,
                         measured_pixels: list[tuple[float, float]],
                         known_world: list[tuple[float, float]]) -> float:
    """How much has the camera/workspace geometry moved since H_prev was fit?

    Re-interpret *this boot's* freshly measured corner-tag pixel positions
    through *last boot's* homography and compare against the corner tags'
    known (fixed) world coordinates. If nothing moved, H_prev still maps
    today's pixels to the right spot and drift is ~0mm. Any real
    displacement of the camera or the work sheet shows up directly in mm --
    no separate pixel<->mm conversion step needed. Takes the worst-case
    corner (not the average) so a single badly-shifted corner isn't diluted.
    """
    worst = 0.0
    for px, world in zip(measured_pixels, known_world):
        predicted = apply_homography(H_prev, px)
        err = math.hypot(predicted[0] - world[0], predicted[1] - world[1])
        worst = max(worst, err)
    return worst


# ── 3. Kinematic parameter calibration (nonlinear least squares) ─────────

@dataclass
class CalibSample:
    servo1_deg: float
    servo2_deg: float
    x_mm: float
    y_mm: float


_PARAM_ORDER = ("L1", "L2", "base_x", "base_y",
                 "servo1_offset_deg", "servo2_offset_deg")

DEFAULT_BOUNDS = dict(
    L1=(80.0, 170.0), L2=(60.0, 130.0),
    base_x=(50.0, 150.0), base_y=(-90.0, 0.0),
    # +-180 (not +-90): the STS3215's raw angle spans a full 0-360 circle,
    # and depending on how a servo happens to be mounted, its zero offset
    # can land anywhere in that circle (e.g. ~179 deg, near a servo's
    # physical center, is a normal offset on real hardware -- not a bug).
    servo1_offset_deg=(-180.0, 180.0), servo2_offset_deg=(-180.0, 180.0),
)


def generate_calibration_targets(params: Optional[ArmParams] = None,
                                  width_mm: float = 200.0, height_mm: float = 150.0,
                                  nx: int = 6, ny: int = 5, margin_mm: float = 15.0,
                                  seed: Optional[int] = None) -> list[IKResult]:
    """Generate a grid of workspace targets for automatic calibration data
    collection, filtered to reachable/in-range servo poses and shuffled.

    Point placement matters more than point count for identifiability:
    - A wide angular spread (targets near both left/right edges) is needed
      to decouple base_x/base_y (translation) from servo1_offset_deg
      (an overall rotation) -- in a narrow angular slice the two look alike.
    - A spread of near/far points is needed to decouple L1 from L2, since
      that requires seeing a range of elbow (theta2) angles.
    - Shuffling avoids a monotonic scan coupling servo backlash/direction
      dependent error with position.
    """
    params = params or ArmParams.nominal()
    xs = np.linspace(margin_mm, width_mm - margin_mm, nx)
    ys = np.linspace(margin_mm, height_mm - margin_mm, ny)
    targets = []
    for x in xs:
        for y in ys:
            r = ik_solve(params, float(x), float(y))
            if r.reachable and 0.0 <= r.servo1_deg <= 180.0 and 0.0 <= r.servo2_deg <= 180.0:
                targets.append(r)
    random.Random(seed).shuffle(targets)
    return targets


def _residuals(vec: np.ndarray, samples: list[CalibSample],
                servo1_dir: int, servo2_dir: int) -> np.ndarray:
    p = ArmParams.from_vector(vec, servo1_dir=servo1_dir, servo2_dir=servo2_dir)
    out = np.empty(2 * len(samples))
    for i, s in enumerate(samples):
        wx, wy = fk_from_servo_angles(p, s.servo1_deg, s.servo2_deg)
        out[2 * i] = wx - s.x_mm
        out[2 * i + 1] = wy - s.y_mm
    return out


@dataclass
class FitReport:
    n_points: int
    rms_error_mm: float
    max_error_mm: float
    per_point_error_mm: list[float]
    params: ArmParams


def fit_kinematics(samples: list[CalibSample],
                    x0: Optional[ArmParams] = None,
                    bounds: Optional[dict] = None) -> FitReport:
    """Jointly solve L1, L2, base position, and servo offsets from a set of
    (measured servo angle pair -> camera-measured mm position) samples.

    6 unknowns => 3 samples is the bare mathematical minimum, but that
    leaves zero residual degrees of freedom (no way to judge fit quality or
    reject a bad sample). Use >=6, ideally 15-30, spread per
    generate_calibration_targets's placement guidance.
    """
    if len(samples) < 6:
        raise ValueError(f"need >=6 calibration samples for a well-posed fit, got {len(samples)}")

    x0 = x0 or ArmParams.nominal()
    bounds = bounds or DEFAULT_BOUNDS
    lower = [bounds[k][0] for k in _PARAM_ORDER]
    upper = [bounds[k][1] for k in _PARAM_ORDER]

    # servo1_dir/servo2_dir are fixed hardware facts, not fit -- carried
    # through from x0 into every trial ArmParams the optimizer builds.
    result = least_squares(_residuals, x0=x0.as_vector(),
                            args=(samples, x0.servo1_dir, x0.servo2_dir),
                            bounds=(lower, upper), loss="soft_l1", f_scale=1.0)

    fitted = ArmParams.from_vector(result.x, servo1_dir=x0.servo1_dir, servo2_dir=x0.servo2_dir)
    errs = [math.hypot(result.fun[2 * i], result.fun[2 * i + 1]) for i in range(len(samples))]
    return FitReport(
        n_points=len(samples),
        rms_error_mm=float(np.sqrt(np.mean(np.square(errs)))),
        max_error_mm=float(max(errs)),
        per_point_error_mm=errs,
        params=fitted,
    )


# ── 4. calib.json persistence ──────────────────────────────────────────

def _default_calib() -> dict:
    p = ArmParams.nominal()
    return {
        "status": "OK",
        "workspace": {
            "width_mm": 200.0, "height_mm": 150.0,
            "corner_tag_ids": {"tl": 0, "tr": 1, "br": 2, "bl": 3},
            "corner_world_mm": {"0": [0, 150], "1": [200, 150], "2": [200, 0], "3": [0, 0]},
            "ee_tag_id": 10,
        },
        "homography": {"H": None, "computed_at": None},
        "kinematics": {**asdict(p), "fit_report": None},
        "thresholds": {"homography_drift_halt_mm": 3.0, "arm_position_halt_mm": 3.0},
        "spotcheck_poses": [{"joint1": 68.0, "joint2": 116.0}],
    }


def _validate_calib(calib: dict) -> None:
    """Reject a malformed/out-of-range calib file instead of silently
    falling back to defaults: a bad offset could drive a servo into its
    mechanical limit on the next move command."""
    k = calib.get("kinematics")
    if not k:
        raise ValueError("calib.json missing 'kinematics' section")
    for key, (lo, hi) in DEFAULT_BOUNDS.items():
        val = k.get(key)
        if val is None or not (lo - 1e-6 <= val <= hi + 1e-6):
            raise ValueError(f"calib.json kinematics.{key}={val} out of expected range [{lo},{hi}]")
    for dir_key in ("servo1_dir", "servo2_dir"):
        val = k.get(dir_key, 1)
        if val not in (1, -1):
            raise ValueError(f"calib.json kinematics.{dir_key}={val} must be 1 or -1")
    for name in ("homography_drift_halt_mm", "arm_position_halt_mm"):
        if calib.get("thresholds", {}).get(name) is None:
            raise ValueError(f"calib.json missing thresholds.{name}")


def load_calib(path: Optional[Path] = None) -> dict:
    # Resolved lazily (rather than bound as a default arg) so tests can
    # monkeypatch DEFAULT_CALIB_PATH without touching the real repo file.
    path = path or DEFAULT_CALIB_PATH
    if not path.exists():
        logger.warning("no calib.json at %s, using nominal CAD defaults", path)
        return _default_calib()
    with open(path) as f:
        calib = json.load(f)
    _validate_calib(calib)
    return calib


def save_calib(calib: dict, path: Optional[Path] = None) -> None:
    """Validate, snapshot the previous file (so a bad write can be rolled
    back by hand), then persist."""
    path = path or DEFAULT_CALIB_PATH
    _validate_calib(calib)
    CALIB_HISTORY_DIR.mkdir(exist_ok=True)
    if path.exists():
        stamp = time.strftime("%Y%m%dT%H%M%S")
        shutil.copy(path, CALIB_HISTORY_DIR / f"calib_{stamp}.json")
    with open(path, "w") as f:
        json.dump(calib, f, indent=2)


def calib_arm_params(calib: dict) -> ArmParams:
    k = calib["kinematics"]
    return ArmParams(L1=k["L1"], L2=k["L2"], base_x=k["base_x"], base_y=k["base_y"],
                      servo1_offset_deg=k["servo1_offset_deg"], servo2_offset_deg=k["servo2_offset_deg"],
                      servo1_dir=k.get("servo1_dir", 1), servo2_dir=k.get("servo2_dir", 1))


# ── 5. Alarm hook (placeholder) ─────────────────────────────────────────

def raise_alarm(message: str, severity: str = "warning", **context) -> None:
    """Placeholder hook: for now this only logs to disk + stderr.
    TODO: wire up a real notification channel (push/email/mqtt/...) later."""
    record = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "severity": severity,
              "message": message, **context}
    with open(ALARMS_LOG_PATH, "a") as f:
        f.write(json.dumps(record) + "\n")
    logger.error("[ALARM:%s] %s %s", severity, message, context)


# ── 6. Boot self-check ──────────────────────────────────────────────────

@dataclass
class SelfCheckResult:
    ok: bool
    reason: Optional[str] = None
    homography_drift_mm: Optional[float] = None
    spotcheck_errors_mm: list = field(default_factory=list)


def run_selfcheck(hw, calib: dict) -> SelfCheckResult:
    """Two-tier boot health check with a single halt threshold per tier:
    below threshold -> self-heal (adopt the freshly measured homography,
    keep working, just log it); at/above threshold -> halt and raise an
    alarm, requiring a manual recalibration before the system runs again.

    `hw` is the black-box hardware handle from arm_hardware.py, expected to
    expose: hw.camera.capture_gray(), hw.detector.detect(frame),
    hw.servos.move_and_wait(targets_deg), hw.servos.get_present_deg(joint).
    """
    ws = calib["workspace"]
    thresholds = calib["thresholds"]

    # ── Tier 1: camera / homography drift ──
    frame = hw.camera.capture_gray()
    detections = hw.detector.detect(frame)

    corner_ids = ws["corner_tag_ids"]
    corner_world = ws["corner_world_mm"]
    missing = [name for name, tid in corner_ids.items() if tid not in detections]
    if missing:
        raise_alarm(f"corner tag(s) missing at boot: {missing}", severity="critical")
        return SelfCheckResult(ok=False, reason=f"missing_corner_tags:{missing}")

    measured_px = [tuple(detections[tid].center) for tid in corner_ids.values()]
    known_world = [tuple(corner_world[str(tid)]) for tid in corner_ids.values()]

    H_prev = calib["homography"].get("H")
    drift_mm = None
    if H_prev is not None:
        drift_mm = homography_drift_mm(np.array(H_prev), measured_px, known_world)
        if drift_mm >= thresholds["homography_drift_halt_mm"]:
            raise_alarm(f"homography drift {drift_mm:.2f}mm >= threshold", severity="critical",
                        drift_mm=drift_mm)
            return SelfCheckResult(ok=False, reason="homography_drift", homography_drift_mm=drift_mm)

    H_new, reproj_rms_px = compute_homography(measured_px, known_world)
    calib["homography"] = {"H": H_new.tolist(),
                            "computed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                            "reproj_rms_px": reproj_rms_px}
    logger.info("homography ok, drift_mm=%s reproj_rms_px=%.2f", drift_mm, reproj_rms_px)

    # ── Tier 2: arm position spot-check (1-2 known poses) ──
    params = calib_arm_params(calib)
    spotcheck_errors: list[float] = []
    for pose in calib["spotcheck_poses"]:
        hw.servos.move_and_wait(pose)
        s1 = hw.servos.get_present_deg("joint1")
        s2 = hw.servos.get_present_deg("joint2")
        predicted = fk_from_servo_angles(params, s1, s2)

        frame = hw.camera.capture_gray()
        detections = hw.detector.detect(frame)
        ee = detections.get(ws["ee_tag_id"])
        if ee is None:
            raise_alarm("end-effector tag not visible during spot-check", severity="warning", pose=pose)
            continue

        measured = apply_homography(H_new, tuple(ee.center))
        err_mm = math.hypot(predicted[0] - measured[0], predicted[1] - measured[1])
        spotcheck_errors.append(err_mm)

        if err_mm >= thresholds["arm_position_halt_mm"]:
            raise_alarm(f"arm position error {err_mm:.2f}mm >= threshold", severity="critical",
                        pose=pose, error_mm=err_mm)
            return SelfCheckResult(ok=False, reason="arm_position_drift",
                                    homography_drift_mm=drift_mm, spotcheck_errors_mm=spotcheck_errors)

    calib["status"] = "OK"
    save_calib(calib)
    return SelfCheckResult(ok=True, homography_drift_mm=drift_mm, spotcheck_errors_mm=spotcheck_errors)
