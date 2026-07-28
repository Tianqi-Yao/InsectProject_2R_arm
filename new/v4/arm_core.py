"""Core kinematics + calib persistence for the 2R arm (v3).

This is the one file meant to be read end-to-end for the arm's own
geometry: everything here is math or a config-validation decision. Hardware
I/O (servo bus protocol, camera capture) lives in arm_hardware.py and is
treated as a black box. Motion/trajectory algorithms live in
motion_planning/ (see jog_controller.py for how a planner is selected and
driven). Rectangle-fitting, photo-grid generation, and straight-line
interpolation -- the actual new logic v3 needed -- live in path_core.py,
not here: this file only knows about the arm itself, not about scanning.

v3 intentionally has NO camera-based calibration (no AprilTag, no
homography, no least-squares fit of L1/L2/base position from vision data --
see ../arm_core.py for that older approach). The reason this is fine: a
rectangle taught by jogging to its four corners and read back through FK is
converted to scan targets by IK using the exact SAME ArmParams -- ik_solve
and fk_from_servo_angles are exact inverses of each other (see below), so
whatever base_x/base_y/servo_offset_deg happen to be, they cancel out
between the "read a corner" step and the "command a scan point" step. Only
L1/L2 (the mm-scale of the whole coordinate frame, which is what makes a
photo's real-world spacing/overlap correct) and servo1_dir/servo2_dir
(which way each servo happens to be wired, a fixed mechanical fact) need to
be right, and both can come straight from CAD/calipers -- see
ArmParams.nominal().
"""

from __future__ import annotations

import json
import logging
import math
import shutil
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger("arm_core")

THIS_DIR = Path(__file__).parent
DEFAULT_CALIB_PATH = THIS_DIR / "calib.json"
CALIB_HISTORY_DIR = THIS_DIR / "calib_history"


# ── 1. Kinematics ─────────────────────────────────────────────────────────
#
# Same 2R planar-arm geometry as ../arm_core.py -- see that file's ArmParams
# docstring for the full reasoning on why servo1_dir/servo2_dir/
# elbow_offset_mm are fixed mechanical facts (not fittable from vision data)
# and why a "base mounting rotation" parameter is deliberately not modeled
# (it's indistinguishable from servo1_offset_deg). None of that reasoning
# changes here -- v3 just never runs the vision-based fit that the
# reasoning was originally protecting.

@dataclass
class ArmParams:
    L1: float
    L2: float
    base_x: float
    base_y: float
    servo1_offset_deg: float
    servo2_offset_deg: float
    servo1_dir: int = 1
    servo2_dir: int = 1
    elbow_offset_mm: float = 0.0

    @classmethod
    def nominal(cls) -> "ArmParams":
        """CAD/tape-measure starting values -- for v3 these ARE the values
        to trust (no vision fit refines them further): measure L1/L2 with
        calipers, note which way each servo is wired (servo1_dir/
        servo2_dir), and leave base_x/base_y/servo_offset_deg at any
        convenient constant (they only fix where the arbitrary (x, y)
        frame sits, not whether a taught rectangle comes out square --
        see this module's docstring)."""
        return cls(L1=125.0, L2=95.0, base_x=100.0, base_y=-45.0,
                   servo1_offset_deg=23.08, servo2_offset_deg=0.0)


@dataclass
class IKResult:
    theta1_deg: float = 0.0
    theta2_deg: float = 0.0
    servo1_deg: float = 0.0
    servo2_deg: float = 0.0
    reachable: bool = False


def _normalize_deg(angle_deg: float) -> float:
    """Wrap to [0, 360) -- matches arm_hardware.Servos' own tick conversion
    (`% TICKS_PER_REV`), so a limit check here agrees with what actually
    gets sent to hardware."""
    return angle_deg % 360.0


def wrap_angle_near(target_deg: float, reference_deg: float) -> float:
    """The angle congruent to target_deg (mod 360) that's closest to
    reference_deg -- used wherever a "distance to travel" between two
    angles is computed (motion_planning/trapezoidal.py, jog_controller.py's
    scan waypoint sequencing), since ik_solve()'s theta1/theta2 come from
    atan2 and have no reason to land near the arm's current angle."""
    return reference_deg + ((target_deg - reference_deg + 180.0) % 360.0 - 180.0)


def rotate_vector(dx: float, dy: float, rotation_deg: float) -> tuple[float, float]:
    """Rotates a 2D vector (dx, dy) by rotation_deg (degrees, CCW) about
    the origin."""
    theta = math.radians(rotation_deg)
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    return (dx * cos_t - dy * sin_t, dx * sin_t + dy * cos_t)


def _point_in_polygon(x: float, y: float, polygon: list) -> bool:
    """Winding-number point-in-polygon test -- see ../arm_core.py's version
    for the full rationale on why winding number (not even-odd) is correct
    for a hand-traced boundary."""
    winding = 0
    x1, y1 = polygon[-1]
    for x2, y2 in polygon:
        if y1 <= y:
            if y2 > y and (x2 - x1) * (y - y1) - (x - x1) * (y2 - y1) > 0:
                winding += 1
        else:
            if y2 <= y and (x2 - x1) * (y - y1) - (x - x1) * (y2 - y1) < 0:
                winding -= 1
        x1, y1 = x2, y2
    return winding != 0


def within_joint_limits(servo1_deg: float, servo2_deg: float,
                         joint_limits: Optional[dict]) -> bool:
    """joint_limits is None (not configured) -> unrestricted. Otherwise
    {"joint1": (lo, hi), "joint2": (lo, hi), "coupled_boundary": [...]} in
    raw servo-degree space -- see ../arm_core.py's version for the full
    docstring on the coupled_boundary polygon and the non-wrapping-range
    assumption."""
    if joint_limits is None:
        return True
    s1n, s2n = _normalize_deg(servo1_deg), _normalize_deg(servo2_deg)
    for name, angle in (("joint1", s1n), ("joint2", s2n)):
        lo, hi = joint_limits[name]
        if not (lo <= angle <= hi):
            return False
    boundary = joint_limits.get("coupled_boundary")
    if boundary:
        if not _point_in_polygon(s1n, s2n, boundary):
            return False
    return True


def _elbow_reach_and_angle(p: ArmParams) -> tuple[float, float]:
    """See ArmParams.elbow_offset_mm: the servo2 mounting offset turns the
    joint1-to-joint2 "link" into a right triangle. Returns
    (reach_mm, angle_offset_deg); angle_offset_deg is exactly 0 when
    elbow_offset_mm is 0."""
    reach = math.hypot(p.L1, p.elbow_offset_mm)
    angle_offset_deg = math.degrees(math.atan2(p.elbow_offset_mm, p.L1))
    return reach, angle_offset_deg


def ik_solve(p: ArmParams, x_ws: float, y_ws: float,
             joint_limits: Optional[dict] = None) -> IKResult:
    """Inverse kinematics: workspace (x, y) in mm -> joint/servo angles.
    Exact inverse of fk_from_servo_angles (see this module's docstring for
    why that exactness is what lets v3 skip vision-based calibration
    entirely)."""
    ax, ay = x_ws - p.base_x, y_ws - p.base_y
    reach, angle_offset_deg = _elbow_reach_and_angle(p)

    d2 = ax * ax + ay * ay
    c2 = (d2 - reach ** 2 - p.L2 ** 2) / (2.0 * reach * p.L2)
    if c2 < -1.0 or c2 > 1.0:
        return IKResult(reachable=False)

    s2 = math.sqrt(1.0 - c2 * c2)  # elbow-up: theta2_std > 0
    theta2_std = math.degrees(math.atan2(s2, c2))
    alpha = math.degrees(math.atan2(ay, ax))
    beta = math.degrees(math.atan2(p.L2 * s2, reach + p.L2 * c2))
    theta1_std = alpha - beta

    theta1 = theta1_std - angle_offset_deg
    theta2 = theta2_std + angle_offset_deg

    servo1 = p.servo1_offset_deg + p.servo1_dir * theta1
    servo2 = p.servo2_offset_deg + p.servo2_dir * theta2
    if not within_joint_limits(servo1, servo2, joint_limits):
        return IKResult(reachable=False)
    return IKResult(theta1, theta2, servo1, servo2, reachable=True)


def fk_from_servo_angles(p: ArmParams, servo1_deg: float, servo2_deg: float) -> tuple[float, float]:
    """Forward kinematics from *measured* servo angles -> workspace (x, y)
    mm. This is what converts a hand-taught corner (a raw encoder reading)
    into the (x, y) frame the rectangle fit / photo grid operate in."""
    theta1 = math.radians(p.servo1_dir * (servo1_deg - p.servo1_offset_deg))
    theta2 = math.radians(p.servo2_dir * (servo2_deg - p.servo2_offset_deg))
    reach, angle_offset_deg = _elbow_reach_and_angle(p)
    angle_offset = math.radians(angle_offset_deg)
    ex = reach * math.cos(theta1 + angle_offset) + p.L2 * math.cos(theta1 + theta2)
    ey = reach * math.sin(theta1 + angle_offset) + p.L2 * math.sin(theta1 + theta2)
    return p.base_x + ex, p.base_y + ey


def fk_joint_positions(p: ArmParams, servo1_deg: float, servo2_deg: float
                        ) -> tuple[tuple[float, float], tuple[float, float]]:
    """Like fk_from_servo_angles, but also returns the elbow position --
    for drawing the two links separately (teach_gui.py/scan_gui.py)."""
    theta1 = math.radians(p.servo1_dir * (servo1_deg - p.servo1_offset_deg))
    theta2 = math.radians(p.servo2_dir * (servo2_deg - p.servo2_offset_deg))
    reach, angle_offset_deg = _elbow_reach_and_angle(p)
    angle_offset = math.radians(angle_offset_deg)
    ex1 = reach * math.cos(theta1 + angle_offset)
    ey1 = reach * math.sin(theta1 + angle_offset)
    ex2 = ex1 + p.L2 * math.cos(theta1 + theta2)
    ey2 = ey1 + p.L2 * math.sin(theta1 + theta2)
    elbow = (p.base_x + ex1, p.base_y + ey1)
    ee = (p.base_x + ex2, p.base_y + ey2)
    return elbow, ee


def servo2_offset_from_known_elbow_angle(servo2_deg: float, servo2_dir: int,
                                          elbow_angle_deg: float, flip: bool = False) -> float:
    """Solve servo2_offset_deg directly from one physical reference pose --
    no camera needed. Fold the arm by hand so the L1-L2 angle, as read by a
    protractor/set-square at the elbow, is `elbow_angle_deg` (180=fully
    extended, 0=fully folded). `servo2_deg` is the real encoder reading at
    that exact pose. `flip`: try True if the result looks mirrored."""
    theta2_target = 180.0 - elbow_angle_deg
    if flip:
        theta2_target = -theta2_target
    return servo2_deg - servo2_dir * theta2_target


# ── 2. calib.json persistence ──────────────────────────────────────────
#
# calib.json is the single source of truth for "everything that describes
# this particular physical rig": the (CAD/caliper-measured) kinematic
# parameters, which bus IDs/port the two servos are on, motion-planning
# tuning knobs, the taught scan rectangle, and the photo-grid settings.

DEFAULT_BOUNDS = dict(
    L1=(50.0, 250.0), L2=(50.0, 250.0),
    base_x=(-500.0, 500.0), base_y=(-500.0, 500.0),
    servo1_offset_deg=(-180.0, 180.0), servo2_offset_deg=(-180.0, 180.0),
    elbow_offset_mm=(-60.0, 60.0),
)


@dataclass
class HardwareConfig:
    servo_port: str = "/dev/cu.usbserial-0001"
    joint_ids: dict = field(default_factory=lambda: {"joint1": 1, "joint2": 2})


@dataclass
class MotionConfig:
    planner_name: str = "trapezoidal"
    # Jogging: single-target moves during teach-in.
    jog_vmax_deg_s: float = 60.0
    jog_amax_deg_s2: float = 120.0
    # Scanning: the photo-grid sweep (see path_core.generate_photo_grid).
    scan_vmax_deg_s: float = 90.0
    scan_amax_deg_s2: float = 180.0
    # How aligned two consecutive segments' directions must be (cosine of
    # the angle between them) to coast through a waypoint instead of
    # stopping -- see jog_controller.py's corner-blending logic. This is
    # what makes the fine interpolation points along a scan row pass
    # through smoothly while a real direction change (or the forced end of
    # a start_scan() call, i.e. a photo stop) still comes to a full stop.
    blend_threshold: float = 0.7
    control_hz: float = 50.0

    @classmethod
    def from_dict(cls, d: dict) -> "MotionConfig":
        defaults = asdict(cls())
        known = {k: v for k, v in d.items() if k in defaults}
        return cls(**{**defaults, **known})


@dataclass
class PhotoConfig:
    """Everything about the photo grid that ISN'T the rectangle itself
    (that's the `rectangle` section, taught separately -- see
    path_core.py)."""
    spacing_x_mm: float = 20.0
    spacing_y_mm: float = 20.0
    margin_mm: float = 0.0
    dwell_s: float = 1.0
    # Cartesian straight-line interpolation resolution between photo
    # stops -- see path_core.interpolate_line_mm. Smaller = straighter
    # (closer to a true cartesian line) but more waypoints to plan/stream.
    max_step_mm: float = 5.0
    photo_dir: str = "photos"

    @classmethod
    def from_dict(cls, d: dict) -> "PhotoConfig":
        defaults = asdict(cls())
        known = {k: v for k, v in d.items() if k in defaults}
        return cls(**{**defaults, **known})


def _default_calib() -> dict:
    p = ArmParams.nominal()
    return {
        "kinematics": asdict(p),
        "hardware": asdict(HardwareConfig()),
        "motion": asdict(MotionConfig()),
        "photo": asdict(PhotoConfig()),
        # None = not taught yet -- see path_core.fit_rect_from_corners /
        # teach_gui.py.
        "rectangle": None,
        # Mechanical dead-zone protection, in raw servo-degree space. None
        # = not yet measured: IK/jog won't reject anything on this basis.
        "joint_limits_deg": None,
    }


def _validate_calib(calib: dict) -> None:
    """Reject a malformed/out-of-range calib file instead of silently
    falling back to defaults: a bad offset could drive a servo into its
    mechanical limit on the next move command."""
    k = calib.get("kinematics")
    if not k:
        raise ValueError("calib.json missing 'kinematics' section")
    for key, (lo, hi) in DEFAULT_BOUNDS.items():
        val = k.get(key, 0.0 if key == "elbow_offset_mm" else None)
        if val is None or not (lo - 1e-6 <= val <= hi + 1e-6):
            raise ValueError(f"calib.json kinematics.{key}={val} out of expected range [{lo},{hi}]")
    for dir_key in ("servo1_dir", "servo2_dir"):
        val = k.get(dir_key, 1)
        if val not in (1, -1):
            raise ValueError(f"calib.json kinematics.{dir_key}={val} must be 1 or -1")

    rect = calib.get("rectangle")
    if rect is not None:
        corners = rect.get("corners_mm")
        if not corners or len(corners) != 4:
            raise ValueError("calib.json rectangle.corners_mm must have exactly 4 [x, y] pairs")
        for c in corners:
            if len(c) != 2 or not all(math.isfinite(v) for v in c):
                raise ValueError(f"calib.json rectangle.corners_mm has an invalid point: {c}")
        for key in ("center_x_mm", "center_y_mm", "width_mm", "height_mm", "rotation_deg"):
            if key not in rect or not math.isfinite(rect[key]):
                raise ValueError(f"calib.json rectangle.{key} missing or not finite")
        if not (rect["width_mm"] > 0 and rect["height_mm"] > 0):
            raise ValueError("calib.json rectangle width_mm/height_mm must be positive")

    photo = calib.get("photo", {})
    for key in ("spacing_x_mm", "spacing_y_mm", "max_step_mm"):
        val = photo.get(key, PhotoConfig.__dataclass_fields__[key].default)
        if not (math.isfinite(val) and val > 0):
            raise ValueError(f"calib.json photo.{key}={val} must be a positive finite number")
    dwell = photo.get("dwell_s", PhotoConfig.__dataclass_fields__["dwell_s"].default)
    if not (math.isfinite(dwell) and dwell >= 0):
        raise ValueError(f"calib.json photo.dwell_s={dwell} must be >= 0")

    joint_limits = calib.get("joint_limits_deg")
    if joint_limits is not None:
        def _check_range(label, pair):
            if not pair or len(pair) != 2:
                raise ValueError(f"calib.json joint_limits_deg.{label} must be a [lo, hi] pair")
            lo, hi = pair
            if not (0.0 <= lo < hi <= 360.0):
                raise ValueError(
                    f"calib.json joint_limits_deg.{label}=[{lo},{hi}] must satisfy "
                    f"0 <= lo < hi <= 360 (a wrapping safe range isn't supported)")
            return lo, hi

        for joint in ("joint1", "joint2"):
            _check_range(joint, joint_limits.get(joint))

        boundary_raw = joint_limits.get("coupled_boundary", [])
        if boundary_raw:
            if len(boundary_raw) < 3:
                raise ValueError(
                    "calib.json joint_limits_deg.coupled_boundary needs >=3 vertices to "
                    "form a closed polygon")
            for i, vertex in enumerate(boundary_raw):
                j1, j2 = vertex.get("joint1"), vertex.get("joint2")
                if j1 is None or j2 is None:
                    raise ValueError(
                        f"calib.json joint_limits_deg.coupled_boundary[{i}] needs both "
                        f"'joint1' and 'joint2'")
                if not (math.isfinite(j1) and math.isfinite(j2)):
                    raise ValueError(
                        f"calib.json joint_limits_deg.coupled_boundary[{i}]=({j1},{j2}) "
                        f"must be finite numbers")


def load_calib(path: Optional[Path] = None) -> dict:
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
                      elbow_offset_mm=k.get("elbow_offset_mm", 0.0),
                      servo1_dir=k.get("servo1_dir", 1), servo2_dir=k.get("servo2_dir", 1))


def calib_hardware_config(calib: dict) -> HardwareConfig:
    h = calib.get("hardware", {})
    defaults = HardwareConfig()
    return HardwareConfig(servo_port=h.get("servo_port", defaults.servo_port),
                           joint_ids=h.get("joint_ids", defaults.joint_ids))


def calib_motion_config(calib: dict) -> MotionConfig:
    return MotionConfig.from_dict(calib.get("motion", {}))


def calib_photo_config(calib: dict) -> PhotoConfig:
    return PhotoConfig.from_dict(calib.get("photo", {}))


def calib_joint_limits(calib: dict) -> Optional[dict]:
    """Returns {"joint1": (lo, hi), "joint2": (lo, hi), "coupled_boundary": [...]}
    in raw servo-degree space, or None if not yet configured."""
    raw = calib.get("joint_limits_deg")
    if raw is None:
        return None
    boundary = [(v["joint1"], v["joint2"]) for v in raw.get("coupled_boundary", [])]
    return {
        "joint1": tuple(raw["joint1"]),
        "joint2": tuple(raw["joint2"]),
        "coupled_boundary": boundary,
    }


def calib_rectangle(calib: dict) -> Optional[tuple[float, float, float, float, float]]:
    """Returns (center_x_mm, center_y_mm, width_mm, height_mm, rotation_deg)
    for the taught scan rectangle, or None if teach-in hasn't been run yet."""
    rect = calib.get("rectangle")
    if rect is None:
        return None
    return (rect["center_x_mm"], rect["center_y_mm"], rect["width_mm"],
            rect["height_mm"], rect["rotation_deg"])
