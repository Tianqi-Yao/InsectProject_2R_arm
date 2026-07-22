"""Core kinematics, vision-fusion, calibration, and self-check logic for the 2R arm.

This is the one file meant to be read end-to-end: everything here is either
geometry/math or a decision (halt vs. continue, accept vs. reject a fit).
Hardware I/O (servo bus protocol, camera capture, AprilTag detection) lives in
arm_hardware.py and is treated as a black box behind a handful of plain
methods (see the `hw` parameter of run_selfcheck, and how main.py uses it).
Motion/trajectory algorithms live in motion_planning/ as swappable plugins
(see jog_controller.py for how a planner is selected and driven).
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
    # The servo2/joint2 mounting offset: L1 ends where servo2's *body* is
    # bolted on, but servo2's *rotation axis* (where L2 actually starts) sits
    # this many mm to the side of L1's own line -- a rigid mechanical fact of
    # how the servo body's footprint doesn't collapse to a point on the link
    # centerline. 0.0 for a build where the two are colinear.
    #
    # MUST be a fixed, independently-measured constant (CAD/calipers), like
    # servo1_dir/servo2_dir -- NOT something fit_kinematics can solve for,
    # even though it looks like an ordinary continuous quantity. Proven by
    # direct substitution: reach=hypot(L1, elbow_offset_mm) and
    # angle_offset=atan2(elbow_offset_mm, L1) are the only things vision data
    # can ever pin down (see _elbow_reach_and_angle) -- the *split* between
    # L1 and elbow_offset_mm for a given reach is exactly absorbable by
    # trading servo1_offset_deg and servo2_offset_deg against each other
    # (shifting theta1 by -angle_offset and theta2 by +angle_offset is
    # indistinguishable from changing which (L1, elbow_offset_mm) pair
    # produced that same angle_offset). Confirmed empirically: fitting
    # synthetic data generated with elbow_offset_mm=28 recovered
    # elbow_offset_mm=4 and a correspondingly shifted L1, with statistically
    # perfect rms_error_mm regardless of the fit's starting guess or how
    # wide/dense the sampled joint angles were -- more or better vision data
    # cannot resolve this, because both parameter sets predict the exact
    # same end-effector position for every joint angle. If this needs
    # firming up, measure the physical center-to-center distance (joint1's
    # rotation axis' line vs. joint2's rotation axis) directly, rather than
    # trusting a fit.
    elbow_offset_mm: float = 0.0

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
    def from_vector(cls, vec, servo1_dir: int = 1, servo2_dir: int = 1,
                     elbow_offset_mm: float = 0.0) -> "ArmParams":
        return cls(*vec, servo1_dir=servo1_dir, servo2_dir=servo2_dir,
                   elbow_offset_mm=elbow_offset_mm)


@dataclass
class IKResult:
    theta1_deg: float = 0.0
    theta2_deg: float = 0.0
    servo1_deg: float = 0.0
    servo2_deg: float = 0.0
    reachable: bool = False


def _normalize_deg(angle_deg: float) -> float:
    """Wrap to [0, 360) -- the same normalization arm_hardware.Servos applies
    when converting a raw servo-degree command to ticks (`% TICKS_PER_REV`),
    so a limit check here agrees with what actually gets sent to hardware."""
    return angle_deg % 360.0


def wrap_angle_near(target_deg: float, reference_deg: float) -> float:
    """The angle congruent to target_deg (mod 360) that's closest to
    reference_deg -- e.g. target_deg=1, reference_deg=359 -> returns 361
    (only 2deg from reference_deg, not the 358deg a raw subtraction would
    suggest). Used wherever a "distance to travel" between two angles is
    computed (motion_planning/trapezoidal.py, jog_controller.py's scan
    waypoint sequencing): ik_solve()'s theta1/theta2 (hence the raw servo
    angle) come from atan2, which has no reason to land near whatever the
    arm's current angle happens to be, so treating a goal angle literally
    (instead of picking its nearest equivalent first) can make a
    trajectory planner sweep almost a full extra revolution to reach a
    target that's physically only a couple degrees away."""
    return reference_deg + ((target_deg - reference_deg + 180.0) % 360.0 - 180.0)


def rotate_vector(dx: float, dy: float, rotation_deg: float) -> tuple[float, float]:
    """Rotates a 2D vector (dx, dy) by rotation_deg (degrees, CCW) about
    the origin. Shared by scan_area_corners() and manual_test/gui.py's
    arrow-key jog (which nudges along the scan area's own tilted axes,
    not the workspace frame's raw x/y, whenever it's been rotated -- see
    arm_core.calib_scan_area)."""
    theta = math.radians(rotation_deg)
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    return (dx * cos_t - dy * sin_t, dx * sin_t + dy * cos_t)


def _point_in_polygon(x: float, y: float, polygon: list) -> bool:
    """Winding-number point-in-polygon test. `polygon` is a list of (x, y)
    vertices, implicitly closed (the last vertex connects back to the
    first) -- works for any simple polygon, convex or not, which is
    exactly why this replaced an earlier per-joint1-interval representation
    that could only ever express a shape that's a single contiguous range
    of joint2 at every joint1 slice.

    Deliberately winding-number (nonzero rule), NOT the simpler even-odd
    ray-casting rule: a hand-traced boundary (see manual_test/
    trace_boundary_gui.py) is recorded densely and continuously, so any
    hesitation/backtrack/jitter along the way is baked into the saved
    vertices verbatim (by design -- nothing is smoothed out). Even-odd
    parity is provably wrong for this: tracing the exact same loop an
    even number of times (e.g. a moment of doubling back over a stretch
    already walked) flips its verdict to "outside" for every point that
    loop encloses, entirely by coincidence of the lap count's parity.
    Winding number doesn't have this failure mode -- any nonzero winding
    (1 lap, 2 laps, or a partial wobble that doesn't add a full lap) is
    still "inside," which matches what a person tracing a boundary by
    hand actually means regardless of how many times their hand happened
    to cross a given edge along the way."""
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
    """joint_limits is None (not configured) -> unrestricted, matches the
    fresh-install default (see arm_core.HardwareConfig's sibling concept,
    calib_joint_limits). Otherwise
    {"joint1": (lo, hi), "joint2": (lo, hi), "coupled_boundary": [...]}
    in raw servo-degree space (same convention as get_present_deg()/
    set_target_deg(), NOT the theta1/theta2 IK convention) -- this is a
    physical/mechanical constraint (the dead zone doesn't care about our
    math convention), and keeping one unit convention across this and the
    servo's own hardware angle-limit registers (see arm_hardware.py) avoids
    a whole class of "which angle am I even looking at" mistakes.

    "joint1"/"joint2" are each joint's own unconditional safe range --
    true no matter what the other joint is doing. "coupled_boundary" (a
    list of (joint1, joint2) vertices, possibly empty) handles the case
    where the distal link's safe range *continuously* shrinks/grows
    depending on where the proximal link currently is (e.g. the far link's
    clearance to a fixed obstacle varies smoothly as the near link sweeps)
    -- something the servo's own hardware angle-limit registers have no
    way to express at all (each servo only knows its own angle, not the
    other one's), so this coupled check is software-only. It's a closed
    polygon traced by hand (see manual_test/trace_boundary_gui.py, or
    main.py's set-joint-limits): release both joints' torque and walk the
    arm around the full perimeter of the safe region once; the traced path
    IS the boundary, no interpolation or derivation applied to it. A
    (servo1, servo2) pose passes this check iff it falls INSIDE that
    traced polygon (_point_in_polygon) -- outside is treated as the dead
    zone.

    IMPORTANT ASSUMPTION: "joint1"/"joint2"'s own ranges are each a single,
    non-wrapping arc (lo <= hi, no wraparound through 0/360), and
    coupled_boundary's vertices don't themselves need to wrap through
    0/360 either. When mounting a servo horn, prefer orienting it so the
    *dead zone* straddles the 0/360 wraparound point rather than the safe
    region -- that keeps the safe region representable by these bounds
    (and the servo's own Min/Max Angle Limit registers, which have the
    same limitation).
    """
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
    """The servo2 mounting offset (ArmParams.elbow_offset_mm) turns the
    joint1-to-joint2 "link" into a right triangle instead of a straight
    line: the true distance from joint1's axis to joint2's axis is
    hypot(L1, elbow_offset_mm), at a fixed angle atan2(elbow_offset_mm, L1)
    off from the direction theta1 alone would give. Returns
    (reach_mm, angle_offset_deg); angle_offset_deg is exactly 0 when
    elbow_offset_mm is 0, so every caller reduces to the plain-colinear
    formula for arms without this offset."""
    reach = math.hypot(p.L1, p.elbow_offset_mm)
    angle_offset_deg = math.degrees(math.atan2(p.elbow_offset_mm, p.L1))
    return reach, angle_offset_deg


def ik_solve(p: ArmParams, x_ws: float, y_ws: float,
             joint_limits: Optional[dict] = None) -> IKResult:
    """Inverse kinematics: workspace (x, y) in mm -> joint/servo angles.
    Direct generalization of firmware/kinematics.h's ik_solve().

    Solved by substitution: with reach = hypot(L1, elbow_offset_mm) standing
    in for L1, the standard colinear-2R closed form gives back
    (theta1_std, theta2_std) for a *virtual* joint1 angle that already
    includes the fixed angle_offset_deg baked in (see _elbow_reach_and_angle
    and fk_from_servo_angles) -- subtracting/adding it back out recovers the
    real theta1/theta2 that correspond to actual servo angles.

    `joint_limits` (see within_joint_limits): when given, a geometrically
    reachable point whose computed servo angle would fall inside a
    configured dead zone / outside the mechanically safe range is also
    reported reachable=False -- callers shouldn't need to separately
    re-check this after calling ik_solve."""
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
    """Forward kinematics from *measured* servo angles -> workspace (x, y) mm.

    This is the one function shared by both the kinematic-parameter fit
    (fit_kinematics) and the boot self-check (run_selfcheck): both boil down
    to "take a real encoder angle pair, predict where the end effector
    should be, compare against what the camera measured."
    """
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
    for drawing the two links separately (e.g. manual_test/gui.py), not
    needed by the fit/self-check, which only care about the end effector.
    "Elbow" here means joint2's actual rotation axis, i.e. where L2 starts
    -- not wherever L1's own link physically ends, if elbow_offset_mm is
    nonzero (see ArmParams.elbow_offset_mm)."""
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
    # elbow_offset_mm is NOT in here / not fit -- see ArmParams' docstring
    # for the exact degeneracy that makes vision-based fitting unable to
    # recover it, no matter how much data. This range is only used to
    # sanity-check a hand-entered calib.json value (_validate_calib).
    elbow_offset_mm=(-60.0, 60.0),
)


def generate_calibration_targets(params: Optional[ArmParams] = None,
                                  width_mm: float = 200.0, height_mm: float = 150.0,
                                  nx: int = 6, ny: int = 5, margin_mm: float = 15.0,
                                  seed: Optional[int] = None,
                                  joint_limits: Optional[dict] = None) -> list[IKResult]:
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

    `joint_limits`: pass calib_joint_limits(calib) to also keep collection
    out of a configured mechanical dead zone (see within_joint_limits).
    Previously this hardcoded a "0 <= servo_deg <= 180" filter left over
    from the older MG90S half-turn servos' hard 0-180deg range -- the
    STS3215 spans a full 0-360deg circle, so that check was both stale and
    not actually protective for this hardware; joint_limits is the real,
    configurable replacement.
    """
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


def generate_scan_path(width_mm: float = 200.0, height_mm: float = 150.0,
                        nx: int = 50, ny: int = 40, margin_mm: float = 20.0,
                        rows_limit: Optional[int] = None,
                        center_x_mm: Optional[float] = None, center_y_mm: Optional[float] = None,
                        rotation_deg: float = 0.0
                        ) -> list[tuple[float, float, str]]:
    """Serpentine (boustrophedon) path across a `width_mm` x `height_mm`
    rectangle for the manual jog tools' scan feature: starts at the
    top-left corner *of the rectangle* (matching the corner_world_mm "tl"
    convention when rotation_deg is 0) and snakes row by row --
    left-to-right, down, right-to-left, down, ... -- so there's no wasted
    travel back to a row's start.

    The grid is built in the rectangle's own local frame (centered on
    (center_x_mm, center_y_mm), local +x/+y aligned with the rectangle's
    own width/height axes before rotation), then rotated by
    `rotation_deg` (degrees, CCW, about that center) and translated into
    the workspace-mm frame -- see manual_test/scan_area_gui.py, which
    fits this rectangle (position, size, AND rotation -- a tilted
    rectangle can cover more of an irregularly-shaped reachable area than
    an axis-aligned one) visually against what's actually reachable.

    `center_x_mm`/`center_y_mm` default to (width_mm/2, height_mm/2) --
    i.e. the rectangle's own bottom-left corner sits at the workspace
    frame's origin with rotation_deg=0, matching this function's
    behavior before the scan area became independent from the
    calibration sheet's own size (see calib_scan_area()). `rows_limit`
    truncates to just the first N rows (same spacing as the full nx*ny
    grid, just fewer of them) -- handy for a quick check before
    committing to the full sweep.

    Previously duplicated near-identically in manual_test/run.py and
    manual_test/gui.py (and had silently drifted out of sync between the
    two copies); now the one place both read from.
    """
    if center_x_mm is None:
        center_x_mm = width_mm / 2.0
    if center_y_mm is None:
        center_y_mm = height_mm / 2.0

    xs_local = [-(width_mm / 2.0) + margin_mm + i * (width_mm - 2 * margin_mm) / (nx - 1)
                for i in range(nx)]
    ys_local = [(height_mm / 2.0) - margin_mm - j * (height_mm - 2 * margin_mm) / (ny - 1)
                for j in range(ny)]
    if rows_limit is not None:
        ys_local = ys_local[:rows_limit]

    theta = math.radians(rotation_deg)
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    path = []
    for row, y_local in enumerate(ys_local):
        row_xs = xs_local if row % 2 == 0 else list(reversed(xs_local))
        for x_local in row_xs:
            x = center_x_mm + x_local * cos_t - y_local * sin_t
            y = center_y_mm + x_local * sin_t + y_local * cos_t
            path.append((x, y, f"row{row + 1}"))
    return path


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

    # servo1_dir/servo2_dir/elbow_offset_mm are fixed hardware facts, not
    # fit -- carried through from x0 into every trial ArmParams the
    # optimizer builds (see ArmParams.elbow_offset_mm's docstring for why
    # the last of those specifically can't be a free parameter here).
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


def servo2_offset_from_known_elbow_angle(servo2_deg: float, servo2_dir: int,
                                          elbow_angle_deg: float, flip: bool = False) -> float:
    """Solve servo2_offset_deg directly from one physical reference pose --
    no camera/vision needed -- for a quick fix when the simulated L1-L2
    angle visibly doesn't match reality (see manual_test/trace_boundary_gui.py's
    'k'/shift+'k' keys).

    Fold the arm by hand so the angle between L1 and L2, as read by a
    protractor/set-square placed at the elbow, is `elbow_angle_deg` -- using
    the usual joint convention: 180 = fully straight/extended, 0 = fully
    folded back on itself (matching theta2=0 meaning "straight" everywhere
    else in this file). `servo2_deg` is the real encoder reading at that
    exact pose.

    Solves for servo2_offset_deg ONLY, not servo1_offset_deg/elbow_offset_mm:
    a fold-to-known-angle pose fixes the *relative* angle between the two
    links, independent of which way the whole arm happens to be pointing,
    so it can't say anything about joint1's absolute orientation
    (servo1_offset_deg) or the elbow's absolute position in the base frame
    (elbow_offset_mm -- which needs a direct physical/CAD measurement, see
    ArmParams' docstring for why vision can't recover it either).

    `flip`: a protractor reading is a magnitude -- it can't tell you which
    rotational direction the elbow was folded to reach it. Try flip=True if
    the first attempt makes the simulated angle look wrong/mirrored instead
    of fixed.
    """
    theta2_target = 180.0 - elbow_angle_deg
    if flip:
        theta2_target = -theta2_target
    return servo2_deg - servo2_dir * theta2_target


# ── 4. calib.json persistence ──────────────────────────────────────────
#
# calib.json is the single source of truth for "everything that describes
# this particular physical rig": the fitted kinematic parameters, which
# bus IDs/port the two servos are on, and the motion-planning tuning
# knobs. Before this, manual_test/run.py and manual_test/gui.py each kept
# their own hardcoded copies of L1/L2/base/offset/dir/joint_ids/scan
# constants -- which had already drifted out of sync with each other (see
# git history). Now both read the same three sections below.

@dataclass
class HardwareConfig:
    servo_port: str = "/dev/cu.usbserial-0001"
    joint_ids: dict = field(default_factory=lambda: {"joint1": 1, "joint2": 2})


@dataclass
class MotionConfig:
    planner_name: str = "trapezoidal"
    # Jogging: single-target moves triggered by arrow keys/nudges.
    jog_vmax_deg_s: float = 60.0
    jog_amax_deg_s2: float = 120.0
    # Scanning: the dense multi-waypoint sweep (see generate_scan_path).
    scan_vmax_deg_s: float = 90.0
    scan_amax_deg_s2: float = 180.0
    scan_nx: int = 50
    scan_ny: int = 40
    scan_rows_limit: Optional[int] = None
    scan_margin_mm: float = 20.0
    # The jog/scan sub-rectangle, in the same workspace-mm coordinate frame
    # the AprilTag homography establishes -- deliberately independent of
    # `workspace.width_mm`/`height_mm` (the calibration SHEET's own size,
    # fixed by where the corner tags are physically stuck down; changing
    # those numbers without moving the tags would desync the homography).
    # Center+size+rotation (not min/max bounds) because it can be tilted:
    # a rotated rectangle can cover more of an irregularly-shaped
    # reachable area than an axis-aligned one. None (any of the first
    # four) means "not configured yet" -> falls back to the full
    # calibration sheet, i.e. today's behavior, so older calib.json files
    # need no changes. See calib_scan_area() and
    # manual_test/scan_area_gui.py, which fits this visually to whatever
    # the arm can actually safely reach (joint_limits_deg), since the
    # calibration sheet's placement has no guarantee of matching that.
    scan_center_x_mm: Optional[float] = None
    scan_center_y_mm: Optional[float] = None
    scan_width_mm: Optional[float] = None
    scan_height_mm: Optional[float] = None
    scan_rotation_deg: float = 0.0
    # How aligned two consecutive scan segments' directions must be
    # (cosine of the angle between them) to coast through the corner
    # instead of stopping -- see jog_controller.py's corner-blending logic.
    blend_threshold: float = 0.7
    control_hz: float = 50.0

    @classmethod
    def from_dict(cls, d: dict) -> "MotionConfig":
        """Merges calib.json's motion section over the defaults. Silently
        drops any key in `d` that isn't a current MotionConfig field --
        e.g. a stale name left over from a schema change (a calib.json
        saved under an older version of this field set, which a renamed/
        removed field would otherwise turn into a hard crash here on
        every load, for a field that isn't even being read for anything
        anymore)."""
        defaults = asdict(cls())
        known = {k: v for k, v in d.items() if k in defaults}
        return cls(**{**defaults, **known})


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
        "hardware": asdict(HardwareConfig()),
        "motion": asdict(MotionConfig()),
        "thresholds": {"homography_drift_halt_mm": 3.0, "arm_position_halt_mm": 3.0},
        "spotcheck_poses": [{"joint1": 68.0, "joint2": 116.0}],
        # Mechanical dead-zone protection, in raw servo-degree space (same
        # convention as get_present_deg()/set_target_deg() -- NOT theta1/
        # theta2). None = not yet measured (fresh install): IK/jog won't
        # reject anything on this basis. Run `main.py set-joint-limits` to
        # measure your rig's safe range and populate this -- see that
        # command's help and within_joint_limits()'s docstring for why a
        # single non-wrapping [lo, hi] arc per joint is what's expected here.
        "joint_limits_deg": None,
    }


def _validate_calib(calib: dict) -> None:
    """Reject a malformed/out-of-range calib file instead of silently
    falling back to defaults: a bad offset could drive a servo into its
    mechanical limit on the next move command.

    `hardware`/`motion` sections are optional (older calib.json files
    predate them) -- calib_hardware_config/calib_motion_config fill in
    defaults for whatever's missing, so they're not required here."""
    k = calib.get("kinematics")
    if not k:
        raise ValueError("calib.json missing 'kinematics' section")
    for key, (lo, hi) in DEFAULT_BOUNDS.items():
        # elbow_offset_mm postdates this field's introduction -- missing
        # it (an older calib.json) means "colinear," not "invalid."
        val = k.get(key, 0.0 if key == "elbow_offset_mm" else None)
        if val is None or not (lo - 1e-6 <= val <= hi + 1e-6):
            raise ValueError(f"calib.json kinematics.{key}={val} out of expected range [{lo},{hi}]")
    for dir_key in ("servo1_dir", "servo2_dir"):
        val = k.get(dir_key, 1)
        if val not in (1, -1):
            raise ValueError(f"calib.json kinematics.{dir_key}={val} must be 1 or -1")
    for name in ("homography_drift_halt_mm", "arm_position_halt_mm"):
        if calib.get("thresholds", {}).get(name) is None:
            raise ValueError(f"calib.json missing thresholds.{name}")

    scan_shape = {k: calib.get("motion", {}).get(k) for k in
                  ("scan_center_x_mm", "scan_center_y_mm", "scan_width_mm", "scan_height_mm")}
    if any(v is not None for v in scan_shape.values()):
        # Partially configured (some but not all four set) is treated the
        # same as "none configured" by calib_scan_area()'s fallback, but
        # is almost certainly a mistake (an incomplete manual edit, or a
        # bug in whatever wrote it) -- catch it here rather than silently
        # falling back to the full calibration sheet.
        if any(v is None for v in scan_shape.values()):
            raise ValueError(
                f"calib.json motion.scan_{{center_x,center_y,width,height}}_mm must be "
                f"either all four set or all four omitted, got {scan_shape}")
        cx, cy, w, h = (scan_shape["scan_center_x_mm"], scan_shape["scan_center_y_mm"],
                        scan_shape["scan_width_mm"], scan_shape["scan_height_mm"])
        if not (math.isfinite(cx) and math.isfinite(cy) and math.isfinite(w) and math.isfinite(h)):
            raise ValueError(f"calib.json motion scan area must be finite numbers, got {scan_shape}")
        if not (w > 0 and h > 0):
            raise ValueError(f"calib.json motion scan_width_mm/scan_height_mm must be "
                              f"positive, got {scan_shape}")
    rotation = calib.get("motion", {}).get("scan_rotation_deg")
    if rotation is not None and not math.isfinite(rotation):
        raise ValueError(f"calib.json motion.scan_rotation_deg={rotation} must be a finite number")

    joint_limits = calib.get("joint_limits_deg")
    if joint_limits is not None:
        def _check_range(label, pair):
            if not pair or len(pair) != 2:
                raise ValueError(f"calib.json joint_limits_deg.{label} must be a [lo, hi] pair")
            lo, hi = pair
            if not (0.0 <= lo < hi <= 360.0):
                raise ValueError(
                    f"calib.json joint_limits_deg.{label}=[{lo},{hi}] must satisfy "
                    f"0 <= lo < hi <= 360 (a wrapping safe range isn't supported -- "
                    f"see within_joint_limits()'s docstring)")
            return lo, hi

        joint_ranges = {}
        for joint in ("joint1", "joint2"):
            joint_ranges[joint] = _check_range(joint, joint_limits.get(joint))

        boundary_raw = joint_limits.get("coupled_boundary", [])
        if boundary_raw:
            # >=3 vertices: the geometric minimum for a polygon to enclose
            # any area at all (see _point_in_polygon).
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
                # Deliberately NOT checking each vertex falls within
                # joint1/joint2's own "unconditional" ranges above: the
                # tool that captures this polygon (manual_test/
                # trace_boundary_gui.py, main.py's set-joint-limits) widens
                # those ranges to the union of themselves and every
                # traced vertex before saving, precisely so this
                # wouldn't be a validation trap here.

        # Defense in depth: a spotcheck pose that's already outside the
        # configured safe range would be a self-inflicted footgun every
        # selfcheck run -- catch it at config-validation time instead.
        # within_joint_limits expects coupled_boundary already parsed into
        # (joint1, joint2) tuples (calib_joint_limits's job), not the raw
        # JSON list-of-dicts shape `joint_limits` still is here.
        parsed_limits = calib_joint_limits(calib)
        for pose in calib.get("spotcheck_poses", []):
            if not within_joint_limits(pose["joint1"], pose["joint2"], parsed_limits):
                raise ValueError(f"calib.json spotcheck_poses entry {pose} violates joint_limits_deg")


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
                      elbow_offset_mm=k.get("elbow_offset_mm", 0.0),
                      servo1_dir=k.get("servo1_dir", 1), servo2_dir=k.get("servo2_dir", 1))


def calib_hardware_config(calib: dict) -> HardwareConfig:
    h = calib.get("hardware", {})
    defaults = HardwareConfig()
    return HardwareConfig(servo_port=h.get("servo_port", defaults.servo_port),
                           joint_ids=h.get("joint_ids", defaults.joint_ids))


def calib_motion_config(calib: dict) -> MotionConfig:
    return MotionConfig.from_dict(calib.get("motion", {}))


def calib_scan_area(calib: dict) -> tuple[float, float, float, float, float]:
    """Returns (center_x_mm, center_y_mm, width_mm, height_mm, rotation_deg)
    for the jog/scan sub-rectangle -- MotionConfig.scan_center_x_mm etc if
    configured, otherwise the full calibration sheet, unrotated
    (width_mm/2, height_mm/2, workspace.width_mm, workspace.height_mm, 0.0),
    i.e. today's behavior. See manual_test/scan_area_gui.py to configure
    this visually (position, size, AND rotation), fit to wherever the arm
    can actually safely reach."""
    mc = calib_motion_config(calib)
    shape = (mc.scan_center_x_mm, mc.scan_center_y_mm, mc.scan_width_mm, mc.scan_height_mm)
    if all(v is not None for v in shape):
        return (*shape, mc.scan_rotation_deg)
    ws = calib["workspace"]
    return (ws["width_mm"] / 2.0, ws["height_mm"] / 2.0, ws["width_mm"], ws["height_mm"], 0.0)


def scan_area_corners(center_x_mm: float, center_y_mm: float, width_mm: float, height_mm: float,
                       rotation_deg: float) -> list[tuple[float, float]]:
    """The 4 corners of a (possibly rotated) scan-area rectangle -- same
    shape calib_scan_area() returns -- in workspace mm, ordered
    bottom-left -> bottom-right -> top-right -> top-left (before
    rotation). Shared by manual_test/gui.py (sizing/drawing its window
    large enough to show the whole area, even the part sticking outside
    the calibration sheet) and manual_test/scan_area_gui.py (drawing +
    corner-handle hit-testing while fitting it)."""
    local = [(-width_mm / 2, -height_mm / 2), (width_mm / 2, -height_mm / 2),
             (width_mm / 2, height_mm / 2), (-width_mm / 2, height_mm / 2)]
    corners = []
    for lx, ly in local:
        rx, ry = rotate_vector(lx, ly, rotation_deg)
        corners.append((center_x_mm + rx, center_y_mm + ry))
    return corners


def calib_joint_limits(calib: dict) -> Optional[dict]:
    """Returns {"joint1": (lo, hi), "joint2": (lo, hi), "coupled_boundary": [...]}
    in raw servo-degree space, or None if not yet configured (fresh
    install -- see _default_calib's comment). "coupled_boundary" is
    returned as a list of (joint1, joint2) vertex tuples, in the order they
    were traced (a closed polygon -- _point_in_polygon implicitly connects
    the last vertex back to the first; order otherwise doesn't matter for
    that check). Pass straight into ik_solve()/within_joint_limits() or
    jog_controller.build_controller()."""
    raw = calib.get("joint_limits_deg")
    if raw is None:
        return None
    boundary = [(v["joint1"], v["joint2"]) for v in raw.get("coupled_boundary", [])]
    return {
        "joint1": tuple(raw["joint1"]),
        "joint2": tuple(raw["joint2"]),
        "coupled_boundary": boundary,
    }


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
