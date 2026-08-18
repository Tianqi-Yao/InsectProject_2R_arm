"""Software joint-limit safety: the second, complementary layer to
servos.Servos's hardware EEPROM angle-limit registers. The hardware layer
protects each servo independently (it has no notion of the other joint's
position); this layer additionally supports a "coupled boundary" -- a
region in (joint1, joint2) space that's unsafe together even though each
angle is individually fine (e.g. self-collision risk that varies smoothly
with both joints).
"""

from __future__ import annotations

from typing import Optional, TypedDict


class JointLimits(TypedDict, total=False):
    joint1: tuple[float, float]
    joint2: tuple[float, float]
    coupled_boundary: list[tuple[float, float]]


def normalize_deg(angle_deg: float) -> float:
    """Wrap to [0, 360) -- the same normalization servos.Servos applies
    when converting a raw servo-degree command to ticks (`% TICKS_PER_REV`),
    so a limit check here agrees with what actually gets sent to hardware."""
    return angle_deg % 360.0


def point_in_polygon(x: float, y: float, polygon: list[tuple[float, float]]) -> bool:
    """Winding-number point-in-polygon test. `polygon` is a list of (x, y)
    vertices, implicitly closed (the last vertex connects back to the
    first) -- works for any simple polygon, convex or not.

    Deliberately winding-number (nonzero rule), NOT the simpler even-odd
    ray-casting rule: a hand-traced boundary is recorded densely and
    continuously, so any hesitation/backtrack/jitter along the way is
    baked into the saved vertices verbatim (by design -- nothing is
    smoothed out). Even-odd parity is provably wrong for this: tracing the
    exact same loop an even number of times (e.g. a moment of doubling back
    over a stretch already walked) flips its verdict to "outside" for every
    point that loop encloses, entirely by coincidence of the lap count's
    parity. Winding number doesn't have this failure mode -- any nonzero
    winding (1 lap, 2 laps, or a partial wobble that doesn't add a full
    lap) is still "inside," which matches what a person tracing a boundary
    by hand actually means, regardless of how many times their hand
    happened to cross a given edge along the way. An earlier
    automatic-derivation approach for this boundary (bin a hand-swept fill
    by joint1, take min/max per bucket) produced visibly wrong results on
    real hardware -- this hand-traced + winding-number approach replaced it."""
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
                         joint_limits: Optional[JointLimits]) -> bool:
    """joint_limits is None (not configured) -> unrestricted, matching a
    fresh install before bring-up has run. Otherwise
    {"joint1": (lo, hi), "joint2": (lo, hi), "coupled_boundary": [...]}
    in raw servo-degree space (same convention as get_present_deg()/
    set_target_deg(), NOT any theta1/theta2 IK convention) -- this is a
    physical/mechanical constraint, and keeping one unit convention across
    this and the servo's own hardware angle-limit registers avoids a whole
    class of "which angle am I even looking at" mistakes.

    "joint1"/"joint2" are each joint's own unconditional safe range -- true
    no matter what the other joint is doing. "coupled_boundary" (a list of
    (joint1, joint2) vertices, possibly absent/empty) handles the case
    where the distal link's safe range continuously shrinks/grows depending
    on where the proximal link currently is -- something the servo's own
    hardware angle-limit registers have no way to express at all, so this
    coupled check is software-only. A (servo1, servo2) pose passes this
    check iff it falls INSIDE that traced polygon (point_in_polygon) --
    outside is treated as the dead zone.

    IMPORTANT ASSUMPTION: "joint1"/"joint2"'s own ranges are each a single,
    non-wrapping arc (lo <= hi, no wraparound through 0/360), and
    coupled_boundary's vertices don't themselves need to wrap through
    0/360 either. When mounting a servo horn, prefer orienting it so the
    *dead zone* straddles the 0/360 wraparound point rather than the safe
    region -- that keeps the safe region representable by these bounds
    (and the servo's own Min/Max Angle Limit registers, which have the
    same limitation)."""
    if joint_limits is None:
        return True
    s1n, s2n = normalize_deg(servo1_deg), normalize_deg(servo2_deg)
    for name, angle in (("joint1", s1n), ("joint2", s2n)):
        lo, hi = joint_limits[name]
        if not (lo <= angle <= hi):
            return False
    boundary = joint_limits.get("coupled_boundary")
    if boundary:
        if not point_in_polygon(s1n, s2n, boundary):
            return False
    return True
