"""Angle-wrapping helpers -- see kinematics_fit/angles.py for the twin of
this module and why every v2 package needs its own copy, even one like
this that never does IK."""

from __future__ import annotations


def wrap_angle_near(target_deg: float, reference_deg: float) -> float:
    """The angle congruent to target_deg (mod 360) that's closest to
    reference_deg -- still needed here even without IK: two independently
    hand-marked points can straddle the 0/360 seam, and replay must take
    the short way between them, not sweep almost a full extra revolution."""
    return reference_deg + ((target_deg - reference_deg + 180.0) % 360.0 - 180.0)
