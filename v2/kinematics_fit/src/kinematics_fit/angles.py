"""Angle-wrapping helpers. Every v2 package that computes a "distance to
travel between two angles" needs its own copy of wrap_angle_near -- atan2-
derived IK solutions have no reason to land near the arm's current angle,
so a naive subtraction can make a trajectory sweep almost a full extra
revolution to reach a target that's physically only a couple degrees away.
This is the single most important function to get right in every
independently-implemented controller.py across v2 -- see kinematics_fit/
card_scan/teach/record_replay's controller.py for where it's used."""

from __future__ import annotations


def normalize_deg(angle_deg: float) -> float:
    """Wrap to [0, 360) -- matches arm_hw_core.servos.Servos' own
    `% TICKS_PER_REV` normalization when converting to ticks, so a limit
    check here agrees with what actually gets sent to hardware."""
    return angle_deg % 360.0


def wrap_angle_near(target_deg: float, reference_deg: float) -> float:
    """The angle congruent to target_deg (mod 360) that's closest to
    reference_deg -- e.g. target_deg=1, reference_deg=359 -> returns 361
    (only 2deg from reference_deg, not the 358deg a raw subtraction would
    suggest)."""
    return reference_deg + ((target_deg - reference_deg + 180.0) % 360.0 - 180.0)
