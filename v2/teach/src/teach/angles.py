"""Angle-wrapping helpers -- see kinematics_fit/angles.py for the twin of
this module and why every v2 package needs its own copy."""

from __future__ import annotations


def normalize_deg(angle_deg: float) -> float:
    """Wrap to [0, 360) -- matches arm_hw_core.servos.Servos' own
    `% TICKS_PER_REV` normalization."""
    return angle_deg % 360.0


def wrap_angle_near(target_deg: float, reference_deg: float) -> float:
    """The angle congruent to target_deg (mod 360) that's closest to
    reference_deg."""
    return reference_deg + ((target_deg - reference_deg + 180.0) % 360.0 - 180.0)
