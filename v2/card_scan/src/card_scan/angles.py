"""Angle-wrapping helpers -- see kinematics_fit/angles.py for the twin of
this module and why every v2 package needs its own copy."""

from __future__ import annotations


def normalize_deg(angle_deg: float) -> float:
    return angle_deg % 360.0


def wrap_angle_near(target_deg: float, reference_deg: float) -> float:
    return reference_deg + ((target_deg - reference_deg + 180.0) % 360.0 - 180.0)


def rotate_vector(dx: float, dy: float, rotation_deg: float) -> tuple[float, float]:
    """Rotates a 2D vector (dx, dy) by rotation_deg (degrees, CCW) about
    the origin -- used by scan.py's rectangle/path math."""
    import math

    theta = math.radians(rotation_deg)
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    return (dx * cos_t - dy * sin_t, dx * sin_t + dy * cos_t)
