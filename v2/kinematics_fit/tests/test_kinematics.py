import pytest

from kinematics_fit.kinematics import ArmParams, fk_from_servo_angles, ik_solve


def test_fk_ik_round_trip_colinear_arm():
    p = ArmParams(L1=125.0, L2=95.0, base_x=100.0, base_y=-45.0,
                   servo1_offset_deg=23.08, servo2_offset_deg=0.0)
    x, y = 150.0, 60.0
    r = ik_solve(p, x, y)
    assert r.reachable
    fx, fy = fk_from_servo_angles(p, r.servo1_deg, r.servo2_deg)
    assert (fx, fy) == pytest.approx((x, y), abs=1e-6)


def test_fk_ik_round_trip_with_inverted_servo_directions():
    p = ArmParams(L1=110.0, L2=80.0, base_x=90.0, base_y=-30.0,
                   servo1_offset_deg=10.0, servo2_offset_deg=170.0,
                   servo1_dir=-1, servo2_dir=-1)
    x, y = 120.0, 40.0
    r = ik_solve(p, x, y)
    assert r.reachable
    fx, fy = fk_from_servo_angles(p, r.servo1_deg, r.servo2_deg)
    assert (fx, fy) == pytest.approx((x, y), abs=1e-6)


@pytest.mark.parametrize("elbow_offset_mm", [0.0, 12.5, -8.0])
def test_fk_ik_round_trip_with_elbow_offset(elbow_offset_mm):
    p = ArmParams(L1=125.0, L2=95.0, base_x=100.0, base_y=-45.0,
                   servo1_offset_deg=23.08, servo2_offset_deg=0.0,
                   elbow_offset_mm=elbow_offset_mm)
    x, y = 140.0, 50.0
    r = ik_solve(p, x, y)
    assert r.reachable
    fx, fy = fk_from_servo_angles(p, r.servo1_deg, r.servo2_deg)
    assert (fx, fy) == pytest.approx((x, y), abs=1e-6)


def test_elbow_offset_zero_reduces_to_colinear_formula():
    p_colinear = ArmParams(L1=125.0, L2=95.0, base_x=100.0, base_y=-45.0,
                             servo1_offset_deg=0.0, servo2_offset_deg=0.0)
    p_zero_offset = ArmParams(**{**p_colinear.__dict__, "elbow_offset_mm": 0.0})
    x1, y1 = fk_from_servo_angles(p_colinear, 40.0, 100.0)
    x2, y2 = fk_from_servo_angles(p_zero_offset, 40.0, 100.0)
    assert (x1, y1) == pytest.approx((x2, y2))


def test_ik_rejects_unreachable_point():
    p = ArmParams.nominal()
    far_away = p.base_x + p.L1 + p.L2 + 1000.0
    r = ik_solve(p, far_away, 0.0)
    assert r.reachable is False


def test_ik_rejects_point_outside_joint_limits():
    p = ArmParams.nominal()
    x, y = 150.0, 30.0
    unrestricted = ik_solve(p, x, y, joint_limits=None)
    assert unrestricted.reachable
    limits = {"joint1": (0.0, 1.0), "joint2": (0.0, 360.0)}  # near-impossible joint1 slice
    restricted = ik_solve(p, x, y, joint_limits=limits)
    assert restricted.reachable is False


def test_base_position_absorbs_a_uniform_rotation_the_same_as_servo1_offset():
    """This is the documented non-identifiability: rotating the whole
    base by delta has the exact same effect on (ex, ey) as shifting
    theta1 by delta -- there's deliberately no separate base_rotation_deg
    field (see kinematics.ArmParams' docstring). Sanity-check that
    shifting servo1_offset_deg alone reproduces a pure end-effector
    rotation about the base, confirming the two really are degenerate."""
    import math

    p1 = ArmParams(L1=100.0, L2=80.0, base_x=0.0, base_y=0.0,
                     servo1_offset_deg=0.0, servo2_offset_deg=0.0)
    delta = 15.0
    p2 = ArmParams(**{**p1.__dict__, "servo1_offset_deg": p1.servo1_offset_deg + delta})

    s1, s2 = 40.0, 60.0
    x1, y1 = fk_from_servo_angles(p1, s1, s2)
    x2, y2 = fk_from_servo_angles(p2, s1, s2)

    theta = math.radians(-delta)
    rx = x1 * math.cos(theta) - y1 * math.sin(theta)
    ry = x1 * math.sin(theta) + y1 * math.cos(theta)
    assert (x2, y2) == pytest.approx((rx, ry), abs=1e-6)
