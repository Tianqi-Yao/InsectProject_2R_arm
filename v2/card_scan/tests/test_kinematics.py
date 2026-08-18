import pytest

from card_scan.kinematics import ArmParams, fk_from_servo_angles, ik_solve


def test_fk_ik_round_trip():
    p = ArmParams(L1=125.0, L2=95.0, base_x=100.0, base_y=-45.0,
                   servo1_offset_deg=23.08, servo2_offset_deg=0.0)
    x, y = 150.0, 60.0
    r = ik_solve(p, x, y)
    assert r.reachable
    assert fk_from_servo_angles(p, r.servo1_deg, r.servo2_deg) == pytest.approx((x, y), abs=1e-6)


def test_ik_rejects_unreachable_point():
    p = ArmParams.nominal()
    r = ik_solve(p, p.base_x + p.L1 + p.L2 + 1000.0, 0.0)
    assert r.reachable is False


def test_ik_rejects_point_outside_joint_limits():
    p = ArmParams.nominal()
    limits = {"joint1": (0.0, 1.0), "joint2": (0.0, 360.0)}
    assert ik_solve(p, 150.0, 30.0, joint_limits=limits).reachable is False
