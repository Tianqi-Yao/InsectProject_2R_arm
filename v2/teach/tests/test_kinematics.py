import pytest

from teach.kinematics import ArmParams, fk_from_servo_angles, ik_solve


def test_fk_ik_round_trip():
    p = ArmParams(L1=125.0, L2=95.0, base_x=100.0, base_y=-45.0,
                   servo1_offset_deg=23.08, servo2_offset_deg=0.0)
    x, y = 150.0, 60.0
    r = ik_solve(p, x, y)
    assert r.reachable
    assert fk_from_servo_angles(p, r.servo1_deg, r.servo2_deg) == pytest.approx((x, y), abs=1e-6)


@pytest.mark.parametrize("elbow_offset_mm", [0.0, 12.5, -8.0])
def test_fk_ik_round_trip_with_elbow_offset(elbow_offset_mm):
    p = ArmParams(L1=125.0, L2=95.0, base_x=100.0, base_y=-45.0,
                   servo1_offset_deg=23.08, servo2_offset_deg=0.0,
                   elbow_offset_mm=elbow_offset_mm)
    x, y = 140.0, 50.0
    r = ik_solve(p, x, y)
    assert r.reachable
    assert fk_from_servo_angles(p, r.servo1_deg, r.servo2_deg) == pytest.approx((x, y), abs=1e-6)


def test_ik_rejects_unreachable_point():
    p = ArmParams.nominal()
    r = ik_solve(p, p.base_x + p.L1 + p.L2 + 1000.0, 0.0)
    assert r.reachable is False
