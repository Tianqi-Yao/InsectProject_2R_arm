import pytest

from kinematics_fit.angles import normalize_deg, wrap_angle_near


def test_normalize_deg_wraps_to_0_360():
    assert normalize_deg(-10.0) == 350.0
    assert normalize_deg(370.0) == 10.0


def test_wrap_angle_near_picks_the_short_way_across_the_seam():
    # target=1, reference=359 -> nearest equivalent is 361 (2deg away),
    # not 1 (358deg away via naive subtraction).
    assert wrap_angle_near(1.0, 359.0) == pytest.approx(361.0)


def test_wrap_angle_near_no_wrap_needed():
    assert wrap_angle_near(50.0, 45.0) == pytest.approx(50.0)


def test_wrap_angle_near_result_is_within_180_of_reference():
    for target in (0.0, 90.0, 179.9, 180.1, 270.0, 359.9):
        for reference in (0.0, 90.0, 180.0, 270.0, 355.0):
            result = wrap_angle_near(target, reference)
            assert abs(result - reference) <= 180.0 + 1e-9
            # result must be congruent to target mod 360 -- compare via
            # minimal circular distance so a diff landing near 360 (the
            # "other side" of the same wraparound point, e.g. 359.999999
            # from floating-point rounding) isn't mistaken for a mismatch.
            diff = (result - target) % 360.0
            circular_dist = min(diff, 360.0 - diff)
            assert circular_dist == pytest.approx(0.0, abs=1e-6)
