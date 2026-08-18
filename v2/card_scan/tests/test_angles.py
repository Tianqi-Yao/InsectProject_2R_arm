import pytest

from card_scan.angles import normalize_deg, rotate_vector, wrap_angle_near


def test_normalize_deg_wraps_to_0_360():
    assert normalize_deg(-10.0) == 350.0


def test_wrap_angle_near_picks_the_short_way_across_the_seam():
    assert wrap_angle_near(1.0, 359.0) == pytest.approx(361.0)


def test_rotate_vector_90_degrees():
    assert rotate_vector(1.0, 0.0, 90.0) == pytest.approx((0.0, 1.0), abs=1e-9)


def test_rotate_vector_zero_is_identity():
    assert rotate_vector(3.0, -4.0, 0.0) == pytest.approx((3.0, -4.0))
