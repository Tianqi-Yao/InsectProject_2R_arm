import pytest

from teach.angles import normalize_deg, wrap_angle_near


def test_normalize_deg_wraps_to_0_360():
    assert normalize_deg(-10.0) == 350.0
    assert normalize_deg(370.0) == 10.0


def test_wrap_angle_near_picks_the_short_way_across_the_seam():
    assert wrap_angle_near(1.0, 359.0) == pytest.approx(361.0)
