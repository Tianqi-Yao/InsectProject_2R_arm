import pytest

from record_replay.angles import wrap_angle_near


def test_wrap_angle_near_picks_the_short_way_across_the_seam():
    assert wrap_angle_near(1.0, 359.0) == pytest.approx(361.0)


def test_wrap_angle_near_no_wrap_needed():
    assert wrap_angle_near(50.0, 45.0) == pytest.approx(50.0)
