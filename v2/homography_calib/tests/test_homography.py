import numpy as np
import pytest

from homography_calib.homography import apply_homography, compute_homography, homography_drift_mm

# A simple axis-aligned scale+translate mapping: world = pixel * 0.5 + (10, 20),
# i.e. pixel (0,0)->world(10,20), pixel(100,0)->world(60,20), etc.
PIXELS = [(0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)]
WORLD = [(10.0, 20.0), (60.0, 20.0), (60.0, 70.0), (10.0, 70.0)]


def test_compute_homography_requires_at_least_4_points():
    with pytest.raises(ValueError, match=">=4"):
        compute_homography(PIXELS[:3], WORLD[:3])


def test_compute_homography_fits_exact_mapping_with_zero_reprojection_error():
    H, rms_px = compute_homography(PIXELS, WORLD)
    assert rms_px == pytest.approx(0.0, abs=1e-6)


def test_apply_homography_matches_known_mapping():
    H, _ = compute_homography(PIXELS, WORLD)
    for px, world in zip(PIXELS, WORLD):
        mapped = apply_homography(H, px)
        assert mapped == pytest.approx(world, abs=1e-6)
    # An interior point not in the fit set should also map correctly for
    # this affine (scale+translate) case.
    assert apply_homography(H, (50.0, 50.0)) == pytest.approx((35.0, 45.0), abs=1e-6)


def test_homography_drift_mm_is_zero_when_nothing_moved():
    H, _ = compute_homography(PIXELS, WORLD)
    drift = homography_drift_mm(H, PIXELS, WORLD)
    assert drift == pytest.approx(0.0, abs=1e-6)


def test_homography_drift_mm_reports_worst_case_corner():
    H, _ = compute_homography(PIXELS, WORLD)
    # Corner 0 unchanged, corner 1 "moved" by reporting a pixel that maps
    # 5mm away from its known world position.
    shifted_pixels = list(PIXELS)
    shifted_pixels[1] = (110.0, 0.0)  # would map to world (65, 20), i.e. 5mm off
    drift = homography_drift_mm(H, shifted_pixels, WORLD)
    assert drift == pytest.approx(5.0, abs=1e-6)
