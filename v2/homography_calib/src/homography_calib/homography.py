"""Pixel<->mm mapping fit from >=4 known (pixel, world_mm) correspondences,
and a drift check to detect the camera or the printed calibration sheet
having moved since the last fit."""

from __future__ import annotations

import math

import cv2
import numpy as np


def compute_homography(pixel_points: list[tuple[float, float]],
                        world_points: list[tuple[float, float]]) -> tuple[np.ndarray, float]:
    """Fit a pixel->mm homography from >=4 known correspondences.
    Returns (H, reprojection_rms_px) so callers can sanity-check fit quality."""
    if len(pixel_points) < 4:
        raise ValueError(f"homography needs >=4 point pairs, got {len(pixel_points)}")
    img_pts = np.array(pixel_points, dtype=np.float64)
    world_pts = np.array(world_points, dtype=np.float64)
    H, _ = cv2.findHomography(img_pts, world_pts, method=0)
    if H is None:
        raise ValueError("cv2.findHomography failed to converge")
    reproj = cv2.perspectiveTransform(img_pts.reshape(-1, 1, 2), H).reshape(-1, 2)
    rms_px = float(np.sqrt(np.mean(np.sum((reproj - world_pts) ** 2, axis=1))))
    return H, rms_px


def apply_homography(H: np.ndarray, pixel_xy: tuple[float, float]) -> tuple[float, float]:
    """Map one pixel coordinate to a workspace mm coordinate through H."""
    pt = np.array([[pixel_xy]], dtype=np.float64)
    out = cv2.perspectiveTransform(pt, H)[0, 0]
    return float(out[0]), float(out[1])


def homography_drift_mm(H_prev: np.ndarray,
                         measured_pixels: list[tuple[float, float]],
                         known_world: list[tuple[float, float]]) -> float:
    """How much has the camera/workspace geometry moved since H_prev was fit?

    Re-interpret freshly measured corner-tag pixel positions through the
    PREVIOUSLY fitted homography and compare against the corner tags' known
    (fixed) world coordinates. If nothing moved, H_prev still maps today's
    pixels to the right spot and drift is ~0mm. Takes the worst-case corner
    (not the average) so a single badly-shifted corner isn't diluted."""
    worst = 0.0
    for px, world in zip(measured_pixels, known_world):
        predicted = apply_homography(H_prev, px)
        err = math.hypot(predicted[0] - world[0], predicted[1] - world[1])
        worst = max(worst, err)
    return worst
