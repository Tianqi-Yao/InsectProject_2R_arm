"""`arm-homography` CLI: fit the pixel<->mm mapping from the 4 corner
AprilTags, and a boot-time self-check (this is Tier 1 of the two-tier
self-check pattern -- see kinematics_fit for Tier 2, the arm-position
spot-check, which needs fitted kinematics this package doesn't have)."""

from __future__ import annotations

import argparse
import sys
import time

from arm_hw_core import hw_state as hws
from arm_hw_core.apriltag import TagDetector
from arm_hw_core.camera import build_camera

from . import config as cfg
from .homography import compute_homography, homography_drift_mm


def _detect_corners(calib: cfg.WorkspaceCalib):
    """Connects the camera, detects all 4 corner tags, and returns
    (pixel_points, world_points) in corner_tag_ids' tl/tr/br/bl order.
    Raises RuntimeError naming any corner tag that wasn't seen."""
    state = hws.load()
    camera = build_camera(state.camera_backend, state.camera_resolution, state.usb_camera_index)
    detector = TagDetector()
    camera.connect()
    try:
        frame = camera.capture_gray()
    finally:
        camera.close()
    detections = detector.detect(frame)

    pixel_points = []
    world_points = calib.corner_world_points()
    for corner, tag_id in calib.corner_tag_ids.items():
        if tag_id not in detections:
            raise RuntimeError(f"corner tag {corner!r} (id={tag_id}) was not detected -- "
                                f"check lighting/focus and that all 4 corner tags are visible")
        pixel_points.append(detections[tag_id].center)
    return pixel_points, world_points


def cmd_fit(args: argparse.Namespace) -> None:
    calib = cfg.load()
    pixel_points, world_points = _detect_corners(calib)
    H, rms_px = compute_homography(pixel_points, world_points)
    calib.H = H.tolist()
    calib.computed_at = time.strftime("%Y-%m-%dT%H:%M:%S")
    calib.reproj_rms_px = rms_px
    cfg.save(calib)
    print(f"homography fit: reprojection RMS = {rms_px:.2f}px, saved to {cfg.DEFAULT_PATH}")
    if rms_px > 3.0:
        print("WARNING: reprojection RMS is high -- check tag detection quality "
              "(focus, lighting, corner_world_mm accuracy) before trusting this fit.")


def cmd_selfcheck(args: argparse.Namespace) -> None:
    import numpy as np

    calib = cfg.load()
    if calib.H is None:
        print("ERROR: no homography fitted yet -- run `arm-homography fit` first")
        sys.exit(1)

    pixel_points, world_points = _detect_corners(calib)
    H_prev = np.array(calib.H)
    drift_mm = homography_drift_mm(H_prev, pixel_points, world_points)

    if drift_mm >= calib.drift_halt_mm:
        print(f"HALT: homography drift {drift_mm:.2f}mm >= threshold "
              f"{calib.drift_halt_mm}mm -- camera or calibration sheet may have moved. "
              f"Run `arm-homography fit` again after confirming/fixing the physical setup.")
        sys.exit(1)

    # Self-heal: adopt the freshly measured homography even though drift
    # was below threshold, so small accumulated drift never compounds
    # silently across many selfcheck runs.
    H_new, rms_px = compute_homography(pixel_points, world_points)
    calib.H = H_new.tolist()
    calib.computed_at = time.strftime("%Y-%m-%dT%H:%M:%S")
    calib.reproj_rms_px = rms_px
    cfg.save(calib)
    print(f"OK: drift={drift_mm:.2f}mm (< {calib.drift_halt_mm}mm threshold), "
          f"self-healed homography saved (reprojection RMS={rms_px:.2f}px)")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="arm-homography")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("fit", help="detect the 4 corner tags and fit+save the homography")
    sub.add_parser("selfcheck", help="re-measure corner tags, self-heal or halt on drift")
    args = parser.parse_args(argv)
    {"fit": cmd_fit, "selfcheck": cmd_selfcheck}[args.command](args)


if __name__ == "__main__":
    main()
