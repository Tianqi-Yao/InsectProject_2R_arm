"""`arm-cardscan` CLI: detect the card (live tag, or manual-corner
fallback), generate a serpentine grid, and drive the arm node by node."""

from __future__ import annotations

import argparse
import time

from arm_hw_core import hw_state as hws
from arm_hw_core.apriltag import TagDetector
from arm_hw_core.camera import build_camera

from . import config as cfg
from .controller import ArmController, MotionParams
from .homography_read import load_homography
from .kinematics_read import load_params
from .scan import PathRunner, detect_card_rect, generate_scan_path, rect_from_corners


def _resolve_card_rect(card_cfg: cfg.CardScanConfig, detections: dict, H):
    if card_cfg.tag_id is not None:
        rect = detect_card_rect(detections, card_cfg.tag_id, card_cfg.width_mm,
                                 card_cfg.height_mm, H)
        if rect is not None:
            return rect
        print("WARNING: card tag not detected this frame.")
    if card_cfg.manual_corner_a_mm is not None:
        print("falling back to manually-taught corners.")
        return rect_from_corners(tuple(card_cfg.manual_corner_a_mm),
                                  tuple(card_cfg.manual_corner_b_mm))
    return None


def cmd_run(args: argparse.Namespace) -> None:
    hw = hws.load()
    card_cfg = cfg.load()
    H = load_homography()
    params = load_params()

    from arm_hw_core.servos import Servos

    servos = Servos(hw.joint_ids)
    servos.connect(hw.servo_port)
    camera = build_camera(hw.camera_backend, hw.camera_resolution, hw.usb_camera_index)
    detector = TagDetector()
    camera.connect()

    try:
        frame = camera.capture_gray()
        detections = detector.detect(frame)
        card_rect = _resolve_card_rect(card_cfg, detections, H)
        if card_rect is None:
            print("ERROR: could not determine the card's rectangle (no tag detected, "
                  "no manual-corner fallback configured)")
            return

        nodes = generate_scan_path(card_rect, rows=card_cfg.rows, cols=card_cfg.cols)
        print(f"card rect: center=({card_rect[0]:.1f},{card_rect[1]:.1f}) "
              f"size=({card_rect[2]:.1f}x{card_rect[3]:.1f})mm rotation={card_rect[4]:.1f}deg")
        print(f"scanning {len(nodes)} node(s), {card_cfg.dwell_s:.1f}s dwell at each...")

        motion = MotionParams()
        controller = ArmController(servos, params, motion, joint_limits=hw.joint_limits_deg)
        if hw.joint_limits_deg is None:
            print("WARNING: no joint_limits_deg configured -- scan will not reject any "
                  "target on mechanical dead-zone grounds.")

        runner = PathRunner(controller, nodes, dwell_s=card_cfg.dwell_s)
        while not runner.done:
            runner.tick(time.monotonic())
            time.sleep(controller.dt)
    finally:
        camera.close()
        servos.close()
    print("scan done.")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="arm-cardscan")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("run", help="detect the card and scan its surface")
    args = parser.parse_args(argv)
    {"run": cmd_run}[args.command](args)


if __name__ == "__main__":
    main()
