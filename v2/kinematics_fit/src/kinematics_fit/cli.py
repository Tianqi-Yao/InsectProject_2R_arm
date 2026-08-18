"""`arm-kinfit` CLI: collect calibration samples and fit L1/L2/base/servo
offsets (`run`), and Tier 2 of the two-tier self-check pattern -- an
arm-position spot-check against known poses (`selfcheck`; Tier 1, the
homography drift check, lives in homography_calib)."""

from __future__ import annotations

import argparse
import math
import sys
import time

from arm_hw_core import hw_state as hws
from arm_hw_core.apriltag import TagDetector
from arm_hw_core.camera import build_camera
from arm_hw_core.servos import Servos

from . import config as cfg
from .controller import ArmController, MotionParams
from .fit import CalibSample, fit_kinematics, generate_calibration_targets
from .homography_read import apply_homography, load_homography
from .kinematics import fk_from_servo_angles


def cmd_run(args: argparse.Namespace) -> None:
    hw = hws.load()
    calib = cfg.load()
    H = load_homography()

    servos = Servos(hw.joint_ids)
    servos.connect(hw.servo_port)
    camera = build_camera(hw.camera_backend, hw.camera_resolution, hw.usb_camera_index)
    detector = TagDetector()
    camera.connect()

    if hw.joint_limits_deg is None:
        print("WARNING: no joint_limits_deg configured (run `arm-hw set-joint-limits` first) -- "
              "collection will not reject any target on mechanical dead-zone grounds.")

    motion = MotionParams(vmax_deg_s=calib.vmax_deg_s, amax_deg_s2=calib.amax_deg_s2,
                           control_hz=calib.control_hz)
    controller = ArmController(servos, motion, joint_limits=hw.joint_limits_deg)

    targets = generate_calibration_targets(
        params=calib.params(), nx=args.nx or calib.grid_nx, ny=args.ny or calib.grid_ny,
        margin_mm=calib.grid_margin_mm, joint_limits=hw.joint_limits_deg)
    print(f"collecting up to {len(targets)} calibration points...")

    settle_s = args.settle_s if args.settle_s is not None else calib.settle_s
    samples = []
    for i, target in enumerate(targets):
        controller.run_to_completion(target.servo1_deg, target.servo2_deg)
        time.sleep(settle_s)  # let physical vibration settle before the camera read
        s1 = servos.get_present_deg("joint1")
        s2 = servos.get_present_deg("joint2")
        frame = camera.capture_gray()
        detections = detector.detect(frame)
        ee = detections.get(calib.ee_tag_id)
        if ee is None:
            print(f"  [{i + 1}/{len(targets)}] end-effector tag not visible, skipping")
            continue
        x, y = apply_homography(H, ee.center)
        samples.append(CalibSample(s1, s2, x, y))
        print(f"  [{i + 1}/{len(targets)}] s1={s1:.1f} s2={s2:.1f} -> ({x:.1f}, {y:.1f})mm")

    camera.close()
    servos.close()

    print(f"\ncollected {len(samples)} usable samples, fitting...")
    report = fit_kinematics(samples, x0=calib.params())

    print("\n--- calibration report ---")
    for i, err in enumerate(sorted(report.per_point_error_mm, reverse=True)):
        print(f"  point {i}: {err:.2f}mm")
    print(f"RMS error : {report.rms_error_mm:.2f}mm")
    print(f"max error : {report.max_error_mm:.2f}mm")
    p = report.params
    print(f"fitted    : L1={p.L1:.2f} L2={p.L2:.2f} base=({p.base_x:.2f},{p.base_y:.2f}) "
          f"offsets=({p.servo1_offset_deg:.2f},{p.servo2_offset_deg:.2f})")
    quality = "good" if report.rms_error_mm < 1.0 else ("ok" if report.rms_error_mm < 3.0 else "poor")
    print(f"quality   : {quality}  (<1mm good / 1-3mm ok / >3mm poor -- recheck tags/mounting)")

    answer = input("\nwrite this fit to kinematics_calib.json? [y/N] ").strip().lower()
    if answer != "y":
        print("not saved.")
        return

    calib.L1, calib.L2, calib.base_x, calib.base_y = p.L1, p.L2, p.base_x, p.base_y
    calib.servo1_offset_deg, calib.servo2_offset_deg = p.servo1_offset_deg, p.servo2_offset_deg
    calib.fit_report = cfg.FitReportSummary(n_points=report.n_points,
                                             rms_error_mm=report.rms_error_mm,
                                             max_error_mm=report.max_error_mm)
    cfg.save(calib)
    print(f"saved to {cfg.DEFAULT_PATH}")


def cmd_selfcheck(args: argparse.Namespace) -> None:
    hw = hws.load()
    calib = cfg.load()
    if not calib.spotcheck_poses:
        print("ERROR: no spotcheck_poses configured in kinematics_calib.json")
        sys.exit(1)
    H = load_homography()

    servos = Servos(hw.joint_ids)
    servos.connect(hw.servo_port)
    camera = build_camera(hw.camera_backend, hw.camera_resolution, hw.usb_camera_index)
    detector = TagDetector()
    camera.connect()

    params = calib.params()
    errors = []
    ok = True
    try:
        for pose in calib.spotcheck_poses:
            servos.move_and_wait(pose)
            s1 = servos.get_present_deg("joint1")
            s2 = servos.get_present_deg("joint2")
            predicted = fk_from_servo_angles(params, s1, s2)

            frame = camera.capture_gray()
            detections = detector.detect(frame)
            ee = detections.get(calib.ee_tag_id)
            if ee is None:
                print(f"WARNING: end-effector tag not visible during spot-check at {pose}")
                continue

            measured = apply_homography(H, ee.center)
            err_mm = math.hypot(predicted[0] - measured[0], predicted[1] - measured[1])
            errors.append(err_mm)
            print(f"  pose={pose} predicted=({predicted[0]:.1f},{predicted[1]:.1f}) "
                  f"measured=({measured[0]:.1f},{measured[1]:.1f}) error={err_mm:.2f}mm")
            if err_mm >= calib.arm_position_halt_mm:
                print(f"HALT: arm position error {err_mm:.2f}mm >= threshold "
                      f"{calib.arm_position_halt_mm}mm")
                ok = False
                break
    finally:
        camera.close()
        servos.close()

    if not ok:
        sys.exit(1)
    print(f"OK: spot-check errors {[round(e, 2) for e in errors]}mm, all under threshold")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="arm-kinfit")
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="collect calibration samples and fit kinematics")
    p_run.add_argument("--nx", type=int, default=None)
    p_run.add_argument("--ny", type=int, default=None)
    p_run.add_argument("--settle-s", type=float, default=None)

    sub.add_parser("selfcheck", help="Tier 2: arm-position spot-check against known poses")

    args = parser.parse_args(argv)
    {"run": cmd_run, "selfcheck": cmd_selfcheck}[args.command](args)


if __name__ == "__main__":
    main()
