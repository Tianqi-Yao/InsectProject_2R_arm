#!/usr/bin/env python3
"""CLI entry point. Each subcommand is thin glue between arm_hardware.py
(the black-box hardware layer) and arm_core.py (all the actual logic) --
no decision-making lives in this file, only wiring and printing."""

import argparse
import sys
import time
from dataclasses import asdict

import numpy as np

import arm_core as core
import arm_hardware as hw


def build_hardware(args) -> hw.ArmHardware:
    # Matches the bus IDs actually flashed on the two servos (see ServoJog.ino).
    return hw.ArmHardware(servo_port=args.port, joint_ids={"joint1": 1, "joint2": 4})


def cmd_test_servo(args):
    h = build_hardware(args)
    h.servos.connect(args.port)
    print("connected. reading present angles...")
    for joint in ("joint1", "joint2"):
        print(f"  {joint}: {h.servos.get_present_deg(joint):.2f} deg")
    current = h.servos.get_present_deg("joint1")
    target = {"joint1": current + 5.0}
    print(f"nudging joint1 by +5deg and waiting to settle: {target}")
    reached = h.servos.move_and_wait(target)
    print(f"reached: {reached}")
    h.servos.close()


def cmd_test_camera(args):
    import cv2

    h = build_hardware(args)
    h.camera.connect()
    frame = h.camera.capture_gray()
    detections = h.detector.detect(frame)
    print(f"detected {len(detections)} tag(s): {sorted(detections)}")

    annotated = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    for tag_id, det in detections.items():
        cx, cy = det.center
        cv2.circle(annotated, (int(cx), int(cy)), 6, (0, 0, 255), 2)
        cv2.putText(annotated, str(tag_id), (int(cx) + 8, int(cy)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    out_path = args.out or "tag_detect.jpg"
    cv2.imwrite(out_path, annotated)
    print(f"annotated image saved to {out_path}")
    h.camera.close()


def cmd_homography(args):
    calib = core.load_calib()
    h = build_hardware(args)
    h.camera.connect()
    frame = h.camera.capture_gray()
    detections = h.detector.detect(frame)

    ws = calib["workspace"]
    corner_ids = ws["corner_tag_ids"]
    corner_world = ws["corner_world_mm"]
    missing = [name for name, tid in corner_ids.items() if tid not in detections]
    if missing:
        print(f"ERROR: corner tag(s) not detected: {missing}")
        h.camera.close()
        sys.exit(1)

    pixels = [detections[tid].center for tid in corner_ids.values()]
    worlds = [tuple(corner_world[str(tid)]) for tid in corner_ids.values()]
    H, rms_px = core.compute_homography(pixels, worlds)
    print(f"homography fit: reprojection RMS = {rms_px:.2f}px")

    calib["homography"] = {"H": H.tolist(),
                            "computed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                            "reproj_rms_px": rms_px}
    core.save_calib(calib)
    print("saved to calib.json")
    h.camera.close()


def cmd_calibrate(args):
    calib = core.load_calib()
    if calib["homography"].get("H") is None:
        print("ERROR: no homography yet -- run `main.py homography` first")
        sys.exit(1)
    H = np.array(calib["homography"]["H"])
    ee_tag_id = calib["workspace"]["ee_tag_id"]

    h = build_hardware(args)
    h.connect()

    targets = core.generate_calibration_targets(
        params=core.calib_arm_params(calib), nx=args.nx, ny=args.ny)
    print(f"collecting up to {len(targets)} calibration points...")

    samples = []
    for i, target in enumerate(targets):
        reached = h.servos.move_and_wait({"joint1": target.servo1_deg, "joint2": target.servo2_deg})
        time.sleep(args.settle_s)
        frame = h.camera.capture_gray()
        detections = h.detector.detect(frame)
        ee = detections.get(ee_tag_id)
        if ee is None:
            print(f"  [{i + 1}/{len(targets)}] end-effector tag not visible, skipping")
            continue
        x, y = core.apply_homography(H, ee.center)
        samples.append(core.CalibSample(reached["joint1"], reached["joint2"], x, y))
        print(f"  [{i + 1}/{len(targets)}] s1={reached['joint1']:.1f} s2={reached['joint2']:.1f} "
              f"-> ({x:.1f}, {y:.1f})mm")

    h.close()

    print(f"\ncollected {len(samples)} usable samples, fitting...")
    report = core.fit_kinematics(samples, x0=core.calib_arm_params(calib))

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

    answer = input("\nwrite this fit to calib.json? [y/N] ").strip().lower()
    if answer != "y":
        print("not saved.")
        return

    calib["kinematics"] = {**asdict(p),
                           "fit_report": {"rms_error_mm": report.rms_error_mm,
                                          "max_error_mm": report.max_error_mm,
                                          "n_points": report.n_points}}
    core.save_calib(calib)
    print("saved to calib.json")


def cmd_selfcheck(args):
    calib = core.load_calib()
    h = build_hardware(args)
    h.connect()
    result = core.run_selfcheck(h, calib)
    h.close()
    if result.ok:
        print(f"OK  homography_drift={result.homography_drift_mm} "
              f"spotcheck_errors={result.spotcheck_errors_mm}")
        sys.exit(0)
    else:
        print(f"HALTED  reason={result.reason}")
        sys.exit(1)


def cmd_jog(args):
    h = build_hardware(args)
    h.servos.connect(args.port)
    print("interactive jog. '1 <deg>' / '2 <deg>' set joint1/joint2 target, "
          "'w' shows current angles, 'q' quits")
    while True:
        try:
            raw = input("jog> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if raw in ("q", "quit"):
            break
        if raw in ("w", "where"):
            print({j: round(h.servos.get_present_deg(j), 2) for j in h.servos.joint_ids})
            continue
        parts = raw.split()
        if len(parts) == 2 and parts[0] in ("1", "2"):
            joint = f"joint{parts[0]}"
            try:
                angle = float(parts[1])
            except ValueError:
                print("usage: 1 <deg> | 2 <deg> | w | q")
                continue
            reached = h.servos.move_and_wait({joint: angle})
            print(f"reached: {reached}")
        else:
            print("usage: 1 <deg> | 2 <deg> | w | q")
    h.servos.close()


def main():
    parser = argparse.ArgumentParser(description="2R arm control CLI")
    parser.add_argument("--port", default="/dev/ttyUSB0", help="servo bus serial port")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("test-servo", help="ping + read/write smoke test")

    p_cam = sub.add_parser("test-camera", help="capture + detect tags, save annotated image")
    p_cam.add_argument("--out", default=None)

    sub.add_parser("homography", help="detect 4 corner tags, fit and save homography")

    p_cal = sub.add_parser("calibrate", help="auto-collect samples and fit kinematic parameters")
    p_cal.add_argument("--nx", type=int, default=6)
    p_cal.add_argument("--ny", type=int, default=5)
    p_cal.add_argument("--settle-s", type=float, default=1.0)

    sub.add_parser("selfcheck", help="boot-time homography drift + arm spot-check")

    sub.add_parser("jog", help="interactive manual angle jog for debugging")

    args = parser.parse_args()
    dispatch = {
        "test-servo": cmd_test_servo,
        "test-camera": cmd_test_camera,
        "homography": cmd_homography,
        "calibrate": cmd_calibrate,
        "selfcheck": cmd_selfcheck,
        "jog": cmd_jog,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
