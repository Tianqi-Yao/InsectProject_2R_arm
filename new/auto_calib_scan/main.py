#!/usr/bin/env python3
"""CLI entry point. Each subcommand is thin glue between arm_hardware.py
(the black-box hardware layer), arm_core.py (kinematics/calibration/self-
check logic), and jog_controller.py (the shared real-time motion
controller) -- no decision-making lives in this file, only wiring and
printing.

`jog` used to be one of three independently hand-rolled point-motion
implementations (see jog_controller.py's docstring); it now drives an
ArmController like the manual_test/ frontends do, instead of snapping
directly to a target via Servos.move_and_wait. `calibrate`'s point-to-
point moves also go through the controller, for the same smoothing/wear
benefit -- but the spot-check moves inside arm_core.run_selfcheck are
left calling Servos.move_and_wait directly: those are one-shot diagnostic
poses, not a jog/scan the user watches, and it isn't worth widening
run_selfcheck's `hw` contract (well-tested, stable) for that.
"""

import argparse
import sys
import time
from dataclasses import asdict

import numpy as np

import arm_core as core
import arm_hardware as hw
import jog_controller as jc


def load_calib_and_hardware(args) -> tuple[dict, hw.ArmHardware, str]:
    """Resolve calib.json + build the ArmHardware bundle, consistently
    across subcommands. --port overrides calib.json's hardware.servo_port
    if given; joint_ids always come from calib.json's hardware section
    (arm_core.HardwareConfig) rather than being hardcoded per-command."""
    calib = core.load_calib()
    hw_cfg = core.calib_hardware_config(calib)
    port = args.port or hw_cfg.servo_port
    h = hw.ArmHardware(servo_port=port, joint_ids=hw_cfg.joint_ids)
    return calib, h, port


def cmd_test_servo(args):
    _calib, h, port = load_calib_and_hardware(args)
    h.servos.connect(port)
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

    _calib, h, _port = load_calib_and_hardware(args)
    h.camera.connect()

    if args.watch:
        print("watching for tags -- move/reprint as needed, Ctrl+C when done "
              "to save a final annotated snapshot...")
        try:
            while True:
                frame = h.camera.capture_gray()
                detections = h.detector.detect(frame)
                stamp = time.strftime("%H:%M:%S")
                if detections:
                    summary = ", ".join(f"id={tid}@({d.center[0]:.0f},{d.center[1]:.0f})"
                                         for tid, d in sorted(detections.items()))
                    print(f"[{stamp}] {len(detections)} tag(s): {summary}")
                else:
                    print(f"[{stamp}] no tags detected")
                time.sleep(args.interval)
        except KeyboardInterrupt:
            print("\nstopped, capturing final snapshot...")

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
    calib, h, _port = load_calib_and_hardware(args)
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
    calib, h, _port = load_calib_and_hardware(args)
    if calib["homography"].get("H") is None:
        print("ERROR: no homography yet -- run `main.py homography` first")
        sys.exit(1)
    H = np.array(calib["homography"]["H"])
    ee_tag_id = calib["workspace"]["ee_tag_id"]

    h.connect()
    controller = jc.build_controller(h.servos, calib)

    targets = core.generate_calibration_targets(
        params=core.calib_arm_params(calib), nx=args.nx, ny=args.ny,
        joint_limits=core.calib_joint_limits(calib))
    print(f"collecting up to {len(targets)} calibration points...")

    samples = []
    for i, target in enumerate(targets):
        controller.run_to_completion(target.servo1_deg, target.servo2_deg)
        time.sleep(args.settle_s)  # let physical vibration settle before the camera read
        s1 = h.servos.get_present_deg("joint1")
        s2 = h.servos.get_present_deg("joint2")
        frame = h.camera.capture_gray()
        detections = h.detector.detect(frame)
        ee = detections.get(ee_tag_id)
        if ee is None:
            print(f"  [{i + 1}/{len(targets)}] end-effector tag not visible, skipping")
            continue
        x, y = core.apply_homography(H, ee.center)
        samples.append(core.CalibSample(s1, s2, x, y))
        print(f"  [{i + 1}/{len(targets)}] s1={s1:.1f} s2={s2:.1f} -> ({x:.1f}, {y:.1f})mm")

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
    calib, h, _port = load_calib_and_hardware(args)
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


def _warn_if_wide_range(joint, lo, hi):
    if hi - lo > 300.0:
        print(f"  WARNING: that's a {hi - lo:.0f}deg-wide range -- if the dead zone "
              f"you're protecting against is narrow, this usually means {joint} "
              f"wandered near/through the 0/360 wraparound point while you were "
              f"moving it (this tool doesn't support a *safe* range that wraps -- "
              f"remount the horn so the DEAD ZONE straddles the wrap instead, and "
              f"try again keeping clear of the wrap point).")


def _measure_range_by_hand(servos, joint, poll_interval_s=0.05):
    """Disable torque on `joint`, live-track the min/max angle visited
    while the user moves it by hand, until Ctrl+C -- printing a
    continuously-updating "current / range seen so far" line so there's
    no need to guess ahead of time which position counts as "the boundary."

    Restores torque before returning -- but first re-syncs the goal
    position to wherever the joint actually is right now. Skipping that
    step would be a real bug: the servo's last GOAL_POSITION is whatever
    it was set to before torque was disabled (unrelated to wherever your
    hand left it), so re-enabling torque without first updating the goal
    would make the joint suddenly snap toward that stale old target the
    instant torque re-engages -- exactly the kind of surprise motion this
    whole feature exists to prevent.

    The joint is left holding wherever you stopped it -- useful on its own
    (e.g. for the coupled-zone flow below, where you deliberately stop at
    the position you want joint1 held at for the next measurement)."""
    servos.set_torque_enabled(joint, False)
    angle = servos.get_present_deg(joint)
    seen_min = seen_max = angle
    try:
        while True:
            angle = servos.get_present_deg(joint)
            seen_min = min(seen_min, angle)
            seen_max = max(seen_max, angle)
            print(f"\r  current: {angle:6.1f} deg   range seen so far: "
                  f"[{seen_min:6.1f}, {seen_max:6.1f}]", end="", flush=True)
            time.sleep(poll_interval_s)
    except KeyboardInterrupt:
        print()  # newline so the next print doesn't overwrite the live line
    finally:
        # Re-sync using `angle` (already updated by the loop above on every
        # iteration, or the initial read if Ctrl+C landed immediately)
        # rather than issuing a fresh read here -- a second read in this
        # `finally` could itself race with a second Ctrl+C and leave
        # torque disabled. Guaranteed to run no matter what happens above.
        servos.set_target_deg(joint, angle)  # hold current position, don't snap to a stale goal
        servos.set_torque_enabled(joint, True)
    return seen_min, seen_max


def _trace_boundary_by_hand(servos, poll_interval_s=0.05):
    """Disable BOTH joints' torque at once and continuously record
    (joint1_deg, joint2_deg) while you walk the arm by hand around the
    FULL PERIMETER of the safe region -- one closed loop, back to about
    where you started -- until Ctrl+C. Every sample gets kept exactly as
    recorded: no binning, smoothing, or min/max derivation -- the traced
    path IS the boundary (arm_core._point_in_polygon later decides
    in/out). This replaced an earlier automatic-derivation approach (bin
    a hand-swept fill by joint1, take min/max per bucket) that produced
    visibly wrong results on real hardware -- tracing the boundary by hand
    and using it as-is is simpler and puts you in full control of what
    gets enforced.

    Same re-sync-before-re-enable safety as _measure_range_by_hand, for
    both joints; Ctrl+C means "I'm done tracing," not a hard abort."""
    for joint in ("joint1", "joint2"):
        servos.set_torque_enabled(joint, False)
    # Seeded before the loop (mirroring _measure_range_by_hand) so `finally`
    # always has a last-known value to resync to, even if Ctrl+C lands
    # before the loop's first read completes.
    s1 = servos.get_present_deg("joint1")
    s2 = servos.get_present_deg("joint2")
    trace = []
    try:
        while True:
            s1 = servos.get_present_deg("joint1")
            s2 = servos.get_present_deg("joint2")
            trace.append((s1, s2))
            print(f"\r  joint1={s1:6.1f}  joint2={s2:6.1f}   ({len(trace)} samples)",
                  end="", flush=True)
            time.sleep(poll_interval_s)
    except KeyboardInterrupt:
        print()
    finally:
        # Re-sync using the last successfully-read s1/s2 (from the loop
        # above, or the seed read if Ctrl+C landed immediately) rather
        # than issuing fresh reads here -- a read in this `finally` could
        # itself race with a second Ctrl+C/comm failure and leave torque
        # disabled on one or both joints. Guaranteed to run regardless.
        servos.set_target_deg("joint1", s1)
        servos.set_target_deg("joint2", s2)
        for joint in ("joint1", "joint2"):
            servos.set_torque_enabled(joint, True)
    return trace


def _plot_boundary_trace(params, boundary, out_path="joint_limits_trace.png"):
    """Render the hand-traced (joint1, joint2) closed-loop boundary as real
    workspace (x, y) mm positions -- via the same forward kinematics
    (arm_core.fk_from_servo_angles) used everywhere else -- and save a PNG
    so you can visually confirm it actually matches your physical dead
    zone. Uses whatever L1/L2/base/offsets calib.json currently holds
    (rough is fine here -- this is a sanity-check plot, not a precision
    one). Drawn exactly in traced order, closed back to the first vertex
    -- this is the boundary as-is, no derivation applied.

    matplotlib is imported lazily here (like cv2 in cmd_test_camera) since
    it's only needed for this one optional step."""
    import math
    import matplotlib
    matplotlib.use("Agg")  # headless-safe: works over SSH with no display
    import matplotlib.pyplot as plt

    loop_xy = [core.fk_from_servo_angles(params, j1, j2) for j1, j2 in boundary]
    loop_xy.append(loop_xy[0])  # close the loop back to the start

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([p[0] for p in loop_xy], [p[1] for p in loop_xy],
            "-o", color="blue", markersize=3, linewidth=1.5, label="traced boundary")
    max_r = params.L1 + params.L2
    steps = [i / 200.0 * 2 * math.pi for i in range(201)]
    ax.plot([params.base_x + max_r * math.cos(t) for t in steps],
            [params.base_y + max_r * math.sin(t) for t in steps],
            "--", color="gray", linewidth=0.5, label="max reach")
    ax.plot(params.base_x, params.base_y, "ks", markersize=8, label="base")
    ax.set_aspect("equal")
    ax.legend(fontsize=8)
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_title("Hand-traced safe-region boundary (workspace mm)")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path
    return out_path


def cmd_set_joint_limits(args):
    """Bring-up procedure for mechanical dead-zone protection: for each
    joint, disable torque, then continuously read the encoder while you
    move it by hand -- the min/max angle actually visited is tracked live
    and reported when you press Ctrl+C for that joint. No need to guess
    which position is "the boundary" ahead of time: just sweep the joint's
    full safe range of motion (with a margin before the dead zone/
    obstruction on each end) and stop when you're satisfied you've covered it.

    Then, optionally, trace a COUPLED boundary: cases where the distal
    link's (joint2's) safe range *continuously* shrinks/grows depending on
    where the proximal link (joint1) is -- e.g. joint2's clearance to a
    fixed obstacle varies smoothly as joint1 sweeps, rather than being a
    fixed zone. Release BOTH joints' torque at once and walk the arm by
    hand around the FULL PERIMETER of the safe region, back to
    (approximately) where you started -- one closed loop
    (_trace_boundary_by_hand). The traced path IS the boundary: every
    sample gets saved as-is, no binning/derivation/smoothing applied to
    it. A pose is later allowed iff it falls INSIDE that traced polygon
    (arm_core._point_in_polygon) -- this handles an arbitrarily-shaped
    safe region, not just one where a single joint2 interval works for
    every joint1 (an earlier automatic-derivation approach couldn't
    represent that, which is why this one doesn't try to be clever at
    all: what you trace is exactly what gets enforced). A PNG diagram of
    the traced polygon, converted to real workspace mm via forward
    kinematics, is saved so you can visually confirm it matches your
    physical dead zone. For a live visual version of this same capture
    (watch the polygon draw itself in real time as you move the arm) use
    manual_test/trace_boundary_gui.py instead.

    Both joints' own "unconditional" ranges (measured before the coupled-
    boundary step, each in one single hand-swept pass) are automatically
    widened to the union of themselves and every traced vertex: hand
    sweeps aren't perfectly repeatable to a fraction of a degree between
    separate passes, so a vertex from the boundary trace can legitimately
    land slightly outside the first sweep's recorded range. Since the
    hardware registers can only hold one fixed range per joint (no notion
    of the other joint's position, or of "which pass" measured it), they
    need to cover every position actually confirmed safe, or they'd refuse
    something just tested.

    calib.json is saved FIRST, before any hardware register writes --
    a serial hiccup during the hardware verification step (which does
    happen occasionally) shouldn't cost you the measurements you just
    took by hand. The two joints' own unconditional ranges are then
    written to the servo's own hardware Min/Max Angle Limit registers
    too (the outermost, most trustworthy layer -- the servo firmware
    itself will then refuse to move past these, regardless of what
    software commands); a failure here is reported as a warning, not
    fatal, since calib.json's copy is already safely saved. The coupled
    boundary is SOFTWARE-ONLY: the STS3215's hardware angle-limit
    registers are per-servo and have no way to express "my limit depends
    on the other servo's position" at all -- if this coupled collision
    risk is serious, a physical mechanical stop is the only true
    hardware-level backstop for that specific case. See arm_hardware.py's
    and arm_core.py's module docstrings for more on why both layers matter.

    All angles here are in raw servo-degree space -- the same number
    `main.py test-servo` / 'w' in the jog REPL show, NOT the IK theta1/
    theta2 convention.
    """
    calib, h, port = load_calib_and_hardware(args)
    h.servos.connect(port)

    print("For each joint: torque will be disabled. Move it BY HAND through its")
    print("full safe range of motion, stopping a bit short of the dead zone/")
    print("obstruction on each end (don't push all the way to the edge --")
    print("leave yourself a margin). Press Ctrl+C when you're done with that joint.\n")

    results = {}
    for joint in ("joint1", "joint2"):
        print(f"--- {joint}: torque OFF, move it by hand now (Ctrl+C when done) ---")
        lo, hi = _measure_range_by_hand(h.servos, joint)
        print(f"  {joint} final safe range: [{lo:.1f}, {hi:.1f}] deg (raw servo angle)")
        _warn_if_wide_range(joint, lo, hi)
        results[joint] = (lo, hi)
        print()

    print("measured limits:", {j: [round(v, 1) for v in r] for j, r in results.items()})

    print("\n--- coupled/relative dead zone (optional) ---")
    print("If joint2's safe range continuously shrinks/grows depending on where")
    print("joint1 is, you can capture that here. This part is SOFTWARE-ONLY")
    print("protection -- the servo's own hardware registers can't express a")
    print("joint1-dependent joint2 limit. If this collision risk is serious, also")
    print("add a physical mechanical stop; don't rely on this configuration alone.\n")

    boundary = []  # list of (joint1_deg, joint2_deg) vertices, in traced order

    if input("trace a closed loop around the safe region's boundary? [y/N] ").strip().lower() == "y":
        print("Both joints' torque will be disabled. Walk the arm by hand around the")
        print("FULL PERIMETER of the safe region -- one continuous loop, back to about")
        print("where you started. Whatever you trace becomes the boundary exactly as")
        print("drawn, with no smoothing/derivation applied. Press Ctrl+C when done.\n")
        boundary = _trace_boundary_by_hand(h.servos)
        print(f"  recorded {len(boundary)} vertices")
        if len(boundary) < 3:
            print("  fewer than 3 vertices can't form a closed polygon -- discarding "
                  "(the two joints' own independent ranges above are unaffected).")
            boundary = []
        else:
            try:
                out_path = _plot_boundary_trace(core.calib_arm_params(calib), boundary)
                print(f"  saved a diagram to {out_path} -- open it and confirm the traced "
                      f"loop matches your physical dead zone (uses calib.json's current "
                      f"L1/L2/base -- rough is fine, this is a sanity-check plot, not a "
                      f"precision one)")
            except Exception as e:  # noqa: BLE001 -- plotting is optional, never fatal
                print(f"  WARNING: could not generate the diagram ({e}) -- the traced "
                      f"boundary itself is unaffected, continuing without it")
        print()

    if boundary:
        # Both joints' "own unconditional range" above were each measured
        # in one single hand-swept pass -- not perfectly repeatable to a
        # fraction of a degree between separate passes, so a traced vertex
        # can legitimately land slightly outside that first sweep's
        # recorded range. Since the hardware register can only hold ONE
        # fixed range per joint (no notion of the other joint's position,
        # or of "which pass" measured it), it needs to cover every
        # position actually confirmed safe, or it would wrongly refuse
        # something just tested.
        j1_lo = min([results["joint1"][0]] + [v[0] for v in boundary])
        j1_hi = max([results["joint1"][1]] + [v[0] for v in boundary])
        if (j1_lo, j1_hi) != results["joint1"]:
            print(f"  widening joint1's own range from [{results['joint1'][0]:.1f},"
                  f"{results['joint1'][1]:.1f}] to [{j1_lo:.1f},{j1_hi:.1f}] to include the "
                  f"traced boundary.")
            results["joint1"] = (j1_lo, j1_hi)

        j2_lo = min([results["joint2"][0]] + [v[1] for v in boundary])
        j2_hi = max([results["joint2"][1]] + [v[1] for v in boundary])
        if (j2_lo, j2_hi) != results["joint2"]:
            print(f"  widening joint2's own range from [{results['joint2'][0]:.1f},"
                  f"{results['joint2'][1]:.1f}] to [{j2_lo:.1f},{j2_hi:.1f}] to cover "
                  f"the traced boundary.")
            results["joint2"] = (j2_lo, j2_hi)

    # Save calib.json FIRST: this is the hard-won, hand-measured data, and
    # shouldn't be lost if the hardware verification step below hits a
    # transient serial error.
    calib["joint_limits_deg"] = {
        "joint1": [round(v, 2) for v in results["joint1"]],
        "joint2": [round(v, 2) for v in results["joint2"]],
        "coupled_boundary": [{"joint1": round(j1, 2), "joint2": round(j2, 2)} for j1, j2 in boundary],
    }
    answer = input("\nsave these to calib.json, and write the two joints' independent "
                    "ranges to the servo's hardware registers? [y/N] ").strip().lower()
    if answer != "y":
        print("not saved.")
        h.servos.close()
        return

    core.save_calib(calib)
    print("saved to calib.json")

    for joint, (lo, hi) in results.items():
        try:
            h.servos.set_hardware_angle_limits(joint, lo, hi)
            readback = h.servos.get_hardware_angle_limits(joint)
            print(f"  {joint}: wrote [{lo:.1f},{hi:.1f}] (hardware register), "
                  f"servo reports back {readback}")
        except IOError as e:
            print(f"  WARNING: {joint} hardware register write/verify failed ({e}). "
                  f"calib.json's software limit is already saved and will still protect "
                  f"this joint in IK/jog, but the hardware backstop may not be set -- try "
                  f"running `main.py set-joint-limits` again, or check the serial connection.")
    if boundary:
        print("  note: the coupled boundary was NOT written to any hardware register (not "
              "possible) -- it's enforced by calib.json's joint_limits_deg only.")

    h.servos.close()


def cmd_jog(args):
    calib, h, port = load_calib_and_hardware(args)
    h.servos.connect(port)
    controller = jc.build_controller(h.servos, calib)
    print("interactive jog (smoothed via the trapezoidal motion planner). "
          "'1 <deg>' / '2 <deg>' set joint1/joint2 target, "
          "'w' shows current angles, 'q' quits")
    while True:
        try:
            raw = input("jog> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if raw in ("q", "quit"):
            break
        if raw in ("w", "where"):
            real = {j: round(h.servos.get_present_deg(j), 2) for j in h.servos.joint_ids}
            print(f"real={real}  commanded={controller.commanded_deg}")
            continue
        parts = raw.split()
        if len(parts) == 2 and parts[0] in ("1", "2"):
            joint = f"joint{parts[0]}"
            try:
                angle = float(parts[1])
            except ValueError:
                print("usage: 1 <deg> | 2 <deg> | w | q")
                continue
            if not controller.set_single_joint_goal(joint, angle):
                print(f"REJECTED: {joint}={angle} is unreachable or outside "
                      f"the configured joint_limits_deg -- nothing sent")
                continue
            while controller.is_moving:
                controller.tick()
                time.sleep(controller.dt)
            print(f"reached: {controller.commanded_deg}")
        else:
            print("usage: 1 <deg> | 2 <deg> | w | q")
    h.servos.close()


def main():
    parser = argparse.ArgumentParser(description="2R arm control CLI")
    parser.add_argument("--port", default=None,
                         help="servo bus serial port (default: calib.json's hardware.servo_port)")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("test-servo", help="ping + read/write smoke test")

    p_cam = sub.add_parser("test-camera", help="capture + detect tags, save annotated image")
    p_cam.add_argument("--out", default=None)
    p_cam.add_argument("--watch", action="store_true",
                        help="loop printing live detections to the terminal before saving "
                             "the final snapshot (Ctrl+C to stop)")
    p_cam.add_argument("--interval", type=float, default=0.5, help="seconds between --watch captures")

    sub.add_parser("homography", help="detect 4 corner tags, fit and save homography")

    p_cal = sub.add_parser("calibrate", help="auto-collect samples and fit kinematic parameters")
    p_cal.add_argument("--nx", type=int, default=6)
    p_cal.add_argument("--ny", type=int, default=5)
    p_cal.add_argument("--settle-s", type=float, default=1.0)

    sub.add_parser("selfcheck", help="boot-time homography drift + arm spot-check")

    sub.add_parser("set-joint-limits",
                    help="bring-up: measure and write mechanical dead-zone safe ranges "
                         "(both the servo's hardware registers and calib.json)")

    sub.add_parser("jog", help="interactive manual angle jog for debugging")

    args = parser.parse_args()
    dispatch = {
        "test-servo": cmd_test_servo,
        "test-camera": cmd_test_camera,
        "homography": cmd_homography,
        "calibrate": cmd_calibrate,
        "selfcheck": cmd_selfcheck,
        "set-joint-limits": cmd_set_joint_limits,
        "jog": cmd_jog,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
