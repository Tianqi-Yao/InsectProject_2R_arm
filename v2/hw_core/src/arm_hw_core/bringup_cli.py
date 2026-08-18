"""`arm-hw` CLI: the safety-critical bring-up procedures that only this
package is allowed to perform (writing the servo's own EEPROM angle-limit
registers, and the software joint_limits_deg every feature package reads
from hw_state.json). Deliberately has no kinematics/FK -- that lives in
each feature package, not here; this package only knows raw servo-degree
space.
"""

from __future__ import annotations

import argparse
import time

from . import hw_state as hws
from .servos import Servos


def _measure_range_by_hand(servos: Servos, joint: str, poll_interval_s: float = 0.05
                            ) -> tuple[float, float]:
    """Disable torque on `joint`, live-track the min/max angle visited
    while the user moves it by hand, until Ctrl+C.

    Restores torque before returning -- but first re-syncs the goal
    position to wherever the joint actually is right now. Skipping that
    step would let the joint snap toward its last (stale) commanded
    position the instant torque re-engages."""
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
        print()
    finally:
        servos.set_target_deg(joint, angle)  # hold current position, don't snap to a stale goal
        servos.set_torque_enabled(joint, True)
    return seen_min, seen_max


def _trace_boundary_by_hand(servos: Servos, poll_interval_s: float = 0.05
                             ) -> list[tuple[float, float]]:
    """Disable BOTH joints' torque at once and continuously record
    (joint1_deg, joint2_deg) while the user walks the arm by hand around
    the FULL PERIMETER of the safe region -- one closed loop -- until
    Ctrl+C. Every sample is kept exactly as recorded: no binning,
    smoothing, or min/max derivation. See limits.point_in_polygon for why
    this is later tested with winding-number, not even-odd."""
    for joint in ("joint1", "joint2"):
        servos.set_torque_enabled(joint, False)
    s1 = servos.get_present_deg("joint1")
    s2 = servos.get_present_deg("joint2")
    trace: list[tuple[float, float]] = []
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
        servos.set_target_deg("joint1", s1)
        servos.set_target_deg("joint2", s2)
        for joint in ("joint1", "joint2"):
            servos.set_torque_enabled(joint, True)
    return trace


def cmd_set_joint_limits(args: argparse.Namespace) -> None:
    state = hws.load()
    servos = Servos(state.joint_ids)
    servos.connect(state.servo_port)

    print("For each joint: torque will be disabled. Move it BY HAND through its")
    print("full safe range of motion, stopping a bit short of the dead zone/")
    print("obstruction on each end. Press Ctrl+C when done with that joint.\n")

    results: dict[str, tuple[float, float]] = {}
    for joint in ("joint1", "joint2"):
        print(f"--- {joint}: torque OFF, move it by hand now (Ctrl+C when done) ---")
        lo, hi = _measure_range_by_hand(servos, joint)
        print(f"  {joint} final safe range: [{lo:.1f}, {hi:.1f}] deg (raw servo angle)")
        results[joint] = (lo, hi)
        print()

    print("measured limits:", {j: [round(v, 1) for v in r] for j, r in results.items()})

    print("\n--- coupled/relative dead zone (optional, software-only) ---")
    print("If joint2's safe range continuously shrinks/grows depending on where")
    print("joint1 is, capture that here. The servo's own hardware registers")
    print("can't express a joint1-dependent joint2 limit -- if this collision")
    print("risk is serious, also add a physical mechanical stop.\n")

    boundary: list[tuple[float, float]] = []
    if input("trace a closed loop around the safe region's boundary? [y/N] ").strip().lower() == "y":
        print("Both joints' torque will be disabled. Walk the arm by hand around the")
        print("FULL PERIMETER of the safe region -- one continuous loop. Ctrl+C when done.\n")
        boundary = _trace_boundary_by_hand(servos)
        print(f"  recorded {len(boundary)} vertices")
        if len(boundary) < 3:
            print("  fewer than 3 vertices can't form a closed polygon -- discarding.")
            boundary = []
        print()

    if boundary:
        # Each joint's "own unconditional range" above was measured in one
        # hand-swept pass -- not perfectly repeatable to a fraction of a
        # degree between separate passes, so a traced vertex can land
        # slightly outside that first sweep's range. The hardware register
        # can only hold one fixed range per joint, so widen to cover
        # everything actually confirmed safe.
        j1_lo = min([results["joint1"][0]] + [v[0] for v in boundary])
        j1_hi = max([results["joint1"][1]] + [v[0] for v in boundary])
        results["joint1"] = (j1_lo, j1_hi)
        j2_lo = min([results["joint2"][0]] + [v[1] for v in boundary])
        j2_hi = max([results["joint2"][1]] + [v[1] for v in boundary])
        results["joint2"] = (j2_lo, j2_hi)

    state.joint_limits_deg = {
        "joint1": [round(v, 2) for v in results["joint1"]],
        "joint2": [round(v, 2) for v in results["joint2"]],
        "coupled_boundary": [[round(j1, 2), round(j2, 2)] for j1, j2 in boundary],
    }

    answer = input("\nsave these to hw_state.json, and write the two joints' independent "
                    "ranges to the servo's hardware registers? [y/N] ").strip().lower()
    if answer != "y":
        print("not saved.")
        servos.close()
        return

    # Save FIRST, before any hardware register writes: this is the
    # hard-won, hand-measured data, and shouldn't be lost if the hardware
    # verification step below hits a transient serial error.
    hws.save(state)
    print(f"saved to {hws.DEFAULT_PATH}")

    for joint, (lo, hi) in results.items():
        try:
            servos.set_hardware_angle_limits(joint, lo, hi)
            readback = servos.get_hardware_angle_limits(joint)
            print(f"  {joint}: wrote [{lo:.1f},{hi:.1f}] (hardware register), "
                  f"servo reports back {readback}")
        except IOError as e:
            print(f"  WARNING: {joint} hardware register write/verify failed ({e}). "
                  f"hw_state.json's software limit is already saved and will still "
                  f"protect this joint, but the hardware backstop may not be set -- "
                  f"try again, or check the serial connection.")
    if boundary:
        print("  note: the coupled boundary is NOT written to any hardware register "
              "(not possible) -- it's enforced by hw_state.json's joint_limits_deg only.")

    servos.close()


def cmd_test_servo(args: argparse.Namespace) -> None:
    state = hws.load()
    servos = Servos(state.joint_ids)
    servos.connect(state.servo_port)
    print("connected. reading present angles...")
    for joint in ("joint1", "joint2"):
        print(f"  {joint}: {servos.get_present_deg(joint):.2f} deg")
    servos.close()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="arm-hw")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("set-joint-limits", help="hand-sweep bring-up: measure + write joint limits")
    sub.add_parser("test-servo", help="ping both servos and print their present angles")
    args = parser.parse_args(argv)
    {"set-joint-limits": cmd_set_joint_limits, "test-servo": cmd_test_servo}[args.command](args)


if __name__ == "__main__":
    main()
