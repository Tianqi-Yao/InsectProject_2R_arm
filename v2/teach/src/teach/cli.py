"""`arm-teach` CLI: interactive hand-teach REPL. No camera dependency --
this mode exists specifically for when AprilTag calibration isn't
available or hasn't been trusted yet."""

from __future__ import annotations

import argparse

from arm_hw_core import hw_state as hws
from arm_hw_core.servos import Servos

from . import config as cfg
from .controller import ArmController, MotionParams
from .kinematics_read import load_params
from .session import TeachSession


def cmd_run(args: argparse.Namespace) -> None:
    hw = hws.load()
    teach_config = cfg.load()
    params = load_params()

    servos = Servos(hw.joint_ids)
    servos.connect(hw.servo_port)
    motion = MotionParams()
    controller = ArmController(servos, motion, joint_limits=hw.joint_limits_deg)
    session = TeachSession(servos, controller, teach_config, params=params)
    session.start()

    print("torque released -- move the arm by hand. Commands:")
    print("  t <label>   mark the current position as <label>")
    print("  g <label>   move (under torque) to a previously taught point")
    print("  list        show taught points")
    print("  del <label> remove a taught point")
    print("  save        write teach.json")
    print("  q           save (if changed) and quit")

    try:
        while True:
            try:
                raw = input("teach> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not raw:
                continue
            parts = raw.split(maxsplit=1)
            cmd, rest = parts[0], (parts[1] if len(parts) > 1 else "").strip()

            if cmd == "q":
                break
            elif cmd == "t":
                if not rest:
                    print("usage: t <label>")
                    continue
                point = session.mark(rest)
                print(f"  marked {point.label!r}: joint1={point.joint1_deg:.1f} "
                      f"joint2={point.joint2_deg:.1f}"
                      + (f"  (x={point.x_mm:.1f} y={point.y_mm:.1f}mm)" if point.x_mm is not None else ""))
            elif cmd == "g":
                if not rest:
                    print("usage: g <label>")
                    continue
                try:
                    reached = session.goto(rest)
                    print(f"  reached joint1={reached[0]:.1f} joint2={reached[1]:.1f}")
                except KeyError as e:
                    print(f"  {e}")
            elif cmd == "list":
                if not teach_config.points:
                    print("  (no taught points yet)")
                for p in teach_config.points:
                    print(f"  {p.label}: joint1={p.joint1_deg:.1f} joint2={p.joint2_deg:.1f}")
            elif cmd == "del":
                if not rest:
                    print("usage: del <label>")
                    continue
                print("  removed" if teach_config.remove(rest) else f"  no such point {rest!r}")
            elif cmd == "save":
                cfg.save(teach_config)
                print(f"  saved to {cfg.DEFAULT_PATH}")
            else:
                print(f"  unknown command {cmd!r}")
    finally:
        session.stop()
        cfg.save(teach_config)
        print(f"saved to {cfg.DEFAULT_PATH}")
        servos.close()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="arm-teach")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("run", help="interactive hand-teach session")
    args = parser.parse_args(argv)
    {"run": cmd_run}[args.command](args)


if __name__ == "__main__":
    main()
