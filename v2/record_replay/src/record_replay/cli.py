"""`arm-replay` CLI: record a joint-space path by hand, replay it (with
optional photo capture per stop), and manage the cron-scheduled unattended
replay timer."""

from __future__ import annotations

import argparse
import shutil
from datetime import datetime
from pathlib import Path

from arm_hw_core import hw_state as hws
from arm_hw_core.camera import build_camera
from arm_hw_core.servos import Servos

from . import config as cfg
from . import cron
from .controller import ArmController, MotionParams
from .recorder import Player, Recorder


def cmd_record(args: argparse.Namespace) -> None:
    hw = hws.load()
    servos = Servos(hw.joint_ids)
    servos.connect(hw.servo_port)

    recorder = Recorder(servos)
    recorder.start()
    print("torque released -- drag the arm to each point you want, marking it as you go.")
    print("Enter = mark the current position as a point; q + Enter = finish.\n")

    path_cfg = cfg.load()
    points = []
    try:
        while True:
            raw = input(f"[{len(points)} point(s) marked] Enter to mark, q to finish: ").strip().lower()
            if raw in ("q", "quit"):
                break
            point = recorder.mark()
            points.append(point)
            print(f"  marked point {len(points)}: ({point.joint1_deg:.1f}, {point.joint2_deg:.1f})")
    except (KeyboardInterrupt, EOFError):
        print("\ninterrupted -- stopping recording")
    finally:
        recorder.stop()
        servos.close()

    path_cfg.points = points
    cfg.save(path_cfg)
    print(f"recorded {len(points)} point(s) -> {cfg.DEFAULT_PATH}")


def cmd_run(args: argparse.Namespace) -> None:
    hw = hws.load()
    path_cfg = cfg.load()
    if not path_cfg.points:
        print(f"{cfg.DEFAULT_PATH} has no recorded points -- run `arm-replay record` first")
        return

    camera = None
    run_dir = None
    if args.photos is not None:
        if not (0.0 < path_cfg.photo_delay_s < path_cfg.dwell_s):
            print(f"ERROR: photo_delay_s ({path_cfg.photo_delay_s}) must be between 0 and "
                  f"dwell_s ({path_cfg.dwell_s}) for a photo to fit inside the dwell")
            return
        run_dir = Path(args.photos) / datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir.mkdir(parents=True, exist_ok=True)
        camera = build_camera(hw.camera_backend, hw.camera_resolution, hw.usb_camera_index)
        camera.connect()

    servos = Servos(hw.joint_ids)
    servos.connect(hw.servo_port)
    motion = MotionParams()
    controller = ArmController(servos, motion)
    player = Player(servos, controller, camera=camera)
    player.prepare()

    print(f"replaying {len(path_cfg.points)} point(s), {path_cfg.dwell_s:.1f}s dwell at each...")
    try:
        for i, point in enumerate(path_cfg.points):
            photo_path = (run_dir / f"point_{i + 1:03d}.jpg") if run_dir is not None else None
            player.goto_and_dwell(point, path_cfg.dwell_s, photo_path=photo_path,
                                   photo_delay_s=path_cfg.photo_delay_s)
            suffix = f" -- photo saved -> {photo_path}" if photo_path is not None else ""
            print(f"  point {i + 1}/{len(path_cfg.points)} done{suffix}")
    finally:
        servos.close()
        if camera is not None:
            camera.close()
    print("replay done.")


def cmd_cron(args: argparse.Namespace) -> None:
    if args.cron_command == "install":
        python_bin = args.python or shutil.which("python3")
        if not python_bin:
            print("ERROR: could not find python3 on PATH -- pass --python")
            return
        arm_replay = shutil.which("arm-replay") or f"{python_bin} -m record_replay.cli"
        photos_arg = f" --photos {args.photos}" if args.photos else ""
        command = f"{arm_replay} run{photos_arg} >> {args.log_file} 2>&1"
        schedule = f"*/{args.interval_min} * * * *"
        line = cron.install(schedule, command)
        print(f"installed: {line}")
    elif args.cron_command == "list":
        line = cron.list_installed()
        print(line if line else "(no arm2r replay timer installed)")
    elif args.cron_command == "uninstall":
        removed = cron.uninstall()
        print("removed." if removed else "(nothing to remove)")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="arm-replay")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("record", help="torque off, hand-drag + mark points, save recorded_path.json")

    p_run = sub.add_parser("run", help="replay the recorded path")
    p_run.add_argument("--photos", type=str, default=None,
                        help="directory to save a per-stop photo into (needs a connected camera)")

    p_cron = sub.add_parser("cron", help="manage the unattended-replay cron timer")
    cron_sub = p_cron.add_subparsers(dest="cron_command", required=True)
    p_install = cron_sub.add_parser("install")
    p_install.add_argument("--interval-min", type=int, default=30)
    p_install.add_argument("--photos", type=str, default=None)
    p_install.add_argument("--python", type=str, default=None)
    p_install.add_argument("--log-file", type=str,
                            default=str(Path.home() / ".config" / "arm2r" / "replay.log"))
    cron_sub.add_parser("list")
    cron_sub.add_parser("uninstall")

    args = parser.parse_args(argv)
    if args.command == "cron":
        cmd_cron(args)
    else:
        {"record": cmd_record, "run": cmd_run}[args.command](args)


if __name__ == "__main__":
    main()
