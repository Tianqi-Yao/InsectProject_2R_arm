"""Minimal terminal-only record/replay: release torque, hand-drag the arm
to each position you want, pressing Enter to mark it as a point; then
re-engage torque and replay through those points in order, pausing
`dwell_s` at each one. No IK/FK, no calibration, no workspace coordinates,
no joint_limits check -- every recorded point was physically visited by
hand already (torque was off when it was marked), so it's inherently
reachable and safe to replay as-is.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import arm_core as core
import arm_hardware as hw
import jog_controller as jc
import motion_planning as mp

DEFAULT_PATH = Path(__file__).parent / "recorded_path.json"
DEFAULT_DWELL_S = 2.0

# No calib.json for v4 -- edit these two if your rig differs (or pass
# --port on the CLI). Everything else (link lengths, offsets, ...) is
# irrelevant here: record/replay never touches kinematics.
DEFAULT_SERVO_PORT = "/dev/cu.usbserial-0001"
DEFAULT_JOINT_IDS = {"joint1": 1, "joint2": 2}


def _connect(servo_port: str, joint_ids: dict) -> hw.Servos:
    servos = hw.Servos(joint_ids)
    servos.connect(servo_port)
    return servos


def _resync_and_relock(servos: hw.Servos) -> None:
    """Sync each joint's goal register to its actual position before
    re-enabling torque, so it doesn't snap toward a stale old target."""
    for joint in ("joint1", "joint2"):
        angle = servos.get_present_deg(joint)
        servos.set_target_deg(joint, angle)
    for joint in ("joint1", "joint2"):
        servos.set_torque_enabled(joint, True)


def record(out_path: Path = DEFAULT_PATH, servo_port: str = DEFAULT_SERVO_PORT,
           joint_ids: dict = None) -> None:
    """Torque off; drag the arm to each position you want, pressing Enter
    (blank input) to mark it as a point -- move on, drag to the next
    position, Enter again, repeat. Type q + Enter to finish."""
    joint_ids = joint_ids or DEFAULT_JOINT_IDS
    servos = _connect(servo_port, joint_ids)
    for joint in ("joint1", "joint2"):
        servos.set_torque_enabled(joint, False)
    print("torque released -- drag the arm to each point you want, marking it as you go.")
    print("Enter = mark the current position as a point; q + Enter = finish.\n")

    points = []
    try:
        while True:
            cmd = input(f"[{len(points)} point(s) marked] Enter to mark, q to finish: ").strip().lower()
            if cmd in ("q", "quit"):
                break
            s1 = servos.get_present_deg("joint1")
            s2 = servos.get_present_deg("joint2")
            points.append([s1, s2])
            print(f"  marked point {len(points)}: ({s1:.1f}, {s2:.1f})")
    except (KeyboardInterrupt, EOFError):
        print("\ninterrupted -- stopping recording")
    finally:
        _resync_and_relock(servos)
        servos.close()

    out_path.write_text(json.dumps(points, indent=2))
    print(f"recorded {len(points)} point(s) -> {out_path}")


def replay(in_path: Path = DEFAULT_PATH, servo_port: str = DEFAULT_SERVO_PORT,
           joint_ids: dict = None, dwell_s: float = DEFAULT_DWELL_S) -> None:
    """Visit each recorded point in order: move there, wait for a full
    stop, pause dwell_s, then move on to the next one."""
    joint_ids = joint_ids or DEFAULT_JOINT_IDS
    points = json.loads(in_path.read_text())
    if not points:
        print(f"{in_path} has no recorded points")
        return

    servos = _connect(servo_port, joint_ids)
    _resync_and_relock(servos)

    params = core.ArmParams.nominal()  # unused by joint-space replay, just satisfies the constructor
    motion_cfg = core.MotionConfig()
    planner = mp.get_planner(motion_cfg.planner_name)
    controller = jc.ArmController(servos, params, planner, motion_cfg, joint_limits=None)
    controller.resync(servos.get_present_deg("joint1"), servos.get_present_deg("joint2"))

    print(f"replaying {len(points)} point(s), {dwell_s:.1f}s dwell at each...")
    try:
        for i, (j1, j2) in enumerate(points):
            controller.set_joint_goal(j1, j2)
            while controller.is_moving:
                controller.tick()
                time.sleep(controller.dt)
            print(f"  point {i + 1}/{len(points)} reached -- dwelling {dwell_s:.1f}s")
            time.sleep(dwell_s)
    finally:
        servos.close()
    print("replay done.")
