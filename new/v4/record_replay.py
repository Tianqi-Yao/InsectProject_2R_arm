"""Minimal terminal-only record/replay: release torque, hand-drag the arm
along a path, record it; then re-engage torque and replay the exact same
joint-angle sequence. No IK/FK, no calibration, no workspace coordinates,
no joint_limits check -- every recorded point was physically visited by
hand already (torque was off when it was sampled), so it's inherently
reachable and safe to replay as-is.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import arm_core as core
import arm_hardware as hw
import jog_controller as jc
import motion_planning as mp

SAMPLE_HZ = 10.0
DEFAULT_PATH = Path(__file__).parent / "recorded_path.json"

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
    joint_ids = joint_ids or DEFAULT_JOINT_IDS
    servos = _connect(servo_port, joint_ids)
    for joint in ("joint1", "joint2"):
        servos.set_torque_enabled(joint, False)
    print("torque released -- drag the arm through the path you want recorded.")

    stop_flag = {"stop": False}

    def _wait_for_enter():
        input("recording at %gHz... press Enter to stop\n" % SAMPLE_HZ)
        stop_flag["stop"] = True

    threading.Thread(target=_wait_for_enter, daemon=True).start()

    points = []
    period = 1.0 / SAMPLE_HZ
    try:
        while not stop_flag["stop"]:
            s1 = servos.get_present_deg("joint1")
            s2 = servos.get_present_deg("joint2")
            points.append([s1, s2])
            time.sleep(period)
    except KeyboardInterrupt:
        print("\ninterrupted -- stopping recording")
    finally:
        _resync_and_relock(servos)
        servos.close()

    out_path.write_text(json.dumps(points, indent=2))
    print(f"recorded {len(points)} points -> {out_path}")


def replay(in_path: Path = DEFAULT_PATH, servo_port: str = DEFAULT_SERVO_PORT,
           joint_ids: dict = None) -> None:
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

    if len(points) == 1:
        controller.set_joint_goal(points[0][0], points[0][1])
    else:
        controller.start_joint_scan(points)

    print(f"replaying {len(points)} points...")
    last_progress = -1
    try:
        while controller.is_moving:
            controller.tick()
            time.sleep(controller.dt)
            completed, total = controller.scan_progress
            if completed != last_progress and total:
                print(f"  {completed}/{total}")
                last_progress = completed
    finally:
        servos.close()
    print("replay done.")
