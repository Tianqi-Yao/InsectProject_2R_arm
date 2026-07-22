"""Quick manual test: jog the arm and run a scan sweep, skipping the
camera/AprilTag/calibration pipeline entirely -- just to physically watch
it move and eyeball whether things are in the right ballpark.

All physical parameters (L1/L2/base/offsets/dirs), hardware settings
(servo port, joint bus IDs), and motion tuning (jog/scan speeds, scan grid
size, corner-blend threshold) come from calib.json -- run `main.py
homography`/`calibrate` first if you have one, or just let this read
calib.json's built-in nominal defaults. This used to hardcode its own copy
of all of those constants (and manual_test/gui.py hardcoded a second,
independently-drifted copy); now both tools -- and main.py -- read the
same single source of truth.

This is a thin adapter over jog_controller.ArmController: it only
translates curses keypresses into controller calls and renders the
controller's state. All of the actual motion planning/smoothing/scanning
logic lives in jog_controller.py + motion_planning/, shared with
manual_test/gui.py and main.py's jog REPL.
"""

import curses
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import arm_core as core          # noqa: E402
import arm_hardware as hw         # noqa: E402
import jog_controller as jc       # noqa: E402

STEP_MIN, STEP_MAX = 0.5, 40.0


def run_jog(controller: jc.ArmController, calib: dict):
    # (center_x, center_y, width, height, rotation_deg) -- see manual_test/scan_area_gui.py
    scan_cx, scan_cy, scan_w, scan_h, scan_rot = core.calib_scan_area(calib)
    scan_path = core.generate_scan_path(
        width_mm=scan_w, height_mm=scan_h,
        nx=controller.motion_cfg.scan_nx, ny=controller.motion_cfg.scan_ny,
        margin_mm=controller.motion_cfg.scan_margin_mm,
        rows_limit=controller.motion_cfg.scan_rows_limit,
        center_x_mm=scan_cx, center_y_mm=scan_cy, rotation_deg=scan_rot)
    home = (scan_cx, scan_cy)

    def loop(stdscr):
        curses.curs_set(0)
        stdscr.keypad(True)
        # Non-blocking with a timeout matching the control period: getch()
        # either returns a pressed key immediately, or -1 after dt_ms with
        # nothing pressed -- either way we tick() every iteration, so an
        # active jog/scan segment keeps advancing smoothly whether or not
        # the user is actively pressing anything.
        stdscr.nodelay(True)
        stdscr.timeout(int(controller.dt * 1000))

        # ArmController only tracks "the joint goal"; the workspace-space
        # target is a frontend concept (see nudge_workspace's docstring),
        # so this script owns it -- seeded from the arm's real current
        # position so the first nudge is relative to wherever it actually
        # is, not a jump to some hardcoded default.
        workspace_target = core.fk_from_servo_angles(controller.params, *controller.commanded_deg)
        step = 5.0

        def draw():
            r = core.ik_solve(controller.params, *workspace_target)
            stdscr.erase()
            stdscr.addstr(0, 0, "arrows: move   [ ]: step size   h: home   "
                                 "t: positioning test   q: quit")
            stdscr.addstr(1, 0, f"step: {step:.1f} mm")
            stdscr.addstr(3, 0, f"target: ({workspace_target[0]:.1f}, {workspace_target[1]:.1f}) mm")
            if r.reachable:
                stdscr.addstr(4, 0, f"theta1={r.theta1_deg:6.1f}  theta2={r.theta2_deg:6.1f}")
                stdscr.addstr(5, 0, f"servo1={r.servo1_deg:6.1f}  servo2={r.servo2_deg:6.1f}")
            else:
                stdscr.addstr(4, 0, "NOT REACHABLE with current L1/L2/base -- ignored")
            if controller.scan_active:
                done, total = controller.scan_progress
                stdscr.addstr(7, 0, f"positioning test: point {done}/{total}  "
                                     f"('q' abort, 'h' home)")
            stdscr.refresh()

        draw()  # show current state only -- nothing sent on startup
        while True:
            key = stdscr.getch()  # returns -1 after dt if nothing pressed

            if controller.scan_active:
                if key in (ord('q'), 27):
                    controller.stop_scan()
                elif key == ord('h'):
                    controller.stop_scan()
                    if controller.set_workspace_goal(*home):
                        workspace_target = home
                # any other key (or none) during a scan: just keep ticking
            elif key in (ord('q'), 27):
                break
            elif key == curses.KEY_UP:
                new_t = controller.nudge_workspace(0.0, step, workspace_target)
                workspace_target = new_t if new_t else workspace_target
            elif key == curses.KEY_DOWN:
                new_t = controller.nudge_workspace(0.0, -step, workspace_target)
                workspace_target = new_t if new_t else workspace_target
            elif key == curses.KEY_LEFT:
                new_t = controller.nudge_workspace(-step, 0.0, workspace_target)
                workspace_target = new_t if new_t else workspace_target
            elif key == curses.KEY_RIGHT:
                new_t = controller.nudge_workspace(step, 0.0, workspace_target)
                workspace_target = new_t if new_t else workspace_target
            elif key in (ord(']'), ord('+')):
                step = min(step * 1.5, STEP_MAX)
            elif key in (ord('['), ord('-')):
                step = max(step / 1.5, STEP_MIN)
            elif key == ord('h'):
                if controller.set_workspace_goal(*home):
                    workspace_target = home
            elif key == ord('t'):
                controller.start_scan(scan_path)

            controller.tick()
            draw()

    curses.wrapper(loop)


def main():
    calib = core.load_calib()
    hw_cfg = core.calib_hardware_config(calib)
    servos = hw.Servos(hw_cfg.joint_ids)
    servos.connect(hw_cfg.servo_port)

    controller = jc.build_controller(servos, calib)
    p = controller.params
    print(f"connected. L1={p.L1} L2={p.L2} base=({p.base_x},{p.base_y}) "
          f"offsets=({p.servo1_offset_deg:.1f},{p.servo2_offset_deg:.1f}) "
          f"dirs=({p.servo1_dir},{p.servo2_dir})")
    print("nothing will move until you press an arrow key or 't' -- "
          "starting jog screen...")
    try:
        run_jog(controller, calib)
    finally:
        servos.close()


if __name__ == "__main__":
    main()
