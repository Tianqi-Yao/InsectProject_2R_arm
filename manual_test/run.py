"""Quick manual test: drive the arm to (x, y) points using L1/L2/base values
you type in below -- no camera, no AprilTag, no calibration, no calib.json.
Just to physically see the arm move and eyeball whether the numbers are in
the right ballpark before bothering with the full vision-calibration setup.

Edit the constants below to match your arm, then run this file. It reuses
the IK geometry from software/arm_core.py and the servo driver from
software/arm_hardware.py, but touches nothing else there (no camera, no
calibration state).
"""

import curses
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "software"))
import arm_core as core   # noqa: E402
import arm_hardware as hw  # noqa: E402

# --- Edit these to match your arm ---
SERVO_PORT = "/dev/cu.usbserial-0001"
JOINT_IDS = {"joint1": 1, "joint2": 2}

L1 = 125.0            # mm, shoulder -> elbow (still a rough CAD guess, not calibrated)
L2 = 95.0              # mm, elbow -> end effector (ditto)
BASE_X = 100.0         # mm, shoulder position in workspace coords (rough guess, ditto)
BASE_Y = -45.0

# Reference pose: L1 and L2 lined up straight (elbow fully extended, our
# theta2=0) is where you set the servos' physical zero via ServoJog, and
# both read ~2040 ticks there. That's the offset. theta1=0 is defined as
# "wherever the shoulder happened to be pointing" at that same pose -- we
# don't know its true relationship to the workspace axes yet (that's what
# real calibration is for), this is just enough to move the arm sensibly.
TICKS_AT_STRAIGHT_LINE = 2040
SERVO1_OFFSET_DEG = TICKS_AT_STRAIGHT_LINE * hw.DEG_PER_TICK
SERVO2_OFFSET_DEG = TICKS_AT_STRAIGHT_LINE * hw.DEG_PER_TICK

# joint1's jog direction matched our math convention; joint2's was reversed.
SERVO1_DIR = 1
SERVO2_DIR = -1
# -------------------------------------

PARAMS = core.ArmParams(L1=L1, L2=L2, base_x=BASE_X, base_y=BASE_Y,
                         servo1_offset_deg=SERVO1_OFFSET_DEG,
                         servo2_offset_deg=SERVO2_OFFSET_DEG,
                         servo1_dir=SERVO1_DIR, servo2_dir=SERVO2_DIR)

# Serpentine (boustrophedon) scan for the 't' positioning test: starts at
# the top-left (x=0 side, y=max/"away from base" side, matching the
# corner_world_mm "tl" convention used elsewhere) and snakes row by row --
# left-to-right, down, right-to-left, down, ... -- covering the workspace
# with no wasted travel back to a row's start. Points unreachable for
# whatever L1/L2/base you've set are skipped with a warning, same as before.
WS_W, WS_H = 200.0, 150.0
SCAN_MARGIN_MM = 20.0
SCAN_NX, SCAN_NY = 5, 4
# Set to e.g. 3 to only run the first 3 rows (same row spacing as the full
# SCAN_NY grid, just fewer of them) -- handy for a quick check before
# committing to the full sweep. None runs every row.
SCAN_ROWS_LIMIT = None


def _generate_serpentine_points(width_mm, height_mm, nx, ny, margin_mm, rows_limit=None):
    xs = [margin_mm + i * (width_mm - 2 * margin_mm) / (nx - 1) for i in range(nx)]
    ys = [height_mm - margin_mm - j * (height_mm - 2 * margin_mm) / (ny - 1) for j in range(ny)]
    if rows_limit is not None:
        ys = ys[:rows_limit]
    points = []
    for row, y in enumerate(ys):
        row_xs = xs if row % 2 == 0 else list(reversed(xs))
        for x in row_xs:
            points.append((x, y, f"row{row + 1}"))
    return points


DEMO_POINTS = _generate_serpentine_points(WS_W, WS_H, SCAN_NX, SCAN_NY, SCAN_MARGIN_MM, SCAN_ROWS_LIMIT)


STEP_MIN, STEP_MAX = 0.5, 40.0

# Arrow-key jog moves: slower than the servo's raw default, plus a bit of
# acceleration ramp (0 = snap straight to speed and stop dead on arrival,
# which is what was feeling too abrupt/shaky). Tune JOG_ACC up for a more
# gradual start/stop, down (toward 0) for snappier response.
JOG_SPEED = 400
JOG_ACC = 30

# How long to wait after sending each scan point before sending the next
# one -- deliberately *not* waiting for the servo to actually settle there.
# With a dense grid the points are only a few mm apart, so continuously
# feeding a slightly-ahead target makes the servo track smoothly instead of
# decelerating to a stop at every point (which is what a settle-and-wait
# per point looks like: visible jitter/stutter). Tune this together with
# SCAN_NX/SCAN_NY: a denser grid wants a shorter interval to stay smooth; a
# sparser one needs a longer interval or it'll dash through without
# actually getting anywhere close to each point.
SCAN_STEP_INTERVAL_S = 0.03

# Slower than the default jog speed (800) on purpose: at full speed the
# servo tries to sprint toward each point and is still accelerating (or
# starting to brake for arrival) when the next one replaces it, which
# looks like jitter even with continuous streaming. A gentler speed suits
# the short hops between closely-spaced scan points better.
SCAN_SPEED = 300


def run_positioning_test(stdscr, servos, state):
    """The 't'-triggered sweep through DEMO_POINTS. Only runs when you ask
    for it -- never automatically -- since a preset sweep can wind cables
    up in ways a small manual nudge won't. Streams targets at a fixed pace
    (see SCAN_STEP_INTERVAL_S) rather than waiting for each one to settle.
    'q' aborts in place; 'h' aborts and sends the arm home."""
    stdscr.nodelay(True)  # so getch() below can poll without blocking
    try:
        for i, (x, y, label) in enumerate(DEMO_POINTS):
            r = core.ik_solve(PARAMS, x, y)
            if r.reachable:
                servos.set_target_deg("joint1", r.servo1_deg, speed=SCAN_SPEED)
                servos.set_target_deg("joint2", r.servo2_deg, speed=SCAN_SPEED)
                state["x"], state["y"] = x, y
            stdscr.erase()
            stdscr.addstr(0, 0, f"positioning test {i + 1}/{len(DEMO_POINTS)}: "
                                 f"{label} ({x:.0f}, {y:.0f})  -- 'q' abort, 'h' home")
            if not r.reachable:
                stdscr.addstr(2, 0, "not reachable, skipping")
            stdscr.refresh()
            key = stdscr.getch()
            if key == ord('q'):
                break
            if key == ord('h'):
                hr = core.ik_solve(PARAMS, HOME_X, HOME_Y)
                if hr.reachable:
                    servos.set_target_deg("joint1", hr.servo1_deg, speed=JOG_SPEED, acc=JOG_ACC)
                    servos.set_target_deg("joint2", hr.servo2_deg, speed=JOG_SPEED, acc=JOG_ACC)
                    state["x"], state["y"] = HOME_X, HOME_Y
                break
            time.sleep(SCAN_STEP_INTERVAL_S)
    finally:
        stdscr.nodelay(False)
    stdscr.erase()
    stdscr.addstr(0, 0, "positioning test done -- press any key to resume jogging")
    stdscr.refresh()
    stdscr.getch()


def run_jog(servos):
    """Arrow-key manual control: Up/Down/Left/Right nudge the target (x, y)
    and the arm follows live. '[' / ']' change step size, 'h' recentres,
    't' runs the DEMO_POINTS positioning test, 'q' quits.

    Nothing is sent to the servos until you press a key -- on startup this
    reads the arm's *actual current* position and just displays it, so the
    first arrow press is a small nudge from wherever it already is, not a
    jump to some hardcoded default.
    """

    def loop(stdscr):
        curses.curs_set(0)
        stdscr.keypad(True)

        s1 = servos.get_present_deg("joint1")
        s2 = servos.get_present_deg("joint2")
        x0, y0 = core.fk_from_servo_angles(PARAMS, s1, s2)
        state = {"x": x0, "y": y0, "step": 5.0}

        def draw(send):
            r = core.ik_solve(PARAMS, state["x"], state["y"])
            stdscr.erase()
            stdscr.addstr(0, 0, "arrows: move   [ ]: step size   h: home   "
                                 "t: positioning test   q: quit")
            stdscr.addstr(1, 0, f"step: {state['step']:.1f} mm")
            stdscr.addstr(3, 0, f"target: ({state['x']:.1f}, {state['y']:.1f}) mm")
            if r.reachable:
                if send:
                    servos.set_target_deg("joint1", r.servo1_deg, speed=JOG_SPEED, acc=JOG_ACC)
                    servos.set_target_deg("joint2", r.servo2_deg, speed=JOG_SPEED, acc=JOG_ACC)
                stdscr.addstr(4, 0, f"theta1={r.theta1_deg:6.1f}  theta2={r.theta2_deg:6.1f}")
                stdscr.addstr(5, 0, f"servo1={r.servo1_deg:6.1f}  servo2={r.servo2_deg:6.1f}")
            else:
                stdscr.addstr(4, 0, "NOT REACHABLE with current L1/L2/base -- ignored")
            stdscr.refresh()

        draw(send=False)  # show current state only, nothing sent on startup
        while True:
            key = stdscr.getch()
            if key in (ord('q'), 27):
                break
            elif key == curses.KEY_UP:
                state["y"] += state["step"]
            elif key == curses.KEY_DOWN:
                state["y"] -= state["step"]
            elif key == curses.KEY_LEFT:
                state["x"] -= state["step"]
            elif key == curses.KEY_RIGHT:
                state["x"] += state["step"]
            elif key in (ord(']'), ord('+')):
                state["step"] = min(state["step"] * 1.5, STEP_MAX)
            elif key in (ord('['), ord('-')):
                state["step"] = max(state["step"] / 1.5, STEP_MIN)
            elif key == ord('h'):
                state["x"], state["y"] = HOME_X, HOME_Y
            elif key == ord('t'):
                run_positioning_test(stdscr, servos, state)
                draw(send=False)  # already there, just refresh the display
                continue
            else:
                continue
            draw(send=True)

    curses.wrapper(loop)


HOME_X, HOME_Y = 100.0, 75.0


def main():
    servos = hw.Servos(JOINT_IDS)
    servos.connect(SERVO_PORT)
    print(f"connected. L1={L1} L2={L2} base=({BASE_X},{BASE_Y}) "
          f"offsets=({SERVO1_OFFSET_DEG:.1f},{SERVO2_OFFSET_DEG:.1f}) "
          f"dirs=({SERVO1_DIR},{SERVO2_DIR})")
    print("nothing will move until you press an arrow key or 't' -- "
          "starting jog screen...")
    try:
        run_jog(servos)
    finally:
        servos.close()


if __name__ == "__main__":
    main()
