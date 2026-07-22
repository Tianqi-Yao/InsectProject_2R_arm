# 2R Arm

A 2-link (2R) planar robotic arm with camera-based automatic calibration
and smooth, planner-driven motion. Two Feetech STS3215-HS bus servos move
the arm in a horizontal plane over a 200x150mm work sheet; an overhead
Raspberry Pi + IMX477 camera detects AprilTags to figure out where things
actually are, so the true link lengths, base position, and servo
zero-offsets/directions don't need to be measured by hand -- they're fit
automatically from vision data. See `QUICKSTART.md` for setup and usage.

This is the second, refactored iteration of the codebase (see "History"
below) -- consolidated into fewer, clearer files after several rounds of
hardware-driven patches (baud rate fixes, a servo-direction bug, three
independently-hand-rolled jog implementations that had drifted out of
sync). If you're looking at `../software/` or `../manual_test/`, those are
the archived previous iteration; this (`new/`) is current.

## Hardware

- 2x Feetech STS3215-HS serial bus servos (magnetic encoder, real position
  feedback -- not open-loop PWM)
- Waveshare ESP32 servo driver board
- Raspberry Pi + IMX477 (Raspberry Pi HQ Camera)
- AprilTags (`tag36h11` family): 4 fixed at the corners of the 200x150mm
  work sheet, 1 mounted on the end effector

## Layout

```
arm_core.py            Core logic: IK/FK, homography, least-squares fit,
                        boot self-check, calib.json schema/persistence --
                        the one file meant to be read end-to-end.
arm_hardware.py         Black-box hardware layer: servo bus register I/O,
                        camera capture, AprilTag detection. No decision
                        logic lives here -- don't worry about its internals.
motion_planning/        Pluggable trajectory planners (see "Motion
                        planning" below). trapezoidal.py is the default;
                        add a new file + one import line to swap in
                        another algorithm.
jog_controller.py       ArmController: the one real-time motion controller
                        shared by all three frontends below (previously
                        three independent, drifted-apart implementations).
main.py                 CLI: test-servo / test-camera / homography /
                        calibrate / selfcheck / set-joint-limits / jog.
manual_test/
  run.py                Curses (terminal) jog + scan tester, skipping the
                        camera/calibration pipeline -- for a quick sanity
                        check that the arm moves as expected.
  gui.py                Same, but pygame-visualized: draws the real
                        (encoder-fed) arm pose alongside the commanded
                        target, side by side.
firmware/
  SerialBridge/          ESP32 firmware for normal operation: a
                        transparent USB<->servo-bus byte relay, so the Pi
                        can talk the SCServo protocol directly.
  ServoJog/              ESP32 firmware for manual wiring/ID testing: its
                        own WiFi hotspot + web jog page, mutually
                        exclusive with SerialBridge (both want the servo
                        UART to themselves).
tests/                   Pure-logic pytest suite -- no hardware needed for
                        any of it (arm_core, motion_planning, jog_controller).
calib.example.json       Example calib.json showing the full schema. The
                        real calib.json is generated at runtime and
                        gitignored (it's rig-specific state, not source).
```

## Motion planning

Smoothing used to be three hand-tuned layers glued together per-frontend:
servo register speed/acc limits, a jog speed/accel constant pair, and a
scan-specific "stream a new target every N ms without waiting for
arrival" trick -- each frontend keeping its own copy, which is exactly how
the scan grid size/speed constants silently drifted out of sync between
`run.py` and `gui.py` (see git history).

Now there's one planner interface (`motion_planning.TrajectoryPlanner`)
and `jog_controller.ArmController` drives it at a fixed control-loop rate
(50Hz by default), for both single-target jogging and multi-waypoint
scanning. The default implementation (`motion_planning/trapezoidal.py`) is
a standard joint-space, two-joint-synchronized trapezoidal velocity
profile -- the same baseline method most robot/CNC controllers use.
Consecutive scan waypoints that continue in roughly the same direction
coast through the corner at cruise speed instead of coming to a full stop
(`MotionConfig.blend_threshold` in `arm_core.py`), which is what actually
fixed the scan-grid jitter this replaced.

To add a different algorithm (e.g. a jerk-limited S-curve): implement
`TrajectoryPlanner` in a new `motion_planning/your_algo.py`, decorate the
class with `@register("your_algo")`, add one import line at the bottom of
`motion_planning/__init__.py`, and switch `calib.json`'s
`motion.planner_name` to `"your_algo"`. Nothing else changes.

## Design notes

- **Why vision-based calibration**: the real connecting-rod lengths and
  the motor base position can't be measured precisely by hand during
  assembly. Rather than guessing, the arm sweeps through a grid of servo
  angles while a camera watches an AprilTag on the end effector;
  `scipy.optimize.least_squares` jointly fits the true L1, L2, base
  position, and servo offsets (`servo1_dir`/`servo2_dir` -- whether a
  joint's raw angle increases the way our math convention expects -- are
  fixed hardware facts confirmed by hand, not fit: a sign flip is a
  reflection that no amount of offset/length tuning can reproduce).
- **Why `elbow_offset_mm` is a fixed constant, not a fit parameter, even
  though it looks like an ordinary length**: on builds where servo2's body
  is bolted at L1's end but its rotation axis sits off to one side of L1's
  line (rather than exactly on it -- a real consequence of the servo
  having physical width, not a design choice), that perpendicular offset
  changes the kinematics (`ArmParams.elbow_offset_mm`, `fk_from_servo_angles`,
  `ik_solve`). It's tempting to let `fit_kinematics` solve for it too --
  vision data already fits L1/L2/base/offsets, why not this? Because it's
  provably degenerate with L1: only `reach = hypot(L1, elbow_offset_mm)` is
  ever identifiable from end-effector position data, never the individual
  split, since trading L1 against elbow_offset_mm at fixed `reach` is
  exactly cancelled out by an equal-and-opposite trade between
  `servo1_offset_deg` and `servo2_offset_deg` -- confirmed empirically:
  fitting synthetic data generated with a real elbow_offset_mm=28
  recovered a wildly different elbow_offset_mm alongside a correspondingly
  shifted L1, with statistically perfect residual error, regardless of how
  much/wide the sampled data was. So this one has to come from an
  independent physical measurement (calipers/CAD, center-to-center between
  the two rotation axes) -- see `ArmParams`'s docstring for the full
  derivation.
- **Why a boot self-check**: the device runs outdoors and restarts daily,
  so it re-verifies its own calibration against the camera every boot.
  Minor drift self-heals (adopts the fresh reading, logs it, keeps
  working); drift past a threshold halts operation and raises an alarm
  (hook only for now) until someone re-calibrates.
- **Why calib.json is the single source of truth for hardware/motion
  settings too** (not just kinematics): three tools independently
  hardcoding the same physical constants is exactly how they drifted out
  of sync before. `arm_core.HardwareConfig`/`MotionConfig` and
  `calib_hardware_config()`/`calib_motion_config()` mean every frontend
  reads the same file.
- **Why mechanical dead-zone protection is two layers, not one**: IK only
  checks geometric reachability (link lengths), it has no idea a physical
  obstruction/dead zone exists. `main.py set-joint-limits` measures a
  joint's safe range by hand and writes it to BOTH the servo's own
  hardware Min/Max Angle Limit registers (`arm_hardware.py` -- the servo
  firmware itself refuses to move past these, regardless of what any
  software, including a bug in this project, commands) AND calib.json's
  `joint_limits_deg` (the software soft limit `ik_solve`/`jog_controller`
  check, for an earlier/clearer rejection). The hardware layer is the one
  that actually matters if the software layer has a bug; the software
  layer is there for a better error message before that.
- **Why coupled/relative dead zones are a hand-traced polygon, not a
  sampled interval curve**: on a 2-link arm, the distal link's safe range
  can *continuously* shrink/grow depending on where the proximal link
  currently is (e.g. joint2's clearance to a fixed obstacle varies
  smoothly as joint1 sweeps, not as a discrete on/off region). Two earlier
  approaches to capturing this both failed on real hardware: tracing a
  thin edge line and guessing which side was constrained collapsed into
  near-degenerate ranges; sweeping the interior and taking min/max per
  joint1 bucket produced a visibly wrong envelope once tested for real.
  `joint_limits_deg`'s `coupled_boundary` is simpler instead: a list of
  (joint1, joint2) vertices, traced by hand in one closed loop around the
  full perimeter of the safe region (release both joints' torque, walk
  the arm around the boundary, back to about where you started --
  `manual_test/trace_boundary_gui.py`'s 'b' key, or `main.py
  set-joint-limits`'s terminal equivalent). No binning, smoothing, or
  derivation is applied -- the traced path IS the boundary, saved exactly
  as recorded. A pose passes iff it falls inside that polygon
  (`_point_in_polygon()` in `arm_core.py`), checked with a winding-number
  rule rather than the simpler even-odd/ray-casting rule specifically
  because a hand-traced path can retrace ground it's already covered
  (hesitation, jitter, doubling back) -- even-odd would flip its verdict
  for an even number of full retraces purely by coincidence of parity;
  winding number treats any nonzero winding as "inside" regardless. This
  representation also handles arbitrarily-shaped (even concave) safe
  regions, which a single joint2 interval per joint1 slice fundamentally
  couldn't. The live GUI window is also a quick way to spot a bad
  `servo2_offset_deg` before ever running the camera-based fit: fold the
  arm to a known L1-L2 angle and press `k`
  (`arm_core.servo2_offset_from_known_elbow_angle`) to solve it from that
  one pose directly. The coupled boundary is still software-only: the
  servo's own hardware angle-limit registers are strictly per-servo --
  there is no register on one STS3215 that can express "my limit depends
  on the other servo's position." For a serious coupled collision risk, a
  physical mechanical stop is the only true hardware-level backstop; the
  software check is an earlier/clearer rejection, not a guarantee
  independent of this code being bug-free.
- **Why the jog/scan area is independent of the calibration sheet's own
  size**: `manual_test/gui.py`/`run.py` used to jog/scan exactly
  `workspace.width_mm`/`height_mm` -- the AprilTag calibration sheet's own
  dimensions. But that sheet's corners are wherever the tags happen to be
  physically stuck down, with no guarantee that rectangle is fully inside
  the arm's actual reachable+safe region (`joint_limits_deg`) -- and it
  can't be "fixed" by changing `workspace.width_mm`/`height_mm`/
  `corner_world_mm` in software, since those numbers have to match the
  physical tag positions or the homography desyncs. `MotionConfig`'s
  `scan_center_x_mm`/`scan_center_y_mm`/`scan_width_mm`/`scan_height_mm`/
  `scan_rotation_deg` (`calib_scan_area()`) describe a separate, optional
  sub-rectangle in that same coordinate frame -- as center+size+rotation
  rather than min/max bounds, specifically because it can be tilted: a
  rotated rectangle can cover more of an irregularly-shaped reachable
  area than an axis-aligned one. Defaults to the full sheet, unrotated,
  until configured; `generate_scan_path()` takes matching
  `center_x_mm`/`center_y_mm`/`rotation_deg` parameters.
  `manual_test/scan_area_gui.py` fits it visually: shades every workspace
  point `ik_solve()` reports reachable (the same three-layer check
  -- IK reach, independent joint ranges, coupled boundary -- every other
  tool already enforces), and lets you drag the rectangle's corner
  handles (resize), a dedicated handle above the top edge (rotate about
  the center), or its interior (move) over that map rather than
  hand-editing numbers.

## History

- `../software/`, `../manual_test/`, `../sim/`, `../ServoDriverST/` are
  archived, earlier iterations -- kept for reference, not run. `sim/` and
  `ServoDriverST/` predate the current STS3215+vision hardware entirely
  (MG90S open-loop servos, no camera); `software/`/`manual_test/` are this
  same design one refactor ago.
