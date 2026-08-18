# 2R Arm

A 2-link (2R) planar robotic arm with camera-based automatic calibration.
Two Feetech STS3215-HS bus servos move the arm in a horizontal plane over a
200x150mm work sheet; an overhead Raspberry Pi + IMX477 camera detects
AprilTags to figure out where things actually are, so the true link lengths,
base position, and servo zero-offsets don't need to be measured by hand --
they're fit automatically from vision data.

## Current development line: `v2/`

**`v2/` is where active work happens.** It's a from-scratch rewrite (see
`v2/README.md` for the full architecture) replacing the fork tree that had
accumulated under `new/` (`new/` itself, `new/auto_calib_scan/`, `new/v3/`,
`new/v4/` -- four independent copies of the same core stack that had drifted
apart). v2 is six independently installable/testable Python packages
sharing only one low-level hardware-safety package (`v2/hw_core/`); nothing
else is shared between them by design. Start there:

```
v2/hw_core/            servo/camera/apriltag drivers + joint-limit safety (the only shared package)
v2/homography_calib/   AprilTag corner-sheet -> pixel<->mm calibration
v2/kinematics_fit/     vision-fitted L1/L2/base/servo-offset calibration
v2/teach/               hand-teach fallback mode, no camera needed
v2/record_replay/       joint-space record/replay + cron-scheduled unattended replay
v2/card_scan/           AprilTag-on-card auto-detected grid scan
v2/firmware/             ESP32 firmware (SerialBridge / ServoJog), carried over as-is
```

## Hardware

- 2x Feetech STS3215-HS serial bus servos (magnetic encoder, real position
  feedback -- not open-loop PWM)
- Waveshare ESP32 servo driver board
- Raspberry Pi + IMX477 (Raspberry Pi HQ Camera)
- AprilTags (`tag36h11` family): 4 fixed at the corners of the 200x150mm
  work sheet, 1 mounted on the end effector

## Design notes

- **Why vision-based calibration**: the real connecting-rod lengths and the
  motor base position can't be measured precisely by hand during assembly.
  Rather than guessing, the arm sweeps through a grid of servo angles while
  a camera watches an AprilTag on the end effector; `scipy.optimize.least_squares`
  then jointly fits the true L1, L2, base position, and servo offsets.
- **Why a boot self-check**: the device runs outdoors and restarts daily, so
  it re-verifies its own calibration against the camera every boot. Minor
  drift self-heals (adopts the fresh reading, logs it, keeps working);
  drift past a threshold halts operation and raises an alarm until someone
  re-calibrates. In v2, this is split into two independent checks:
  `homography_calib`'s drift self-check and `kinematics_fit`'s
  arm-position spot-check.

## Archived / historical (not part of active development)

- **`new/`** (and its `auto_calib_scan/`/`v3/`/`v4/` subfolders) -- the
  generation `v2/` replaces. Kept for reference; see each subfolder's own
  `README`/`QUICKSTART` for what it tried.
- **`software/`** -- the generation before `new/`. Superseded (top-level
  `QUICKSTART.md` documents this generation specifically, not `v2/`).
- **`sim/`** -- pure simulation + open-loop MG90S PWM servos, no camera or
  real hardware at all. Predates the STS3215 + vision stack; its IK/FK
  formulas were the historical starting point for the arm_core math that
  both `new/` and `v2/` build on.
- **`ServoDriverST/`** -- Waveshare's stock example firmware for the driver
  board. Superseded by `SerialBridge/`/`ServoJog/` (and their `v2/firmware/`
  copy above).
