# 2R Arm

A 2-link (2R) planar robotic arm with camera-based automatic calibration.
Two Feetech STS3215-HS bus servos move the arm in a horizontal plane over a
200x150mm work sheet; an overhead Raspberry Pi + IMX477 camera detects
AprilTags to figure out where things actually are, so the true link lengths,
base position, and servo zero-offsets don't need to be measured by hand --
they're fit automatically from vision data. See `QUICKSTART.md` for setup
and usage.

## Hardware

- 2x Feetech STS3215-HS serial bus servos (magnetic encoder, real position
  feedback -- not open-loop PWM)
- Waveshare ESP32 servo driver board
- Raspberry Pi + IMX477 (Raspberry Pi HQ Camera)
- AprilTags (`tag36h11` family): 4 fixed at the corners of the 200x150mm
  work sheet, 1 mounted on the end effector

## Layout

```
software/           Main control + calibration stack (Python, runs on the Pi)
  arm_core.py          Core logic: IK/FK, homography, least-squares fit,
                        boot self-check -- the one file meant to be read
                        end-to-end.
  arm_hardware.py      Black-box hardware layer: servo bus, camera,
                        AprilTag detector. No decision logic lives here.
  main.py              CLI entry point (test-servo / test-camera /
                        homography / calibrate / selfcheck / jog).
  tests/               Pure-logic unit tests, no hardware needed.

SerialBridge/        ESP32 firmware for normal operation: a transparent
                      USB<->servo-bus byte relay, so the Pi can talk the
                      SCServo protocol directly via scservo_sdk.

ServoJog/             ESP32 firmware for manual testing: hosts its own
                      WiFi hotspot + web page with press-and-hold jog
                      buttons for both servos, plus a "set servo ID" tool.
                      Useful for wiring/ID troubleshooting; not used
                      during normal operation (mutually exclusive with
                      SerialBridge -- both want the servo UART to
                      themselves).

ServoDriverST/        Waveshare's stock example firmware for the driver
                      board. Superseded by SerialBridge/ServoJog above;
                      kept only for reference.

sim/                  Earlier iteration: pure simulation + open-loop
                      MG90S PWM servos, no camera. Superseded by the
                      STS3215 + vision stack above; kept only for
                      reference (its IK/FK formulas were the starting
                      point for arm_core.py).
```

## Design notes

- **Why vision-based calibration**: the real connecting-rod lengths and the
  motor base position can't be measured precisely by hand during assembly.
  Rather than guessing, the arm sweeps through a grid of servo angles while
  a camera watches an AprilTag on the end effector; `scipy.optimize.least_squares`
  then jointly fits the true L1, L2, base position, and servo offsets.
- **Why a boot self-check**: the device runs outdoors and restarts daily, so
  it re-verifies its own calibration against the camera every boot. Minor
  drift self-heals (adopts the fresh reading, logs it, keeps working);
  drift past a threshold halts operation and raises an alarm (hook only for
  now) until someone re-calibrates.
