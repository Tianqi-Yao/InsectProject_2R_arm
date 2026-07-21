# Quickstart

## 1. Flash the ESP32 driver board

For normal operation (Pi controls servos directly):

- Open `SerialBridge/SerialBridge.ino` in the Arduino IDE and flash it.
- This makes the board a transparent USB<->servo-bus relay -- no WiFi, no
  logic of its own. The Pi talks the SCServo protocol straight through it.

If you need to check wiring, verify a servo responds, or change a servo's
bus ID, flash `ServoJog/ServoJog.ino` instead (temporarily):

- It hosts its own WiFi hotspot `ArmServoCtrl` (password `12345678`).
  Connect your phone to it, browse to `192.168.4.1`.
- Press-and-hold buttons jog each joint; there's also a "Set ID" field to
  permanently reassign a servo's bus ID.
- Don't leave this flashed for normal operation -- it and `SerialBridge`
  both want exclusive control of the servo UART, only one can run at a time.
  Re-flash `SerialBridge.ino` when you're done testing.

## 2. Set up the Raspberry Pi

```bash
sudo apt install -y python3-picamera2   # camera lib -- apt, not pip
cd software
pip install -r requirements.txt
```

Confirm the servo SDK exposes the API `arm_hardware.py` expects (package
naming varies between forks):

```bash
python3 -c "import scservo_sdk as s; print([n for n in dir(s) if not n.startswith('_')])"
```
Look for `PortHandler`, `sms_sts`, `ReadPos`, `WritePosEx`. If your install
names things differently, that's the one file to adjust
(`software/arm_hardware.py`) -- nothing else depends on the exact names.

You'll also need a few printed AprilTags, `tag36h11` family (from
AprilRobotics' `apriltag-imgs` repo, or any tag36h11 generator): 4 for the
corners of the 200x150mm work sheet, 1 for the end effector.

## 3. Test in isolation

**Camera + tag detection** (no servos needed):
```bash
python3 main.py test-camera --watch
```
Prints live detections to the terminal (Ctrl+C to stop) and saves an
annotated snapshot (`tag_detect.jpg`) so you can check tag ID + position
visually. If the camera itself won't open, confirm `libcamera-hello` works
first -- that isolates a hardware/driver problem from a Python one.

**Servos** (no camera needed):
```bash
ls /dev/ttyUSB* /dev/ttyACM*          # find the port
python3 main.py test-servo --port /dev/ttyUSB0
```
Pings both joints, reads their real encoder angles, nudges joint1 by +5deg
and confirms it settles there.

**Manual jog** (debugging either side of a move):
```bash
python3 main.py jog --port /dev/ttyUSB0
```

**Pure logic** (no hardware at all):
```bash
cd software && pytest tests/
```

## 4. Combined calibration

Once both camera and servos check out individually:

1. Stick the 4 corner AprilTags at the work sheet's corners, mount the
   end-effector tag flush with the sheet's height (a small downturned tab
   works -- keeps it on the same plane the corner tags define, avoiding
   parallax from any camera tilt).
2. ```bash
   python3 main.py homography --port /dev/ttyUSB0
   ```
   Detects the 4 corner tags, fits the pixel->mm mapping, saves it.
3. ```bash
   python3 main.py calibrate --port /dev/ttyUSB0
   ```
   Auto-sweeps a grid of servo angles, reads real encoder positions, watches
   the end-effector tag, and fits L1/L2/base position/servo offsets. Prints
   a per-point error report and asks for confirmation before saving
   (RMS < 1mm: good, 1-3mm: usable, > 3mm: recheck tags/mounting first).
4. ```bash
   python3 main.py selfcheck --port /dev/ttyUSB0
   ```
   Boot-time health check: re-verifies homography and does a couple of
   spot-check moves. This is what should run automatically on every restart
   once things are working (e.g. via systemd).

All calibration state lives in `software/calib.json` (git-ignored --
machine/robot-specific, regenerate locally rather than commit it).
