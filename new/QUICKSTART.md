# Quickstart

## 1. Flash the ESP32 driver board

For normal operation (Pi/Mac controls servos directly):

- Open `firmware/SerialBridge/SerialBridge.ino` in the Arduino IDE and flash it.
- This makes the board a transparent USB<->servo-bus relay -- no WiFi, no
  logic of its own. The host talks the SCServo protocol straight through it.

If you need to check wiring, verify a servo responds, or change a servo's
bus ID, flash `firmware/ServoJog/ServoJog.ino` instead (temporarily):

- It hosts its own WiFi hotspot `ArmServoCtrl` (password `12345678`).
  Connect your phone to it, browse to `192.168.4.1`.
- Press-and-hold buttons jog each joint; there's an OLED status display and
  a "Set ID" field to permanently reassign a servo's bus ID.
- Don't leave this flashed for normal operation -- it and `SerialBridge`
  both want exclusive control of the servo UART, only one can run at a time.
  Re-flash `SerialBridge.ino` when you're done testing.

## 2. Set up the host (Raspberry Pi or dev machine)

```bash
sudo apt install -y python3-picamera2   # camera lib -- apt, not pip (Pi only)
cd new
pip install -r requirements.txt
```

Confirm the servo SDK exposes the API `arm_hardware.py` expects (package
naming varies between forks):

```bash
python3 -c "import scservo_sdk as s; print([n for n in dir(s) if not n.startswith('_')])"
```
Look for `PortHandler` and `PacketHandler`. If your install exposes
something different, that's the one file to adjust
(`arm_hardware.py`) -- nothing else depends on the exact names.

You'll also need a few printed AprilTags, `tag36h11` family (from
AprilRobotics' `apriltag-imgs` repo, or any tag36h11 generator): 4 for the
corners of the 200x150mm work sheet, 1 for the end effector.

## 3. Configure calib.json for your rig

Copy `calib.example.json` to `calib.json` and edit `hardware.servo_port`
(and `hardware.joint_ids` if your servo bus IDs differ) to match your
setup. Everything else -- kinematics, motion tuning -- has reasonable
defaults and will be refined once you run calibration (step 6).

If your build has servo2's body bolted at L1's end but its rotation axis
sitting off to one side of L1's own line (common with bus servos -- the
body has real width, so the shaft isn't exactly on the link centerline),
set `kinematics.elbow_offset_mm` to that offset in mm now (measure the
center-to-center distance between the two rotation axes directly --
calipers/CAD, not a vision fit: step 6's calibration deliberately does NOT
touch this value, because it's mathematically impossible to recover from
end-effector position data alone, no matter how much you collect -- see
README.md's design notes if you want the derivation). Leave it at `0.0`
if the two are colinear.

## 4. Set mechanical safety limits (do this before any jogging/scanning)

If your arm has a mechanical dead zone (a range a joint physically
shouldn't enter -- e.g. it would collide with the base or the other
link), configure that now, before running anything that actually moves
the arm around (step 5 onward all do):

```bash
python3 main.py set-joint-limits
```

For each joint: torque is disabled so you can move it by hand through its
full safe range while the terminal live-tracks the min/max angle visited.
Press Ctrl+C when you've covered that joint's range (leave a margin before
the dead zone -- don't push all the way to the edge), and it moves on to
the next joint. Confirms before writing anything.

This writes to BOTH the servo's own hardware Min/Max Angle Limit
registers (the servo firmware itself then refuses to move past these, no
matter what software commands -- the most trustworthy layer) AND
`calib.json`'s `joint_limits_deg` (a software soft limit `main.py jog`/
`manual_test/*` also check, for an earlier, clearer rejection message).
See `README.md`'s design notes for why both layers exist.

**If joint2's safe range continuously shrinks/grows depending on where
joint1 is** (not a fixed zone, but a smoothly-varying relationship -- e.g.
joint2's clearance to a fixed obstacle changes as joint1 sweeps), you need
to trace a "coupled boundary" for this: a closed loop, drawn by hand,
around the entire perimeter of the safe region. Two earlier automatic
approaches (trace an edge and guess which side is limited; sweep the
interior and take min/max per joint1 bucket) both produced visibly wrong
results once tested on real hardware -- this one applies no
derivation/smoothing at all: what you trace is exactly what gets saved
and enforced.

**Recommended, if you have a display:**
```bash
python3 manual_test/trace_boundary_gui.py
```
A live pygame window. Press `b` to start recording, then walk the arm by
hand (both joints' torque released) around the FULL PERIMETER of the safe
region -- one continuous loop, back to about where you started -- while
the window draws the traced path live (converted to real workspace mm).
Press `b` again to stop. `c` clears the trace and stops recording, in
case you want to start over. Press `s`/Enter to save straight to
calib.json AND write a screenshot of the window at that moment
(`joint_limits_trace.png`, in the current directory) -- a plain
screen-capture of whatever's drawn, not a separately generated plot, so
it's exactly what you saved. `q`/ESC to quit without saving. Requires
`joint_limits_deg` to already exist (run `python3 main.py
set-joint-limits`'s two-joint independent-range sweep above first if you
haven't -- you can answer `N` to its coupled-boundary question and use
this tool for that part instead).

After saving, press `r` to REPLAY the saved boundary: torque re-engages
and the arm is actively driven all the way around it and back, so you can
watch the real arm (and the simulated one) trace exactly what got saved,
as a sanity check. This drives the arm right at the edge of the
configured dead zone with no inward safety margin, so watch closely and
press `r` again (or `q`/ESC) to stop early if anything looks off.

The same window also doubles as a live sanity check of your kinematics:
the drawn arm reflects whatever calib.json's L1/L2/offsets currently
believe, so if it visibly doesn't match the real arm's pose (e.g. real
L1-L2 angle is 90deg but the drawing shows something else -- expected if
you haven't run `main.py calibrate` yet), fold the arm by hand to a known
L1-L2 angle (read with a protractor/set-square at the elbow -- default
reference is 90deg, override with `--elbow-ref-deg`) and press `k` to
solve `servo2_offset_deg` directly from that one pose, no camera needed.
This only fixes servo2_offset_deg (the *relative* elbow angle) -- it can't
determine `servo1_offset_deg` (needs an absolute-orientation reference) or
`elbow_offset_mm` (needs a direct physical measurement -- see README.md's
design notes for why vision can't recover that one either). A bare angle
reading can't tell which rotational direction the elbow was folded in --
if `k` makes the drawing look wrong/mirrored instead of fixed, try
`shift+k` instead.

**Terminal-only fallback** (no display / SSH), inside `main.py
set-joint-limits`: after the two joints' independent ranges are measured,
answer `y` to "trace a closed loop around the safe region's boundary?" --
same release-both-joints-and-walk-the-perimeter motion as above, recorded
continuously until Ctrl+C (that's the "start"/"stop" -- calling the
prompt is the start, Ctrl+C is the stop). It also saves a PNG diagram
(`joint_limits_trace.png`) plotting the traced loop in real workspace
(x, y) mm, using calib.json's current L1/L2/base -- open it afterward and
confirm the shape roughly matches your arm's physical dead zone (this is
a sanity-check plot, not a precision one).

**This whole coupled-boundary layer is software-only protection** -- the
servo's hardware registers are per-servo and can't express "my limit
depends on the other servo's position" at all, so if this collision risk
is serious, add a physical mechanical stop too; don't rely on this
configuration alone.

calib.json is saved before any hardware register writes, so a transient
serial hiccup during the hardware verification step won't cost you the
measurements you just took by hand; a failed hardware write is reported
as a warning, not fatal.

If any traced boundary vertex falls slightly outside the two joints'
independent ranges measured earlier, that's expected (not a mistake) --
hand sweeps aren't perfectly repeatable to a fraction of a degree between
separate passes. The tool automatically widens both joints' ranges to the
union of themselves and every traced vertex before writing the hardware
registers, since those registers can only hold one fixed range per joint
(no notion of the other joint's position, or of which pass measured it)
and need to cover everything you actually confirmed safe.

## 5. Test in isolation

**No hardware at all** (pure logic):
```bash
cd new && pytest tests/
```

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
python3 main.py test-servo
```
Pings both joints, reads their real encoder angles, nudges joint1 by +5deg
and confirms it settles there. Uses `calib.json`'s `hardware.servo_port`;
pass `--port /dev/ttyXXXX` to override for a one-off run.

**Manual jog + scan, no camera/calibration** (curses terminal, or pygame
with a real-vs-target visualization):
```bash
python3 manual_test/run.py     # terminal
python3 manual_test/gui.py     # visual (needs a display)
```
Arrow keys jog the end effector (nothing moves until you press a key) --
along the scan area's own tilted edges if you've rotated it (step 7),
not raw world x/y, so pressing "up" always tracks one wall of the
rectangle you're looking at -- `[`/`]` change step size, `h` returns to
the scan area's center, `t` runs a serpentine scan sweep (`h` interrupts
it and heads home, `q` aborts in place), `q`/ESC quits. Both tools read
the same `calib.json` -- if you change scan density/speed, edit
`calib.json`'s `motion` section once, not two separate hardcoded copies.

**Interactive raw-angle jog** (for low-level debugging):
```bash
python3 main.py jog
```
`1 <deg>` / `2 <deg>` move joint1/joint2 to an absolute angle (smoothly,
via the same motion planner), `w` shows real vs. commanded position, `q` quits.

## 6. Combined calibration

Once both camera and servos check out individually:

1. Stick the 4 corner AprilTags at the work sheet's corners, mount the
   end-effector tag flush with the sheet's height (a small downturned tab
   works -- keeps it on the same plane the corner tags define, avoiding
   parallax from any camera tilt).
2. ```bash
   python3 main.py homography
   ```
   Detects the 4 corner tags, fits the pixel->mm mapping, saves it.
3. ```bash
   python3 main.py calibrate
   ```
   Auto-sweeps a grid of servo angles (smoothly, via the motion planner),
   reads real encoder positions, watches the end-effector tag, and fits
   L1/L2/base position/servo offsets. Prints a per-point error report and
   asks for confirmation before saving (RMS < 1mm: good, 1-3mm: usable,
   > 3mm: recheck tags/mounting first).
4. ```bash
   python3 main.py selfcheck
   ```
   Boot-time health check: re-verifies homography and does a couple of
   spot-check moves. This is what should run automatically on every restart
   once things are working (e.g. via systemd).

All calibration state lives in `calib.json` (git-ignored --
machine/robot-specific, regenerate locally rather than commit it;
`calib.example.json` documents the schema).

## 7. Fit the jog/scan area to what's actually reachable

`manual_test/gui.py`/`run.py`'s jog/scan rectangle used to just be the
AprilTag calibration sheet's own size (`workspace.width_mm`/`height_mm`)
-- but that sheet's corners are wherever you physically stuck the tags,
with no guarantee that matches the arm's actual reachable+safe region
(`joint_limits_deg`, from step 4). If part of that rectangle turned out
to be unreachable when jogging/scanning, fit a smaller, independently-
positioned sub-rectangle instead:

```bash
python3 manual_test/scan_area_gui.py
```

A live pygame window shades every reachable+safe workspace point green
(the same check `main.py jog`/scans already enforce -- IK reach + the
independent joint ranges + the coupled boundary, all at once), draws the
calibration sheet's own outline for reference (fixed, not editable here --
changing it would need physically moving the AprilTags and rerunning
`main.py homography`), and overlays the current jog/scan rectangle
(defaults to the full sheet if you haven't configured this before), which
can be tilted -- a rotated rectangle often covers more of an irregularly-
shaped reachable area than an axis-aligned one:

- drag a **corner handle** (yellow dot) to resize, symmetrically about
  the current center
- drag the **purple handle** above the top edge to rotate about the
  center
- drag **inside** the rectangle (away from any handle) to move it

The border turns green when everything inside is reachable, red if any of
it isn't. `f` resets to the full sheet (unrotated), `s`/Enter saves to
calib.json AND writes a screenshot (`scan_area.png`, in the current
directory -- a plain capture of the window, so what's saved is exactly
what's in the picture, same convention as `trace_boundary_gui.py`), `q`/
ESC quits without saving. Doesn't touch torque or move the arm at all --
it only polls encoder angles to draw the current pose for reference, so
it's safe to run any time; release torque by hand first (e.g. via
`trace_boundary_gui.py`) if you want to walk the arm around to spot-check
specific positions against the map.

This does NOT change `workspace.width_mm`/`height_mm`/`corner_world_mm`
(the calibration sheet's own size -- a physical fact this tool can't and
doesn't try to change); it saves a separate sub-rectangle
(`motion.scan_center_x_mm`/`scan_center_y_mm`/`scan_width_mm`/
`scan_height_mm`/`scan_rotation_deg`) that `generate_scan_path()` uses
instead, falling back to the full sheet (unrotated) if never configured.
