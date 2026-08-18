# arm-hw-core

Shared low-level hardware layer for the 2R arm's v2 codebase: STS3215 bus
servo register I/O, software + hardware joint-limit safety, camera (USB
debug / Pi CSI), and AprilTag detection.

**This is the only package the five v2 feature packages
(`homography_calib`, `kinematics_fit`, `card_scan`, `teach`,
`record_replay`) are allowed to import.** Everything else — kinematics,
motion planning, calibration fitting, scan logic, GUIs — is deliberately
reimplemented separately in each of them. See the v2 top-level plan for why:
this project went through several rounds of accidental whole-stack forking
(different directories silently drifting apart), and the fix chosen for v2
is to make forking *explicit* per feature while keeping the one genuinely
high-risk layer (servo EEPROM writes, joint-limit safety) in a single
place, so a safety fix only needs to be made once.

## Install (each feature package does this)

```bash
pip install -e ../hw_core
```

## What lives here

- `servos.py` — STS3215 control-table register I/O (`Servos`), including
  the EEPROM Min/Max Angle Limit write sequence (`set_hardware_angle_limits`)
  that is the outermost, firmware-enforced safety layer.
- `limits.py` — the complementary software layer: independent per-joint
  ranges plus an optional hand-traced "coupled boundary" polygon
  (`within_joint_limits`, winding-number `point_in_polygon`).
- `camera.py` — dual-backend camera (`usb` via OpenCV for desktop
  debugging, `picamera2` for Raspberry Pi deployment), selected by config,
  not code.
- `apriltag.py` — thin `pupil_apriltags` wrapper.
- `hw_state.py` — persistence for rig-level facts every feature package
  reads (serial port, joint IDs, camera backend, joint limits) —
  `~/.config/arm2r/hw_state.json` by default.
- `bringup_cli.py` (`arm-hw` command) — the safety bring-up procedure:
  `arm-hw set-joint-limits` (hand-sweep + optional coupled-boundary trace,
  writes both `hw_state.json` and the servo's hardware registers),
  `arm-hw test-servo` (connectivity smoke test).

## Testing

```bash
pip install -e ".[dev]"
pytest
```

All tests run against fakes (`scservo_sdk`, `cv2`) — no real hardware
needed. `set-joint-limits`'s actual bring-up behavior (EEPROM writes, the
resync-before-relock safety) must still be exercised by hand on the real
arm at least once; see the v2 plan's stage 1.
