# 2R arm control software — v2

Six independent Python packages, each installable and testable on its own. This
replaces the accumulated `new/` / `new/auto_calib_scan/` / `new/v3/` / `new/v4/`
fork tree in the parent directory — see the git history / project notes for why.

## Architecture

**One shared package, `hw_core`, and nothing else is shared.** `hw_core`
(`arm-hw-core`) owns the only genuinely high-risk layer: STS3215 servo
register I/O (including the EEPROM angle-limit write sequence), software +
hardware joint-limit safety, camera, and AprilTag detection. Every other
package below depends on `hw_core` and *only* on `hw_core` — never on each
other's code.

| package | installs as | provides |
|---|---|---|
| `hw_core/` | `arm-hw-core` (lib) + `arm-hw` (CLI) | servo/camera/apriltag drivers, joint-limit safety, `arm-hw set-joint-limits` |
| `homography_calib/` | `homography-calib` (lib) + `arm-homography` (CLI) | 4-corner-tag pixel↔mm homography fit + drift self-check |
| `kinematics_fit/` | `kinematics-fit` (lib) + `arm-kinfit` (CLI) | vision-fitted L1/L2/base/servo-offset calibration + arm-position spot-check |
| `teach/` | `teach` (lib) + `arm-teach` (CLI) | hand-teach fallback mode, no camera needed |
| `record_replay/` | `record-replay` (lib) + `arm-replay` (CLI) | joint-space record/replay + cron-scheduled unattended replay |
| `card_scan/` | `card-scan` (lib) + `arm-cardscan` (CLI) | AprilTag-on-card auto-detected grid scan |

Each of `kinematics_fit/`, `teach/`, `record_replay/`, `card_scan/` carries
its **own copy** of FK/IK (where relevant), angle-wrapping, and a
trapezoidal-planner motion controller (with `resync()` — see below). This
is a deliberate, informed trade: debugging isolation (a bug in one
package's controller can't silently affect another's) in exchange for
duplicated maintenance. It is not the same thing as the old repo's
*accidental* forking — see each package's own module docstrings for the
concrete bug (`resync()` missing from one fork, present in another) that
motivated doing this on purpose instead.

### Cross-package data contract: JSON files, never code

Packages that need another package's output (`kinematics_fit` and
`card_scan` need `homography_calib`'s fitted matrix; `card_scan` needs
`kinematics_fit`'s fitted params; `teach`/`card_scan` optionally read
`kinematics_fit`'s output for display/IK accuracy) read the producing
package's output JSON file directly — reimplementing the few lines needed
to parse it (see each package's `*_read.py` modules) rather than importing
that package. All state lives under `~/.config/arm2r/`:

```
~/.config/arm2r/
  hw_state.json           # hw_core        — port, joint ids, camera backend, joint limits
  workspace_calib.json    # homography_calib — corner tag ids/world coords, fitted H
  kinematics_calib.json   # kinematics_fit  — fitted L1/L2/base/offsets, ee_tag_id, spotcheck poses
  teach.json              # teach           — named hand-taught points
  recorded_path.json      # record_replay   — ordered hand-recorded joint-space points
  card_scan.json          # card_scan       — card tag id/size, grid, manual-corner fallback
  */_history/              # auto-backup-before-overwrite, one per package, disposable
```

## Install (per package, editable + dev deps)

```bash
cd hw_core            && pip install -e ".[dev]"
cd ../homography_calib && pip install -e ../hw_core -e ".[dev]"
cd ../kinematics_fit    && pip install -e ../hw_core -e ".[dev]"
cd ../teach             && pip install -e ../hw_core -e ".[dev]"
cd ../record_replay     && pip install -e ../hw_core -e ".[dev]"
cd ../card_scan         && pip install -e ../hw_core -e ".[dev]"
```

Each package's own `pytest` suite runs against fakes/mocks — no real
hardware needed. Bring-up steps that genuinely need the real arm+camera
(EEPROM writes, the resync-before-relock safety, actual calibration fits)
must still be exercised by hand at least once; see each package's own
module docstrings and CLI `--help`.

## Suggested bring-up order

1. `hw_core`: `arm-hw test-servo`, then `arm-hw set-joint-limits` on the bench.
2. `homography_calib`: `arm-homography fit`.
3. `kinematics_fit`: `arm-kinfit run`, then `arm-kinfit selfcheck`.
4. `teach` and `record_replay` need only step 1 and can be validated in parallel with step 3.
5. `card_scan` last — it depends on both 2 and 3's output files.

## Firmware

`firmware/` (ESP32 `SerialBridge`/`ServoJog`) is carried over as-is from the
parent directory — not part of this Python rewrite's scope.
