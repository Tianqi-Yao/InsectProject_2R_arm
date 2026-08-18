"""Shared low-level hardware layer for the 2R arm's five v2 feature
packages: STS3215 servo register I/O, software+hardware joint-limit
safety, camera, AprilTag detection.

This is the ONLY package the five feature packages (homography_calib,
kinematics_fit, card_scan, teach, record_replay) are allowed to depend on.
Everything else -- kinematics, motion planning, calibration fitting, scan
logic, GUIs -- is deliberately reimplemented separately in each of them; see
v2's top-level plan for why (debugging isolation traded for duplicated
maintenance, a conscious choice, not accidental drift)."""
