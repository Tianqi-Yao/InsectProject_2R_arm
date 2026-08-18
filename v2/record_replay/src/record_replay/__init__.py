"""Minimal joint-space record/replay: release torque, hand-drag the arm to
each position you want, mark it; then re-engage torque and replay through
those points in order, dwelling at each -- optionally snapping a photo
partway through the dwell. No IK/FK, no calibration, no workspace
coordinates: every recorded point was physically visited by hand already
(torque was off when it was marked), so it's inherently reachable and safe
to replay as-is. Own copy of the motion controller -- does not import from
any other v2 feature package, only from arm_hw_core.
"""
