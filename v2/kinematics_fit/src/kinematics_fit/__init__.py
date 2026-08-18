"""2R arm kinematics + vision-fitted calibration (L1, L2, base position,
servo zero offsets). This package owns its own copy of the FK/IK model and
a minimal motion controller -- it does not import from any other v2
feature package, only from arm_hw_core (see that package's README for why).
It reads homography_calib's workspace_calib.json (read-only) to convert
detected end-effector pixel positions to mm.
"""
