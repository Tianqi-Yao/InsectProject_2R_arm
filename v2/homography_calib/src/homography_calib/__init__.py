"""AprilTag corner-sheet homography calibration: pixel<->mm mapping for the
2R arm's workspace. Depends only on arm_hw_core (camera + AprilTag
detection) -- does not move the arm and has no kinematics of its own, so it
can be built and validated before the arm's own calibration exists.
"""
