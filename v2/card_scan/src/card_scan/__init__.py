"""AprilTag-on-card auto-detected grid scan: the card's position/size/
rotation is detected live from a tag stuck to it (see scan.detect_card_rect),
converted to a serpentine node path, and visited with corner-blended
multi-waypoint motion. Depends on the OUTPUT FILES of homography_calib and
kinematics_fit (read-only) but not their code -- see homography_read.py/
kinematics_read.py. Own copy of kinematics/motion, depends only on
arm_hw_core otherwise.
"""
