import pytest

from kinematics_fit.controller import ArmController, MotionParams, plan_segment


class FakeServos:
    """Duck-types the subset of arm_hw_core.servos.Servos that
    controller.ArmController needs, without any real serial/hardware I/O."""

    def __init__(self, start=(0.0, 0.0)):
        self.present = {"joint1": start[0], "joint2": start[1]}
        self.commanded_log = []

    def get_present_deg(self, joint):
        return self.present[joint]

    def set_target_deg(self, joint, angle_deg, speed=800, acc=0):
        self.commanded_log.append((joint, angle_deg))
        self.present[joint] = angle_deg  # perfect tracking, no dynamics -- fine for unit tests


def _controller(start=(0.0, 0.0), joint_limits=None):
    servos = FakeServos(start)
    motion = MotionParams(vmax_deg_s=90.0, amax_deg_s2=180.0, control_hz=50.0)
    return ArmController(servos, motion, joint_limits=joint_limits), servos


def test_initial_commanded_position_seeds_from_real_servo_readback():
    controller, _ = _controller(start=(12.0, -34.0))
    assert controller.commanded_deg == (12.0, -34.0)


def test_set_joint_goal_rejected_outside_joint_limits_leaves_state_unchanged():
    limits = {"joint1": (0.0, 10.0), "joint2": (0.0, 360.0)}
    controller, _ = _controller(start=(5.0, 5.0), joint_limits=limits)
    accepted = controller.set_joint_goal(500.0, 5.0)  # far outside joint1's range
    assert accepted is False
    assert controller.is_moving is False
    assert controller.commanded_deg == (5.0, 5.0)


def test_run_to_completion_drives_the_arm_to_the_goal():
    controller, servos = _controller(start=(0.0, 0.0))
    reached = controller.run_to_completion(45.0, -30.0, timeout_s=5.0)
    assert reached == pytest.approx((45.0, -30.0), abs=1e-6)
    assert controller.is_moving is False


def test_run_to_completion_takes_the_short_way_across_the_360_seam():
    controller, servos = _controller(start=(359.0, 0.0))
    controller.run_to_completion(1.0, 0.0, timeout_s=5.0)
    # Every intermediate commanded angle for joint1 should stay within a
    # couple degrees of 359/361 -- never sweep the long way around through ~180.
    joint1_targets = [angle for joint, angle in servos.commanded_log if joint == "joint1"]
    assert all(355.0 <= a <= 365.0 for a in joint1_targets)


def test_resync_resets_commanded_position_and_drops_in_flight_queue():
    """Regression test for the exact bug documented in controller.py's
    resync() docstring, and previously found in the old codebase (present
    in v4/jog_controller.py, missing from the top-level copy): after a
    hand-move while torque was released, resync() must make the controller
    plan its next segment from the arm's ACTUAL new position, not from
    wherever it last remembered commanding the arm to."""
    controller, servos = _controller(start=(0.0, 0.0))
    controller.set_joint_goal(90.0, 90.0)
    assert controller.is_moving is True  # a segment is queued, not yet ticked through

    # Simulate a hand-move while torque was off: the servo is now
    # somewhere totally different from what the controller last commanded.
    servos.present["joint1"] = 200.0
    servos.present["joint2"] = 200.0
    controller.resync(200.0, 200.0)

    assert controller.commanded_deg == (200.0, 200.0)
    assert controller.is_moving is False  # the stale in-flight queue must be dropped

    # The next move must start from the resynced position, not silently
    # replan a phantom jump back through the pre-resync goal (90, 90).
    controller.set_joint_goal(210.0, 210.0)
    first_step = controller.tick()
    assert abs(first_step[0] - 200.0) < 20.0  # small step from 200, not a jump to/through 90
    assert abs(first_step[1] - 200.0) < 20.0


def test_plan_segment_lands_exactly_on_goal():
    samples = plan_segment((0.0, 0.0), (37.0, -12.0), (0.0, 0.0), (0.0, 0.0),
                            (90.0, 90.0), (180.0, 180.0), 0.02)
    assert samples[-1] == pytest.approx((37.0, -12.0))


def test_plan_segment_zero_distance_returns_single_sample():
    samples = plan_segment((10.0, 10.0), (10.0, 10.0), (0.0, 0.0), (0.0, 0.0),
                            (90.0, 90.0), (180.0, 180.0), 0.02)
    assert samples == [(10.0, 10.0)]
