import pytest

from record_replay.controller import ArmController, MotionParams, plan_segment


class FakeServos:
    def __init__(self, start=(0.0, 0.0)):
        self.present = {"joint1": start[0], "joint2": start[1]}
        self.commanded_log = []

    def get_present_deg(self, joint):
        return self.present[joint]

    def set_target_deg(self, joint, angle_deg, speed=800, acc=0):
        self.commanded_log.append((joint, angle_deg))
        self.present[joint] = angle_deg


def _controller(start=(0.0, 0.0)):
    servos = FakeServos(start)
    motion = MotionParams(vmax_deg_s=90.0, amax_deg_s2=180.0, control_hz=50.0)
    return ArmController(servos, motion), servos


def test_run_to_completion_drives_the_arm_to_the_goal():
    controller, _ = _controller()
    reached = controller.run_to_completion(45.0, -30.0, timeout_s=5.0)
    assert reached == pytest.approx((45.0, -30.0), abs=1e-6)


def test_run_to_completion_takes_the_short_way_across_the_360_seam():
    controller, servos = _controller(start=(359.0, 0.0))
    controller.run_to_completion(1.0, 0.0, timeout_s=5.0)
    joint1_targets = [angle for joint, angle in servos.commanded_log if joint == "joint1"]
    assert all(355.0 <= a <= 365.0 for a in joint1_targets)


def test_resync_resets_commanded_position_and_drops_in_flight_queue():
    """Every replay run in this package begins right after a recording
    session (torque off, arm hand-moved) -- resync() being correct here is
    not an edge case, it's the normal startup path (see recorder.Player.prepare,
    which calls this before the first move)."""
    controller, servos = _controller(start=(0.0, 0.0))
    controller.set_joint_goal(90.0, 90.0)
    assert controller.is_moving is True

    servos.present["joint1"] = 300.0
    servos.present["joint2"] = 300.0
    controller.resync(300.0, 300.0)

    assert controller.commanded_deg == (300.0, 300.0)
    assert controller.is_moving is False

    controller.set_joint_goal(310.0, 310.0)
    first_step = controller.tick()
    assert abs(first_step[0] - 300.0) < 20.0
    assert abs(first_step[1] - 300.0) < 20.0


def test_plan_segment_lands_exactly_on_goal():
    samples = plan_segment((0.0, 0.0), (37.0, -12.0), (0.0, 0.0), (0.0, 0.0),
                            (90.0, 90.0), (180.0, 180.0), 0.02)
    assert samples[-1] == pytest.approx((37.0, -12.0))
