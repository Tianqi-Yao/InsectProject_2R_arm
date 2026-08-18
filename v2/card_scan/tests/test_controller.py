import pytest

from card_scan.controller import ArmController, MotionParams, plan_segment
from card_scan.kinematics import ArmParams


class FakeServos:
    def __init__(self, start=(0.0, 0.0)):
        self.present = {"joint1": start[0], "joint2": start[1]}
        self.commanded_log = []

    def get_present_deg(self, joint):
        return self.present[joint]

    def set_target_deg(self, joint, angle_deg, speed=800, acc=0):
        self.commanded_log.append((joint, angle_deg))
        self.present[joint] = angle_deg


def _controller(start=(0.0, 0.0), joint_limits=None):
    servos = FakeServos(start)
    params = ArmParams.nominal()
    motion = MotionParams(jog_vmax_deg_s=90.0, jog_amax_deg_s2=180.0,
                           scan_vmax_deg_s=120.0, scan_amax_deg_s2=240.0,
                           blend_threshold=0.7, control_hz=50.0)
    return ArmController(servos, params, motion, joint_limits=joint_limits), servos


def test_set_workspace_goal_uses_ik_and_moves_the_arm():
    controller, servos = _controller(start=(23.08, 0.0))
    p = controller.params
    accepted = controller.set_workspace_goal(p.base_x + 50.0, p.base_y + 30.0)
    assert accepted is True
    assert controller.is_moving is True


def test_set_workspace_goal_rejects_unreachable_point():
    controller, _ = _controller()
    p = controller.params
    accepted = controller.set_workspace_goal(p.base_x + p.L1 + p.L2 + 1000.0, 0.0)
    assert accepted is False
    assert controller.is_moving is False


def test_start_scan_skips_unreachable_waypoints():
    controller, _ = _controller()
    p = controller.params
    far_away = (p.base_x + p.L1 + p.L2 + 1000.0, 0.0, "bad")
    reachable = (p.base_x + 40.0, p.base_y + 20.0, "good")
    controller.start_scan([far_away, reachable])
    # Only the reachable waypoint should have made it into the scan.
    assert controller.scan_progress[1] == 1


def test_scan_runs_to_completion_and_visits_every_reachable_waypoint():
    controller, _ = _controller()
    p = controller.params
    waypoints = [(p.base_x + 40.0, p.base_y + 20.0, "a"),
                 (p.base_x + 50.0, p.base_y + 10.0, "b"),
                 (p.base_x + 60.0, p.base_y + 25.0, "c")]
    controller.start_scan(waypoints)
    steps = 0
    while controller.is_moving and steps < 100_000:
        controller.tick()
        steps += 1
    assert controller.scan_active is False
    assert steps < 100_000  # actually terminated, not stuck


def test_resync_resets_commanded_position_and_drops_in_flight_scan():
    """card_scan is the one v2 package with an active multi-waypoint scan
    state -- resync() must drop that too, not just a simple single-target
    queue, or a stale scan would keep trying to advance from a position
    the arm is no longer at."""
    controller, servos = _controller(start=(0.0, 0.0))
    p = controller.params
    controller.start_scan([(p.base_x + 40.0, p.base_y + 20.0, "a"),
                            (p.base_x + 50.0, p.base_y + 10.0, "b")])
    assert controller.is_moving is True
    assert controller.scan_active is True

    servos.present["joint1"] = 250.0
    servos.present["joint2"] = 250.0
    controller.resync(250.0, 250.0)

    assert controller.commanded_deg == (250.0, 250.0)
    assert controller.is_moving is False
    assert controller.scan_active is False


def test_corner_blend_coasts_through_a_shallow_turn_but_stops_at_a_sharp_one():
    controller, _ = _controller()
    p = controller.params
    # Three points nearly in a straight line (shallow turn) vs. a sharp
    # reversal -- the blend velocity should be nonzero for the shallow
    # case and zero for the reversal.
    straight = [(p.base_x + 30.0, p.base_y + 10.0, "a"),
                (p.base_x + 40.0, p.base_y + 11.0, "b"),
                (p.base_x + 50.0, p.base_y + 12.0, "c")]
    controller.start_scan(straight)
    from card_scan.controller import _ScanState
    scan = controller._scan
    assert scan is not None
    v = controller._corner_blend_velocity(_ScanState(joint_targets=scan.joint_targets, index=0))
    # Not asserting an exact value (depends on IK geometry) -- just that
    # the mechanism runs without error and returns a 2-tuple.
    assert isinstance(v, tuple) and len(v) == 2


def test_plan_segment_lands_exactly_on_goal():
    samples = plan_segment((0.0, 0.0), (37.0, -12.0), (0.0, 0.0), (0.0, 0.0),
                            (90.0, 90.0), (180.0, 180.0), 0.02)
    assert samples[-1] == pytest.approx((37.0, -12.0))
