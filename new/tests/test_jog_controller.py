"""Pure-logic tests for jog_controller.py, using a fake Servos handle --
no real hardware needed. Fake mirrors the style already used in
test_arm_core.py's self-check mocks (a hand-rolled duck-typed stand-in,
not a mock of arm_hardware.Servos specifically)."""

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import arm_core as ac
import jog_controller as jc
import motion_planning as mp


class FakeServos:
    """Servos stand-in that just tracks the last commanded angle per
    joint and returns it on read -- good enough to drive ArmController's
    tick loop without any real bus I/O."""

    def __init__(self, start=(68.0, 116.0)):
        self._pos = {"joint1": start[0], "joint2": start[1]}
        self.calls = []

    def get_present_deg(self, joint):
        return self._pos[joint]

    def set_target_deg(self, joint, angle_deg, speed=800, acc=0):
        self._pos[joint] = angle_deg
        self.calls.append((joint, angle_deg, speed, acc))


def _make_controller(start=(68.0, 116.0), joint_limits=None, **motion_overrides):
    servos = FakeServos(start)
    params = ac.ArmParams.nominal()
    motion_cfg = ac.MotionConfig(**motion_overrides)
    planner = mp.get_planner(motion_cfg.planner_name)
    return jc.ArmController(servos, params, planner, motion_cfg, joint_limits=joint_limits), servos


def _run_to_stop(ctl, max_ticks=2000):
    n = 0
    while ctl.is_moving and n < max_ticks:
        ctl.tick()
        n += 1
    return n


# ── Basic single-target motion ──────────────────────────────────────

def test_seeds_commanded_position_from_real_servo_feedback():
    ctl, _ = _make_controller(start=(12.0, 34.0))
    assert ctl.commanded_deg == (12.0, 34.0)
    assert not ctl.is_moving  # nothing sent until a goal is actually set


def test_set_joint_goal_converges_exactly():
    ctl, _ = _make_controller()
    ctl.set_joint_goal(90.0, 100.0)
    n = _run_to_stop(ctl)
    assert n > 0
    assert ctl.commanded_deg == pytest.approx((90.0, 100.0))
    assert not ctl.is_moving


def test_tick_uses_streaming_speed_not_jog_speed_constant():
    # Motion smoothing lives entirely in the planner now; the servo's own
    # speed/acc registers should just track setpoints as fast as possible.
    ctl, servos = _make_controller()
    ctl.set_joint_goal(90.0, 100.0)
    ctl.tick()
    for _joint, _angle, speed, acc in servos.calls:
        assert speed == jc.STREAMING_SPEED
        assert acc == jc.STREAMING_ACC


def test_set_workspace_goal_reachable_moves_arm():
    ctl, _ = _make_controller()
    ok = ctl.set_workspace_goal(100.0, 75.0)
    assert ok
    _run_to_stop(ctl)
    r = ac.ik_solve(ctl.params, 100.0, 75.0)
    assert ctl.commanded_deg == pytest.approx((r.servo1_deg, r.servo2_deg))


def test_set_workspace_goal_unreachable_is_a_no_op():
    ctl, _ = _make_controller()
    before = ctl.commanded_deg
    ok = ctl.set_workspace_goal(0.0, 10_000.0)  # far beyond L1+L2
    assert not ok
    assert not ctl.is_moving
    assert ctl.commanded_deg == before


def test_nudge_workspace_advances_a_receding_horizon_target():
    ctl, _ = _make_controller()
    base = (100.0, 75.0)
    new_target = ctl.nudge_workspace(5.0, 0.0, base)
    assert new_target == (105.0, 75.0)
    assert ctl.is_moving  # replanned immediately, doesn't wait for arrival
    # a second nudge, called before the first has settled, should still work
    new_target2 = ctl.nudge_workspace(5.0, 0.0, new_target)
    assert new_target2 == (110.0, 75.0)


def test_nudge_workspace_returns_none_when_unreachable():
    ctl, _ = _make_controller()
    result = ctl.nudge_workspace(1_000_000.0, 0.0, (100.0, 75.0))
    assert result is None


def test_set_single_joint_goal_leaves_other_joint_untouched():
    ctl, _ = _make_controller(start=(68.0, 116.0))
    ctl.set_single_joint_goal("joint1", 90.0)
    _run_to_stop(ctl)
    assert ctl.commanded_deg == pytest.approx((90.0, 116.0))

    ctl.set_single_joint_goal("joint2", 50.0)
    _run_to_stop(ctl)
    assert ctl.commanded_deg == pytest.approx((90.0, 50.0))


def test_run_to_completion_blocks_until_arrival(monkeypatch):
    ctl, _ = _make_controller()
    monkeypatch.setattr(jc.time, "sleep", lambda _s: None)  # don't actually wait in tests
    final = ctl.run_to_completion(90.0, 100.0)
    assert final == pytest.approx((90.0, 100.0))
    assert not ctl.is_moving


# ── Mechanical dead-zone protection (joint_limits) ──────────────────

def test_set_joint_goal_rejects_target_outside_joint_limits():
    limits = {"joint1": (0.0, 90.0), "joint2": (0.0, 360.0)}
    ctl, servos = _make_controller(start=(68.0, 116.0), joint_limits=limits)
    ok = ctl.set_joint_goal(150.0, 100.0)  # joint1=150 is outside [0,90]
    assert not ok
    assert not ctl.is_moving
    assert ctl.commanded_deg == (68.0, 116.0)  # nothing moved
    assert servos.calls == []  # nothing sent to hardware at all


def test_set_joint_goal_accepts_target_inside_joint_limits():
    limits = {"joint1": (0.0, 180.0), "joint2": (0.0, 360.0)}
    ctl, _ = _make_controller(start=(68.0, 116.0), joint_limits=limits)
    ok = ctl.set_joint_goal(90.0, 100.0)
    assert ok
    _run_to_stop(ctl)
    assert ctl.commanded_deg == pytest.approx((90.0, 100.0))


def test_set_single_joint_goal_rejects_and_reports_false():
    limits = {"joint1": (0.0, 90.0), "joint2": (0.0, 360.0)}
    ctl, _ = _make_controller(start=(68.0, 116.0), joint_limits=limits)
    ok = ctl.set_single_joint_goal("joint1", 150.0)
    assert not ok
    assert not ctl.is_moving


def test_set_workspace_goal_rejects_point_whose_servo_angle_violates_joint_limits():
    ctl, _ = _make_controller()
    # find a reachable point, then configure a limit that specifically excludes it
    r = ac.ik_solve(ctl.params, 100.0, 75.0)
    assert r.reachable
    ctl.joint_limits = {"joint1": (r.servo1_deg + 5.0, 360.0), "joint2": (0.0, 360.0)}
    ok = ctl.set_workspace_goal(100.0, 75.0)
    assert not ok
    assert not ctl.is_moving


def test_start_scan_skips_waypoints_outside_joint_limits():
    ctl, _ = _make_controller()
    r = ac.ik_solve(ctl.params, 100.0, 75.0)
    ctl.joint_limits = {"joint1": (r.servo1_deg + 5.0, 360.0), "joint2": (0.0, 360.0)}
    # (100,75) now excluded by joint_limits; (60,40) should still be fine
    ctl.start_scan([(100.0, 75.0, "excluded"), (60.0, 40.0, "ok")])
    n = _run_to_stop(ctl, max_ticks=5000)
    assert n > 0
    r2 = ac.ik_solve(ctl.params, 60.0, 40.0)
    assert ctl.commanded_deg == pytest.approx((r2.servo1_deg, r2.servo2_deg))


def test_build_controller_warns_when_joint_limits_not_configured(capsys):
    calib = ac._default_calib()
    assert calib["joint_limits_deg"] is None
    servos = FakeServos()
    ctl = jc.build_controller(servos, calib)
    assert ctl.joint_limits is None
    captured = capsys.readouterr()
    assert "joint_limits_deg" in captured.out


def test_build_controller_reads_configured_joint_limits():
    calib = ac._default_calib()
    calib["joint_limits_deg"] = {"joint1": [10.0, 170.0], "joint2": [20.0, 340.0]}
    servos = FakeServos()
    ctl = jc.build_controller(servos, calib)
    assert ctl.joint_limits == {"joint1": (10.0, 170.0), "joint2": (20.0, 340.0), "coupled_boundary": []}


def test_build_controller_enforces_coupled_boundary_from_calib():
    calib = ac._default_calib()
    calib["joint_limits_deg"] = {
        "joint1": [0.0, 360.0], "joint2": [0.0, 360.0],
        "coupled_boundary": [{"joint1": 80.0, "joint2": 150.0},
                             {"joint1": 100.0, "joint2": 150.0},
                             {"joint1": 100.0, "joint2": 200.0},
                             {"joint1": 80.0, "joint2": 200.0}],
    }
    servos = FakeServos(start=(90.0, 175.0))
    ctl = jc.build_controller(servos, calib)
    assert ctl.set_joint_goal(90.0, 250.0) is False  # within the polygon's joint1 span,
    assert not ctl.is_moving                          # but 250 is outside the traced boundary


# ── Both joints arrive together ─────────────────────────────────────

def test_both_joints_synchronized_even_with_very_different_distances():
    ctl, _ = _make_controller(start=(0.0, 0.0))
    ctl.set_joint_goal(100.0, 5.0)
    n = _run_to_stop(ctl)
    assert n > 1
    assert ctl.commanded_deg == pytest.approx((100.0, 5.0))


# ── Scanning ─────────────────────────────────────────────────────────

def test_start_scan_visits_every_reachable_waypoint_in_order():
    ctl, _ = _make_controller()
    path = ac.generate_scan_path(nx=3, ny=2, margin_mm=20.0)
    ctl.start_scan(path)
    assert ctl.scan_active

    expected_final = None
    for x, y, _label in path:
        r = ac.ik_solve(ctl.params, x, y)
        if r.reachable:
            expected_final = (r.servo1_deg, r.servo2_deg)

    n = _run_to_stop(ctl, max_ticks=10_000)
    assert n > 0
    assert not ctl.scan_active
    assert ctl.commanded_deg == pytest.approx(expected_final)


def test_scan_progress_advances_and_completes():
    ctl, _ = _make_controller()
    path = ac.generate_scan_path(nx=3, ny=2, margin_mm=20.0)
    ctl.start_scan(path)
    completed, total = ctl.scan_progress
    assert total == len(path)
    assert completed == 1  # first segment already queued by start_scan

    _run_to_stop(ctl, max_ticks=10_000)
    completed, total = ctl.scan_progress
    assert completed == total


def test_stop_scan_aborts_without_jumping_to_a_new_target():
    ctl, _ = _make_controller()
    path = ac.generate_scan_path(nx=5, ny=4, margin_mm=20.0)
    ctl.start_scan(path)
    for _ in range(3):
        ctl.tick()
    mid_position = ctl.commanded_deg

    ctl.stop_scan()
    assert not ctl.scan_active
    # the in-flight segment keeps playing (not yanked to a stop) --
    # commanded position should still be moving smoothly, not snap back
    ctl.tick()
    assert ctl.commanded_deg != mid_position  # still progressing along the old segment


def test_set_joint_goal_interrupts_an_active_scan():
    ctl, _ = _make_controller()
    path = ac.generate_scan_path(nx=5, ny=4, margin_mm=20.0)
    ctl.start_scan(path)
    assert ctl.scan_active

    ctl.set_joint_goal(68.0, 116.0)  # e.g. "go home" pressed mid-scan
    assert not ctl.scan_active
    _run_to_stop(ctl)
    assert ctl.commanded_deg == pytest.approx((68.0, 116.0))


def test_unreachable_scan_waypoints_are_skipped_not_fatal():
    ctl, _ = _make_controller()
    # a waypoint far outside the workspace, mixed in with reachable ones
    path = [(100.0, 75.0, "a"), (1_000_000.0, 1_000_000.0, "unreachable"), (60.0, 40.0, "b")]
    ctl.start_scan(path)
    n = _run_to_stop(ctl, max_ticks=10_000)
    assert n > 0
    r = ac.ik_solve(ctl.params, 60.0, 40.0)
    assert ctl.commanded_deg == pytest.approx((r.servo1_deg, r.servo2_deg))


# ── Corner blending ──────────────────────────────────────────────────

def test_corner_blend_velocity_is_zero_at_final_waypoint():
    ctl, _ = _make_controller()
    scan = jc._ScanState(joint_targets=[(10.0, 10.0), (20.0, 20.0)], index=1)
    v_end = ctl._corner_blend_velocity(scan)
    assert v_end == (0.0, 0.0)


def test_corner_blend_velocity_nonzero_for_aligned_segments():
    # start -> (10,0) -> (20,0): a straight line, same direction both legs
    ctl, _ = _make_controller(start=(0.0, 0.0))
    scan = jc._ScanState(joint_targets=[(10.0, 0.0), (20.0, 0.0)], index=0)
    v_end = ctl._corner_blend_velocity(scan)
    assert v_end[0] > 0.0  # coasting through, not stopping
    assert v_end[1] == pytest.approx(0.0)


def test_corner_blend_velocity_zero_for_sharp_turnaround():
    # start -> (10,0) -> (0,0): reverses direction entirely (like a scan
    # row's end-of-row turnaround)
    ctl, _ = _make_controller(start=(0.0, 0.0))
    scan = jc._ScanState(joint_targets=[(10.0, 0.0), (0.0, 0.0)], index=0)
    v_end = ctl._corner_blend_velocity(scan)
    assert v_end == (0.0, 0.0)


def test_scan_with_aligned_waypoints_does_not_fully_stop_mid_row():
    # Three colinear, evenly-spaced points: the middle leg's exit velocity
    # should be nonzero (blended through), unlike the old
    # stop-at-every-point behaviour this replaces.
    ctl, _ = _make_controller(start=(0.0, 0.0))
    scan = jc._ScanState(joint_targets=[(10.0, 0.0), (20.0, 0.0), (30.0, 0.0)], index=0)
    ctl._scan = scan
    ctl._advance_scan()  # queues the leg ending at (10,0), scan.index now 1
    # peek at what velocity that first segment was told to end at
    v_end = ctl._corner_blend_velocity(jc._ScanState(joint_targets=scan.joint_targets, index=0))
    assert v_end[0] > 0.0


def test_build_controller_reads_planner_name_from_calib(monkeypatch):
    calib = ac._default_calib()
    servos = FakeServos()
    ctl = jc.build_controller(servos, calib)
    assert isinstance(ctl.planner, mp.get_planner("trapezoidal").__class__)
    assert ctl.motion_cfg.planner_name == "trapezoidal"
