"""Pure-logic tests for jog_controller.py, using a fake Servos handle --
no real hardware needed. Covers the corner-blending behavior
path_core.PhotoScanRunner's whole "interpolation points coast through,
photo points fully stop" design depends on (see path_core.py's module
docstring), plus resync() and the live dt property added for
control_gui.py's torque-toggle / PARAMS-mode workflows."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import arm_core as ac  # noqa: E402
import jog_controller as jc  # noqa: E402
import motion_planning as mp  # noqa: E402
import path_core as pc  # noqa: E402


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


def _run_to_stop(ctl, max_ticks=10_000):
    n = 0
    while ctl.is_moving and n < max_ticks:
        ctl.tick()
        n += 1
    return n


# ── Basic single-target motion ──────────────────────────────────────

def test_seeds_commanded_position_from_real_servo_feedback():
    ctl, _ = _make_controller(start=(12.0, 34.0))
    assert ctl.commanded_deg == (12.0, 34.0)
    assert not ctl.is_moving


def test_set_joint_goal_converges_exactly():
    ctl, _ = _make_controller()
    ctl.set_joint_goal(90.0, 100.0)
    n = _run_to_stop(ctl)
    assert n > 0
    assert ctl.commanded_deg == pytest.approx((90.0, 100.0))
    assert not ctl.is_moving


def test_set_workspace_goal_unreachable_is_a_no_op():
    ctl, _ = _make_controller()
    before = ctl.commanded_deg
    ok = ctl.set_workspace_goal(0.0, 10_000.0)
    assert not ok
    assert not ctl.is_moving
    assert ctl.commanded_deg == before


def test_nudge_workspace_advances_a_receding_horizon_target():
    ctl, _ = _make_controller()
    base = (100.0, 75.0)
    new_target = ctl.nudge_workspace(5.0, 0.0, base)
    assert new_target == (105.0, 75.0)
    assert ctl.is_moving


# ── dt live-tracks motion_cfg.control_hz ──────────────────────────────

def test_dt_reflects_live_control_hz_edits():
    ctl, _ = _make_controller()
    assert ctl.dt == pytest.approx(1.0 / ctl.motion_cfg.control_hz)
    ctl.motion_cfg.control_hz = 25.0  # e.g. control_gui.py's PARAMS mode editing in place
    assert ctl.dt == pytest.approx(1.0 / 25.0)


# ── resync() ───────────────────────────────────────────────────────────

def test_resync_updates_commanded_and_goal():
    ctl, _ = _make_controller(start=(10.0, 20.0))
    ctl.resync(99.0, 88.0)
    assert ctl.commanded_deg == (99.0, 88.0)
    assert not ctl.is_moving


def test_resync_clears_in_flight_queue_and_scan():
    ctl, _ = _make_controller(start=(0.0, 0.0))
    path = pc.generate_scan_path(width_mm=100.0, height_mm=80.0, nx=5, ny=4, margin_mm=10.0)
    ctl.start_scan(path)
    assert ctl.is_moving
    ctl.resync(42.0, 43.0)
    assert not ctl.is_moving
    assert not ctl.scan_active
    assert ctl.commanded_deg == (42.0, 43.0)


def test_resync_prevents_a_stale_jump_after_a_hand_drag():
    # Simulates control_gui.py's exact hazard: torque released (arm hand-
    # dragged far away without any tick() to update _commanded), torque
    # re-engaged. Without resync(), the next goal would plan a segment
    # from the OLD pre-drag position, streaming a phantom move through it.
    ctl, servos = _make_controller(start=(10.0, 10.0))
    # Hand-drag happens physically -- the servo's real position changes,
    # but nothing calls tick(), so ctl._commanded is still (10, 10).
    servos._pos["joint1"] = 170.0
    servos._pos["joint2"] = 190.0
    assert ctl.commanded_deg == (10.0, 10.0)  # stale, as expected pre-fix

    ctl.resync(servos.get_present_deg("joint1"), servos.get_present_deg("joint2"))
    assert ctl.commanded_deg == (170.0, 190.0)

    ctl.set_joint_goal(175.0, 195.0)  # a small real nudge from the true position
    n = _run_to_stop(ctl)
    # every streamed setpoint should stay close to the real start -- no
    # detour back through the stale (10, 10)
    for _joint, angle, _speed, _acc in servos.calls:
        assert 0.0 <= angle <= 200.0
    assert n > 0


# ── Corner blending (the property path_core.PhotoScanRunner relies on) ──

def test_corner_blend_velocity_is_zero_at_final_waypoint():
    ctl, _ = _make_controller()
    scan = jc._ScanState(joint_targets=[(10.0, 10.0), (20.0, 20.0)], index=1)
    v_end = ctl._corner_blend_velocity(scan)
    assert v_end == (0.0, 0.0)


def test_corner_blend_velocity_nonzero_for_aligned_segments():
    ctl, _ = _make_controller(start=(0.0, 0.0))
    scan = jc._ScanState(joint_targets=[(10.0, 0.0), (20.0, 0.0)], index=0)
    v_end = ctl._corner_blend_velocity(scan)
    assert v_end[0] > 0.0
    assert v_end[1] == pytest.approx(0.0)


def test_corner_blend_velocity_zero_for_sharp_turnaround():
    ctl, _ = _make_controller(start=(0.0, 0.0))
    scan = jc._ScanState(joint_targets=[(10.0, 0.0), (0.0, 0.0)], index=0)
    v_end = ctl._corner_blend_velocity(scan)
    assert v_end == (0.0, 0.0)


def test_start_scan_with_fine_interpolated_waypoints_only_stops_at_the_last_one():
    # Mirrors exactly what path_core.PhotoScanRunner feeds start_scan():
    # a run of collinear cartesian-interpolated points ending at a photo
    # point. Every interior waypoint's exit velocity must be nonzero
    # (coast through); only the LAST one may be zero (full stop).
    # Seed the controller's commanded position at the segment's own start
    # point (100, 75) -- matching how path_core.PhotoScanRunner actually
    # drives this: each segment starts from wherever the arm just arrived
    # (the previous photo point), never from an unrelated pose.
    nominal = ac.ArmParams.nominal()
    start_ik = ac.ik_solve(nominal, 100.0, 75.0)
    assert start_ik.reachable
    ctl, _ = _make_controller(start=(start_ik.servo1_deg, start_ik.servo2_deg))
    segment = pc.interpolate_line_mm((100.0, 75.0), (140.0, 75.0), max_step_mm=5.0)
    joint_targets = []
    for x, y in segment:
        r = ac.ik_solve(ctl.params, x, y)
        assert r.reachable
        joint_targets.append((r.servo1_deg, r.servo2_deg))
    scan = jc._ScanState(joint_targets=joint_targets, index=0)
    for i in range(len(joint_targets) - 1):
        v_end = ctl._corner_blend_velocity(jc._ScanState(joint_targets=joint_targets, index=i))
        assert v_end != (0.0, 0.0), f"waypoint {i} should coast, not stop"
    v_end_last = ctl._corner_blend_velocity(
        jc._ScanState(joint_targets=joint_targets, index=len(joint_targets) - 1))
    assert v_end_last == (0.0, 0.0)


def test_start_scan_visits_every_reachable_waypoint_in_order():
    ctl, _ = _make_controller()
    # center_x/y matches the nominal arm's own calibration workspace (see
    # ArmParams.nominal()'s base_x/base_y) -- path_core.generate_scan_path
    # defaults to centering on (0, 0), unlike ../arm_core.py's older
    # version, since every real v3 caller passes an explicit rectangle
    # center; a test that omits it needs to supply one that's actually
    # reachable instead.
    path = pc.generate_scan_path(width_mm=200.0, height_mm=150.0, nx=3, ny=2, margin_mm=20.0,
                                  center_x_mm=100.0, center_y_mm=75.0)
    ctl.start_scan(path)
    assert ctl.scan_active

    expected_final = None
    for x, y, _label in path:
        r = ac.ik_solve(ctl.params, x, y)
        if r.reachable:
            expected_final = (r.servo1_deg, r.servo2_deg)

    n = _run_to_stop(ctl)
    assert n > 0
    assert not ctl.scan_active
    assert ctl.commanded_deg == pytest.approx(expected_final)


def test_unreachable_scan_waypoints_are_skipped_not_fatal():
    ctl, _ = _make_controller()
    path = [(100.0, 75.0, "a"), (1_000_000.0, 1_000_000.0, "unreachable"), (60.0, 40.0, "b")]
    ctl.start_scan(path)
    n = _run_to_stop(ctl)
    assert n > 0
    r = ac.ik_solve(ctl.params, 60.0, 40.0)
    assert ctl.commanded_deg == pytest.approx((r.servo1_deg, r.servo2_deg))


def test_stop_scan_aborts_without_jumping_to_a_new_target():
    ctl, _ = _make_controller()
    path = pc.generate_scan_path(width_mm=200.0, height_mm=150.0, nx=5, ny=4, margin_mm=20.0)
    ctl.start_scan(path)
    for _ in range(3):
        ctl.tick()
    mid_position = ctl.commanded_deg

    ctl.stop_scan()
    assert not ctl.scan_active
    ctl.tick()
    assert ctl.commanded_deg != mid_position  # in-flight segment still progressing


# ── goal_deg: receding-horizon base for repeated incremental jogs ─────

def test_goal_deg_reflects_the_last_commanded_target_immediately():
    ctl, _ = _make_controller(start=(10.0, 20.0))
    assert ctl.goal_deg == (10.0, 20.0)
    ctl.set_joint_goal(90.0, 100.0)
    # goal_deg updates the instant a goal is set, unlike commanded_deg
    # (which only creeps toward it tick by tick).
    assert ctl.goal_deg == (90.0, 100.0)
    assert ctl.commanded_deg != (90.0, 100.0)


def test_goal_deg_accumulates_correctly_across_rapid_repeated_steps():
    # Simulates control_gui.py's direct joint-jog: each new goal is
    # derived from the PREVIOUS goal_deg, not from commanded_deg/present
    # angle -- so five 2-degree steps fired faster than the arm can
    # physically settle still add up to a full 10-degree move, not get
    # stuck re-deriving from a lagging position each time.
    ctl, _ = _make_controller(start=(0.0, 0.0))
    for _ in range(5):
        j1, j2 = ctl.goal_deg
        ctl.set_joint_goal(j1 + 2.0, j2)
        ctl.tick()  # only one tick per step -- nowhere near arrival
    assert ctl.goal_deg == pytest.approx((10.0, 0.0))
    assert ctl.commanded_deg[0] < 10.0  # confirms the arm hadn't caught up


# ── Mechanical dead-zone protection (joint_limits) ──────────────────

def test_set_joint_goal_rejects_target_outside_joint_limits():
    limits = {"joint1": (0.0, 90.0), "joint2": (0.0, 360.0)}
    ctl, servos = _make_controller(start=(68.0, 116.0), joint_limits=limits)
    ok = ctl.set_joint_goal(150.0, 100.0)
    assert not ok
    assert not ctl.is_moving
    assert ctl.commanded_deg == (68.0, 116.0)
    assert servos.calls == []
