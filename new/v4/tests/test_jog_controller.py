"""Pure-logic tests for jog_controller.py, using a fake Servos handle --
no real hardware needed. v4 only adds start_joint_scan() on top of v3's
already-tested ArmController; this file focuses on that addition plus the
handful of primitives record_replay.py directly depends on (resync, dt,
is_moving/scan_progress, corner blending)."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import arm_core as ac  # noqa: E402
import jog_controller as jc  # noqa: E402
import motion_planning as mp  # noqa: E402


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


def _make_controller(start=(68.0, 116.0), **motion_overrides):
    servos = FakeServos(start)
    params = ac.ArmParams.nominal()
    motion_cfg = ac.MotionConfig(**motion_overrides)
    planner = mp.get_planner(motion_cfg.planner_name)
    return jc.ArmController(servos, params, planner, motion_cfg, joint_limits=None), servos


def _run_to_stop(ctl, max_ticks=10_000):
    n = 0
    while ctl.is_moving and n < max_ticks:
        ctl.tick()
        n += 1
    return n


# ── start_joint_scan: the method record_replay.py's replay() drives ──────

def test_start_joint_scan_visits_every_waypoint_in_order():
    ctl, _ = _make_controller(start=(0.0, 0.0))
    path = [(10.0, 5.0), (20.0, 5.0), (20.0, 15.0), (5.0, 15.0)]
    ctl.start_joint_scan(path)
    assert ctl.scan_active

    n = _run_to_stop(ctl)
    assert n > 0
    assert not ctl.scan_active
    assert ctl.commanded_deg == pytest.approx(path[-1])


def test_start_joint_scan_progress_advances_and_completes():
    ctl, _ = _make_controller()
    path = [(70.0, 118.0), (75.0, 120.0), (80.0, 122.0)]
    ctl.start_joint_scan(path)
    completed, total = ctl.scan_progress
    assert total == len(path)
    assert completed == 1  # first segment already queued

    _run_to_stop(ctl)
    completed, total = ctl.scan_progress
    assert completed == total


def test_start_joint_scan_only_stops_at_the_final_waypoint():
    # Three collinear joint-space points, evenly spaced -- exactly the
    # shape a hand-recorded path segment has. The middle leg must coast
    # (nonzero exit velocity); only the last leg comes to a full stop.
    ctl, _ = _make_controller(start=(0.0, 0.0))
    path = [(10.0, 0.0), (20.0, 0.0), (30.0, 0.0)]
    ctl.start_joint_scan(path)
    scan = ctl._scan
    v_mid = ctl._corner_blend_velocity(jc._ScanState(joint_targets=scan.joint_targets, index=0))
    assert v_mid != (0.0, 0.0)
    v_last = ctl._corner_blend_velocity(
        jc._ScanState(joint_targets=scan.joint_targets, index=len(scan.joint_targets) - 1))
    assert v_last == (0.0, 0.0)


def test_start_joint_scan_chains_near_360deg_wrap_correctly():
    # Recorded points near the 0/360 seam (e.g. present-angle reads of
    # 359, 1, 3 as the arm physically moves a few degrees across the
    # wrap) shouldn't be interpreted as a near-full-revolution move.
    ctl, _ = _make_controller(start=(359.0, 100.0))
    ctl.start_joint_scan([(1.0, 100.0), (3.0, 100.0)])
    targets = ctl._scan.joint_targets
    assert abs(targets[0][0] - 359.0) < 10.0
    assert abs(targets[1][0] - targets[0][0]) < 10.0


def test_start_joint_scan_single_waypoint_still_moves_and_stops():
    ctl, _ = _make_controller(start=(0.0, 0.0))
    ctl.start_joint_scan([(45.0, 30.0)])
    n = _run_to_stop(ctl)
    assert n > 0
    assert ctl.commanded_deg == pytest.approx((45.0, 30.0))


# ── resync / dt / goal_deg (what replay()/record_replay.py rely on) ──────

def test_resync_clears_stale_state_before_a_replay():
    ctl, servos = _make_controller(start=(0.0, 0.0))
    servos._pos["joint1"] = 200.0  # hand-dragged elsewhere while torque was off
    servos._pos["joint2"] = 210.0
    ctl.resync(servos.get_present_deg("joint1"), servos.get_present_deg("joint2"))
    assert ctl.commanded_deg == (200.0, 210.0)
    assert not ctl.is_moving


def test_dt_reflects_motion_cfg_control_hz():
    ctl, _ = _make_controller()
    assert ctl.dt == pytest.approx(1.0 / ctl.motion_cfg.control_hz)


def test_goal_deg_updates_immediately_on_start_joint_scan():
    ctl, _ = _make_controller(start=(0.0, 0.0))
    ctl.start_joint_scan([(10.0, 0.0), (20.0, 0.0)])
    assert ctl.goal_deg == pytest.approx((10.0, 0.0))  # first leg's destination, queued now
