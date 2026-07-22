"""Pure-math tests for motion_planning/. No hardware needed."""

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import motion_planning as mp

DT = 1.0 / 50.0  # 50Hz control loop


def _velocities(samples, start, dt):
    """Finite-difference velocity estimate for each axis across a sample
    sequence, prefixed with `start` so the first sample's velocity is
    measured too."""
    pts = [start] + samples
    vels = []
    for i in range(1, len(pts)):
        vels.append(((pts[i][0] - pts[i - 1][0]) / dt, (pts[i][1] - pts[i - 1][1]) / dt))
    return vels


def _accelerations(vels, dt):
    accs = []
    for i in range(1, len(vels)):
        accs.append(((vels[i][0] - vels[i - 1][0]) / dt, (vels[i][1] - vels[i - 1][1]) / dt))
    return accs


# ── Registry ─────────────────────────────────────────────────────────

def test_registry_has_trapezoidal():
    assert "trapezoidal" in mp.available_planners()
    planner = mp.get_planner("trapezoidal")
    assert isinstance(planner, mp.TrajectoryPlanner)


def test_registry_rejects_unknown_name():
    with pytest.raises(ValueError):
        mp.get_planner("does_not_exist")


# ── Trapezoidal planner ─────────────────────────────────────────────

def test_lands_exactly_on_goal():
    planner = mp.get_planner("trapezoidal")
    start, goal = (0.0, 0.0), (37.0, -14.0)
    samples = planner.plan_segment(start, goal, (0.0, 0.0), (0.0, 0.0),
                                    (60.0, 60.0), (120.0, 120.0), DT)
    assert samples[-1] == pytest.approx(goal)


def test_wraps_around_360_instead_of_sweeping_the_long_way():
    # start=359, goal=1 is a 2deg move physically (goal wraps to 361,
    # right next to start) -- NOT the 358deg a raw subtraction would
    # compute. Regression test for a real bug: without wrapping the goal
    # to its nearest equivalent first, the arm would sweep almost an
    # entire extra revolution to reach a target only a couple degrees
    # away.
    planner = mp.get_planner("trapezoidal")
    start, goal = (359.0, 0.0), (1.0, 0.0)
    samples = planner.plan_segment(start, goal, (0.0, 0.0), (0.0, 0.0),
                                    (60.0, 60.0), (120.0, 120.0), DT)
    for j1, _j2 in samples:
        assert 358.0 <= j1 <= 362.0, f"swept way outside the ~2deg move: {j1}"
    # lands on the equivalent angle nearest start (361), not the literal
    # goal (1) -- that's what set_target_deg()'s own %360 wrapping expects.
    assert samples[-1][0] == pytest.approx(361.0)
    assert samples[-1][0] % 360.0 == pytest.approx(1.0)


def test_zero_distance_returns_single_sample_at_goal():
    planner = mp.get_planner("trapezoidal")
    samples = planner.plan_segment((5.0, 5.0), (5.0, 5.0), (0.0, 0.0), (0.0, 0.0),
                                    (60.0, 60.0), (120.0, 120.0), DT)
    assert samples == [(5.0, 5.0)]


def test_both_joints_arrive_at_the_same_sample_index():
    # The whole point of synchronization: neither joint should finish
    # early and sit idle while the other keeps moving.
    planner = mp.get_planner("trapezoidal")
    start, goal = (0.0, 0.0), (100.0, 5.0)  # very different distances
    samples = planner.plan_segment(start, goal, (0.0, 0.0), (0.0, 0.0),
                                    (60.0, 60.0), (120.0, 120.0), DT)
    # joint2 (short move) must not reach its final value before the last sample
    j2_values = [s[1] for s in samples]
    assert j2_values[-2] != pytest.approx(goal[1], abs=1e-9) or len(samples) == 1


def test_velocity_never_exceeds_vmax():
    planner = mp.get_planner("trapezoidal")
    start, goal = (0.0, 0.0), (90.0, -60.0)
    vmax = (60.0, 60.0)
    samples = planner.plan_segment(start, goal, (0.0, 0.0), (0.0, 0.0),
                                    vmax, (200.0, 200.0), DT)
    vels = _velocities(samples, start, DT)
    for vx, vy in vels:
        assert abs(vx) <= vmax[0] + 1e-6
        assert abs(vy) <= vmax[1] + 1e-6


def test_acceleration_never_exceeds_amax():
    planner = mp.get_planner("trapezoidal")
    start, goal = (0.0, 0.0), (90.0, -60.0)
    amax = (120.0, 120.0)
    samples = planner.plan_segment(start, goal, (0.0, 0.0), (0.0, 0.0),
                                    (60.0, 60.0), amax, DT)
    vels = _velocities(samples, start, DT)
    accs = _accelerations(vels, DT)
    # allow generous slack for finite-difference/discretization noise
    for ax, ay in accs:
        assert abs(ax) <= amax[0] * 1.15
        assert abs(ay) <= amax[1] * 1.15


def test_triangular_profile_for_short_distance():
    # Distance too short to reach vmax before needing to decelerate --
    # should still land exactly on goal and never exceed vmax.
    planner = mp.get_planner("trapezoidal")
    start, goal = (0.0, 0.0), (2.0, 0.0)
    vmax, amax = (100.0, 100.0), (50.0, 50.0)
    samples = planner.plan_segment(start, goal, (0.0, 0.0), (0.0, 0.0), vmax, amax, DT)
    assert samples[-1] == pytest.approx(goal)
    vels = _velocities(samples, start, DT)
    peak_v = max(abs(v[0]) for v in vels)
    assert peak_v < vmax[0]  # never reached vmax -- confirms triangular, not trapezoidal


def test_path_is_a_straight_line_in_joint_space():
    # angle_i(t) = start_i + s(t)*(goal_i - start_i) for both joints with
    # the SAME s(t) means every sample lies on the straight line from
    # start to goal in joint space.
    planner = mp.get_planner("trapezoidal")
    start, goal = (10.0, -30.0), (70.0, 90.0)
    samples = planner.plan_segment(start, goal, (0.0, 0.0), (0.0, 0.0),
                                    (60.0, 60.0), (120.0, 120.0), DT)
    dx, dy = goal[0] - start[0], goal[1] - start[1]
    for jx, jy in samples:
        # cross product of (sample-start) and (goal-start) should be ~0
        cross = (jx - start[0]) * dy - (jy - start[1]) * dx
        assert abs(cross) < 1e-6


def test_nonzero_boundary_velocity_for_corner_blending():
    # A segment that should NOT come to a full stop at the end (v_end
    # nonzero) -- this is what corner blending relies on.
    planner = mp.get_planner("trapezoidal")
    start, goal = (0.0, 0.0), (50.0, 0.0)
    vmax, amax = (60.0, 60.0), (120.0, 120.0)
    v_end = (30.0, 0.0)
    samples = planner.plan_segment(start, goal, (0.0, 0.0), v_end, vmax, amax, DT)
    assert samples[-1] == pytest.approx(goal)
    # velocity at the last step should be close to v_end, not 0
    vels = _velocities(samples, start, DT)
    assert vels[-1][0] == pytest.approx(v_end[0], abs=3.0)


def test_nonzero_start_velocity_continues_smoothly():
    # A segment picking up from a nonzero entry velocity (as if handed off
    # from a previous segment's nonzero v_end) shouldn't start with a
    # velocity discontinuity/jerk back to 0.
    planner = mp.get_planner("trapezoidal")
    start, goal = (0.0, 0.0), (50.0, 0.0)
    vmax, amax = (60.0, 60.0), (120.0, 120.0)
    v_start = (30.0, 0.0)
    samples = planner.plan_segment(start, goal, v_start, (0.0, 0.0), vmax, amax, DT)
    vels = _velocities(samples, start, DT)
    assert vels[0][0] == pytest.approx(v_start[0], abs=3.0)
