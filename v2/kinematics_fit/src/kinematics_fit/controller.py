"""Minimal motion controller for kinematics_fit: single-target trapezoidal
point-to-point moves, used to visit calibration target poses one at a time
(see fit.generate_calibration_targets) and to fold the arm to a spot-check
pose. No corner blending / multi-waypoint scanning here -- unlike
card_scan, kinematics_fit never needs to move smoothly through a chain of
waypoints, so that complexity is left out (see card_scan/controller.py for
the version that needs it).

This is v2's own, independently-implemented copy of the trapezoidal
planner + real-time controller pattern -- see kinematics_fit/README or the
v2 top-level plan for why each feature package carries its own copy rather
than sharing this code.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Optional

from arm_hw_core.limits import JointLimits, within_joint_limits
from arm_hw_core.servos import STREAMING_ACC, STREAMING_SPEED, Servos

from .angles import wrap_angle_near

TwoTuple = tuple


def _solve_single_axis(distance: float, v_entry: float, v_exit: float,
                        vmax: float, amax: float) -> tuple[float, float, float, float, float]:
    """Returns (t1, t2, t3, v_peak, T) for one joint's accel/cruise/decel
    phases covering `distance` (>=0), starting at v_entry and ending at
    v_exit (both clamped to vmax). Standard bounded-acceleration trapezoid
    (falls back to triangular when the distance is too short to reach vmax)."""
    v_entry = max(0.0, min(v_entry, vmax))
    v_exit = max(0.0, min(v_exit, vmax))

    v_peak_sq = amax * distance + 0.5 * (v_entry ** 2 + v_exit ** 2)
    v_peak = math.sqrt(max(v_peak_sq, 0.0))

    if v_peak <= vmax:
        t1 = (v_peak - v_entry) / amax if v_peak > v_entry else 0.0
        t3 = (v_peak - v_exit) / amax if v_peak > v_exit else 0.0
        t2 = 0.0
    else:
        v_peak = vmax
        t1 = (vmax - v_entry) / amax
        t3 = (vmax - v_exit) / amax
        d1 = (v_entry + vmax) / 2.0 * t1
        d3 = (vmax + v_exit) / 2.0 * t3
        d2 = max(distance - d1 - d3, 0.0)
        t2 = d2 / vmax if vmax > 0 else 0.0

    return t1, t2, t3, v_peak, t1 + t2 + t3


def _progress_at(t: float, t1: float, t2: float, v_entry: float,
                  v_peak: float, amax: float, distance: float) -> float:
    if distance <= 0.0:
        return 1.0
    if t <= t1:
        pos = v_entry * t + 0.5 * amax * t * t
    elif t <= t1 + t2:
        d1 = v_entry * t1 + 0.5 * amax * t1 * t1
        pos = d1 + v_peak * (t - t1)
    else:
        d1 = v_entry * t1 + 0.5 * amax * t1 * t1
        d2 = v_peak * t2
        td = t - t1 - t2
        pos = d1 + d2 + v_peak * td - 0.5 * amax * td * td
    return min(pos / distance, 1.0)


def plan_segment(start_deg: TwoTuple, goal_deg: TwoTuple, v_start_deg_s: TwoTuple,
                  v_end_deg_s: TwoTuple, vmax_deg_s: TwoTuple, amax_deg_s2: TwoTuple,
                  dt_s: float) -> list[TwoTuple]:
    """Two-joint-synchronized trapezoidal segment: both joints solved
    independently, the longer (dominant) joint's own progress curve drives
    both so the path stays a straight line in joint space with an exact
    simultaneous arrival."""
    goal_deg = tuple(wrap_angle_near(goal_deg[i], start_deg[i]) for i in (0, 1))
    D = [goal_deg[i] - start_deg[i] for i in (0, 1)]
    absD = [abs(d) for d in D]

    if absD[0] == 0.0 and absD[1] == 0.0:
        return [tuple(goal_deg)]

    axes = [_solve_single_axis(absD[i], abs(v_start_deg_s[i]), abs(v_end_deg_s[i]),
                                vmax_deg_s[i], amax_deg_s2[i]) for i in (0, 1)]
    dominant = 0 if axes[0][4] >= axes[1][4] else 1
    t1, t2, t3, v_peak, T = axes[dominant]
    v_entry_dom = max(0.0, min(abs(v_start_deg_s[dominant]), vmax_deg_s[dominant]))
    amax_dom = amax_deg_s2[dominant]
    D_dom = absD[dominant]

    n_steps = max(1, round(T / dt_s)) if T > 0 else 1
    step = T / n_steps if T > 0 else dt_s
    samples = []
    for k in range(1, n_steps + 1):
        t = min(k * step, T)
        s = _progress_at(t, t1, t2, v_entry_dom, v_peak, amax_dom, D_dom)
        samples.append((start_deg[0] + s * D[0], start_deg[1] + s * D[1]))
    samples[-1] = tuple(goal_deg)  # exact landing, guards against fp roundoff
    return samples


@dataclass
class MotionParams:
    vmax_deg_s: float = 60.0
    amax_deg_s2: float = 120.0
    control_hz: float = 50.0


class ArmController:
    """Drives real-time joint motion by feeding planned trapezoidal
    segments to a Servos handle. `tick()` is meant to be called once per
    control-loop iteration; `run_to_completion` is a blocking convenience
    for scripted callers (calibration data collection, self-check)."""

    def __init__(self, servos: Servos, motion: MotionParams,
                 joint_limits: Optional[JointLimits] = None):
        self.servos = servos
        self.motion = motion
        self.dt = 1.0 / motion.control_hz
        self.joint_limits = joint_limits

        s1 = servos.get_present_deg("joint1")
        s2 = servos.get_present_deg("joint2")
        self._commanded: TwoTuple = (s1, s2)
        self._velocity: TwoTuple = (0.0, 0.0)
        self._joint_goal: TwoTuple = (s1, s2)
        self._queue: list = []

    @property
    def commanded_deg(self) -> TwoTuple:
        return self._commanded

    @property
    def is_moving(self) -> bool:
        return bool(self._queue)

    def set_joint_goal(self, j1_deg: float, j2_deg: float) -> bool:
        """Returns False (no-op) if the target violates self.joint_limits."""
        if not within_joint_limits(j1_deg, j2_deg, self.joint_limits):
            return False
        vmax = (self.motion.vmax_deg_s,) * 2
        amax = (self.motion.amax_deg_s2,) * 2
        self._joint_goal = (j1_deg, j2_deg)
        self._queue = plan_segment(
            self._commanded, self._joint_goal, self._velocity, (0.0, 0.0), vmax, amax, self.dt)
        return True

    def tick(self) -> TwoTuple:
        if self._queue:
            nxt = self._queue.pop(0)
            self._velocity = ((nxt[0] - self._commanded[0]) / self.dt,
                               (nxt[1] - self._commanded[1]) / self.dt)
            self._commanded = nxt
            self.servos.set_target_deg("joint1", nxt[0], speed=STREAMING_SPEED, acc=STREAMING_ACC)
            self.servos.set_target_deg("joint2", nxt[1], speed=STREAMING_SPEED, acc=STREAMING_ACC)
        return self._commanded

    def run_to_completion(self, j1_deg: float, j2_deg: float, timeout_s: float = 8.0) -> TwoTuple:
        self.set_joint_goal(j1_deg, j2_deg)
        deadline = time.monotonic() + timeout_s
        while self.is_moving and time.monotonic() < deadline:
            self.tick()
            time.sleep(self.dt)
        return self._commanded

    def resync(self, j1_deg: float, j2_deg: float) -> None:
        """Reset all internal motion state to (j1_deg, j2_deg), as if the
        controller had just been constructed at that pose. Call this after
        torque re-engages following any period the arm could have been
        hand-moved while free: self._commanded only ever updates from
        inside tick() while something is queued, so it silently goes stale
        the moment torque is released and the arm is moved by hand.
        Without this, the next goal-setting call plans a segment starting
        from that stale remembered pose, not from where the arm actually
        now is -- streaming a phantom jump back through it before
        continuing on to the real target."""
        self._commanded = (j1_deg, j2_deg)
        self._joint_goal = (j1_deg, j2_deg)
        self._velocity = (0.0, 0.0)
        self._queue = []
