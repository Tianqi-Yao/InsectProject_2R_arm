"""Motion controller for card_scan: single-target trapezoidal moves PLUS
multi-waypoint scanning with corner blending (consecutive scan segments
that continue in roughly the same joint-space direction coast through the
corner at cruise speed instead of stopping) -- card_scan is the one v2
package that actually needs this, since a dense scan grid looks jittery
otherwise (every waypoint would be a full accelerate-decelerate-to-zero
cycle). See kinematics_fit/controller.py for the simpler, single-target-only
twin of this module used by packages that never scan a multi-point path."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Optional

from arm_hw_core.limits import JointLimits, within_joint_limits
from arm_hw_core.servos import STREAMING_ACC, STREAMING_SPEED, Servos

from .angles import wrap_angle_near
from .kinematics import ArmParams, ik_solve

TwoTuple = tuple


def _solve_single_axis(distance: float, v_entry: float, v_exit: float,
                        vmax: float, amax: float) -> tuple[float, float, float, float, float]:
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
    samples[-1] = tuple(goal_deg)
    return samples


@dataclass
class MotionParams:
    jog_vmax_deg_s: float = 60.0
    jog_amax_deg_s2: float = 120.0
    scan_vmax_deg_s: float = 90.0
    scan_amax_deg_s2: float = 180.0
    blend_threshold: float = 0.7
    control_hz: float = 50.0


@dataclass
class _ScanState:
    joint_targets: list = field(default_factory=list)
    index: int = 0

    @property
    def active(self) -> bool:
        return self.index < len(self.joint_targets)


class ArmController:
    def __init__(self, servos: Servos, params: ArmParams, motion: MotionParams,
                 joint_limits: Optional[JointLimits] = None):
        self.servos = servos
        self.params = params
        self.motion = motion
        self.dt = 1.0 / motion.control_hz
        self.joint_limits = joint_limits

        s1 = servos.get_present_deg("joint1")
        s2 = servos.get_present_deg("joint2")
        self._commanded: TwoTuple = (s1, s2)
        self._velocity: TwoTuple = (0.0, 0.0)
        self._joint_goal: TwoTuple = (s1, s2)
        self._queue: list = []
        self._scan: Optional[_ScanState] = None

    @property
    def commanded_deg(self) -> TwoTuple:
        return self._commanded

    @property
    def is_moving(self) -> bool:
        return bool(self._queue) or self.scan_active

    @property
    def scan_active(self) -> bool:
        return self._scan is not None and self._scan.active

    @property
    def scan_progress(self) -> TwoTuple:
        if self._scan is None:
            return (0, 0)
        return (self._scan.index, len(self._scan.joint_targets))

    def set_joint_goal(self, j1_deg: float, j2_deg: float) -> bool:
        if not within_joint_limits(j1_deg, j2_deg, self.joint_limits):
            return False
        self._scan = None
        vmax = (self.motion.jog_vmax_deg_s,) * 2
        amax = (self.motion.jog_amax_deg_s2,) * 2
        self._joint_goal = (j1_deg, j2_deg)
        self._queue = plan_segment(
            self._commanded, self._joint_goal, self._velocity, (0.0, 0.0), vmax, amax, self.dt)
        return True

    def set_workspace_goal(self, x_mm: float, y_mm: float) -> bool:
        r = ik_solve(self.params, x_mm, y_mm, joint_limits=self.joint_limits)
        if not r.reachable:
            return False
        return self.set_joint_goal(r.servo1_deg, r.servo2_deg)

    def start_scan(self, waypoints: list) -> None:
        """`waypoints`: (x_mm, y_mm) or (x_mm, y_mm, label) tuples, as
        produced by scan.generate_scan_path. Waypoints that are unreachable
        or violate self.joint_limits are skipped with a warning."""
        joint_targets = []
        for x, y, *_ in waypoints:
            r = ik_solve(self.params, x, y, joint_limits=self.joint_limits)
            if r.reachable:
                joint_targets.append((r.servo1_deg, r.servo2_deg))

        # Chain each waypoint to its nearest equivalent (mod 360) relative
        # to the PREVIOUS waypoint (seeded from the arm's current commanded
        # position), so an independent ik_solve() call per waypoint never
        # introduces a spurious ~360deg jump between two waypoints that are
        # actually close together.
        prev = self._commanded
        for i, (j1, j2) in enumerate(joint_targets):
            prev = (wrap_angle_near(j1, prev[0]), wrap_angle_near(j2, prev[1]))
            joint_targets[i] = prev
        self._scan = _ScanState(joint_targets=joint_targets, index=0)
        self._queue = []
        self._advance_scan()

    def stop_scan(self) -> None:
        self._scan = None

    def _segment_direction(self, a: TwoTuple, b: TwoTuple) -> TwoTuple:
        dx, dy = b[0] - a[0], b[1] - a[1]
        norm = math.hypot(dx, dy)
        if norm < 1e-9:
            return (0.0, 0.0)
        return (dx / norm, dy / norm)

    def _corner_blend_velocity(self, scan: _ScanState) -> TwoTuple:
        is_last = scan.index + 1 >= len(scan.joint_targets)
        if is_last:
            return (0.0, 0.0)
        goal = scan.joint_targets[scan.index]
        nxt = scan.joint_targets[scan.index + 1]
        dir_cur = self._segment_direction(self._commanded, goal)
        dir_next = self._segment_direction(goal, nxt)
        cos_theta = dir_cur[0] * dir_next[0] + dir_cur[1] * dir_next[1]
        if cos_theta > self.motion.blend_threshold:
            cruise = self.motion.scan_vmax_deg_s
            return (cruise * dir_cur[0], cruise * dir_cur[1])
        return (0.0, 0.0)

    def _advance_scan(self) -> None:
        scan = self._scan
        if scan is None or not scan.active:
            self._scan = None
            return
        goal = scan.joint_targets[scan.index]
        v_end = self._corner_blend_velocity(scan)
        vmax = (self.motion.scan_vmax_deg_s,) * 2
        amax = (self.motion.scan_amax_deg_s2,) * 2
        self._queue = plan_segment(
            self._commanded, goal, self._velocity, v_end, vmax, amax, self.dt)
        self._joint_goal = goal
        scan.index += 1

    def tick(self) -> TwoTuple:
        if not self._queue and self.scan_active:
            self._advance_scan()

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
        """See kinematics_fit/controller.py's resync() docstring. Also
        drops any in-flight scan, since it was planned against the same
        now-stale state."""
        self._commanded = (j1_deg, j2_deg)
        self._joint_goal = (j1_deg, j2_deg)
        self._velocity = (0.0, 0.0)
        self._queue = []
        self._scan = None
