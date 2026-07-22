"""Shared real-time motion controller for jogging and scanning the arm.

Before this file existed, there were three independent, hand-rolled point-
motion implementations: software/main.py's text-REPL jog (raw joint
angles, one-shot move_and_wait, no smoothing), manual_test/run.py's curses
arrow-key jog + scan (its own speed/acc constants and a "stream a new
target every 30ms without waiting for arrival" trick), and
manual_test/gui.py's pygame version of the same thing -- reimplemented
independently, which is exactly how its scan grid/speed/row-limit
constants silently drifted out of sync with run.py's copies (see git
history). All three also only ever came to a *complete* stop between
points, which is why the dense scan grid looked jittery: every waypoint
was a full accelerate-decelerate-to-zero cycle.

ArmController below is the one implementation all three frontends now
call into. It owns:
  - talking to a TrajectoryPlanner (motion_planning/) to turn "move from
    here to there" into a smooth, time-sampled sequence of joint
    setpoints, replacing the old register-limits + fixed-interval hack;
  - workspace (mm) <-> joint (servo degrees) convenience via arm_core's
    IK/FK, since the manual test tools think in workspace coordinates;
  - multi-waypoint scanning with corner blending: consecutive scan
    segments that continue in roughly the same joint-space direction
    coast through the corner at cruise speed instead of stopping, and
    only decelerate to a full stop where the path actually turns around
    (e.g. end of a scan row) -- see _corner_blend_velocity.

Frontends (main.py, manual_test/run.py, manual_test/gui.py) are thin
adapters: they translate keypresses to calls on this controller and read
back `commanded_deg`/`scan_progress` to render. None of them plan a
trajectory or touch servo speed/acc registers directly.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Optional

import motion_planning as mp
from arm_core import (ArmParams, MotionConfig, calib_arm_params, calib_joint_limits,
                       calib_motion_config, ik_solve, within_joint_limits)
from arm_hardware import STREAMING_ACC, STREAMING_SPEED

logger = logging.getLogger("jog_controller")

TwoTuple = tuple


@dataclass
class _ScanState:
    joint_targets: list = field(default_factory=list)
    index: int = 0

    @property
    def active(self) -> bool:
        return self.index < len(self.joint_targets)


class ArmController:
    """Drives real-time joint motion by continuously feeding a
    TrajectoryPlanner's planned segments to a Servos handle at a fixed
    control-loop rate. `tick()` is meant to be called once per frame/loop
    iteration by whatever event loop the frontend already has (curses
    getch loop, pygame event loop, or a simple sleep loop for a REPL)."""

    def __init__(self, servos, params: ArmParams, planner: mp.TrajectoryPlanner,
                 motion_cfg: MotionConfig, joint_limits: Optional[dict] = None):
        self.servos = servos
        self.params = params
        self.planner = planner
        self.motion_cfg = motion_cfg
        self.dt = 1.0 / motion_cfg.control_hz
        # Mechanical dead-zone protection (see arm_core.within_joint_limits).
        # None (not configured) means every goal below is only checked
        # against IK reachability, not any physical safe-range boundary --
        # see main.py's set-joint-limits command and this project's
        # servo-hardware Min/Max Angle Limit registers (arm_hardware.py)
        # for the outer, hardware-enforced layer of this same protection.
        self.joint_limits = joint_limits

        s1 = servos.get_present_deg("joint1")
        s2 = servos.get_present_deg("joint2")
        # Nothing is commanded until a caller actually asks for a move --
        # seeding from the servos' real current position (rather than some
        # hardcoded default) means the very first jog step is a small nudge
        # from wherever the arm physically is, not a jump.
        self._commanded: TwoTuple = (s1, s2)
        self._velocity: TwoTuple = (0.0, 0.0)
        self._joint_goal: TwoTuple = (s1, s2)
        self._queue: list = []
        self._scan: Optional[_ScanState] = None

    # ── State readback (for frontends to render) ───────────────────────

    @property
    def commanded_deg(self) -> TwoTuple:
        """Where the planner is telling the servos to be *right now* --
        not necessarily where they've physically arrived yet. This is the
        "ghost"/target position manual_test/gui.py draws."""
        return self._commanded

    @property
    def is_moving(self) -> bool:
        return bool(self._queue) or self.scan_active

    @property
    def scan_active(self) -> bool:
        return self._scan is not None and self._scan.active

    @property
    def scan_progress(self) -> TwoTuple:
        """(completed_count, total_count) for a "point i/N" display."""
        if self._scan is None:
            return (0, 0)
        return (self._scan.index, len(self._scan.joint_targets))

    # ── Single-target motion ────────────────────────────────────────────

    def set_joint_goal(self, j1_deg: float, j2_deg: float) -> bool:
        """Plan a fresh segment from wherever the arm is currently
        commanded (with whatever velocity it currently has) to
        (j1_deg, j2_deg), replacing anything queued. Interrupts an active
        scan -- a manual goal always takes precedence, matching the
        expectation that pressing e.g. 'h' mid-scan should immediately
        head home rather than wait for the current scan leg to finish.

        Returns False (no-op, nothing planned/sent) if the target violates
        self.joint_limits -- this is the lowest common choke point every
        goal-setting method funnels through (set_workspace_goal already
        filters via ik_solve, but set_single_joint_goal's raw-angle input
        doesn't go through IK at all, so the check is enforced here too,
        not just there)."""
        if not within_joint_limits(j1_deg, j2_deg, self.joint_limits):
            logger.warning("joint goal (%.1f, %.1f) rejected: outside configured "
                            "joint_limits_deg", j1_deg, j2_deg)
            return False
        self._scan = None
        vmax = (self.motion_cfg.jog_vmax_deg_s,) * 2
        amax = (self.motion_cfg.jog_amax_deg_s2,) * 2
        self._joint_goal = (j1_deg, j2_deg)
        self._queue = self.planner.plan_segment(
            self._commanded, self._joint_goal, self._velocity, (0.0, 0.0), vmax, amax, self.dt)
        return True

    def set_single_joint_goal(self, joint: str, angle_deg: float) -> bool:
        """Set one joint's absolute target, leaving the other joint's
        target unchanged -- matches the old main.py jog REPL's semantics
        ('1 <deg>' moves joint1 only). Returns False if rejected (see
        set_joint_goal) -- this is the highest-risk direct entry point
        (a raw absolute angle typed by an operator, no IK sanity check at
        all otherwise), so callers should check this return value."""
        goal = list(self._joint_goal)
        goal[0 if joint == "joint1" else 1] = angle_deg
        return self.set_joint_goal(goal[0], goal[1])

    def set_workspace_goal(self, x_mm: float, y_mm: float) -> bool:
        """IK-based convenience for frontends that think in workspace mm.
        Returns False (no-op, nothing sent) if the point is unreachable OR
        violates self.joint_limits (ik_solve checks both)."""
        r = ik_solve(self.params, x_mm, y_mm, joint_limits=self.joint_limits)
        if not r.reachable:
            return False
        return self.set_joint_goal(r.servo1_deg, r.servo2_deg)

    def nudge_workspace(self, dx_mm: float, dy_mm: float, base: TwoTuple) -> Optional[TwoTuple]:
        """Move a workspace-space target by a relative amount (one arrow-
        key press) starting from `base` (the frontend's own idea of "where
        the target currently is" -- see e.g. manual_test/run.py, which
        tracks this in its own state dict since ArmController intentionally
        doesn't own "the workspace target," only "the joint goal").

        Replanning from the arm's current commanded position/velocity
        (rather than waiting for the previous nudge to finish) is what
        makes a held key read as continuous motion instead of stepping and
        waiting -- a receding-horizon target.

        Returns the new (x, y) target if reachable (caller should store it
        back into `base` for the next nudge), or None if this nudge would
        go somewhere unreachable (caller should keep the old base as-is).
        """
        tx, ty = base[0] + dx_mm, base[1] + dy_mm
        if self.set_workspace_goal(tx, ty):
            return (tx, ty)
        return None

    # ── Multi-waypoint scanning with corner blending ───────────────────

    def start_scan(self, waypoints: list) -> None:
        """Begin a scan over `waypoints` (as produced by
        arm_core.generate_scan_path: (x_mm, y_mm) or (x_mm, y_mm, label)
        tuples). Waypoints that are unreachable OR would violate
        self.joint_limits are skipped with a warning, same as the old
        per-frontend behaviour. tick() drives the scan forward segment by
        segment; call stop_scan() to abort early."""
        joint_targets = []
        for x, y, *_ in waypoints:
            r = ik_solve(self.params, x, y, joint_limits=self.joint_limits)
            if r.reachable:
                joint_targets.append((r.servo1_deg, r.servo2_deg))
            else:
                logger.warning("scan waypoint (%.1f, %.1f) unreachable or outside "
                                "joint_limits_deg, skipping", x, y)
        self._scan = _ScanState(joint_targets=joint_targets, index=0)
        self._queue = []
        self._advance_scan()

    def stop_scan(self) -> None:
        """Abort the active scan immediately. Does NOT clear the current
        in-flight segment -- whatever's already queued keeps playing out
        via tick() (e.g. so a subsequent set_joint_goal/set_workspace_goal
        call, like heading home, smoothly continues from the arm's actual
        current velocity rather than yanking to a dead stop first)."""
        self._scan = None

    def _segment_direction(self, a: TwoTuple, b: TwoTuple) -> TwoTuple:
        dx, dy = b[0] - a[0], b[1] - a[1]
        norm = math.hypot(dx, dy)
        if norm < 1e-9:
            return (0.0, 0.0)
        return (dx / norm, dy / norm)

    def _corner_blend_velocity(self, scan: _ScanState) -> TwoTuple:
        """Exit velocity for the segment ending at scan.joint_targets[scan.index]:
        nonzero (coast through the corner at cruise speed) if the next
        segment continues in roughly the same joint-space direction
        (cos of the angle between them > blend_threshold); zero (come to a
        full stop) at an actual turnaround, e.g. the end of a scan row.
        This is a simplified two-joint version of the "corner
        deceleration/coasting" every CNC controller does -- not a full
        look-ahead planner, just a one-segment-ahead check."""
        is_last = scan.index + 1 >= len(scan.joint_targets)
        if is_last:
            return (0.0, 0.0)
        goal = scan.joint_targets[scan.index]
        nxt = scan.joint_targets[scan.index + 1]
        dir_cur = self._segment_direction(self._commanded, goal)
        dir_next = self._segment_direction(goal, nxt)
        cos_theta = dir_cur[0] * dir_next[0] + dir_cur[1] * dir_next[1]
        if cos_theta > self.motion_cfg.blend_threshold:
            cruise = self.motion_cfg.scan_vmax_deg_s
            return (cruise * dir_cur[0], cruise * dir_cur[1])
        return (0.0, 0.0)

    def _advance_scan(self) -> None:
        scan = self._scan
        if scan is None or not scan.active:
            self._scan = None
            return
        goal = scan.joint_targets[scan.index]
        v_end = self._corner_blend_velocity(scan)
        vmax = (self.motion_cfg.scan_vmax_deg_s,) * 2
        amax = (self.motion_cfg.scan_amax_deg_s2,) * 2
        self._queue = self.planner.plan_segment(
            self._commanded, goal, self._velocity, v_end, vmax, amax, self.dt)
        self._joint_goal = goal
        scan.index += 1

    # ── Control loop step ────────────────────────────────────────────────

    def tick(self) -> TwoTuple:
        """Advance one control-loop step: pop the next planned setpoint
        (refilling from an active scan if the queue just ran dry) and
        command it to the servos at STREAMING_SPEED/STREAMING_ACC -- see
        arm_hardware's module docstring for why the servo's own
        speed/acc limiting needs to get out of the way here (all the
        smoothing already happened in the planner). Returns the newly
        commanded (joint1_deg, joint2_deg)."""
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
        """Blocking convenience for callers without their own event loop
        already ticking the controller (e.g. main.py's jog REPL): plan a
        segment and tick it through to arrival, sleeping between ticks."""
        self.set_joint_goal(j1_deg, j2_deg)
        deadline = time.monotonic() + timeout_s
        while self.is_moving and time.monotonic() < deadline:
            self.tick()
            time.sleep(self.dt)
        return self._commanded


def build_controller(servos, calib: dict) -> ArmController:
    """Wires up an ArmController from calib.json's kinematics/motion/
    joint_limits_deg sections -- the few lines every frontend needs, in
    one place instead of copy-pasted three times."""
    params = calib_arm_params(calib)
    motion_cfg = calib_motion_config(calib)
    joint_limits = calib_joint_limits(calib)
    if joint_limits is None:
        msg = ("no joint_limits_deg configured in calib.json -- IK/jog will not "
               "reject any target on mechanical dead-zone grounds. Run "
               "`main.py set-joint-limits` once to measure and configure this "
               "(and, more importantly, the servo's own hardware angle-limit "
               "registers, which protect even against a bug in this software).")
        logger.warning(msg)
        print(f"WARNING: {msg}")
    planner = mp.get_planner(motion_cfg.planner_name)
    return ArmController(servos, params, planner, motion_cfg, joint_limits=joint_limits)
