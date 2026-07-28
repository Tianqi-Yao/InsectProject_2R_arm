"""Single comprehensive pygame control panel for the v3 arm: one process,
one window, five modes (Tab / Shift+Tab to cycle, F1-F5 to jump directly):

  JOG    per-joint torque lock/unlock, arrow-key workspace jogging, direct
         per-joint jogging.
  TEACH  hand-teach the 4 scan-rectangle corners (torque auto-released).
  PATH   record an arbitrary hand-driven path (torque auto-released while
         recording), save/load/rename/delete it by name, play it back.
  SCAN   generate + preview + run the serpentine photo grid over the
         taught rectangle (see path_core.PhotoScanRunner), with a
         dry-run mode and a thumbnail of the last captured photo.
  PARAMS live-editable view of every numeric knob in calib.json
         (kinematics/motion/photo/hardware/joint_limits) -- arrow keys to
         adjust by a step, Enter to type an exact value, 's' to persist.

This replaces the separate teach_gui.py/scan_gui.py processes: everything
that used to require restarting a different tool now lives in one running
session, so the same servo connection, camera connection, and live-edited
parameters carry across modes without reconnecting or reloading anything.

Torque toggling always resyncs ArmController's internal state to the
servos' real position (see jog_controller.ArmController.resync) and
clears any stale jog target -- without this, jogging right after a
teach-in drag could plan a segment starting from wherever the controller
last THOUGHT it was, streaming a phantom jump back through that stale
position before continuing to the real target.

Global keys (work in every mode):
  Tab / Shift+Tab   cycle modes forward/backward
  F1..F5            jump directly to JOG/TEACH/PATH/SCAN/PARAMS
  z / x             toggle joint1 / joint2 torque (blocked while moving)
  Space             abort whatever motion is currently in progress
  q / Esc           quit (torque re-engages, synced, before exit)
"""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass, field, fields as dataclass_fields
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, Optional

import pygame

import arm_core as core
import arm_hardware as hw
import jog_controller as jc
import motion_planning as mp
import path_core as pc

POLL_INTERVAL_S = 0.05
FPS = 60

BG = (28, 30, 40)
GRID = (48, 54, 72)
LINK1_C = (80, 160, 255)
LINK2_C = (255, 120, 55)
JOINT_C = (220, 220, 220)
EE_C = (55, 215, 95)
BASE_C = (200, 100, 55)
CORNER_C = (255, 210, 60)
RECT_C = (170, 90, 220)
PHOTO_C = (90, 200, 220)
PHOTO_DONE_C = (80, 220, 130)
PATH_C = (120, 200, 255)
TEXT_C = (200, 210, 225)
LABEL_C = (110, 130, 155)
OK_C = (80, 220, 130)
WARN_C = (220, 80, 80)
LOCKED_C = (80, 220, 130)
FREE_C = (255, 190, 60)
SEL_C = (255, 230, 140)


# ── Modes ──────────────────────────────────────────────────────────────

MODE_JOG, MODE_TEACH, MODE_PATH, MODE_SCAN, MODE_PARAMS = range(5)
MODE_NAMES = ["JOG", "TEACH", "PATH", "SCAN", "PARAMS"]
FUNCTION_KEY_MODES = {
    pygame.K_F1: MODE_JOG, pygame.K_F2: MODE_TEACH, pygame.K_F3: MODE_PATH,
    pygame.K_F4: MODE_SCAN, pygame.K_F5: MODE_PARAMS,
}
CAPTURE_KEYS = {pygame.K_1: 0, pygame.K_2: 1, pygame.K_3: 2, pygame.K_4: 3}


@dataclass
class Layout:
    """Screen-space layout, computed once from the arm's params at
    startup. Deliberately NOT recomputed if L1/L2/base_x/base_y get
    live-edited in PARAMS mode later -- the drawing might not perfectly
    re-center/re-scale after such an edit, but every actual IK/FK/motion
    computation still uses the live ArmParams object, not this Layout, so
    that's a cosmetic limitation only."""
    base_x: float
    base_y: float
    max_r_mm: float
    scale: float = 1.0
    margin_px: int = 40
    panel_w: int = 340

    def __post_init__(self):
        canvas_px = 560
        self.scale = canvas_px / (2 * self.max_r_mm)

    @property
    def win_w(self) -> int:
        return int(2 * self.max_r_mm * self.scale) + 2 * self.margin_px + self.panel_w

    @property
    def win_h(self) -> int:
        return int(2 * self.max_r_mm * self.scale) + 2 * self.margin_px

    def ws2s(self, wx: float, wy: float) -> tuple:
        cx = self.margin_px + self.max_r_mm * self.scale
        cy = self.margin_px + self.max_r_mm * self.scale
        return (int(cx + (wx - self.base_x) * self.scale),
                int(cy - (wy - self.base_y) * self.scale))


# ── Torque helpers ────────────────────────────────────────────────────

def _resync_and_relock(servos: hw.Servos) -> None:
    """Sync goal to current position before re-enabling torque, so the
    servo doesn't snap toward a stale old target."""
    for joint in ("joint1", "joint2"):
        try:
            angle = servos.get_present_deg(joint)
            servos.set_target_deg(joint, angle)
        except Exception:
            pass
    for joint in ("joint1", "joint2"):
        servos.set_torque_enabled(joint, True)


def _toggle_torque(servos: hw.Servos, controller: jc.ArmController, torque: dict,
                    state: "AppState", joint: str) -> None:
    new_state = not torque[joint]
    if new_state:
        # Re-syncing the servo's own goal register to its actual (hand-
        # moved) position before re-enabling torque is what stops IT from
        # snapping back toward whatever stale target it had before torque
        # was released.
        angle = servos.get_present_deg(joint)
        servos.set_target_deg(joint, angle)
    servos.set_torque_enabled(joint, new_state)
    torque[joint] = new_state
    # The jog target and the ArmController's own internal commanded/goal
    # state can now be stale too -- the arm may have been dragged anywhere
    # by hand while torque was off (control_gui.py's TEACH/PATH modes both
    # rely on that), and ArmController never tracks the real servo
    # position on its own while idle (see jog_controller.ArmController.
    # resync's docstring). Reset both on ANY torque transition, not just
    # when both joints end up locked: a single-joint direct jog
    # (set_single_joint_goal) always resends BOTH joints' targets, so even
    # locking just one joint needs the pair back in sync first.
    state.jog_target = None
    if new_state:
        s1 = servos.get_present_deg("joint1")
        s2 = servos.get_present_deg("joint2")
        controller.resync(s1, s2)


# ── PARAMS mode: generic live-editable field list ────────────────────

@dataclass
class Field:
    section: str
    label: str
    kind: str  # "float" | "int" | "sign" | "text"
    get: Callable[[], object]
    set: Callable[[object], None]
    step: float = 1.0
    fmt: str = "{:.2f}"


def _attr_field(obj, name: str, section: str, label: str, kind: str = "float",
                 step: float = 1.0, fmt: str = "{:.2f}") -> Field:
    return Field(section, label, kind,
                 get=lambda: getattr(obj, name), set=lambda v: setattr(obj, name, v),
                 step=step, fmt=fmt)


def build_fields(calib: dict, params: core.ArmParams, motion_cfg: core.MotionConfig,
                  photo_cfg: core.PhotoConfig, hw_cfg: core.HardwareConfig) -> list[Field]:
    fields = [
        _attr_field(params, "L1", "kinematics", "L1 (mm)", step=1.0),
        _attr_field(params, "L2", "kinematics", "L2 (mm)", step=1.0),
        _attr_field(params, "base_x", "kinematics", "base_x (mm)", step=1.0),
        _attr_field(params, "base_y", "kinematics", "base_y (mm)", step=1.0),
        _attr_field(params, "servo1_offset_deg", "kinematics", "servo1_offset (deg)", step=0.1),
        _attr_field(params, "servo2_offset_deg", "kinematics", "servo2_offset (deg)", step=0.1),
        _attr_field(params, "elbow_offset_mm", "kinematics", "elbow_offset (mm)", step=0.5),
        Field("kinematics", "servo1_dir (+1/-1)", "sign",
              get=lambda: params.servo1_dir, set=lambda v: setattr(params, "servo1_dir", v)),
        Field("kinematics", "servo2_dir (+1/-1)", "sign",
              get=lambda: params.servo2_dir, set=lambda v: setattr(params, "servo2_dir", v)),

        _attr_field(motion_cfg, "jog_vmax_deg_s", "motion", "jog_vmax (deg/s)", step=5.0),
        _attr_field(motion_cfg, "jog_amax_deg_s2", "motion", "jog_amax (deg/s2)", step=10.0),
        _attr_field(motion_cfg, "scan_vmax_deg_s", "motion", "scan_vmax (deg/s)", step=5.0),
        _attr_field(motion_cfg, "scan_amax_deg_s2", "motion", "scan_amax (deg/s2)", step=10.0),
        _attr_field(motion_cfg, "blend_threshold", "motion", "blend_threshold", step=0.05),
        _attr_field(motion_cfg, "control_hz", "motion", "control_hz", step=5.0),

        _attr_field(photo_cfg, "spacing_x_mm", "photo", "spacing_x (mm)", step=1.0),
        _attr_field(photo_cfg, "spacing_y_mm", "photo", "spacing_y (mm)", step=1.0),
        _attr_field(photo_cfg, "margin_mm", "photo", "margin (mm)", step=1.0),
        _attr_field(photo_cfg, "dwell_s", "photo", "dwell (s)", step=0.1),
        _attr_field(photo_cfg, "max_step_mm", "photo", "max_step (mm)", step=0.5),
        Field("photo", "photo_dir", "text",
              get=lambda: photo_cfg.photo_dir, set=lambda v: setattr(photo_cfg, "photo_dir", v)),

        Field("hardware*", "servo_port", "text",
              get=lambda: hw_cfg.servo_port, set=lambda v: setattr(hw_cfg, "servo_port", v)),
        Field("hardware*", "joint1 bus id", "int",
              get=lambda: hw_cfg.joint_ids["joint1"], step=1,
              set=lambda v: hw_cfg.joint_ids.__setitem__("joint1", int(v))),
        Field("hardware*", "joint2 bus id", "int",
              get=lambda: hw_cfg.joint_ids["joint2"], step=1,
              set=lambda v: hw_cfg.joint_ids.__setitem__("joint2", int(v))),
    ]

    jl = calib.get("joint_limits_deg")
    if jl is not None:
        for joint in ("joint1", "joint2"):
            fields.append(Field(
                "joint_limits", f"{joint} lo (deg)", "float", step=1.0,
                get=lambda j=joint: calib["joint_limits_deg"][j][0],
                set=lambda v, j=joint: calib["joint_limits_deg"][j].__setitem__(0, v)))
            fields.append(Field(
                "joint_limits", f"{joint} hi (deg)", "float", step=1.0,
                get=lambda j=joint: calib["joint_limits_deg"][j][1],
                set=lambda v, j=joint: calib["joint_limits_deg"][j].__setitem__(1, v)))
    return fields


def _adjust_field(f: Field, direction: int) -> None:
    if f.kind == "sign":
        f.set(-f.get())
    elif f.kind == "text":
        pass
    elif f.kind == "int":
        f.set(int(f.get()) + int(f.step) * direction)
    else:
        f.set(f.get() + f.step * direction)


def _format_field(f: Field) -> str:
    if f.kind == "sign":
        v = f.get()
        return "+1" if v > 0 else "-1"
    if f.kind in ("text", "int"):
        return str(f.get())
    return f.fmt.format(f.get())


def _copy_dataclass_fields(dst, src) -> None:
    for fd in dataclass_fields(dst):
        setattr(dst, fd.name, getattr(src, fd.name))


def _sync_calib_from_live(ctx: "Ctx") -> None:
    """Write every live-edited in-memory config object back into ctx.calib
    -- call this before EVERY core.save_calib(), regardless of which mode
    triggered the save. Without it, a save that only touches one section
    directly (e.g. TEACH mode setting calib["rectangle"]) would persist
    that fresh section alongside a STALE on-disk kinematics/motion/photo
    section from before any not-yet-saved PARAMS-mode edit -- and a
    taught rectangle's corners were computed through FK using the live
    ctx.params, so saving different kinematics than what was actually used
    to teach it would silently desync the geometry on the next reload (see
    arm_core.py's module docstring on why teach and scan must share the
    same params)."""
    ctx.calib["kinematics"] = asdict(ctx.params)
    ctx.calib["hardware"] = asdict(ctx.hw_cfg)
    ctx.calib["motion"] = asdict(ctx.motion_cfg)
    ctx.calib["photo"] = asdict(ctx.photo_cfg)


# ── Text input modal (path naming, exact-value entry) ────────────────

class TextInput:
    def __init__(self, prompt: str, initial: str = ""):
        self.prompt = prompt
        self.buffer = initial
        self.active = True
        self.result: Optional[str] = None

    def handle_key(self, event) -> None:
        if event.key == pygame.K_RETURN:
            self.result = self.buffer
            self.active = False
        elif event.key == pygame.K_ESCAPE:
            self.result = None
            self.active = False
        elif event.key == pygame.K_BACKSPACE:
            self.buffer = self.buffer[:-1]
        elif event.unicode and event.unicode.isprintable():
            self.buffer += event.unicode


# ── Shared app context + per-session state ────────────────────────────

@dataclass
class Ctx:
    calib: dict
    params: core.ArmParams
    motion_cfg: core.MotionConfig
    photo_cfg: core.PhotoConfig
    hw_cfg: core.HardwareConfig
    servos: hw.Servos
    camera: hw.Camera
    controller: jc.ArmController
    torque: dict
    paths: dict
    fields: list = field(default_factory=list)
    camera_connected: bool = False
    camera_error: Optional[str] = None


@dataclass
class AppState:
    mode: int = MODE_JOG
    status_msg: str = ""
    status_until: float = 0.0
    last_poll: float = 0.0
    s1: float = 0.0
    s2: float = 0.0
    jog_target: Optional[tuple] = None
    jog_step_mm: float = 5.0
    joint_step_deg: float = 2.0
    corners: list = field(default_factory=lambda: [None, None, None, None])
    recording: bool = False
    recorded_points: list = field(default_factory=list)
    path_names: list = field(default_factory=list)
    selected_path_idx: int = 0
    playback_active: bool = False
    runner: Optional[pc.PhotoScanRunner] = None
    scan_out_dir: Optional[Path] = None
    dry_run: bool = False
    field_idx: int = 0
    text_input: Optional[TextInput] = None
    text_input_purpose: Optional[str] = None
    text_input_field: Optional[Field] = None
    rename_old_name: Optional[str] = None
    repeat_key: Optional[int] = None
    repeat_next_at: float = 0.0
    log: list = field(default_factory=list)
    last_photo_surface: Optional[object] = None
    last_photo_label: str = ""


def _set_status(state: AppState, msg: str, ttl: float = 3.0) -> None:
    state.status_msg = msg
    state.status_until = time.monotonic() + ttl
    state.log.append(msg)
    del state.log[:-6]


def _switch_mode(ctx: Ctx, state: AppState, new_mode: int) -> None:
    if new_mode == MODE_TEACH and ctx.controller.is_moving:
        # Auto-releasing torque out from under an in-flight scan/playback
        # would strand the arm mid-motion while the runner keeps believing
        # it's progressing -- and PhotoScanRunner would go on to "arrive"
        # and capture photos of wherever the arm actually stopped being
        # driven, not where it visually looks like it is. Refuse the
        # switch entirely (stay in the current mode); TEACH is still
        # reachable once the motion finishes or is aborted (Space).
        _set_status(state, "arm is moving -- press Space to stop before teaching")
        return

    if state.mode == MODE_PATH and state.recording:
        state.recording = False
        _set_status(state, f"stopped recording ({len(state.recorded_points)} points)")
    state.mode = new_mode
    if new_mode == MODE_TEACH:
        for j in ("joint1", "joint2"):
            if ctx.torque[j]:
                _toggle_torque(ctx.servos, ctx.controller, ctx.torque, state, j)


# ── JOG mode ───────────────────────────────────────────────────────────

def _handle_jog_key(event, ctx: Ctx, state: AppState) -> None:
    both_locked = ctx.torque["joint1"] and ctx.torque["joint2"]
    if event.key == pygame.K_LEFTBRACKET:
        state.jog_step_mm = max(1.0, state.jog_step_mm - 1.0)
    elif event.key == pygame.K_RIGHTBRACKET:
        state.jog_step_mm = min(50.0, state.jog_step_mm + 1.0)
    elif event.key == pygame.K_MINUS:
        state.joint_step_deg = max(0.5, state.joint_step_deg - 0.5)
    elif event.key == pygame.K_EQUALS:
        state.joint_step_deg = min(20.0, state.joint_step_deg + 0.5)
    elif event.key in (pygame.K_UP, pygame.K_DOWN, pygame.K_LEFT, pygame.K_RIGHT):
        if not both_locked:
            _set_status(state, "torque off -- press z/x to lock both joints before jogging")
            return
        if state.jog_target is None:
            state.jog_target = core.fk_from_servo_angles(ctx.params, state.s1, state.s2)
        dx, dy = {
            pygame.K_UP: (0.0, state.jog_step_mm), pygame.K_DOWN: (0.0, -state.jog_step_mm),
            pygame.K_LEFT: (-state.jog_step_mm, 0.0), pygame.K_RIGHT: (state.jog_step_mm, 0.0),
        }[event.key]
        new = ctx.controller.nudge_workspace(dx, dy, state.jog_target)
        if new is None:
            _set_status(state, "unreachable -- nudge ignored")
        else:
            state.jog_target = new
    elif event.key in (pygame.K_i, pygame.K_k, pygame.K_o, pygame.K_l):
        joint = "joint1" if event.key in (pygame.K_i, pygame.K_k) else "joint2"
        if not ctx.torque[joint]:
            lock_key = "z" if joint == "joint1" else "x"
            _set_status(state, f"{joint} torque off -- press {lock_key} to lock it first")
            return
        # Step from the controller's own last-commanded GOAL, not a
        # physically-measured present angle (state.s1/s2): the present
        # angle lags behind while the arm is still travelling toward the
        # previous step, so re-deriving from it on every repeat would make
        # a held key feel sticky instead of accumulating smoothly (see
        # ArmController.goal_deg's docstring).
        j1, j2 = ctx.controller.goal_deg
        delta = -state.joint_step_deg if event.key in (pygame.K_i, pygame.K_o) else state.joint_step_deg
        if joint == "joint1":
            j1 += delta
        else:
            j2 += delta
        ctx.controller.set_joint_goal(j1, j2)
        # Keep the workspace jog accumulator (state.jog_target) in sync so
        # a subsequent arrow-key nudge continues from THIS step, not from
        # wherever workspace-jogging last left off before this direct
        # joint jog ran (set_joint_goal already keeps goal_deg in sync the
        # other way around, via set_workspace_goal -> set_joint_goal).
        state.jog_target = core.fk_from_servo_angles(ctx.params, j1, j2)


def _draw_jog_panel(lines: list, ctx: Ctx, state: AppState) -> None:
    lines += [
        "arrows       workspace jog (needs both locked)",
        "[ / ]        workspace jog step -/+",
        "i/k  o/l     joint1 -/+   joint2 -/+",
        "- / =        joint step -/+",
        "",
        f"workspace jog step: {state.jog_step_mm:.1f} mm",
        f"joint jog step:     {state.joint_step_deg:.1f} deg",
    ]


# ── TEACH mode ─────────────────────────────────────────────────────────

def _handle_teach_key(event, ctx: Ctx, state: AppState) -> None:
    if event.key in CAPTURE_KEYS:
        idx = CAPTURE_KEYS[event.key]
        x, y = core.fk_from_servo_angles(ctx.params, state.s1, state.s2)
        state.corners[idx] = (state.s1, state.s2, x, y)
        _set_status(state, f"corner {idx + 1} recorded: ({x:.1f}, {y:.1f}) mm", 2.0)
    elif event.key == pygame.K_c:
        state.corners = [None, None, None, None]
        _set_status(state, "cleared", 2.0)
    elif event.key in (pygame.K_s, pygame.K_RETURN):
        if any(c is None for c in state.corners):
            _set_status(state, "need all 4 corners before saving")
        else:
            corners_mm = [(c[2], c[3]) for c in state.corners]
            cx, cy, w, h, rot = pc.fit_rect_from_corners(corners_mm)
            ctx.calib["rectangle"] = {
                "corners_mm": [list(p) for p in corners_mm],
                "center_x_mm": cx, "center_y_mm": cy,
                "width_mm": w, "height_mm": h, "rotation_deg": rot,
            }
            _sync_calib_from_live(ctx)
            try:
                core.save_calib(ctx.calib)
                _set_status(state, f"saved: center=({cx:.1f},{cy:.1f}) {w:.1f}x{h:.1f}mm "
                                    f"rot={rot:.1f}deg", 5.0)
            except ValueError as e:
                _set_status(state, f"save failed: {e}", 5.0)


def _draw_teach_overlay(screen, layout: Layout, ctx: Ctx, state: AppState) -> None:
    for i, c in enumerate(state.corners):
        if c is None:
            continue
        pt = layout.ws2s(c[2], c[3])
        pygame.draw.circle(screen, CORNER_C, pt, 6)
    if all(c is not None for c in state.corners):
        corners_mm = [(c[2], c[3]) for c in state.corners]
        cx, cy, w, h, rot = pc.fit_rect_from_corners(corners_mm)
        rect_pts = [layout.ws2s(x, y) for x, y in pc.rect_corners(cx, cy, w, h, rot)]
        pygame.draw.polygon(screen, RECT_C, rect_pts, width=2)
        grid = pc.generate_photo_grid((cx, cy, w, h, rot), ctx.photo_cfg.spacing_x_mm,
                                       ctx.photo_cfg.spacing_y_mm, ctx.photo_cfg.margin_mm)
        for x, y, _label in grid:
            pygame.draw.circle(screen, PHOTO_C, layout.ws2s(x, y), 2)


def _draw_teach_panel(lines: list, state: AppState) -> None:
    lines += [
        "1/2/3/4  record corner (walk the perimeter in order)",
        "c        clear all corners",
        "s/Enter  save rectangle to calib.json",
        "",
        "corners:",
    ]
    for i, c in enumerate(state.corners):
        status = f"({c[2]:.1f}, {c[3]:.1f})" if c is not None else "-- not recorded --"
        lines.append(f"  {i + 1}: {status}")


# ── PATH mode ──────────────────────────────────────────────────────────

def _handle_path_key(event, ctx: Ctx, state: AppState) -> None:
    if event.key == pygame.K_b:
        if not state.recording:
            for j in ("joint1", "joint2"):
                if ctx.torque[j]:
                    _toggle_torque(ctx.servos, ctx.controller, ctx.torque, state, j)
            state.recorded_points = []
            state.recording = True
            _set_status(state, "recording...", 2.0)
        else:
            state.recording = False
            _set_status(state, f"stopped recording ({len(state.recorded_points)} points)")
    elif event.key == pygame.K_p:
        if not state.recorded_points:
            _set_status(state, "no recorded points to play back")
        elif ctx.controller.is_moving:
            _set_status(state, "already moving")
        else:
            for j in ("joint1", "joint2"):
                if not ctx.torque[j]:
                    _toggle_torque(ctx.servos, ctx.controller, ctx.torque, state, j)
            if len(state.recorded_points) >= 2:
                ctx.controller.start_scan(state.recorded_points)
                state.playback_active = True
                _set_status(state, "playing back...", 2.0)
            elif ctx.controller.set_workspace_goal(*state.recorded_points[0]):
                state.playback_active = True
                _set_status(state, "playing back...", 2.0)
            else:
                _set_status(state, "recorded point is unreachable -- nothing to play back")
    elif event.key == pygame.K_c:
        state.recorded_points = []
        _set_status(state, "cleared buffer", 2.0)
    elif event.key == pygame.K_s:
        if not state.recorded_points:
            _set_status(state, "nothing recorded to save")
        else:
            state.text_input = TextInput("path name")
            state.text_input_purpose = "save_path"
    elif event.key == pygame.K_UP:
        state.selected_path_idx = max(0, state.selected_path_idx - 1)
    elif event.key == pygame.K_DOWN:
        state.selected_path_idx = min(max(0, len(state.path_names) - 1), state.selected_path_idx + 1)
    elif event.key == pygame.K_l:
        if state.path_names:
            name = state.path_names[state.selected_path_idx]
            state.recorded_points = list(ctx.paths[name])
            _set_status(state, f"loaded '{name}' ({len(state.recorded_points)} pts)")
    elif event.key == pygame.K_DELETE:
        if state.path_names:
            name = state.path_names[state.selected_path_idx]
            del ctx.paths[name]
            pc.save_paths(ctx.paths)
            state.path_names = sorted(ctx.paths)
            state.selected_path_idx = min(state.selected_path_idx, max(0, len(state.path_names) - 1))
            _set_status(state, f"deleted '{name}'")
    elif event.key == pygame.K_r:
        if state.path_names:
            old_name = state.path_names[state.selected_path_idx]
            state.text_input = TextInput("rename to", initial=old_name)
            state.text_input_purpose = "rename_path"
            state.rename_old_name = old_name


def _draw_path_overlay(screen, layout: Layout, state: AppState) -> None:
    if len(state.recorded_points) > 1:
        pts = [layout.ws2s(x, y) for x, y in state.recorded_points]
        pygame.draw.lines(screen, PATH_C, False, pts, 2)


def _draw_path_panel(lines: list, state: AppState) -> None:
    lines += [
        "b        toggle recording (torque auto-released)",
        "p        play back current buffer",
        "c        clear current buffer",
        "s        save current buffer as a named path",
        "up/down  select a saved path   l  load selected",
        "r        rename selected      delete  delete selected",
        "",
        f"recording: {'YES' if state.recording else 'no'}   "
        f"buffer: {len(state.recorded_points)} points",
        "",
        "saved paths:",
    ]
    if not state.path_names:
        lines.append("  (none yet)")
    for i, name in enumerate(state.path_names):
        marker = ">" if i == state.selected_path_idx else " "
        lines.append(f" {marker}{name}")


# ── SCAN mode ──────────────────────────────────────────────────────────

THUMB_MAX_W = 150
THUMB_MAX_H = 110


def _load_thumbnail(path: Path):
    """Best-effort load + aspect-preserving downscale for the SCAN-mode
    "last capture" preview. Returns None (rather than raising) on any
    failure -- a preview thumbnail is a nice-to-have, not something that
    should ever interrupt an in-progress scan over a decode error."""
    try:
        img = pygame.image.load(str(path))
        w, h = img.get_size()
        scale = min(THUMB_MAX_W / w, THUMB_MAX_H / h, 1.0)
        return pygame.transform.smoothscale(img, (max(1, int(w * scale)), max(1, int(h * scale))))
    except Exception:  # noqa: BLE001 -- decode/format errors, never fatal here
        return None


def _make_capture_callback(camera: hw.Camera, out_dir: Path, state: "AppState"):
    def on_arrive(index: int, x_mm: float, y_mm: float, label: str) -> None:
        path = out_dir / f"{index:03d}_{label}.jpg"
        camera.capture_and_save(path)
        print(f"[v3 scan] captured {path.name} at ({x_mm:.1f}, {y_mm:.1f}) mm")
        state.last_photo_surface = _load_thumbnail(path)
        state.last_photo_label = f"{label} #{index}"
    return on_arrive


def _handle_scan_key(event, ctx: Ctx, state: AppState) -> None:
    if event.key == pygame.K_n:
        state.dry_run = not state.dry_run
        _set_status(state, f"dry run (no capture): {'ON' if state.dry_run else 'off'}", 2.0)
    elif event.key in (pygame.K_g, pygame.K_RETURN):
        rect = core.calib_rectangle(ctx.calib)
        if rect is None:
            _set_status(state, "no rectangle taught yet -- use TEACH mode first")
            return
        if ctx.controller.is_moving:
            _set_status(state, "already moving")
            return
        for j in ("joint1", "joint2"):
            if not ctx.torque[j]:
                _toggle_torque(ctx.servos, ctx.controller, ctx.torque, state, j)
        photo_points = pc.generate_photo_grid(
            rect, ctx.photo_cfg.spacing_x_mm, ctx.photo_cfg.spacing_y_mm, ctx.photo_cfg.margin_mm)
        if not state.dry_run and not ctx.camera_connected:
            try:
                ctx.camera.connect()
                ctx.camera_connected = True
            except Exception as e:  # noqa: BLE001 -- camera driver is Pi-only/deferred
                ctx.camera_error = str(e)
                _set_status(state, f"camera unavailable ({e}) -- press 'n' for a dry run", 5.0)
                return
        state.last_photo_surface = None
        state.last_photo_label = ""
        if state.dry_run:
            def on_arrive(index, x, y, label):
                print(f"[v3 scan][dry run] point {index} ({label}): x={x:.1f} y={y:.1f} mm")
            state.scan_out_dir = None
        else:
            stamp = time.strftime("%Y%m%dT%H%M%S")
            state.scan_out_dir = Path(ctx.photo_cfg.photo_dir) / stamp
            state.scan_out_dir.mkdir(parents=True, exist_ok=True)
            on_arrive = _make_capture_callback(ctx.camera, state.scan_out_dir, state)
        state.runner = pc.PhotoScanRunner(
            ctx.controller, photo_points, max_step_mm=ctx.photo_cfg.max_step_mm,
            dwell_s=ctx.photo_cfg.dwell_s, on_arrive=on_arrive)
        _set_status(state, f"scanning {len(photo_points)} points"
                            + (f" -> {state.scan_out_dir}" if state.scan_out_dir else " (dry run)"), 3.0)


def _draw_scan_overlay(screen, layout: Layout, ctx: Ctx, state: AppState) -> None:
    rect = core.calib_rectangle(ctx.calib)
    if rect is None:
        return
    cx, cy, w, h, rot = rect
    rect_pts = [layout.ws2s(x, y) for x, y in pc.rect_corners(cx, cy, w, h, rot)]
    pygame.draw.polygon(screen, RECT_C, rect_pts, width=2)

    if state.runner is not None:
        photo_points = state.runner.photo_points
        visited = state.runner.index
    else:
        photo_points = pc.generate_photo_grid(rect, ctx.photo_cfg.spacing_x_mm,
                                               ctx.photo_cfg.spacing_y_mm, ctx.photo_cfg.margin_mm)
        visited = -1
    for i, (x, y, _label) in enumerate(photo_points):
        color = PHOTO_DONE_C if i < visited else PHOTO_C
        pygame.draw.circle(screen, color, layout.ws2s(x, y), 3)


def _draw_scan_panel(lines: list, ctx: Ctx, state: AppState) -> None:
    rect = core.calib_rectangle(ctx.calib)
    lines.append("g/Enter  start scan   n  toggle dry-run (no capture)")
    if rect is None:
        lines.append("")
        lines.append("no rectangle taught yet -- switch to TEACH mode")
        return
    cx, cy, w, h, rot = rect
    photo_points = (state.runner.photo_points if state.runner is not None else
                    pc.generate_photo_grid(rect, ctx.photo_cfg.spacing_x_mm,
                                            ctx.photo_cfg.spacing_y_mm, ctx.photo_cfg.margin_mm))
    lines += [
        "",
        f"rectangle: {w:.1f} x {h:.1f} mm, rot={rot:.1f}deg",
        f"photo points: {len(photo_points)} (spacing "
        f"{ctx.photo_cfg.spacing_x_mm:.1f}x{ctx.photo_cfg.spacing_y_mm:.1f}mm)",
        f"dry run: {'ON' if state.dry_run else 'off'}",
    ]
    if ctx.camera_connected:
        lines.append("camera: connected")
    elif ctx.camera_error is not None:
        lines.append(f"camera: unavailable ({ctx.camera_error})")
    else:
        lines.append("camera: not connected yet (connects on first real scan)")
    if state.runner is not None:
        progress = f"{min(state.runner.index + 1, len(photo_points))}/{len(photo_points)}"
        lines.append(f"progress: {progress}" + ("  DONE" if state.runner.done else ""))


# ── PARAMS mode (full-window) ─────────────────────────────────────────

def _handle_params_key(event, ctx: Ctx, state: AppState) -> None:
    if event.key == pygame.K_UP:
        state.field_idx = max(0, state.field_idx - 1)
    elif event.key == pygame.K_DOWN:
        state.field_idx = min(len(ctx.fields) - 1, state.field_idx + 1)
    elif event.key == pygame.K_LEFT:
        _adjust_field(ctx.fields[state.field_idx], -1)
        ctx.controller.joint_limits = core.calib_joint_limits(ctx.calib)
    elif event.key == pygame.K_RIGHT:
        _adjust_field(ctx.fields[state.field_idx], +1)
        ctx.controller.joint_limits = core.calib_joint_limits(ctx.calib)
    elif event.key == pygame.K_RETURN:
        f = ctx.fields[state.field_idx]
        if f.kind == "sign":
            f.set(-f.get())
        else:
            state.text_input = TextInput(f.label, initial=_format_field(f))
            state.text_input_purpose = "edit_field"
            state.text_input_field = f
    elif event.key == pygame.K_i and ctx.calib.get("joint_limits_deg") is None:
        ctx.calib["joint_limits_deg"] = {"joint1": [0.0, 360.0], "joint2": [0.0, 360.0],
                                          "coupled_boundary": []}
        ctx.controller.joint_limits = core.calib_joint_limits(ctx.calib)
        ctx.fields = build_fields(ctx.calib, ctx.params, ctx.motion_cfg, ctx.photo_cfg, ctx.hw_cfg)
        _set_status(state, "joint_limits_deg initialized to 0-360 (unrestricted)")
    elif event.key == pygame.K_s:
        _sync_calib_from_live(ctx)
        try:
            core.save_calib(ctx.calib)
            _set_status(state, "saved calib.json", 3.0)
        except ValueError as e:
            _set_status(state, f"save failed: {e}", 5.0)
    elif event.key == pygame.K_r:
        if ctx.controller.is_moving:
            # ctx.params/motion_cfg/photo_cfg are mutated IN PLACE (the
            # same objects ctx.controller holds references to, not
            # replaced) -- reloading mid-scan would silently swap the
            # kinematics used to interpret the REMAINING waypoints out
            # from under an already-in-progress PhotoScanRunner, which
            # planned earlier segments (and the whole photo grid) against
            # the OLD params. A single incremental field nudge is small
            # enough to tolerate live; a bulk reload isn't.
            _set_status(state, "arm is moving -- press Space to stop before reloading")
            return
        fresh = core.load_calib()
        ctx.calib.clear()
        ctx.calib.update(fresh)
        _copy_dataclass_fields(ctx.params, core.calib_arm_params(ctx.calib))
        _copy_dataclass_fields(ctx.hw_cfg, core.calib_hardware_config(ctx.calib))
        _copy_dataclass_fields(ctx.motion_cfg, core.calib_motion_config(ctx.calib))
        _copy_dataclass_fields(ctx.photo_cfg, core.calib_photo_config(ctx.calib))
        ctx.controller.joint_limits = core.calib_joint_limits(ctx.calib)
        ctx.fields = build_fields(ctx.calib, ctx.params, ctx.motion_cfg, ctx.photo_cfg, ctx.hw_cfg)
        state.field_idx = min(state.field_idx, len(ctx.fields) - 1)
        _set_status(state, "reloaded calib.json from disk (live edits discarded)", 3.0)


def _draw_params_fullscreen(screen, layout: Layout, ctx: Ctx, state: AppState, font, sfont) -> None:
    screen.fill(BG)
    header = ("PARAMS -- up/down select, left/right adjust, Enter=type exact value / "
              "flip sign, s=save, r=reload" +
              ("" if ctx.calib.get("joint_limits_deg") is not None else ", i=init joint limits"))
    screen.blit(sfont.render(header, True, LABEL_C), (10, 8))
    screen.blit(sfont.render("* hardware section only takes effect after restart", True, LABEL_C),
                (10, 26))

    row_h = 20
    top = 52
    visible_rows = max(1, (layout.win_h - top - 10) // row_h)
    n = len(ctx.fields)
    start = max(0, min(state.field_idx - visible_rows // 2, max(0, n - visible_rows)))
    end = min(n, start + visible_rows)

    if start > 0:
        screen.blit(sfont.render(f"^ {start} more above", True, LABEL_C), (10, top - 2))
    for row, i in enumerate(range(start, end)):
        f = ctx.fields[i]
        selected = i == state.field_idx
        text = f"{'>' if selected else ' '} [{f.section}] {f.label}: {_format_field(f)}"
        color = SEL_C if selected else TEXT_C
        screen.blit(font.render(text, True, color), (10, top + row * row_h))
    if end < n:
        screen.blit(sfont.render(f"v {n - end} more below", True, LABEL_C),
                    (10, top + visible_rows * row_h))


# ── Main loop ──────────────────────────────────────────────────────────

def _apply_text_input(ctx: Ctx, state: AppState) -> None:
    ti = state.text_input
    if ti.result is not None:
        if state.text_input_purpose == "edit_field":
            f = state.text_input_field
            try:
                if f.kind == "int":
                    f.set(int(ti.result))
                elif f.kind == "text":
                    f.set(ti.result)
                else:
                    f.set(float(ti.result))
                ctx.controller.joint_limits = core.calib_joint_limits(ctx.calib)
            except ValueError:
                _set_status(state, f"invalid value: {ti.result!r}")
        elif state.text_input_purpose == "save_path":
            name = ti.result.strip()
            if name:
                ctx.paths[name] = list(state.recorded_points)
                pc.save_paths(ctx.paths)
                state.path_names = sorted(ctx.paths)
                _set_status(state, f"saved path '{name}' ({len(state.recorded_points)} pts)")
        elif state.text_input_purpose == "rename_path":
            new_name = ti.result.strip()
            old_name = state.rename_old_name
            if new_name and old_name in ctx.paths and new_name != old_name:
                ctx.paths[new_name] = ctx.paths.pop(old_name)
                pc.save_paths(ctx.paths)
                state.path_names = sorted(ctx.paths)
                state.selected_path_idx = state.path_names.index(new_name)
                _set_status(state, f"renamed '{old_name}' -> '{new_name}'")
    state.text_input = None
    state.text_input_purpose = None
    state.text_input_field = None
    state.rename_old_name = None


JOG_REPEATABLE_KEYS = (pygame.K_UP, pygame.K_DOWN, pygame.K_LEFT, pygame.K_RIGHT,
                       pygame.K_i, pygame.K_k, pygame.K_o, pygame.K_l)
PARAMS_REPEATABLE_KEYS = (pygame.K_UP, pygame.K_DOWN, pygame.K_LEFT, pygame.K_RIGHT)
REPEAT_DELAY_S = 0.32
REPEAT_INTERVAL_S = 0.05


def _poll_held_repeat(ctx: Ctx, state: AppState, now: float) -> None:
    """Continuous-hold "repeat" for navigation/jog keys ONLY, driven by
    polling pygame.key.get_pressed() once per frame instead of pygame's
    own global key.set_repeat(). set_repeat applies to every key
    uniformly, which is the wrong tool here: holding a one-shot or
    destructive key a beat too long (e.g. 'b' to toggle PATH recording,
    Delete to remove a saved path) would silently re-fire it several times
    and land in a different end state than a single tap intended.
    Restricting repeat to this explicit allowlist keeps every other key
    strictly one-fire-per-physical-press while still letting jogging and
    PARAMS-mode navigation feel smooth when held down."""
    if state.text_input is not None:
        _poll_text_input_backspace(state, now)
        return
    if state.mode == MODE_JOG:
        candidates = JOG_REPEATABLE_KEYS
    elif state.mode == MODE_PARAMS:
        candidates = PARAMS_REPEATABLE_KEYS
    else:
        state.repeat_key = None
        return
    pressed = pygame.key.get_pressed()
    held = next((k for k in candidates if pressed[k]), None)
    if held is None:
        state.repeat_key = None
        return
    if held != state.repeat_key:
        # Just started being held -- the original KEYDOWN already fired
        # this once; only schedule the FIRST repeat from here.
        state.repeat_key = held
        state.repeat_next_at = now + REPEAT_DELAY_S
        return
    if now < state.repeat_next_at:
        return
    state.repeat_next_at = now + REPEAT_INTERVAL_S
    fake_event = SimpleNamespace(key=held, mod=0, unicode="")
    if state.mode == MODE_JOG:
        _handle_jog_key(fake_event, ctx, state)
    else:
        _handle_params_key(fake_event, ctx, state)


def _poll_text_input_backspace(state: AppState, now: float) -> None:
    pressed = pygame.key.get_pressed()
    if not pressed[pygame.K_BACKSPACE]:
        state.repeat_key = None
        return
    if state.repeat_key != pygame.K_BACKSPACE:
        state.repeat_key = pygame.K_BACKSPACE
        state.repeat_next_at = now + REPEAT_DELAY_S
        return
    if now < state.repeat_next_at:
        return
    state.repeat_next_at = now + REPEAT_INTERVAL_S
    state.text_input.buffer = state.text_input.buffer[:-1]


def main():
    calib = core.load_calib()
    params = core.calib_arm_params(calib)
    hw_cfg = core.calib_hardware_config(calib)
    motion_cfg = core.calib_motion_config(calib)
    photo_cfg = core.calib_photo_config(calib)
    joint_limits = core.calib_joint_limits(calib)

    servos = hw.Servos(hw_cfg.joint_ids)
    servos.connect(hw_cfg.servo_port)
    camera = hw.Camera()
    planner = mp.get_planner(motion_cfg.planner_name)
    controller = jc.ArmController(servos, params, planner, motion_cfg, joint_limits=joint_limits)

    torque = {"joint1": True, "joint2": True}
    paths = pc.load_paths()

    ctx = Ctx(calib=calib, params=params, motion_cfg=motion_cfg, photo_cfg=photo_cfg,
              hw_cfg=hw_cfg, servos=servos, camera=camera, controller=controller,
              torque=torque, paths=paths)
    ctx.fields = build_fields(calib, params, motion_cfg, photo_cfg, hw_cfg)

    state = AppState()
    state.path_names = sorted(paths)
    state.s1 = servos.get_present_deg("joint1")
    state.s2 = servos.get_present_deg("joint2")

    layout = Layout(base_x=params.base_x, base_y=params.base_y, max_r_mm=params.L1 + params.L2)

    pygame.init()
    # Deliberately NOT using pygame.key.set_repeat(): it applies to every
    # key uniformly, which would make one-shot/destructive keys (b, s,
    # Delete, g, 1-4, ...) re-fire if held a beat too long. Continuous
    # repeat for jog/navigation keys only is handled by _poll_held_repeat()
    # every frame instead -- see its docstring.
    pygame.display.set_caption("2R Arm v3 -- control panel")
    screen = pygame.display.set_mode((layout.win_w, layout.win_h))
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("menlo,consolas,monospace", 17)
    sfont = pygame.font.SysFont("menlo,consolas,monospace", 13)

    running = True
    try:
        while running:
            now = time.monotonic()

            for event in pygame.event.get():
                if state.text_input is not None:
                    if event.type == pygame.QUIT:
                        running = False
                    elif event.type == pygame.KEYDOWN:
                        state.text_input.handle_key(event)
                        if not state.text_input.active:
                            _apply_text_input(ctx, state)
                    continue
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_q, pygame.K_ESCAPE):
                        running = False
                    elif event.key == pygame.K_TAB:
                        direction = -1 if (event.mod & pygame.KMOD_SHIFT) else 1
                        _switch_mode(ctx, state, (state.mode + direction) % len(MODE_NAMES))
                    elif event.key in FUNCTION_KEY_MODES:
                        _switch_mode(ctx, state, FUNCTION_KEY_MODES[event.key])
                    elif event.key == pygame.K_SPACE and controller.is_moving:
                        controller.stop_scan()
                        state.runner = None
                        state.playback_active = False
                        _set_status(state, "motion aborted")
                    elif event.key in (pygame.K_z, pygame.K_x) and not controller.is_moving:
                        joint = "joint1" if event.key == pygame.K_z else "joint2"
                        _toggle_torque(servos, controller, torque, state, joint)
                    elif state.mode == MODE_JOG:
                        _handle_jog_key(event, ctx, state)
                    elif state.mode == MODE_TEACH:
                        _handle_teach_key(event, ctx, state)
                    elif state.mode == MODE_PATH:
                        _handle_path_key(event, ctx, state)
                    elif state.mode == MODE_SCAN:
                        _handle_scan_key(event, ctx, state)
                    elif state.mode == MODE_PARAMS:
                        _handle_params_key(event, ctx, state)

            _poll_held_repeat(ctx, state, now)

            if now - state.last_poll >= POLL_INTERVAL_S:
                state.last_poll = now
                state.s1 = servos.get_present_deg("joint1")
                state.s2 = servos.get_present_deg("joint2")

            controller.tick()

            if state.recording:
                pt = core.fk_from_servo_angles(params, state.s1, state.s2)
                if not state.recorded_points or math.dist(pt, state.recorded_points[-1]) >= 1.0:
                    state.recorded_points.append(pt)

            if state.runner is not None:
                state.runner.tick(now)

            if state.playback_active and not controller.is_moving:
                state.playback_active = False
                _set_status(state, "playback finished")

            if state.mode == MODE_PARAMS:
                _draw_params_fullscreen(screen, layout, ctx, state, font, sfont)
            else:
                screen.fill(BG)
                pygame.draw.circle(screen, GRID, layout.ws2s(params.base_x, params.base_y),
                                    int((params.L1 + params.L2) * layout.scale), width=1)
                pygame.draw.circle(screen, BASE_C, layout.ws2s(params.base_x, params.base_y), 6)

                if state.mode == MODE_TEACH:
                    _draw_teach_overlay(screen, layout, ctx, state)
                elif state.mode == MODE_PATH:
                    _draw_path_overlay(screen, layout, state)
                elif state.mode == MODE_SCAN:
                    _draw_scan_overlay(screen, layout, ctx, state)

                elbow, ee = core.fk_joint_positions(params, state.s1, state.s2)
                base_pt = layout.ws2s(params.base_x, params.base_y)
                elbow_pt = layout.ws2s(*elbow)
                ee_pt = layout.ws2s(*ee)
                pygame.draw.line(screen, LINK1_C, base_pt, elbow_pt, 4)
                pygame.draw.line(screen, LINK2_C, elbow_pt, ee_pt, 4)
                pygame.draw.circle(screen, JOINT_C, elbow_pt, 5)
                pygame.draw.circle(screen, EE_C, ee_pt, 7)

                panel_x = layout.win_w - layout.panel_w + 10
                t1_c = LOCKED_C if torque["joint1"] else FREE_C
                t2_c = LOCKED_C if torque["joint2"] else FREE_C
                mode_line = f"[{MODE_NAMES[state.mode]}]  (Tab/F1-F5 to switch)"
                screen.blit(font.render(mode_line, True, TEXT_C), (panel_x, 12))
                y = 40
                screen.blit(font.render("z  joint1:", True, TEXT_C), (panel_x, y))
                screen.blit(font.render("LOCKED" if torque["joint1"] else "FREE", True, t1_c),
                            (panel_x + 110, y))
                y += 22
                screen.blit(font.render("x  joint2:", True, TEXT_C), (panel_x, y))
                screen.blit(font.render("LOCKED" if torque["joint2"] else "FREE", True, t2_c),
                            (panel_x + 110, y))
                y += 30
                screen.blit(font.render(f"joint1={state.s1:.1f}deg joint2={state.s2:.1f}deg",
                                         True, TEXT_C), (panel_x, y))
                y += 20
                screen.blit(font.render(f"end effector: ({ee[0]:.1f}, {ee[1]:.1f}) mm",
                                         True, TEXT_C), (panel_x, y))
                y += 30

                lines: list = []
                if state.mode == MODE_JOG:
                    _draw_jog_panel(lines, ctx, state)
                elif state.mode == MODE_TEACH:
                    _draw_teach_panel(lines, state)
                elif state.mode == MODE_PATH:
                    _draw_path_panel(lines, state)
                elif state.mode == MODE_SCAN:
                    _draw_scan_panel(lines, ctx, state)
                for line in lines:
                    screen.blit(sfont.render(line, True, TEXT_C), (panel_x, y))
                    y += 19

                if state.mode == MODE_SCAN and state.last_photo_surface is not None:
                    y += 8
                    screen.blit(sfont.render(f"last capture: {state.last_photo_label}", True, LABEL_C),
                                (panel_x, y))
                    y += 16
                    screen.blit(state.last_photo_surface, (panel_x, y))
                    y += state.last_photo_surface.get_height() + 8

                if state.status_msg and now < state.status_until:
                    color = WARN_C if any(w in state.status_msg for w in
                                           ("failed", "need", "unreachable", "off", "unavailable")) else OK_C
                    screen.blit(sfont.render(state.status_msg, True, color), (panel_x, y + 6))

                # Recent activity log, newest first, anchored to the
                # bottom of the panel so it doesn't jump around as the
                # mode-specific content above grows/shrinks.
                log_lines = list(reversed(state.log[:-1]))[:4]
                log_y = layout.win_h - 20 * len(log_lines) - 12
                for i, msg in enumerate(log_lines):
                    screen.blit(sfont.render(msg, True, LABEL_C), (panel_x, log_y + i * 18))

            if state.text_input is not None:
                # Drawn last, on top of whichever mode just rendered (including
                # PARAMS, which fills the whole window) -- otherwise a
                # PARAMS-mode Enter-to-edit prompt would be invisible.
                box = pygame.Rect(layout.margin_px, layout.win_h // 2 - 20,
                                   layout.win_w - 2 * layout.margin_px, 40)
                pygame.draw.rect(screen, (50, 54, 70), box)
                pygame.draw.rect(screen, SEL_C, box, width=1)
                prompt_surf = font.render(
                    f"{state.text_input.prompt}: {state.text_input.buffer}_", True, TEXT_C)
                screen.blit(prompt_surf, (box.x + 8, box.y + 10))

            pygame.display.flip()
            clock.tick(FPS)
    finally:
        _resync_and_relock(servos)
        servos.close()
        if ctx.camera_connected:
            camera.close()
        pygame.quit()


if __name__ == "__main__":
    main()
