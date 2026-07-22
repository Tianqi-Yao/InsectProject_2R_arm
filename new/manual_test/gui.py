"""Live pygame visualizer + jog control.

Draws two overlaid arms every frame:
  - solid  = computed from the *real* servo encoder feedback, polled
             periodically -- this is what the code currently believes the
             physical arm looks like right now.
  - ghost  = the motion controller's current *commanded* position -- the
             planned trajectory's instantaneous point, not just "the final
             target snapped there."

Put this window next to the real arm and compare by eye: if the solid
arm's pose doesn't match what you're looking at, that's L1/L2/base/offset/
direction being wrong, not a bug in this viewer.

All physical parameters, hardware settings, and motion tuning come from
calib.json -- see manual_test/run.py's module docstring for why (this
used to hardcode an independent copy of every constant run.py also
hardcoded, and the two copies had already drifted apart -- see git
history's SCAN_NX/SCAN_SPEED/etc. mismatches).

This is a thin adapter over jog_controller.ArmController, same as run.py:
translates pygame events into controller calls and renders controller
state. All the actual motion planning/smoothing/scanning logic lives in
jog_controller.py + motion_planning/, shared with run.py and main.py's
jog REPL.
"""

import sys
import time
from dataclasses import dataclass
from pathlib import Path

import pygame

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import arm_core as core          # noqa: E402
import arm_hardware as hw         # noqa: E402
import jog_controller as jc       # noqa: E402

STEP_MIN, STEP_MAX = 0.5, 40.0
POLL_INTERVAL_S = 0.08  # how often to re-read real encoder positions
FPS = 60

BG = (28, 30, 40)
WS_FILL = (38, 46, 62)
WS_BORDER = (70, 110, 190)
GRID = (48, 54, 72)
LINK1_C = (80, 160, 255)
LINK2_C = (255, 120, 55)
JOINT_C = (220, 220, 220)
EE_OK = (55, 215, 95)
BASE_C = (200, 100, 55)
GHOST_LINK1 = (80, 160, 255, 100)
GHOST_LINK2 = (255, 120, 55, 100)
GHOST_JOINT = (220, 220, 220, 100)
TARGET_C = (255, 230, 80)
SCAN_AREA_C = (255, 210, 60)
TEXT_C = (200, 210, 225)
LABEL_C = (110, 130, 155)
HIGHLIGHT = (255, 220, 80)
EE_ERR = (220, 55, 55)


@dataclass
class Layout:
    """Screen-space layout for a given workspace-mm bounding box -- pulled
    out of what used to be module-level SCALE/WS_OX/WS_OY/WIN_W/WIN_H
    constants sized for a hardcoded 200x150mm workspace, so the window
    sizes itself to whatever calib.json's workspace actually is.

    `origin_x_mm`/`origin_y_mm` is the workspace point mapped to the
    canvas's bottom-left -- 0,0 reproduces the old assumption that
    everything of interest starts at the calibration sheet's own corner,
    but that's no longer always true (see `fit()`): the jog/scan area can
    now be a rotated rectangle sitting partly or wholly outside the
    sheet, and the window needs to be big enough to show all of it, not
    just clip it at the sheet's own edge."""
    origin_x_mm: float
    origin_y_mm: float
    span_width_mm: float
    span_height_mm: float
    scale: float = 2.0
    ws_ox: int = 55
    ws_oy: int = 30

    @property
    def win_w(self) -> int:
        return int(self.ws_ox + self.span_width_mm * self.scale) + 260

    @property
    def win_h(self) -> int:
        return int(self.ws_oy + self.span_height_mm * self.scale) + 40

    def ws2s(self, wx: float, wy: float) -> tuple:
        """Workspace mm -> screen px (Y flipped: workspace +Y = screen up)."""
        return (int(self.ws_ox + (wx - self.origin_x_mm) * self.scale),
                int(self.ws_oy + (self.origin_y_mm + self.span_height_mm - wy) * self.scale))

    @classmethod
    def fit(cls, sheet_width_mm: float, sheet_height_mm: float, scan_area, margin_mm: float = 15.0,
            **kwargs) -> "Layout":
        """Sized and positioned to cover BOTH the calibration sheet
        ((0,0)-(sheet_width_mm,sheet_height_mm), fixed by the physical
        AprilTag placement) and the jog/scan area (arm_core.calib_scan_area
        -- independently positioned/sized/rotated, can extend outside the
        sheet), with a little margin so edges/handles aren't drawn flush
        against the window border. If the scan area has never been
        configured (still equals the full sheet), this is identical to
        the old sheet-only sizing."""
        cx, cy, w, h, rotation_deg = scan_area
        all_pts = (core.scan_area_corners(cx, cy, w, h, rotation_deg)
                   + [(0.0, 0.0), (sheet_width_mm, 0.0),
                      (sheet_width_mm, sheet_height_mm), (0.0, sheet_height_mm)])
        min_x = min(p[0] for p in all_pts) - margin_mm
        max_x = max(p[0] for p in all_pts) + margin_mm
        min_y = min(p[1] for p in all_pts) - margin_mm
        max_y = max(p[1] for p in all_pts) + margin_mm
        return cls(origin_x_mm=min_x, origin_y_mm=min_y,
                    span_width_mm=max_x - min_x, span_height_mm=max_y - min_y, **kwargs)


def draw_workspace(surf, layout: Layout, sheet_width_mm: float, sheet_height_mm: float):
    """Draws the AprilTag calibration sheet's own rectangle -- its true
    (0,0)-(sheet_width_mm,sheet_height_mm) extent, which may now be only
    part of a larger canvas (see Layout.fit)."""
    top_left = layout.ws2s(0.0, sheet_height_mm)
    w, h = int(sheet_width_mm * layout.scale), int(sheet_height_mm * layout.scale)
    pygame.draw.rect(surf, WS_FILL, (top_left[0], top_left[1], w, h))
    for gx in range(0, int(sheet_width_mm) + 1, 25):
        pygame.draw.line(surf, GRID, layout.ws2s(gx, 0), layout.ws2s(gx, sheet_height_mm))
    for gy in range(0, int(sheet_height_mm) + 1, 25):
        pygame.draw.line(surf, GRID, layout.ws2s(0, gy), layout.ws2s(sheet_width_mm, gy))
    pygame.draw.rect(surf, WS_BORDER, (top_left[0], top_left[1], w, h), 2)


def draw_scan_area(surf, layout: Layout, scan_area):
    """Highlights the jog/scan sub-rectangle -- see arm_core.calib_scan_area().
    When it hasn't been configured (the sub-rectangle equals the full
    sheet, unrotated), this just retraces the sheet's own border and is
    easy to ignore. Can be tilted (scan_rotation_deg != 0) and/or extend
    outside the sheet -- Layout.fit already sized the window to show all
    of it -- so this draws the actual rotated quadrilateral, not an
    axis-aligned pygame.draw.rect."""
    cx, cy, w, h, rotation_deg = scan_area
    points = [layout.ws2s(*c) for c in core.scan_area_corners(cx, cy, w, h, rotation_deg)]
    pygame.draw.polygon(surf, SCAN_AREA_C, points, 2)


def draw_solid_arm(surf, layout: Layout, base_xy, elbow, ee):
    p_base, p_elbow, p_ee = layout.ws2s(*base_xy), layout.ws2s(*elbow), layout.ws2s(*ee)
    pygame.draw.line(surf, LINK1_C, p_base, p_elbow, 6)
    pygame.draw.line(surf, LINK2_C, p_elbow, p_ee, 5)
    pygame.draw.circle(surf, JOINT_C, p_base, 9)
    pygame.draw.circle(surf, BASE_C, p_base, 9, 2)
    pygame.draw.circle(surf, JOINT_C, p_elbow, 7)
    pygame.draw.circle(surf, EE_OK, p_ee, 8)
    pygame.draw.circle(surf, (255, 255, 255), p_ee, 8, 2)


def draw_ghost_arm(surf, layout: Layout, base_xy, elbow, ee):
    overlay = pygame.Surface((layout.win_w, layout.win_h), pygame.SRCALPHA)
    p_base, p_elbow, p_ee = layout.ws2s(*base_xy), layout.ws2s(*elbow), layout.ws2s(*ee)
    pygame.draw.line(overlay, GHOST_LINK1, p_base, p_elbow, 4)
    pygame.draw.line(overlay, GHOST_LINK2, p_elbow, p_ee, 3)
    pygame.draw.circle(overlay, GHOST_JOINT, p_elbow, 5)
    pygame.draw.circle(overlay, (*TARGET_C, 160), p_ee, 6, 2)
    surf.blit(overlay, (0, 0))


def draw_panel(surf, layout: Layout, font, sfont, info):
    px = int(layout.ws_ox + layout.span_width_mm * layout.scale) + 24
    py = 28

    def row(text, dy, color=TEXT_C, f=None):
        nonlocal py
        surf.blit((f or font).render(text, True, color), (px, py))
        py += dy

    row("REAL vs TARGET", 22, HIGHLIGHT)
    row("solid = real (encoders)", 15, LABEL_C, sfont)
    row("ghost = commanded/planned", 22, LABEL_C, sfont)

    row("Real", 13, LABEL_C, sfont)
    row(f"s1={info['real_s1']:6.1f}  s2={info['real_s2']:6.1f}", 16)
    row(f"x={info['real_x']:6.1f}  y={info['real_y']:6.1f} mm", 22)

    row("Target", 13, LABEL_C, sfont)
    if info["target_reachable"]:
        row(f"s1={info['target_s1']:6.1f}  s2={info['target_s2']:6.1f}", 16)
        row(f"x={info['target_x']:6.1f}  y={info['target_y']:6.1f} mm", 22)
    else:
        row("NOT REACHABLE", 16, EE_ERR)
        row(" ", 22)

    row("Step", 13, LABEL_C, sfont)
    row(f"{info['step']:.1f} mm", 22)

    if info["scan_active"]:
        row("Positioning test", 13, LABEL_C, sfont)
        done, total = info["scan_progress"]
        row(f"point {done}/{total}", 22, HIGHLIGHT)
    else:
        row(" ", 13)
        row(" ", 22)

    row("-- Keys --", 13, LABEL_C, sfont)
    for line in ["arrows   move target", "[ ]      step size",
                 "h        home", "t        positioning test",
                 "q / ESC  quit"]:
        row(line, 16, LABEL_C, sfont)


def main():
    calib = core.load_calib()
    hw_cfg = core.calib_hardware_config(calib)
    servos = hw.Servos(hw_cfg.joint_ids)
    servos.connect(hw_cfg.servo_port)

    controller = jc.build_controller(servos, calib)
    params = controller.params
    ws = calib["workspace"]
    # (center_x, center_y, width, height, rotation_deg) -- see manual_test/scan_area_gui.py
    scan_area = core.calib_scan_area(calib)
    scan_cx, scan_cy, scan_w, scan_h, scan_rot = scan_area
    # Sized to show BOTH the calibration sheet and the scan area, even the
    # part of it (if any) sticking outside the sheet -- see Layout.fit's
    # docstring for why this can no longer just be the sheet's own size.
    layout = Layout.fit(ws["width_mm"], ws["height_mm"], scan_area)
    home = (scan_cx, scan_cy)
    scan_path = core.generate_scan_path(
        width_mm=scan_w, height_mm=scan_h,
        nx=controller.motion_cfg.scan_nx, ny=controller.motion_cfg.scan_ny,
        margin_mm=controller.motion_cfg.scan_margin_mm,
        rows_limit=controller.motion_cfg.scan_rows_limit,
        center_x_mm=scan_cx, center_y_mm=scan_cy, rotation_deg=scan_rot)

    pygame.init()
    pygame.display.set_caption("2R Arm -- real vs target")
    screen = pygame.display.set_mode((layout.win_w, layout.win_h))
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("menlo,consolas,monospace", 18)
    sfont = pygame.font.SysFont("menlo,consolas,monospace", 13)

    real_s1 = servos.get_present_deg("joint1")
    real_s2 = servos.get_present_deg("joint2")
    real_elbow, real_ee = core.fk_joint_positions(params, real_s1, real_s2)

    # ArmController only tracks "the joint goal"; the workspace-space
    # target is a frontend concept the panel displays and arrow keys
    # nudge -- seeded from the arm's real current position so the first
    # nudge is relative to wherever it actually is, not a jump.
    workspace_target = core.fk_from_servo_angles(params, *controller.commanded_deg)
    step = 5.0
    last_poll = time.monotonic()

    running = True
    try:
        while running:
            now = time.monotonic()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_q, pygame.K_ESCAPE):
                        if controller.scan_active:
                            controller.stop_scan()
                        else:
                            running = False
                    elif event.key == pygame.K_h:
                        # 'h' always wins, even mid-scan: interrupts the
                        # positioning test and heads straight home.
                        controller.stop_scan()
                        if controller.set_workspace_goal(*home):
                            workspace_target = home
                    elif controller.scan_active:
                        pass  # ignore other keys while a scan is running
                    elif event.key == pygame.K_LEFTBRACKET:
                        step = max(step / 1.5, STEP_MIN)
                    elif event.key == pygame.K_RIGHTBRACKET:
                        step = min(step * 1.5, STEP_MAX)
                    elif event.key == pygame.K_t:
                        controller.start_scan(scan_path)
                    elif event.key == pygame.K_UP:
                        new_t = controller.nudge_workspace(0.0, step, workspace_target)
                        workspace_target = new_t if new_t else workspace_target
                    elif event.key == pygame.K_DOWN:
                        new_t = controller.nudge_workspace(0.0, -step, workspace_target)
                        workspace_target = new_t if new_t else workspace_target
                    elif event.key == pygame.K_LEFT:
                        new_t = controller.nudge_workspace(-step, 0.0, workspace_target)
                        workspace_target = new_t if new_t else workspace_target
                    elif event.key == pygame.K_RIGHT:
                        new_t = controller.nudge_workspace(step, 0.0, workspace_target)
                        workspace_target = new_t if new_t else workspace_target

            if now - last_poll >= POLL_INTERVAL_S:
                real_s1 = servos.get_present_deg("joint1")
                real_s2 = servos.get_present_deg("joint2")
                real_elbow, real_ee = core.fk_joint_positions(params, real_s1, real_s2)
                last_poll = now

            controller.tick()
            cmd_s1, cmd_s2 = controller.commanded_deg
            cmd_elbow, cmd_ee = core.fk_joint_positions(params, cmd_s1, cmd_s2)
            target_result = core.ik_solve(params, *workspace_target)

            screen.fill(BG)
            draw_workspace(screen, layout, ws["width_mm"], ws["height_mm"])
            draw_scan_area(screen, layout, scan_area)
            draw_ghost_arm(screen, layout, (params.base_x, params.base_y), cmd_elbow, cmd_ee)
            draw_solid_arm(screen, layout, (params.base_x, params.base_y), real_elbow, real_ee)
            draw_panel(screen, layout, font, sfont, {
                "real_s1": real_s1, "real_s2": real_s2,
                "real_x": real_ee[0], "real_y": real_ee[1],
                "target_reachable": target_result.reachable,
                "target_s1": target_result.servo1_deg,
                "target_s2": target_result.servo2_deg,
                "target_x": workspace_target[0], "target_y": workspace_target[1],
                "step": step,
                "scan_active": controller.scan_active,
                "scan_progress": controller.scan_progress,
            })
            pygame.display.flip()
            clock.tick(FPS)
    finally:
        pygame.quit()
        servos.close()


if __name__ == "__main__":
    main()
