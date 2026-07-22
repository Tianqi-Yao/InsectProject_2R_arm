"""Live pygame tool for fitting the jog/scan sub-rectangle (see
arm_core.calib_scan_area) to wherever the arm can actually safely reach --
by dragging it (position, size, AND rotation) with the mouse over a live
map of the reachable+safe area, rather than hand-editing numbers. A
tilted rectangle can cover more of an irregularly-shaped reachable area
than an axis-aligned one, so rotation is a first-class part of what gets
saved, not just position/size.

Background: manual_test/gui.py and run.py's jog/scan rectangle used to be
exactly the AprilTag calibration sheet's own size (workspace.width_mm/
height_mm) -- but that sheet's placement is a physical fact (wherever the
four corner tags happen to be stuck down), with no guarantee it matches
the arm's actual reachable+safe region (joint_limits_deg). This tool lets
the jog/scan area be a smaller, independently-positioned (and now
independently-rotated) sub-rectangle within that same coordinate frame,
fitted to what's really usable, without touching the calibration sheet's
own numbers (doing that without moving the physical tags would desync
the homography).

The shaded background is computed once at startup (not per frame): for a
grid of workspace (x, y) points, arm_core.ik_solve(params, x, y,
joint_limits=joint_limits) says whether that point is reachable --
already accounting for IK reachability, joint1/joint2's independent
ranges, AND the coupled_boundary polygon all at once, so this reuses the
exact same check every other tool here enforces, rather than
re-deriving the region's shape by hand.

Drag any of the rectangle's 4 corner handles to resize (symmetric about
the current center), the small handle above the top edge to rotate about
the center, or drag inside the rectangle (away from any handle) to move
it. The border turns green when every sampled point inside the rectangle
is reachable, red if any isn't.

's'/Enter saves to calib.json AND writes a screenshot (SCREENSHOT_PATH,
"scan_area.png") -- a plain pygame.image.save of whatever's currently
drawn, so what's saved is exactly what's in the picture, same convention
as manual_test/trace_boundary_gui.py.

Doesn't touch torque at all (this tool only needs to poll encoder angles
for the live reference pose, never move the arm) -- whatever state torque
was already in when you start this stays that way; release it by hand
first (e.g. via manual_test/trace_boundary_gui.py) if you want to walk
the arm around to sanity-check specific spots against the drawn map.
"""

import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import pygame

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import arm_core as core          # noqa: E402
import arm_hardware as hw         # noqa: E402

FPS = 60
POLL_INTERVAL_S = 0.08
HANDLE_SNAP_PX = 12
MIN_RECT_MM = 10.0
SHADING_STEP_PX = 4
ROTATE_HANDLE_OFFSET_MM = 25.0
SCREENSHOT_PATH = "scan_area.png"

BG = (28, 30, 40)
REACHABLE_C = (40, 62, 46)
SHEET_BORDER_C = (70, 110, 190)
LINK1_C = (80, 160, 255)
LINK2_C = (255, 120, 55)
JOINT_C = (220, 220, 220)
EE_C = (55, 215, 95)
BASE_C = (200, 100, 55)
RECT_OK_C = (80, 220, 130)
RECT_WARN_C = (220, 80, 80)
HANDLE_C = (255, 220, 80)
ROTATE_HANDLE_C = (170, 90, 220)
TEXT_C = (200, 210, 225)
LABEL_C = (110, 130, 155)
HIGHLIGHT = (255, 220, 80)
OK_C = (80, 220, 130)


@dataclass
class Layout:
    """Screen-space layout centered on the base, sized to the arm's full
    reach circle -- same convention as trace_boundary_gui.py, since the
    reachable area can extend well outside the (possibly much smaller,
    possibly offset) calibration sheet."""
    base_x: float
    base_y: float
    max_r_mm: float
    scale: float = 1.0
    margin_px: int = 40
    panel_w: int = 280

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

    def s2ws(self, px: float, py: float) -> tuple:
        cx = self.margin_px + self.max_r_mm * self.scale
        cy = self.margin_px + self.max_r_mm * self.scale
        return ((px - cx) / self.scale + self.base_x,
                self.base_y - (py - cy) / self.scale)


@dataclass
class DragState:
    mode: str = ""             # "" | "handle" | "move" | "rotate"
    handle: str = ""           # "tl" | "tr" | "br" | "bl" (which corner, if mode=="handle")
    start_rect: tuple = ()     # rect (cx, cy, w, h, rotation_deg) when the drag began
    start_mouse_ws: tuple = () # mouse's workspace position when the drag began (mode=="move")


def _local_to_world(rect, lx, ly) -> tuple:
    cx, cy, _w, _h, rotation_deg = rect
    theta = math.radians(rotation_deg)
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    return (cx + lx * cos_t - ly * sin_t, cy + lx * sin_t + ly * cos_t)


def _world_to_local(rect, wx, wy) -> tuple:
    cx, cy, _w, _h, rotation_deg = rect
    theta = math.radians(-rotation_deg)  # inverse rotation
    dx, dy = wx - cx, wy - cy
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    return (dx * cos_t - dy * sin_t, dx * sin_t + dy * cos_t)


def _corner_positions(rect) -> dict:
    """Named corners (for handle hit-testing), backed by the same shared
    rotation math manual_test/gui.py uses to size its window around this
    same rectangle -- see arm_core.scan_area_corners()."""
    corners = core.scan_area_corners(*rect)  # [bl, br, tr, tl], see that function's docstring
    return dict(zip(("bl", "br", "tr", "tl"), corners))


def _rotate_handle_position(rect) -> tuple:
    _cx, _cy, _w, h, _rot = rect
    return _local_to_world(rect, 0.0, h / 2 + ROTATE_HANDLE_OFFSET_MM)


def _compute_reachable_surface(layout: Layout, params, joint_limits) -> pygame.Surface:
    """Precomputed once at startup: shade every reachable+safe workspace
    point (arm_core.ik_solve, the same check every other tool enforces)
    across the canvas. A live per-frame recompute isn't needed since
    nothing about the arm's kinematics/limits changes while this tool
    runs (they're only ever changed by other tools)."""
    surf = pygame.Surface((layout.win_w, layout.win_h))
    surf.fill(BG)
    for py in range(0, layout.win_h, SHADING_STEP_PX):
        for px in range(0, layout.win_w, SHADING_STEP_PX):
            wx, wy = layout.s2ws(px, py)
            if core.ik_solve(params, wx, wy, joint_limits=joint_limits).reachable:
                pygame.draw.rect(surf, REACHABLE_C, (px, py, SHADING_STEP_PX, SHADING_STEP_PX))
    return surf


def _rect_is_fully_reachable(rect, params, joint_limits, samples=5) -> bool:
    _cx, _cy, w, h, _rot = rect
    for i in range(samples):
        for j in range(samples):
            lx = -w / 2 + w * i / (samples - 1)
            ly = -h / 2 + h * j / (samples - 1)
            wx, wy = _local_to_world(rect, lx, ly)
            if not core.ik_solve(params, wx, wy, joint_limits=joint_limits).reachable:
                return False
    return True


def _hit_test(rect, layout, click_xy):
    """Returns ("rotate", None) | ("handle", corner_name) | ("move", None)
    | None -- checked in that priority order (the rotate handle sits
    outside the rectangle so it can't be confused with a corner)."""
    cx, cy = click_xy

    rp = layout.ws2s(*_rotate_handle_position(rect))
    if (rp[0] - cx) ** 2 + (rp[1] - cy) ** 2 <= HANDLE_SNAP_PX ** 2:
        return ("rotate", None)

    for name, (wx, wy) in _corner_positions(rect).items():
        px, py = layout.ws2s(wx, wy)
        if (px - cx) ** 2 + (py - cy) ** 2 <= HANDLE_SNAP_PX ** 2:
            return ("handle", name)

    lx, ly = _world_to_local(rect, *layout.s2ws(cx, cy))
    _rcx, _rcy, w, h, _rot = rect
    if abs(lx) <= w / 2 and abs(ly) <= h / 2:
        return ("move", None)
    return None


def _apply_handle_drag(rect, handle, wx, wy) -> tuple:
    """Symmetric resize about the current center: the dragged corner's
    LOCAL position (in the rectangle's own, possibly-rotated frame) sets
    the new half-width/half-height directly, same magnitude on the
    opposite side too. Simpler than anchoring the opposite corner (which
    would also have to shift the center) and still gives full control."""
    cx, cy, _w, _h, rotation_deg = rect
    lx, ly = _world_to_local(rect, wx, wy)
    new_w = max(2 * abs(lx), MIN_RECT_MM)
    new_h = max(2 * abs(ly), MIN_RECT_MM)
    return (cx, cy, new_w, new_h, rotation_deg)


def _apply_rotate_drag(rect, wx, wy) -> tuple:
    cx, cy, w, h, _rot = rect
    angle_deg = math.degrees(math.atan2(wy - cy, wx - cx)) - 90.0
    return (cx, cy, w, h, angle_deg)


def main():
    calib = core.load_calib()
    limits = core.calib_joint_limits(calib)
    if limits is None:
        print("NOTE: calib.json has no joint_limits_deg configured yet -- the shaded "
              "map below only reflects IK reach (link lengths), not any mechanical "
              "dead zone. Run `python3 main.py set-joint-limits` (and optionally "
              "manual_test/trace_boundary_gui.py) first for an accurate map.")
    params = core.calib_arm_params(calib)
    hw_cfg = core.calib_hardware_config(calib)
    ws = calib["workspace"]
    sheet_rect = (ws["width_mm"] / 2.0, ws["height_mm"] / 2.0, ws["width_mm"], ws["height_mm"], 0.0)

    servos = hw.Servos(hw_cfg.joint_ids)
    servos.connect(hw_cfg.servo_port)

    max_r = params.L1 + params.L2
    layout = Layout(base_x=params.base_x, base_y=params.base_y, max_r_mm=max_r)

    pygame.init()
    pygame.display.set_caption("2R Arm -- fit jog/scan area")
    screen = pygame.display.set_mode((layout.win_w, layout.win_h))
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("menlo,consolas,monospace", 18)
    sfont = pygame.font.SysFont("menlo,consolas,monospace", 13)

    print("computing reachable-area map...")
    reachable_surface = _compute_reachable_surface(layout, params, limits)
    print("done")

    rect = core.calib_scan_area(calib)  # (center_x, center_y, width, height, rotation_deg)
    drag = DragState()
    save_msg = ""
    save_msg_until = 0.0
    screenshot_pending = False
    last_poll = 0.0
    s1 = servos.get_present_deg("joint1")
    s2 = servos.get_present_deg("joint2")

    running = True
    try:
        while running:
            now = time.monotonic()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_q, pygame.K_ESCAPE):
                        running = False
                    elif event.key == pygame.K_f:
                        rect = sheet_rect
                        save_msg = "reset to the full calibration sheet"
                        save_msg_until = now + 2.0
                    elif event.key in (pygame.K_s, pygame.K_RETURN):
                        cx, cy, w, h, rotation_deg = rect
                        calib.setdefault("motion", {})
                        calib["motion"]["scan_center_x_mm"] = round(cx, 2)
                        calib["motion"]["scan_center_y_mm"] = round(cy, 2)
                        calib["motion"]["scan_width_mm"] = round(w, 2)
                        calib["motion"]["scan_height_mm"] = round(h, 2)
                        calib["motion"]["scan_rotation_deg"] = round(rotation_deg, 2)
                        core.save_calib(calib)
                        screenshot_pending = True
                        save_msg = f"saved scan area to calib.json + {SCREENSHOT_PATH}"
                        print(f"saved scan area to calib.json: center=({cx:.1f},{cy:.1f}) "
                              f"size={w:.1f}x{h:.1f} rotation={rotation_deg:.1f}deg -- "
                              f"manual_test/gui.py and run.py will use this instead of "
                              f"the full calibration sheet")
                        save_msg_until = now + 3.0
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    hit = _hit_test(rect, layout, event.pos)
                    if hit is not None:
                        mode, handle = hit
                        drag = DragState(mode=mode, handle=handle or "", start_rect=rect,
                                          start_mouse_ws=layout.s2ws(*event.pos))
                elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                    drag = DragState()
                elif event.type == pygame.MOUSEMOTION and drag.mode:
                    wx, wy = layout.s2ws(*event.pos)
                    if drag.mode == "handle":
                        rect = _apply_handle_drag(drag.start_rect, drag.handle, wx, wy)
                    elif drag.mode == "rotate":
                        rect = _apply_rotate_drag(drag.start_rect, wx, wy)
                    elif drag.mode == "move":
                        # Translate the center by however far the mouse has
                        # moved (in workspace mm) since the drag began.
                        cx0, cy0, w0, h0, rot0 = drag.start_rect
                        dx, dy = wx - drag.start_mouse_ws[0], wy - drag.start_mouse_ws[1]
                        rect = (cx0 + dx, cy0 + dy, w0, h0, rot0)

            if now - last_poll >= POLL_INTERVAL_S:
                s1 = servos.get_present_deg("joint1")
                s2 = servos.get_present_deg("joint2")
                last_poll = now

            screen.blit(reachable_surface, (0, 0))

            # calibration sheet outline -- fixed reference, not editable here
            sheet_corners = [layout.ws2s(*xy) for xy in _corner_positions(sheet_rect).values()]
            pygame.draw.polygon(screen, SHEET_BORDER_C, sheet_corners, 1)

            # the editable scan rectangle
            rect_ok = _rect_is_fully_reachable(rect, params, limits)
            rect_corners = [layout.ws2s(*xy) for xy in _corner_positions(rect).values()]
            pygame.draw.polygon(screen, RECT_OK_C if rect_ok else RECT_WARN_C, rect_corners, 3)
            for wx, wy in _corner_positions(rect).values():
                pygame.draw.circle(screen, HANDLE_C, layout.ws2s(wx, wy), 6)
            rot_handle_xy = layout.ws2s(*_rotate_handle_position(rect))
            pygame.draw.line(screen, ROTATE_HANDLE_C, layout.ws2s(rect[0], rect[1]), rot_handle_xy, 1)
            pygame.draw.circle(screen, ROTATE_HANDLE_C, rot_handle_xy, 7)

            # current (real, encoder-read) arm pose, for reference
            elbow, ee = core.fk_joint_positions(params, s1, s2)
            p_base, p_elbow, p_ee = (layout.ws2s(params.base_x, params.base_y),
                                      layout.ws2s(*elbow), layout.ws2s(*ee))
            pygame.draw.line(screen, LINK1_C, p_base, p_elbow, 6)
            pygame.draw.line(screen, LINK2_C, p_elbow, p_ee, 5)
            pygame.draw.circle(screen, JOINT_C, p_elbow, 7)
            pygame.draw.circle(screen, EE_C, p_ee, 8)
            pygame.draw.circle(screen, BASE_C, p_base, 9, 2)

            # panel
            px, py = int(2 * max_r * layout.scale) + layout.margin_px * 2, 28

            def row(text, dy, color=TEXT_C, f=None):
                nonlocal py
                screen.blit((f or font).render(text, True, color), (px, py))
                py += dy

            row("FIT JOG/SCAN AREA", 22, HIGHLIGHT)
            row(f"joint1={s1:6.1f}  joint2={s2:6.1f}", 22, TEXT_C, sfont)
            row(f"scan area: {rect[2]:.0f} x {rect[3]:.0f} mm  rot={rect[4]:.1f} deg",
                16, LABEL_C, sfont)
            row(f"center=({rect[0]:.1f},{rect[1]:.1f})", 16, LABEL_C, sfont)
            row("fully reachable" if rect_ok else "SOME OF THIS AREA IS UNREACHABLE", 22,
                RECT_OK_C if rect_ok else RECT_WARN_C, sfont)
            if save_msg and now < save_msg_until:
                row(save_msg, 22, OK_C, sfont)
            else:
                row(" ", 22)
            row("green background = reachable+safe", 16, LABEL_C, sfont)
            row("blue outline = calibration sheet", 22, LABEL_C, sfont)
            row("-- Keys/mouse --", 13, LABEL_C, sfont)
            for line in ["drag corner       resize (about center)",
                         "drag purple dot   rotate about center",
                         "drag inside       move the scan area",
                         "f                 reset to full sheet",
                         f"s/Enter           save calib.json + {SCREENSHOT_PATH}",
                         "q/ESC             quit (no save)"]:
                row(line, 16, LABEL_C, sfont)

            pygame.display.flip()
            if screenshot_pending:
                try:
                    pygame.image.save(screen, SCREENSHOT_PATH)
                    print(f"saved a screenshot of the fitted scan area to {SCREENSHOT_PATH}")
                except Exception as e:  # noqa: BLE001 -- a screenshot is optional, never fatal
                    print(f"WARNING: could not save {SCREENSHOT_PATH} ({e}) -- "
                          f"calib.json's data is unaffected")
                screenshot_pending = False
            clock.tick(FPS)
    finally:
        pygame.quit()
        servos.close()


if __name__ == "__main__":
    main()
