"""Live pygame tool for fitting the jog/scan area to wherever the arm can
actually safely reach -- now as four recorded corner points instead of a
single center/size/rotation rectangle.

Background: manual_test/gui.py and run.py's jog/scan rectangle used to be
exactly the AprilTag calibration sheet's own size (workspace.width_mm/
height_mm) -- but that sheet's placement is a physical fact (wherever the
four corner tags happen to be stuck down), with no guarantee it matches
the arm's actual reachable+safe region (joint_limits_deg). This tool lets
the jog/scan area be a smaller, independently-positioned shape within that
same coordinate frame, fitted to what's really usable, without touching
the calibration sheet's own numbers (doing that without moving the
physical tags would desync the homography). Four recorded points define
the usable boundary directly, which is a better fit when the reachable
region is visually warped into an irregular quadrilateral.

The shaded background is computed once at startup (not per frame): for a
grid of workspace (x, y) points, arm_core.ik_solve(params, x, y,
joint_limits=joint_limits) says whether that point is reachable --
already accounting for IK reachability, joint1/joint2's independent
ranges, AND the coupled_boundary polygon all at once, so this reuses the
exact same check every other tool here enforces, rather than
re-deriving the region's shape by hand.

Drag any of the 4 corner handles to reposition that corner, or drag inside
the shape (away from any handle) to move the whole quadrilateral. Press
1/2/3/4 to record the current end-effector position into corner 1/2/3/4
(bottom-left, bottom-right, top-right, top-left). The border turns green
when every sampled point inside the quadrilateral is reachable, red if
any isn't.

Hold Ctrl and use Left/Right to select which kinematics parameter you're
tuning, then Ctrl+Up/Down to nudge it in small steps. The four recorded
corners are stored as joint-angle samples and reprojected live through
the current arm parameters, so you can keep tweaking L1/L2/base/offsets
until the drawn shape becomes a rectangle.

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
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import pygame

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import arm_core as core          # noqa: E402
import arm_hardware as hw         # noqa: E402
import jog_controller as jc      # noqa: E402

FPS = 60
POLL_INTERVAL_S = 0.08
HANDLE_SNAP_PX = 12
MIN_RECT_MM = 10.0
SHADING_STEP_PX = 4
SCREENSHOT_PATH = "scan_area.png"
STEP_MM = 5.0
PARAM_STEP_MM = 0.5
PARAM_STEP_DEG = 0.2
PARAM_ORDER = (
    ("L1", "mm"),
    ("L2", "mm"),
    ("base_x", "mm"),
    ("base_y", "mm"),
    ("servo1_offset_deg", "deg"),
    ("servo2_offset_deg", "deg"),
    ("elbow_offset_mm", "mm"),
)

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
    mode: str = ""             # "" | "corner" | "move"
    handle: str = ""           # "bl" | "br" | "tr" | "tl"
    start_points: Optional[dict] = None   # {corner_name: (x, y)} when the drag began
    start_mouse_ws: tuple = ()  # mouse's workspace position when the drag began (mode=="move")

    def __post_init__(self):
        if self.start_points is None:
            self.start_points = {}


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


def _quad_order() -> tuple[str, ...]:
    return ("bl", "br", "tr", "tl")


def _quad_points(points: dict) -> list[tuple[float, float]]:
    return [points[name] for name in _quad_order()]


def _point_in_polygon(x: float, y: float, polygon: list[tuple[float, float]]) -> bool:
    if len(polygon) < 3:
        return False
    inside = False
    x1, y1 = polygon[-1]
    for x2, y2 in polygon:
        if ((y1 > y) != (y2 > y)) and (
            x < (x2 - x1) * (y - y1) / ((y2 - y1) or 1e-12) + x1
        ):
            inside = not inside
        x1, y1 = x2, y2
    return inside


def _quad_is_fully_reachable(points: dict, params, joint_limits, samples=6) -> bool:
    pts = _quad_points(points)
    min_x = min(x for x, _y in pts)
    max_x = max(x for x, _y in pts)
    min_y = min(y for _x, y in pts)
    max_y = max(y for _x, y in pts)
    for i in range(samples):
        for j in range(samples):
            wx = min_x + (max_x - min_x) * i / (samples - 1)
            wy = min_y + (max_y - min_y) * j / (samples - 1)
            if _point_in_polygon(wx, wy, pts):
                if not core.ik_solve(params, wx, wy, joint_limits=joint_limits).reachable:
                    return False
    return True


def _move_points(points: dict, dx: float, dy: float) -> dict:
    return {name: (x + dx, y + dy) for name, (x, y) in points.items()}


def _quad_center(points: dict) -> tuple[float, float]:
    pts = _quad_points(points)
    return (sum(x for x, _y in pts) / 4.0, sum(y for _x, y in pts) / 4.0)


def _quad_to_rect(points: dict) -> tuple[float, float, float, float, float]:
    """Compatibility approximation for older tools that still read the
    legacy center/size/rotation fields. The saved corner points are the
    canonical representation; this is only a best-effort bridge."""
    bl, br, tr, tl = (_quad_points(points))
    cx, cy = _quad_center(points)
    width = (math.dist(bl, br) + math.dist(tl, tr)) / 2.0
    height = (math.dist(bl, tl) + math.dist(br, tr)) / 2.0
    rotation_deg = math.degrees(math.atan2(br[1] - bl[1], br[0] - bl[0]))
    return (cx, cy, max(width, MIN_RECT_MM), max(height, MIN_RECT_MM), rotation_deg)


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


def _hit_test(points, layout, click_xy):
    """Returns ("corner", corner_name) | ("move", None) | None."""
    cx, cy = click_xy

    for name, (wx, wy) in points.items():
        px, py = layout.ws2s(wx, wy)
        if (px - cx) ** 2 + (py - cy) ** 2 <= HANDLE_SNAP_PX ** 2:
            return ("corner", name)

    wx, wy = layout.s2ws(cx, cy)
    if _point_in_polygon(wx, wy, _quad_points(points)):
        return ("move", None)
    return None


def _corner_joint_keys(name: str) -> tuple[str, str]:
    return (f"scan_corner_{name}_joint1_deg", f"scan_corner_{name}_joint2_deg")


def _corner_points_from_samples(params, corner_samples: dict) -> dict:
    points = {}
    for name, (s1, s2) in corner_samples.items():
        points[name] = core.fk_from_servo_angles(params, s1, s2)
    return points


def _samples_from_points(points: dict, params, joint_limits) -> dict:
    samples = {}
    for name, (x, y) in points.items():
        result = core.ik_solve(params, x, y, joint_limits=joint_limits)
        if result.reachable:
            samples[name] = (result.servo1_deg, result.servo2_deg)
    return samples


def _load_saved_samples(calib: dict, params, fallback_rect: tuple[float, float, float, float, float],
                        joint_limits) -> dict:
    motion = calib.get("motion", {})
    sample_keys = {name: _corner_joint_keys(name) for name in _quad_order()}
    samples = {}
    if all(motion.get(x_key) is not None and motion.get(y_key) is not None
           for x_key, y_key in sample_keys.values()):
        for name, (s1_key, s2_key) in sample_keys.items():
            samples[name] = (float(motion[s1_key]), float(motion[s2_key]))
        return samples

    workspace_keys = {
        "bl": ("scan_corner_bl_x_mm", "scan_corner_bl_y_mm"),
        "br": ("scan_corner_br_x_mm", "scan_corner_br_y_mm"),
        "tr": ("scan_corner_tr_x_mm", "scan_corner_tr_y_mm"),
        "tl": ("scan_corner_tl_x_mm", "scan_corner_tl_y_mm"),
    }
    workspace_points = {}
    for name, (x_key, y_key) in workspace_keys.items():
        x = motion.get(x_key)
        y = motion.get(y_key)
        if x is None or y is None:
            return _samples_from_points(_corner_positions(fallback_rect), params, joint_limits)
        workspace_points[name] = (float(x), float(y))
    samples = _samples_from_points(workspace_points, params, joint_limits)
    if len(samples) != 4:
        return _samples_from_points(_corner_positions(fallback_rect), params, joint_limits)
    return samples


def _save_state(calib: dict, params, corner_samples: dict, points: dict) -> None:
    rect = _quad_to_rect(points)
    cx, cy, w, h, rotation_deg = rect
    fit_report = calib.get("kinematics", {}).get("fit_report")
    calib["kinematics"] = {**asdict(params), "fit_report": fit_report}
    calib.setdefault("motion", {})
    motion = calib["motion"]
    for name, (s1, s2) in corner_samples.items():
        s1_key, s2_key = _corner_joint_keys(name)
        motion[s1_key] = round(s1, 2)
        motion[s2_key] = round(s2, 2)
    for name, (x, y) in points.items():
        motion[f"scan_corner_{name}_x_mm"] = round(x, 2)
        motion[f"scan_corner_{name}_y_mm"] = round(y, 2)
    motion["scan_center_x_mm"] = round(cx, 2)
    motion["scan_center_y_mm"] = round(cy, 2)
    motion["scan_width_mm"] = round(w, 2)
    motion["scan_height_mm"] = round(h, 2)
    motion["scan_rotation_deg"] = round(rotation_deg, 2)


def _tweak_param(params, name: str, delta: float) -> None:
    current = getattr(params, name)
    updated = current + delta
    if name in ("L1", "L2"):
        updated = max(updated, 1.0)
    elif name == "elbow_offset_mm":
        updated = max(updated, 0.0)
    setattr(params, name, updated)


def main():
    calib = core.load_calib()
    limits = core.calib_joint_limits(calib)
    if limits is None:
        print("NOTE: calib.json has no joint_limits_deg configured yet -- the shaded "
              "map below only reflects IK reach (link lengths), not any mechanical "
              "dead zone. Run `python3 main.py set-joint-limits` (and optionally "
              "manual_test/trace_boundary_gui.py) first for an accurate map.")
    hw_cfg = core.calib_hardware_config(calib)
    ws = calib["workspace"]
    sheet_rect = (ws["width_mm"] / 2.0, ws["height_mm"] / 2.0, ws["width_mm"], ws["height_mm"], 0.0)

    servos = hw.Servos(hw_cfg.joint_ids)
    servos.connect(hw_cfg.servo_port)
    controller = jc.build_controller(servos, calib)
    params = controller.params

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

    rect = core.calib_scan_area(calib)  # legacy fallback for older files
    corner_samples = _load_saved_samples(calib, params, rect, limits)
    drag = DragState()
    save_msg = ""
    save_msg_until = 0.0
    screenshot_pending = False
    last_poll = 0.0
    s1 = servos.get_present_deg("joint1")
    s2 = servos.get_present_deg("joint2")
    workspace_target = core.fk_from_servo_angles(params, *controller.commanded_deg)
    step = STEP_MM
    selected_param = 0

    def refresh_points() -> dict:
        return _corner_points_from_samples(params, corner_samples)

    running = True
    try:
        while running:
            now = time.monotonic()
            points = refresh_points()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_q, pygame.K_ESCAPE):
                        running = False
                    elif event.key == pygame.K_f:
                        corner_samples = _samples_from_points(_corner_positions(sheet_rect), params, None)
                        points = refresh_points()
                        save_msg = "reset to the full calibration sheet"
                        save_msg_until = now + 2.0
                    elif event.key == pygame.K_LEFTBRACKET:
                        step = max(step / 1.5, 0.5)
                    elif event.key == pygame.K_RIGHTBRACKET:
                        step = min(step * 1.5, 40.0)
                    elif event.mod & pygame.KMOD_CTRL and event.key in (pygame.K_LEFT, pygame.K_RIGHT):
                        selected_param = (selected_param - 1) % len(PARAM_ORDER) if event.key == pygame.K_LEFT else (
                            selected_param + 1) % len(PARAM_ORDER)
                    elif event.mod & pygame.KMOD_CTRL and event.key in (pygame.K_UP, pygame.K_DOWN):
                        param_name, unit = PARAM_ORDER[selected_param]
                        delta = PARAM_STEP_DEG if unit == "deg" else PARAM_STEP_MM
                        if event.key == pygame.K_DOWN:
                            delta = -delta
                        _tweak_param(params, param_name, delta)
                        points = refresh_points()
                        workspace_target = core.fk_from_servo_angles(params, *controller.commanded_deg)
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
                    elif event.key in (pygame.K_s, pygame.K_RETURN):
                        points = refresh_points()
                        _save_state(calib, params, corner_samples, points)
                        core.save_calib(calib)
                        screenshot_pending = True
                        rect = _quad_to_rect(points)
                        save_msg = f"saved scan corners to calib.json + {SCREENSHOT_PATH}"
                        print("saved scan corners to calib.json: "
                              f"bl={points['bl']} br={points['br']} tr={points['tr']} tl={points['tl']} -- "
                              f"kinematics and corner joint samples were also saved")
                        save_msg_until = now + 3.0
                    elif event.key in (pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4):
                        corner_map = {
                            pygame.K_1: "bl",
                            pygame.K_2: "br",
                            pygame.K_3: "tr",
                            pygame.K_4: "tl",
                        }
                        name = corner_map[event.key]
                        live_s1 = servos.get_present_deg("joint1")
                        live_s2 = servos.get_present_deg("joint2")
                        corner_samples[name] = (live_s1, live_s2)
                        points = refresh_points()
                        save_msg = f"recorded {name.upper()} from the live end-effector pose"
                        save_msg_until = now + 2.0
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    hit = _hit_test(points, layout, event.pos)
                    if hit is not None:
                        mode, handle = hit
                        drag = DragState(mode=mode, handle=handle or "", start_points=points.copy(),
                                          start_mouse_ws=layout.s2ws(*event.pos))
                elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                    drag = DragState()
                elif event.type == pygame.MOUSEMOTION and drag.mode:
                    wx, wy = layout.s2ws(*event.pos)
                    if drag.mode == "corner":
                        result = core.ik_solve(params, wx, wy, joint_limits=limits)
                        if result.reachable:
                            corner_samples[drag.handle] = (result.servo1_deg, result.servo2_deg)
                            points = refresh_points()
                    elif drag.mode == "move":
                        dx, dy = wx - drag.start_mouse_ws[0], wy - drag.start_mouse_ws[1]
                        if drag.start_points is not None:
                            moved_points = _move_points(drag.start_points, dx, dy)
                            moved_samples = _samples_from_points(moved_points, params, limits)
                            if len(moved_samples) == 4:
                                corner_samples = moved_samples
                                points = refresh_points()

            if now - last_poll >= POLL_INTERVAL_S:
                s1 = servos.get_present_deg("joint1")
                s2 = servos.get_present_deg("joint2")
                last_poll = now

            controller.tick()

            screen.blit(reachable_surface, (0, 0))

            # calibration sheet outline -- fixed reference, not editable here
            sheet_corners = [layout.ws2s(*xy) for xy in _corner_positions(sheet_rect).values()]
            pygame.draw.polygon(screen, SHEET_BORDER_C, sheet_corners, 1)

            # the editable scan quadrilateral
            quad_ok = _quad_is_fully_reachable(points, params, limits)
            quad_corners = [layout.ws2s(*xy) for xy in _quad_points(points)]
            pygame.draw.polygon(screen, RECT_OK_C if quad_ok else RECT_WARN_C, quad_corners, 3)
            for wx, wy in _quad_points(points):
                pygame.draw.circle(screen, HANDLE_C, layout.ws2s(wx, wy), 6)

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
            param_name, param_unit = PARAM_ORDER[selected_param]
            row(f"edit {param_name}={getattr(params, param_name):.2f}{param_unit}  step={'0.2deg' if param_unit == 'deg' else '0.5mm'}",
                16, LABEL_C, sfont)
            row(f"step={step:.1f}mm  target=({workspace_target[0]:.1f},{workspace_target[1]:.1f})", 16,
                LABEL_C, sfont)
            row(f"BL={points['bl'][0]:.1f},{points['bl'][1]:.1f}", 14, LABEL_C, sfont)
            row(f"BR={points['br'][0]:.1f},{points['br'][1]:.1f}", 14, LABEL_C, sfont)
            row(f"TR={points['tr'][0]:.1f},{points['tr'][1]:.1f}", 14, LABEL_C, sfont)
            row(f"TL={points['tl'][0]:.1f},{points['tl'][1]:.1f}", 16, LABEL_C, sfont)
            row("fully reachable" if quad_ok else "SOME OF THIS AREA IS UNREACHABLE", 22,
                RECT_OK_C if quad_ok else RECT_WARN_C, sfont)
            if save_msg and now < save_msg_until:
                row(save_msg, 22, OK_C, sfont)
            else:
                row(" ", 22)
            row("green background = reachable+safe", 16, LABEL_C, sfont)
            row("blue outline = calibration sheet", 22, LABEL_C, sfont)
            row("-- Keys/mouse --", 13, LABEL_C, sfont)
            for line in ["1/2/3/4           record BL/BR/TR/TL from live pose",
                         "Ctrl+Left/Right   select kinematics parameter",
                         "Ctrl+Up/Down      tweak selected parameter",
                         "arrows            jog the arm in workspace mm",
                         "[ / ]             change jog step size",
                         "drag corner       move that corner",
                         "drag inside       move the quadrilateral",
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
