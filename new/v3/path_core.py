"""Pure logic for teaching a scan rectangle and running a photo-grid scan
over it: no pygame, no direct hardware I/O. This is the module that holds
everything ../arm_core.py's older AprilTag/homography approach doesn't need
anymore -- the rectangle now comes from four hand-taught corners (read
through FK, see arm_core.fk_from_servo_angles) instead of a live tag
detection.

Two distinct kinds of points flow through here, and keeping them distinct
is the whole point of this module:
  - photo points (generate_photo_grid): where the arm must come to a
    complete stop, settle, and trigger a capture.
  - motion/interpolation points (interpolate_line_mm / build_motion_plan):
    exist ONLY to keep the end effector on a true cartesian straight line
    between two photo points -- never stopped at, never trigger a capture.

The split is made to work for free by jog_controller.ArmController's
existing corner-blending scan mode (see PhotoScanRunner's docstring below
for exactly how): nothing about ArmController or motion_planning/
trapezoidal.py needed to change for this project.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Callable, Optional

import arm_core as core

THIS_DIR = Path(__file__).parent
DEFAULT_PATHS_PATH = THIS_DIR / "paths.json"

# ── 1. Rectangle: fit from 4 taught corners, and corner geometry for drawing ──


def fit_rect_from_corners(corners: list[tuple[float, float]]
                           ) -> tuple[float, float, float, float, float]:
    """Fit a rotated rectangle (center_x_mm, center_y_mm, width_mm,
    height_mm, rotation_deg) to 4 hand-taught corner points, given in order
    walking the perimeter (either winding direction) -- e.g. teach_gui.py's
    '1'/'2'/'3'/'4' keys, pressed while physically moving the end effector
    around the four corners of the card/area in sequence, NOT diagonally
    across it.

    width/height are the average of the two opposite-edge lengths (so a
    slightly imperfect-by-hand quadrilateral still gets a sane, symmetric
    rectangle instead of overfitting to one edge); rotation is taken from
    the first edge (corners[0] -> corners[1]). This mirrors the
    already-field-tested approach in the older auto_calib_scan/manual_test
    tools (their _quad_to_rect / _fit_scan_rect_from_corners) -- a full
    orthogonal least-squares fit would be more rigorous but isn't needed at
    this project's precision target."""
    if len(corners) != 4:
        raise ValueError(f"need exactly 4 corners to fit a rectangle, got {len(corners)}")
    p0, p1, p2, p3 = corners
    cx = sum(p[0] for p in corners) / 4.0
    cy = sum(p[1] for p in corners) / 4.0
    width = (math.dist(p0, p1) + math.dist(p3, p2)) / 2.0
    height = (math.dist(p1, p2) + math.dist(p0, p3)) / 2.0
    rotation_deg = math.degrees(math.atan2(p1[1] - p0[1], p1[0] - p0[0]))
    return (float(cx), float(cy), max(width, 1e-6), max(height, 1e-6), float(rotation_deg))


def rect_corners(cx: float, cy: float, w: float, h: float, rotation_deg: float
                  ) -> list[tuple[float, float]]:
    """The 4 corners of a (possibly rotated) rectangle in workspace mm,
    ordered bottom-left -> bottom-right -> top-right -> top-left (before
    rotation) -- for drawing (teach_gui.py/scan_gui.py)."""
    local = [(-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2), (-w / 2, h / 2)]
    corners = []
    for lx, ly in local:
        rx, ry = core.rotate_vector(lx, ly, rotation_deg)
        corners.append((cx + rx, cy + ry))
    return corners


# ── 2. Serpentine photo grid over the rectangle ──────────────────────────


def generate_scan_path(width_mm: float, height_mm: float, nx: int, ny: int,
                        margin_mm: float = 0.0,
                        center_x_mm: Optional[float] = None, center_y_mm: Optional[float] = None,
                        rotation_deg: float = 0.0) -> list[tuple[float, float, str]]:
    """Serpentine (boustrophedon) grid across a `width_mm` x `height_mm`
    rectangle: starts at the top-left corner and snakes row by row --
    left-to-right, down, right-to-left, down, ... The grid is built in the
    rectangle's own local frame (centered on (center_x_mm, center_y_mm),
    local +x/+y aligned with the rectangle's own width/height axes before
    rotation), then rotated by `rotation_deg` (degrees, CCW, about that
    center) -- same shape/convention as fit_rect_from_corners's output, so
    a taught rectangle's tilt carries straight through to the photo grid.
    """
    if nx < 2 or ny < 2:
        raise ValueError(f"nx and ny must both be >=2, got nx={nx}, ny={ny}")
    if center_x_mm is None:
        center_x_mm = 0.0
    if center_y_mm is None:
        center_y_mm = 0.0

    xs_local = [-(width_mm / 2.0) + margin_mm + i * (width_mm - 2 * margin_mm) / (nx - 1)
                for i in range(nx)]
    ys_local = [(height_mm / 2.0) - margin_mm - j * (height_mm - 2 * margin_mm) / (ny - 1)
                for j in range(ny)]

    path = []
    for row, y_local in enumerate(ys_local):
        row_xs = xs_local if row % 2 == 0 else list(reversed(xs_local))
        for x_local in row_xs:
            x, y = core.rotate_vector(x_local, y_local, rotation_deg)
            path.append((center_x_mm + x, center_y_mm + y, f"row{row + 1}"))
    return path


def generate_photo_grid(rect: tuple[float, float, float, float, float],
                         spacing_x_mm: float, spacing_y_mm: float,
                         margin_mm: float = 0.0) -> list[tuple[float, float, str]]:
    """rect = (center_x_mm, center_y_mm, width_mm, height_mm, rotation_deg)
    -- see fit_rect_from_corners. `spacing_x_mm`/`spacing_y_mm` is the
    desired distance between adjacent photo points (pick these from your
    camera's field of view at working height and the overlap you want,
    e.g. spacing = fov_mm * (1 - overlap_fraction)); point counts are
    derived from that spacing and handed to generate_scan_path, which does
    the actual grid/serpentine-ordering math."""
    cx, cy, w, h, rotation_deg = rect
    eff_w = max(w - 2 * margin_mm, 1e-6)
    eff_h = max(h - 2 * margin_mm, 1e-6)
    nx = max(2, round(eff_w / spacing_x_mm) + 1)
    ny = max(2, round(eff_h / spacing_y_mm) + 1)
    return generate_scan_path(width_mm=w, height_mm=h, nx=nx, ny=ny, margin_mm=margin_mm,
                               center_x_mm=cx, center_y_mm=cy, rotation_deg=rotation_deg)


# ── 3. Cartesian straight-line interpolation ─────────────────────────────


def interpolate_line_mm(p0: tuple[float, float], p1: tuple[float, float],
                         max_step_mm: float) -> list[tuple[float, float]]:
    """Subdivide the straight line from p0 to p1 into evenly spaced steps,
    each <= max_step_mm. Returns the intermediate points AND p1 itself
    (NOT p0 -- so consecutive segments' outputs can be concatenated without
    a duplicated point), so callers can feed this directly as motion
    waypoints. A degenerate zero-length segment (p0 == p1) returns [p1]
    unchanged -- still a valid single waypoint to arrive at/stop on."""
    if max_step_mm <= 0:
        raise ValueError(f"max_step_mm must be positive, got {max_step_mm}")
    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
    dist = math.hypot(dx, dy)
    if dist < 1e-9:
        return [p1]
    n_steps = max(1, math.ceil(dist / max_step_mm))
    return [(p0[0] + dx * i / n_steps, p0[1] + dy * i / n_steps) for i in range(1, n_steps + 1)]


def build_motion_plan(waypoints: list[tuple[float, float]],
                       max_step_mm: float) -> list[list[tuple[float, float]]]:
    """waypoints (>=2 points, in travel order) -> one interpolated segment
    per consecutive pair, each ending exactly at the next waypoint. Each
    returned segment is meant to be fed straight into
    jog_controller.ArmController.start_scan() -- see PhotoScanRunner."""
    if len(waypoints) < 2:
        raise ValueError(f"need >=2 waypoints to build a motion plan, got {len(waypoints)}")
    return [interpolate_line_mm(waypoints[i], waypoints[i + 1], max_step_mm)
            for i in range(len(waypoints) - 1)]


# ── 4. Segmented runner: stop+dwell+capture only at real photo points ────


def default_on_arrive(index: int, x_mm: float, y_mm: float, label: str) -> None:
    """Placeholder invoked once per photo point, after the arm has fully
    stopped and dwelled -- reserved for camera-capture code. Replace via
    PhotoScanRunner's on_arrive parameter (see scan_gui.py)."""
    print(f"[v3] photo point {index} ({label}): x={x_mm:.1f} y={y_mm:.1f} mm -- "
          f"(no on_arrive callback given, camera capture not wired up)")


class PhotoScanRunner:
    """Drives `controller` through `photo_points` in order, stopping and
    dwelling at every one of them (and only them):

    - The first photo point is reached via a plain
      `controller.set_workspace_goal()` -- getting there from wherever the
      arm happens to be idling isn't part of any scan line, so it doesn't
      need straight-line interpolation.
    - Every photo point after that is reached via
      `controller.start_scan(build_motion_plan(...)[i])`: a list of fine
      cartesian-interpolated waypoints (see interpolate_line_mm) ending
      exactly at that photo point. ArmController's existing corner-blending
      (jog_controller.py's _corner_blend_velocity) coasts straight through
      every interior interpolation waypoint at cruise speed, because they
      all continue in the same direction along the line -- and decelerates
      to a FULL stop exactly at the segment's last waypoint, because
      _corner_blend_velocity always treats the last waypoint of a
      start_scan() call as the end of the road (`is_last` check). Since
      each segment's last waypoint is always the next photo point, this
      gets "coast through interpolation points, stop at photo points" for
      free, with zero changes to ArmController/trapezoidal.py.

    `on_arrive` fires once per photo point, but only AFTER the arm has
    been stopped there for the full `dwell_s` settle time -- stop, settle,
    THEN capture, then immediately move on (see scan_gui.py, which passes
    a callback that triggers Camera.capture_and_save; there's no separate
    post-capture wait, since a still capture is already fast relative to
    the settle time).

    `controller` only needs to duck-type set_workspace_goal(x,y),
    start_scan(waypoints), tick(), and is_moving -- jog_controller.
    ArmController satisfies this directly, and tests can pass a fake.

    Call tick(now) once per frame/loop iteration (now = time.monotonic()).
    `done` becomes True once every photo point has been visited and fully
    dwelled."""

    def __init__(self, controller, photo_points: list[tuple[float, float, str]],
                 max_step_mm: float, dwell_s: float,
                 on_arrive: Optional[Callable] = None):
        if not photo_points:
            raise ValueError("photo_points must be non-empty")
        self.controller = controller
        self.photo_points = list(photo_points)
        self.max_step_mm = max_step_mm
        self.dwell_s = dwell_s
        self.on_arrive = on_arrive or default_on_arrive
        self.index = 0
        self.arrived_at: Optional[float] = None
        self.done = False
        x0, y0, _label = self.photo_points[0]
        self.controller.set_workspace_goal(x0, y0)

    def _advance(self) -> None:
        self.index += 1
        if self.index >= len(self.photo_points):
            self.done = True
            return
        x_prev, y_prev, _ = self.photo_points[self.index - 1]
        x, y, _ = self.photo_points[self.index]
        segment = interpolate_line_mm((x_prev, y_prev), (x, y), self.max_step_mm)
        self.controller.start_scan(segment)
        self.arrived_at = None

    def tick(self, now: float) -> None:
        if self.done:
            return
        self.controller.tick()
        if self.controller.is_moving:
            return
        if self.arrived_at is None:
            # Just came to a full stop -- start the settle timer. Capture
            # doesn't fire yet: the user's requirement is stop, THEN
            # settle, THEN photograph, not photograph-then-settle.
            self.arrived_at = now
            return
        if now - self.arrived_at >= self.dwell_s:
            x, y, label = self.photo_points[self.index]
            self.on_arrive(self.index, x, y, label)
            self._advance()


# ── 5. Named recorded-path persistence (control_gui.py's PATH mode) ──────
#
# Separate from calib.json on purpose: these are hand-driven macros a user
# records/replays ad hoc (control_gui.py), not rig configuration.

def load_paths(path: Optional[Path] = None) -> dict[str, list[tuple[float, float]]]:
    path = path or DEFAULT_PATHS_PATH
    if not path.exists():
        return {}
    with open(path) as f:
        raw = json.load(f)
    return {name: [tuple(p) for p in pts] for name, pts in raw.items()}


def save_paths(paths: dict[str, list[tuple[float, float]]], path: Optional[Path] = None) -> None:
    path = path or DEFAULT_PATHS_PATH
    with open(path, "w") as f:
        json.dump({name: [list(p) for p in pts] for name, pts in paths.items()}, f, indent=2)
