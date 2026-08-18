"""Card-rectangle detection, serpentine path generation, and the
node-by-node scan runner."""

from __future__ import annotations

import math
from typing import Callable, Optional

import numpy as np

from .angles import rotate_vector
from .homography_read import apply_homography

# (center_x_mm, center_y_mm, width_mm, height_mm, rotation_deg)
CardRect = tuple[float, float, float, float, float]


def detect_card_rect(detections: dict, card_tag_id: int, card_width_mm: float,
                      card_height_mm: float, H: np.ndarray) -> Optional[CardRect]:
    """Turns one AprilTag detection (stuck to a physical card) into a scan
    rectangle. `card_width_mm`/`card_height_mm` are the card's own physical
    size (measured by hand -- the tag is much smaller than the card it's
    stuck to, so its size can't be derived from the tag itself). `H` is the
    already-fit pixel->mm homography.

    Returns None if `card_tag_id` isn't in `detections` this frame -- the
    card may be transiently out of frame; callers should keep the previous
    rectangle rather than treat this as fatal.

    ASSUMPTION: the tag is mounted so its own edges align with the card's
    edges -- if it's glued on crooked, rotation_deg is off by that fixed
    amount (see project docs for how to physically avoid this)."""
    det = detections.get(card_tag_id)
    if det is None:
        return None
    center_x, center_y = apply_homography(H, tuple(det.center))
    c0 = apply_homography(H, tuple(det.corners[0]))
    c1 = apply_homography(H, tuple(det.corners[1]))
    rotation_deg = math.degrees(math.atan2(c1[1] - c0[1], c1[0] - c0[0]))
    return (center_x, center_y, card_width_mm, card_height_mm, rotation_deg)


def rect_from_corners(corner_a: tuple[float, float], corner_b: tuple[float, float],
                       rotation_deg: float = 0.0) -> CardRect:
    """Manual fallback for detect_card_rect() when the camera can't
    reliably auto-detect the card's tag: two hand-jogged-and-recorded
    opposite corners (world mm) -> a scan rectangle. `rotation_deg`
    defaults to 0 (axis-aligned corners, the common case for a manually
    jogged rectangle) but can be supplied if the card is known to sit at
    an angle in the workspace frame."""
    lx1, ly1 = rotate_vector(corner_a[0], corner_a[1], -rotation_deg)
    lx2, ly2 = rotate_vector(corner_b[0], corner_b[1], -rotation_deg)
    local_cx, local_cy = (lx1 + lx2) / 2.0, (ly1 + ly2) / 2.0
    width, height = abs(lx2 - lx1), abs(ly2 - ly1)
    world_cx, world_cy = rotate_vector(local_cx, local_cy, rotation_deg)
    return (world_cx, world_cy, width, height, rotation_deg)


def generate_scan_path(card_rect: CardRect, rows: int, cols: int
                        ) -> list[tuple[float, float, str]]:
    """Serpentine (boustrophedon) path across `card_rect`: left-to-right,
    down, right-to-left, down, ... -- no wasted travel back to a row's
    start. `cols` points per row (along width), `rows` rows (along
    height). margin=0 -- nodes reach all the way to the card's own
    detected edges."""
    if rows < 2 or cols < 2:
        raise ValueError(f"rows and cols must both be >=2, got rows={rows}, cols={cols}")
    center_x, center_y, width, height, rotation_deg = card_rect

    xs_local = [-(width / 2.0) + i * width / (cols - 1) for i in range(cols)]
    ys_local = [(height / 2.0) - j * height / (rows - 1) for j in range(rows)]

    path = []
    for row, y_local in enumerate(ys_local):
        row_xs = xs_local if row % 2 == 0 else list(reversed(xs_local))
        for x_local in row_xs:
            dx, dy = rotate_vector(x_local, y_local, rotation_deg)
            path.append((center_x + dx, center_y + dy, f"row{row + 1}"))
    return path


def default_on_arrive(index: int, x_mm: float, y_mm: float, label: str) -> None:
    """Placeholder invoked once per node, after the arm has fully stopped
    and dwelled -- reserved for real camera-capture code. Replace via
    PathRunner's on_arrive parameter."""
    print(f"[card_scan] node {index} ({label}): x={x_mm:.1f} y={y_mm:.1f}mm -- "
          f"(camera capture not wired up; pass a different on_arrive callback)")


class PathRunner:
    """Drives `controller` through `nodes` in order, dwelling `dwell_s`
    seconds at each after it fully stops, calling `on_arrive` once per node
    right when it arrives (before the dwell starts). `controller` only
    needs to duck-type set_workspace_goal(x,y), tick(), and is_moving --
    controller.ArmController satisfies this directly, and tests can pass a
    bare fake instead.

    Call tick(now) once per frame/loop iteration (now = time.monotonic()).
    `done` becomes True once every node has been visited and fully
    dwelled."""

    def __init__(self, controller, nodes: list[tuple[float, float, str]],
                 dwell_s: float, on_arrive: Optional[Callable] = None):
        self.controller = controller
        self.nodes = list(nodes)
        self.dwell_s = dwell_s
        self.on_arrive = on_arrive or default_on_arrive
        self.index = -1
        self.arrived_at: Optional[float] = None
        self.done = False
        self._advance()

    def _advance(self) -> None:
        self.index += 1
        if self.index >= len(self.nodes):
            self.done = True
            return
        x, y, _label = self.nodes[self.index]
        self.controller.set_workspace_goal(x, y)
        self.arrived_at = None

    def tick(self, now: float) -> None:
        if self.done:
            return
        self.controller.tick()
        if self.controller.is_moving:
            return
        if self.arrived_at is None:
            self.arrived_at = now
            x, y, label = self.nodes[self.index]
            self.on_arrive(self.index, x, y, label)
        elif now - self.arrived_at >= self.dwell_s:
            self._advance()
