"""Pure logic for the fixed-path inspection tool: no pygame, no AprilTag/
homography, no vision at all. This tool exists precisely to sidestep the
whole camera calibration pipeline -- see path_gui.py's module docstring --
so it only ever reads calib.json's kinematics/hardware/motion/
joint_limits_deg sections (arm_core.calib_arm_params/calib_hardware_config/
calib_motion_config/calib_joint_limits), the same four sections
manual_test/gui.py and manual_test/run.py already treat as independent of
the calibration *sheet*/homography concept.

The rectangle is defined by two corners taught by jogging the real arm
(see path_gui.py), not typed in or dragged with a mouse -- rect_from_corners
turns whatever two points into a center+width+height for
arm_core.generate_scan_path, which already implements the serpentine
(boustrophedon) grid this tool needs; this module doesn't reimplement it.

PathRunner is the one genuinely new piece: a "visit each node, stop, dwell,
call a hook" state machine. jog_controller.ArmController's own scanning
(start_scan/_advance_scan) is deliberately not reused for this, because
it's built to *coast through* corners at cruise speed and never fully stop
between waypoints (see its module docstring) -- the opposite of what a
per-node dwell-and-photograph tool needs. PathRunner instead drives the
controller the same way manual_test/trace_boundary_gui.py's replay loop
does: one set_workspace_goal per node, ridden to a full stop via
repeated tick() calls (that tool's replay never dwells -- it advances the
instant is_moving goes False -- so the dwell/on_arrive bookkeeping here is
new, not copied from there).
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import arm_core as core  # noqa: E402

THIS_DIR = Path(__file__).parent
PATH_CONFIG_PATH = THIS_DIR / "path_config.json"

TwoTuple = tuple


@dataclass
class PathConfig:
    """Everything needed to regenerate a fixed node path: the taught
    rectangle (as two corners, in whatever order they were recorded -- see
    rect_from_corners) plus the grid density and per-node dwell time.
    corner_a_mm/corner_b_mm are None until taught (see path_gui.py's '1'/
    '2' keys)."""
    corner_a_mm: Optional[TwoTuple] = None
    corner_b_mm: Optional[TwoTuple] = None
    rows: int = 3
    cols: int = 3
    dwell_s: float = 1.0


def load_path_config(path: Optional[Path] = None) -> PathConfig:
    """Missing file (first run) -> defaults, same convention as
    arm_core.load_calib. Silently drops any unrecognized key, same
    tolerant-merge convention as arm_core.MotionConfig.from_dict, so a
    future field rename here doesn't turn an old path_config.json into a
    hard crash."""
    path = path or PATH_CONFIG_PATH
    if not path.exists():
        return PathConfig()
    with open(path) as f:
        d = json.load(f)
    defaults = asdict(PathConfig())
    known = {k: v for k, v in d.items() if k in defaults}
    cfg = PathConfig(**{**defaults, **known})
    if cfg.corner_a_mm is not None:
        cfg.corner_a_mm = tuple(cfg.corner_a_mm)
    if cfg.corner_b_mm is not None:
        cfg.corner_b_mm = tuple(cfg.corner_b_mm)
    return cfg


def save_path_config(cfg: PathConfig, path: Optional[Path] = None) -> None:
    path = path or PATH_CONFIG_PATH
    with open(path, "w") as f:
        json.dump(asdict(cfg), f, indent=2)


def rect_from_corners(corner_a: TwoTuple, corner_b: TwoTuple) -> tuple[float, float, float, float]:
    """Two corners, either order -> (center_x_mm, center_y_mm, width_mm,
    height_mm). Doesn't care which corner is "first"/"start" vs "end" --
    only their bounding box matters."""
    ax, ay = corner_a
    bx, by = corner_b
    return ((ax + bx) / 2.0, (ay + by) / 2.0, abs(bx - ax), abs(by - ay))


def spacing_mm(cfg: PathConfig) -> tuple[float, float]:
    """(col_spacing_mm, row_spacing_mm) for the current rows/cols against
    the taught rectangle -- display-only, derived fresh whenever rows/cols
    change (see the plan's "rows/cols first, spacing is read-only" choice).
    (0.0, 0.0) if the rectangle isn't taught yet."""
    if cfg.corner_a_mm is None or cfg.corner_b_mm is None:
        return (0.0, 0.0)
    _cx, _cy, width, height = rect_from_corners(cfg.corner_a_mm, cfg.corner_b_mm)
    col_spacing = width / (cfg.cols - 1) if cfg.cols > 1 else 0.0
    row_spacing = height / (cfg.rows - 1) if cfg.rows > 1 else 0.0
    return (col_spacing, row_spacing)


def generate_node_path(cfg: PathConfig) -> list[tuple[float, float, str]]:
    """Both corners + rows/cols -> a serpentine node list, via
    arm_core.generate_scan_path (nx=cols: points per row/along width,
    ny=rows: number of rows/along height -- see that function's source).
    margin_mm=0.0 so nodes reach all the way to the taught corners
    themselves, and rotation_deg=0.0 since this tool has no rotation
    concept (the rectangle is always axis-aligned to the two corners)."""
    if cfg.corner_a_mm is None or cfg.corner_b_mm is None:
        raise ValueError("both corner_a_mm and corner_b_mm must be taught before generating a path")
    if cfg.rows < 2 or cfg.cols < 2:
        raise ValueError(f"rows and cols must both be >=2, got rows={cfg.rows}, cols={cfg.cols}")
    center_x, center_y, width, height = rect_from_corners(cfg.corner_a_mm, cfg.corner_b_mm)
    return core.generate_scan_path(
        width_mm=width, height_mm=height, nx=cfg.cols, ny=cfg.rows,
        margin_mm=0.0, center_x_mm=center_x, center_y_mm=center_y, rotation_deg=0.0)


def default_on_arrive(index: int, x_mm: float, y_mm: float, label: str) -> None:
    """Placeholder invoked once per node, after the arm has fully stopped
    and dwelled -- reserved for real camera-capture code later (see this
    tool's whole reason for existing: a fixed, repeatable inspection path).
    For now this only logs, so a run is still observable without a camera
    wired up yet."""
    print(f"[fixed_path_scan] node {index} ({label}): x={x_mm:.1f} y={y_mm:.1f} mm -- "
          f"(camera capture not wired up yet; replace default_on_arrive or pass "
          f"a different on_arrive callback into PathRunner)")


class PathRunner:
    """Drives `controller` through `nodes` in order, one at a time,
    dwelling `dwell_s` seconds at each after it fully stops, calling
    `on_arrive` once per node right when it arrives (before the dwell
    starts). `controller` only needs to duck-type set_workspace_goal(x,y),
    tick(), and is_moving -- jog_controller.ArmController satisfies this
    directly, and tests can pass a bare fake instead.

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
