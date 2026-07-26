"""Pure logic for the auto-calibrated card scanner: no pygame, no direct
hardware I/O. This is a self-contained fork of fixed_path_scan/path_core.py
(see /Users/.../new/fixed_path_scan/ -- copied rather than imported, per
this folder's whole reason for existing: an independent variant to compare
against later, not sharing files with the rest of the project) with one
central change: the scan rectangle is no longer taught by jogging the arm
to two corners -- it's detected live from an AprilTag stuck to the
physical card (see arm_core.detect_card_rect), so it tracks wherever the
card actually is instead of trusting hand-taught points against
potentially-imprecise kinematics.

CardScanConfig only holds rows/cols/dwell_s -- no corner_a_mm/corner_b_mm,
since there's nothing to teach anymore. generate_node_path() takes the
freshly-detected card_rect (center_x, center_y, width_mm, height_mm,
rotation_deg -- the exact same shape arm_core.calib_scan_area()/
scan_area_corners() already use) directly, straight from
arm_core.generate_scan_path -- no corner->rectangle conversion step is
needed here (that was sub_rect_from_corners's job in fixed_path_scan,
specific to turning two taught points into a rectangle; a detected card
already IS one).

PathRunner/default_on_arrive are copied verbatim from
fixed_path_scan/path_core.py -- the "visit each node, stop, dwell, call a
hook" state machine is unchanged by where the rectangle came from.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Optional

import arm_core as core

THIS_DIR = Path(__file__).parent
CARD_SCAN_CONFIG_PATH = THIS_DIR / "card_scan_config.json"


@dataclass
class CardScanConfig:
    """Grid density + per-node dwell time -- everything about a card scan
    that ISN'T the rectangle itself (that comes fresh from live tag
    detection every run, see arm_core.detect_card_rect, so there's nothing
    about the card's position/size to persist here)."""
    rows: int = 3
    cols: int = 3
    dwell_s: float = 1.0


def load_card_scan_config(path: Optional[Path] = None) -> CardScanConfig:
    """Missing file (first run) -> defaults, same convention as
    arm_core.load_calib. Silently drops any unrecognized key, same
    tolerant-merge convention as arm_core.MotionConfig.from_dict."""
    path = path or CARD_SCAN_CONFIG_PATH
    if not path.exists():
        return CardScanConfig()
    with open(path) as f:
        d = json.load(f)
    defaults = asdict(CardScanConfig())
    known = {k: v for k, v in d.items() if k in defaults}
    return CardScanConfig(**{**defaults, **known})


def save_card_scan_config(cfg: CardScanConfig, path: Optional[Path] = None) -> None:
    path = path or CARD_SCAN_CONFIG_PATH
    with open(path, "w") as f:
        json.dump(asdict(cfg), f, indent=2)


def generate_node_path(cfg: CardScanConfig, card_rect: tuple) -> list[tuple[float, float, str]]:
    """card_rect (center_x_mm, center_y_mm, width_mm, height_mm,
    rotation_deg -- see arm_core.detect_card_rect) + rows/cols -> a
    serpentine node list, via arm_core.generate_scan_path (nx=cols: points
    per row/along width, ny=rows: number of rows/along height -- see that
    function's source). margin_mm=0.0 so nodes reach all the way to the
    card's own detected edges."""
    if cfg.rows < 2 or cfg.cols < 2:
        raise ValueError(f"rows and cols must both be >=2, got rows={cfg.rows}, cols={cfg.cols}")
    center_x, center_y, width, height, rotation_deg = card_rect
    return core.generate_scan_path(
        width_mm=width, height_mm=height, nx=cfg.cols, ny=cfg.rows,
        margin_mm=0.0, center_x_mm=center_x, center_y_mm=center_y, rotation_deg=rotation_deg)


def default_on_arrive(index: int, x_mm: float, y_mm: float, label: str) -> None:
    """Placeholder invoked once per node, after the arm has fully stopped
    and dwelled -- reserved for real camera-capture code later. For now
    this only logs, so a run is still observable without a camera wired
    up yet."""
    print(f"[auto_calib_scan] node {index} ({label}): x={x_mm:.1f} y={y_mm:.1f} mm -- "
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
