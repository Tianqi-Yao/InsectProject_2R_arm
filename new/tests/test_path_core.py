"""Pure-logic tests for fixed_path_scan/path_core.py -- no pygame, no
hardware, no calib.json needed. Covers the two genuinely new pieces this
tool adds on top of arm_core.generate_scan_path: turning two taught
corners into a center+width+height rectangle, and the dwell-and-callback
PathRunner state machine (which jog_controller.ArmController's own
start_scan/_advance_scan deliberately doesn't provide -- see that
function's docstring on corner-blend coasting)."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "fixed_path_scan"))
import path_core as pc  # noqa: E402


# ── rect_from_corners ──────────────────────────────────────────────────

def test_rect_from_corners_bottom_left_to_top_right():
    cx, cy, w, h = pc.rect_from_corners((0.0, 0.0), (100.0, 60.0))
    assert (cx, cy, w, h) == pytest.approx((50.0, 30.0, 100.0, 60.0))


def test_rect_from_corners_order_independent():
    a = pc.rect_from_corners((10.0, 80.0), (90.0, 20.0))
    b = pc.rect_from_corners((90.0, 20.0), (10.0, 80.0))
    assert a == pytest.approx(b)
    assert a == pytest.approx((50.0, 50.0, 80.0, 60.0))


# ── spacing_mm ──────────────────────────────────────────────────────────

def test_spacing_mm_before_corners_taught_is_zero():
    cfg = pc.PathConfig()
    assert pc.spacing_mm(cfg) == (0.0, 0.0)


def test_spacing_mm_matches_grid_dimensions():
    cfg = pc.PathConfig(corner_a_mm=(0.0, 0.0), corner_b_mm=(100.0, 40.0), rows=5, cols=3)
    col_sp, row_sp = pc.spacing_mm(cfg)
    assert col_sp == pytest.approx(100.0 / 2)  # 3 cols -> 2 gaps across width 100
    assert row_sp == pytest.approx(40.0 / 4)   # 5 rows -> 4 gaps across height 40


# ── generate_node_path ────────────────────────────────────────────────

def test_generate_node_path_requires_both_corners():
    cfg = pc.PathConfig(corner_a_mm=(0.0, 0.0), corner_b_mm=None)
    with pytest.raises(ValueError):
        pc.generate_node_path(cfg)


def test_generate_node_path_requires_at_least_2x2():
    cfg = pc.PathConfig(corner_a_mm=(0.0, 0.0), corner_b_mm=(100.0, 100.0), rows=1, cols=3)
    with pytest.raises(ValueError):
        pc.generate_node_path(cfg)


def test_generate_node_path_count_and_corners():
    cfg = pc.PathConfig(corner_a_mm=(0.0, 0.0), corner_b_mm=(100.0, 60.0), rows=3, cols=4)
    nodes = pc.generate_node_path(cfg)
    assert len(nodes) == 12
    xs = [n[0] for n in nodes]
    ys = [n[1] for n in nodes]
    assert min(xs) == pytest.approx(0.0)
    assert max(xs) == pytest.approx(100.0)
    assert min(ys) == pytest.approx(0.0)
    assert max(ys) == pytest.approx(60.0)


def test_generate_node_path_is_serpentine():
    # row 1 left->right, row 2 right->left, matching arm_core.generate_scan_path.
    cfg = pc.PathConfig(corner_a_mm=(0.0, 0.0), corner_b_mm=(30.0, 10.0), rows=2, cols=3)
    nodes = pc.generate_node_path(cfg)
    row1_xs = [n[0] for n in nodes[0:3]]
    row2_xs = [n[0] for n in nodes[3:6]]
    assert row1_xs == sorted(row1_xs)
    assert row2_xs == sorted(row2_xs, reverse=True)


# ── PathRunner ──────────────────────────────────────────────────────────

class _FakeController:
    """Reaches every goal after exactly `ticks_to_arrive` tick() calls --
    just enough for PathRunner to exercise its "still moving" vs "arrived"
    branches without needing any real kinematics/servo I/O."""

    def __init__(self, ticks_to_arrive=2):
        self.ticks_to_arrive = ticks_to_arrive
        self._remaining = 0
        self.goals = []

    def set_workspace_goal(self, x, y):
        self.goals.append((x, y))
        self._remaining = self.ticks_to_arrive
        return True

    def tick(self):
        if self._remaining > 0:
            self._remaining -= 1

    @property
    def is_moving(self):
        return self._remaining > 0


def test_path_runner_visits_every_node_in_order():
    nodes = [(0.0, 0.0, "row1"), (10.0, 0.0, "row1"), (10.0, 5.0, "row2")]
    controller = _FakeController(ticks_to_arrive=1)
    runner = pc.PathRunner(controller, nodes, dwell_s=0.0, on_arrive=lambda *a: None)

    now = 0.0
    # first goal issued synchronously in __init__
    assert controller.goals == [(0.0, 0.0)]
    for _ in range(20):
        runner.tick(now)
        now += 0.01
        if runner.done:
            break
    assert runner.done
    assert controller.goals == [(0.0, 0.0), (10.0, 0.0), (10.0, 5.0)]


def test_path_runner_calls_on_arrive_once_per_node_with_correct_args():
    nodes = [(1.0, 2.0, "a"), (3.0, 4.0, "b")]
    controller = _FakeController(ticks_to_arrive=0)  # arrives immediately
    calls = []
    runner = pc.PathRunner(controller, nodes, dwell_s=0.0, on_arrive=lambda i, x, y, l: calls.append((i, x, y, l)))

    now = 0.0
    for _ in range(10):
        runner.tick(now)
        now += 0.01
        if runner.done:
            break
    assert runner.done
    assert calls == [(0, 1.0, 2.0, "a"), (1, 3.0, 4.0, "b")]


def test_path_runner_dwells_before_advancing():
    nodes = [(0.0, 0.0, "a"), (1.0, 1.0, "b")]
    controller = _FakeController(ticks_to_arrive=0)
    runner = pc.PathRunner(controller, nodes, dwell_s=1.0, on_arrive=lambda *a: None)

    runner.tick(now=0.0)  # arrives immediately, dwell starts at t=0
    assert controller.goals == [(0.0, 0.0)]  # hasn't advanced yet
    runner.tick(now=0.5)  # still dwelling
    assert controller.goals == [(0.0, 0.0)]
    runner.tick(now=1.5)  # dwell satisfied -> advances to node 2
    assert controller.goals == [(0.0, 0.0), (1.0, 1.0)]


def test_path_runner_empty_node_list_is_immediately_done():
    controller = _FakeController()
    runner = pc.PathRunner(controller, [], dwell_s=1.0)
    assert runner.done
    assert controller.goals == []


# ── load/save round trip ────────────────────────────────────────────────

def test_path_config_round_trip(tmp_path):
    path = tmp_path / "path_config.json"
    cfg = pc.PathConfig(corner_a_mm=(1.5, 2.5), corner_b_mm=(50.0, 60.0), rows=4, cols=6, dwell_s=2.0)
    pc.save_path_config(cfg, path)
    loaded = pc.load_path_config(path)
    assert loaded == cfg


def test_path_config_missing_file_returns_defaults(tmp_path):
    path = tmp_path / "does_not_exist.json"
    loaded = pc.load_path_config(path)
    assert loaded == pc.PathConfig()


def test_path_config_ignores_stale_unknown_fields(tmp_path):
    import json
    path = tmp_path / "path_config.json"
    with open(path, "w") as f:
        json.dump({"rows": 5, "cols": 4, "dwell_s": 1.5, "some_removed_field": 123}, f)
    loaded = pc.load_path_config(path)
    assert loaded.rows == 5
    assert loaded.cols == 4
    assert loaded.dwell_s == pytest.approx(1.5)
