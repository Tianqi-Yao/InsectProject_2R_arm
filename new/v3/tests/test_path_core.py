"""Pure-logic tests for path_core.py. No hardware needed."""

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import path_core as pc  # noqa: E402


# ── 1. fit_rect_from_corners ─────────────────────────────────────────────

def test_fit_rect_from_corners_axis_aligned():
    # A plain 100x60 rectangle centered on (50, 30), corners in perimeter
    # order bl -> br -> tr -> tl.
    corners = [(0, 0), (100, 0), (100, 60), (0, 60)]
    cx, cy, w, h, rot = pc.fit_rect_from_corners(corners)
    assert cx == pytest.approx(50.0)
    assert cy == pytest.approx(30.0)
    assert w == pytest.approx(100.0)
    assert h == pytest.approx(60.0)
    assert rot == pytest.approx(0.0, abs=1e-6)


def test_fit_rect_from_corners_rotated():
    # Same 100x60 rectangle, rotated 30deg CCW about its own center (50,30).
    base = [(0, 0), (100, 0), (100, 60), (0, 60)]
    theta = math.radians(30.0)
    cx0, cy0 = 50.0, 30.0
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    corners = []
    for x, y in base:
        dx, dy = x - cx0, y - cy0
        rx = dx * cos_t - dy * sin_t
        ry = dx * sin_t + dy * cos_t
        corners.append((cx0 + rx, cy0 + ry))
    cx, cy, w, h, rot = pc.fit_rect_from_corners(corners)
    assert cx == pytest.approx(50.0, abs=1e-6)
    assert cy == pytest.approx(30.0, abs=1e-6)
    assert w == pytest.approx(100.0, abs=1e-6)
    assert h == pytest.approx(60.0, abs=1e-6)
    assert rot == pytest.approx(30.0, abs=1e-6)


def test_fit_rect_from_corners_slightly_off_is_reasonable():
    # A hand-taught quad that's close to, but not exactly, a rectangle.
    corners = [(0, 0), (101, -1), (99, 61), (-2, 59)]
    cx, cy, w, h, rot = pc.fit_rect_from_corners(corners)
    assert w == pytest.approx(100.0, abs=3.0)
    assert h == pytest.approx(60.0, abs=3.0)
    assert rot == pytest.approx(0.0, abs=3.0)


def test_fit_rect_from_corners_requires_exactly_four():
    with pytest.raises(ValueError):
        pc.fit_rect_from_corners([(0, 0), (1, 0), (1, 1)])


def test_rect_corners_round_trip_through_fit():
    corners = pc.rect_corners(cx=10.0, cy=-5.0, w=40.0, h=25.0, rotation_deg=15.0)
    assert len(corners) == 4
    cx, cy, w, h, rot = pc.fit_rect_from_corners(corners)
    assert cx == pytest.approx(10.0, abs=1e-6)
    assert cy == pytest.approx(-5.0, abs=1e-6)
    assert w == pytest.approx(40.0, abs=1e-6)
    assert h == pytest.approx(25.0, abs=1e-6)
    assert rot == pytest.approx(15.0, abs=1e-6)


# ── 2. interpolate_line_mm / build_motion_plan ───────────────────────────

def test_interpolate_line_mm_step_bound_and_endpoint():
    p0, p1 = (0.0, 0.0), (23.0, 0.0)
    pts = pc.interpolate_line_mm(p0, p1, max_step_mm=5.0)
    assert pts[-1] == pytest.approx(p1)
    prev = p0
    for pt in pts:
        step = math.dist(prev, pt)
        assert step <= 5.0 + 1e-9
        prev = pt


def test_interpolate_line_mm_is_collinear():
    p0, p1 = (5.0, 5.0), (25.0, 45.0)
    pts = pc.interpolate_line_mm(p0, p1, max_step_mm=3.0)
    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
    for x, y in pts:
        # cross product of (p1-p0) and (pt-p0) should be ~0 for collinearity
        cross = dx * (y - p0[1]) - dy * (x - p0[0])
        assert cross == pytest.approx(0.0, abs=1e-6)


def test_interpolate_line_mm_zero_length_returns_single_point():
    p = (12.0, 34.0)
    assert pc.interpolate_line_mm(p, p, max_step_mm=5.0) == [p]


def test_interpolate_line_mm_rejects_nonpositive_step():
    with pytest.raises(ValueError):
        pc.interpolate_line_mm((0, 0), (1, 1), max_step_mm=0.0)


def test_build_motion_plan_segments_concatenate_without_duplicates():
    waypoints = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0)]
    plan = pc.build_motion_plan(waypoints, max_step_mm=4.0)
    assert len(plan) == 2
    assert plan[0][-1] == pytest.approx(waypoints[1])
    assert plan[1][-1] == pytest.approx(waypoints[2])
    # no segment includes its own start point
    assert waypoints[0] not in plan[0]
    assert waypoints[1] not in plan[1]


def test_build_motion_plan_requires_at_least_two_waypoints():
    with pytest.raises(ValueError):
        pc.build_motion_plan([(0.0, 0.0)], max_step_mm=5.0)


# ── 3. generate_photo_grid ────────────────────────────────────────────────

def test_generate_photo_grid_point_count_and_spacing():
    rect = (0.0, 0.0, 100.0, 60.0, 0.0)
    grid = pc.generate_photo_grid(rect, spacing_x_mm=25.0, spacing_y_mm=20.0)
    # width 100 / spacing 25 -> 4 intervals -> 5 columns; height 60/20 -> 3 -> 4 rows
    xs_row1 = sorted({round(x, 6) for x, y, label in grid if label == "row1"})
    assert len(xs_row1) == 5
    assert xs_row1[1] - xs_row1[0] == pytest.approx(25.0)
    rows = sorted({label for _, _, label in grid})
    assert len(rows) == 4


def test_generate_photo_grid_is_serpentine():
    rect = (0.0, 0.0, 100.0, 60.0, 0.0)
    grid = pc.generate_photo_grid(rect, spacing_x_mm=50.0, spacing_y_mm=60.0)
    xs_by_row = {}
    for x, y, label in grid:
        xs_by_row.setdefault(label, []).append(x)
    rows = list(xs_by_row.values())
    assert len(rows) == 2
    # consecutive rows must run in opposite x-direction (serpentine)
    assert (rows[0][0] < rows[0][-1]) != (rows[1][0] < rows[1][-1])
    # the end of row1 and the start of row2 should be adjacent (no long jump back)
    assert abs(rows[0][-1] - rows[1][0]) < abs(rows[0][-1] - rows[0][0])


def test_generate_photo_grid_rotates_with_rectangle():
    rect = (0.0, 0.0, 100.0, 60.0, 90.0)
    grid = pc.generate_photo_grid(rect, spacing_x_mm=25.0, spacing_y_mm=20.0)
    # rotated 90deg: what was the width (x) axis is now along y, and vice versa
    xs = [x for x, y, _ in grid]
    ys = [y for x, y, _ in grid]
    assert max(xs) - min(xs) == pytest.approx(60.0, abs=1e-6)
    assert max(ys) - min(ys) == pytest.approx(100.0, abs=1e-6)


# ── 4. PhotoScanRunner ────────────────────────────────────────────────────

class FakeController:
    """Duck-types set_workspace_goal/start_scan/tick/is_moving. Every
    "move" is instantaneous (arrives on the very next tick()) so the test
    can drive the state machine deterministically without real timing."""

    def __init__(self):
        self.start_scan_calls: list[list] = []
        self.single_goal_calls: list[tuple[float, float]] = []
        self._pending = False

    def set_workspace_goal(self, x, y):
        self.single_goal_calls.append((x, y))
        self._pending = True
        return True

    def start_scan(self, waypoints):
        self.start_scan_calls.append(list(waypoints))
        self._pending = True

    def tick(self):
        # Simulate "arrives by the next tick after being commanded".
        if self._pending:
            self._pending = False

    @property
    def is_moving(self):
        return self._pending


def test_photo_scan_runner_visits_every_point_once_and_only_at_stops():
    controller = FakeController()
    photo_points = [(0.0, 0.0, "row1"), (10.0, 0.0, "row1"), (10.0, 10.0, "row2")]
    arrivals = []
    runner = pc.PhotoScanRunner(controller, photo_points, max_step_mm=3.0, dwell_s=0.5,
                                 on_arrive=lambda i, x, y, label: arrivals.append((i, x, y, label)))

    now = 0.0
    # First point reached via a plain single-target goal, not start_scan.
    assert controller.single_goal_calls == [(0.0, 0.0)]
    assert controller.start_scan_calls == []

    while not runner.done:
        now += 0.1
        runner.tick(now)

    assert len(arrivals) == 3
    assert [a[3] for a in arrivals] == ["row1", "row1", "row2"]
    for i, (idx, x, y, label) in enumerate(arrivals):
        assert (x, y, label) == photo_points[i]

    # Two start_scan segments for the two moves after the initial approach.
    assert len(controller.start_scan_calls) == 2
    # Every segment's last waypoint is exactly the photo point it leads to.
    assert controller.start_scan_calls[0][-1] == pytest.approx((10.0, 0.0))
    assert controller.start_scan_calls[1][-1] == pytest.approx((10.0, 10.0))
    # Interpolated segments have more than one waypoint (not a single jump).
    assert len(controller.start_scan_calls[0]) > 1


def test_photo_scan_runner_dwells_before_advancing():
    # FakeController "arrives" synchronously inside a single tick() call
    # (no multi-tick travel simulation), so the arm stops on the very
    # first runner.tick() call after a point's goal was issued -- but
    # on_arrive (capture) only fires once the FULL dwell_s settle time has
    # elapsed since that stop: stop, THEN settle, THEN capture+advance.
    controller = FakeController()
    photo_points = [(0.0, 0.0, "a"), (5.0, 0.0, "b")]
    arrivals = []
    runner = pc.PhotoScanRunner(controller, photo_points, max_step_mm=2.0, dwell_s=1.0,
                                 on_arrive=lambda i, x, y, label: arrivals.append((i, label)))

    runner.tick(now=0.1)  # stops at point 0 -> settle timer starts, no capture yet
    assert arrivals == []
    assert controller.start_scan_calls == []

    runner.tick(now=0.5)  # dwell_s=1.0 not yet elapsed since arrived_at=0.1
    assert arrivals == []
    assert not runner.done

    runner.tick(now=1.2)  # dwell_s elapsed (1.2-0.1 >= 1.0) -> captures + issues next segment
    assert arrivals == [(0, "a")]
    assert len(controller.start_scan_calls) == 1
    assert not runner.done

    runner.tick(now=1.3)  # segment resolves synchronously -> stops at point 1, settle starts
    assert arrivals == [(0, "a")]  # not captured yet
    assert not runner.done

    runner.tick(now=1.4)  # dwell_s=1.0 not yet elapsed since arrived_at=1.3
    assert arrivals == [(0, "a")]
    assert not runner.done

    runner.tick(now=2.5)  # dwell_s elapsed -> captures point 1, advances past the last point -> done
    assert arrivals == [(0, "a"), (1, "b")]
    assert runner.done


def test_photo_scan_runner_requires_nonempty_points():
    with pytest.raises(ValueError):
        pc.PhotoScanRunner(FakeController(), [], max_step_mm=2.0, dwell_s=1.0)


def test_photo_scan_runner_single_point_completes_after_dwell():
    controller = FakeController()
    runner = pc.PhotoScanRunner(controller, [(1.0, 2.0, "only")], max_step_mm=2.0, dwell_s=0.2)
    runner.tick(now=0.0)  # arrives immediately, fires default on_arrive, arrived_at=0.0
    assert not runner.done
    runner.tick(now=0.1)  # dwell_s=0.2 not yet elapsed
    assert not runner.done
    runner.tick(now=0.4)  # dwell_s elapsed -> no further points -> done
    assert runner.done
    assert controller.start_scan_calls == []


# ── 5. load_paths / save_paths ────────────────────────────────────────────

def test_save_and_load_paths_round_trip(tmp_path):
    p = tmp_path / "paths.json"
    paths = {"boundary": [(1.0, 2.0), (3.5, 4.5)], "empty": []}
    pc.save_paths(paths, path=p)
    loaded = pc.load_paths(path=p)
    assert loaded == paths


def test_load_paths_missing_file_returns_empty_dict(tmp_path):
    assert pc.load_paths(path=tmp_path / "does_not_exist.json") == {}
