import numpy as np
import pytest
from arm_hw_core.apriltag import Detection

from card_scan.scan import (PathRunner, detect_card_rect, generate_scan_path,
                             rect_from_corners)

IDENTITY_H = np.eye(3)


def test_detect_card_rect_returns_none_when_tag_not_seen():
    assert detect_card_rect({}, card_tag_id=20, card_width_mm=85.6,
                             card_height_mm=54.0, H=IDENTITY_H) is None


def test_detect_card_rect_axis_aligned_tag():
    # A tag whose corner 0 -> corner 1 edge runs along +x (no rotation).
    det = Detection(tag_id=20, center=(100.0, 50.0),
                     corners=[(90.0, 40.0), (110.0, 40.0), (110.0, 60.0), (90.0, 60.0)])
    rect = detect_card_rect({20: det}, card_tag_id=20, card_width_mm=85.6,
                             card_height_mm=54.0, H=IDENTITY_H)
    cx, cy, w, h, rotation_deg = rect
    assert (cx, cy) == pytest.approx((100.0, 50.0))
    assert (w, h) == (85.6, 54.0)
    assert rotation_deg == pytest.approx(0.0, abs=1e-6)


def test_detect_card_rect_rotated_tag():
    # Corner 0 -> corner 1 edge runs along +y instead of +x -> 90deg rotation.
    det = Detection(tag_id=20, center=(0.0, 0.0),
                     corners=[(0.0, -10.0), (0.0, 10.0), (20.0, 10.0), (20.0, -10.0)])
    rect = detect_card_rect({20: det}, card_tag_id=20, card_width_mm=85.6,
                             card_height_mm=54.0, H=IDENTITY_H)
    assert rect[4] == pytest.approx(90.0, abs=1e-6)


def test_rect_from_corners_axis_aligned():
    rect = rect_from_corners((0.0, 0.0), (100.0, 50.0))
    cx, cy, w, h, rotation_deg = rect
    assert (cx, cy) == pytest.approx((50.0, 25.0))
    assert (w, h) == pytest.approx((100.0, 50.0))
    assert rotation_deg == 0.0


def test_generate_scan_path_requires_at_least_2x2():
    rect = (0.0, 0.0, 100.0, 50.0, 0.0)
    with pytest.raises(ValueError, match=">=2"):
        generate_scan_path(rect, rows=1, cols=3)


def test_generate_scan_path_covers_all_four_corners_axis_aligned():
    rect = (0.0, 0.0, 100.0, 50.0, 0.0)
    nodes = generate_scan_path(rect, rows=2, cols=2)
    points = {(round(x, 6), round(y, 6)) for x, y, _label in nodes}
    assert points == {(-50.0, 25.0), (50.0, 25.0), (50.0, -25.0), (-50.0, -25.0)}


def test_generate_scan_path_is_serpentine_not_a_raster_return():
    rect = (0.0, 0.0, 100.0, 50.0, 0.0)
    nodes = generate_scan_path(rect, rows=2, cols=3)
    row0_xs = [x for x, y, label in nodes if label == "row1"]
    row1_xs = [x for x, y, label in nodes if label == "row2"]
    assert row0_xs == sorted(row0_xs)              # left to right
    assert row1_xs == sorted(row1_xs, reverse=True)  # then right to left


class FakeController:
    """Duck-types set_workspace_goal/tick/is_moving for PathRunner."""

    def __init__(self):
        self.goals = []
        self._moving_ticks_left = 0

    def set_workspace_goal(self, x, y):
        self.goals.append((x, y))
        self._moving_ticks_left = 3  # simulate 3 ticks of travel per node
        return True

    def tick(self):
        if self._moving_ticks_left > 0:
            self._moving_ticks_left -= 1

    @property
    def is_moving(self):
        return self._moving_ticks_left > 0


def test_path_runner_visits_every_node_and_dwells():
    controller = FakeController()
    nodes = [(0.0, 0.0, "a"), (10.0, 0.0, "b")]
    arrivals = []
    runner = PathRunner(controller, nodes, dwell_s=1.0,
                         on_arrive=lambda i, x, y, label: arrivals.append(label))

    t = 0.0
    while not runner.done:
        t += 0.1
        runner.tick(t)

    assert controller.goals == [(0.0, 0.0), (10.0, 0.0)]
    assert arrivals == ["a", "b"]


def test_path_runner_calls_on_arrive_once_per_node_not_every_tick():
    controller = FakeController()
    nodes = [(0.0, 0.0, "a")]
    calls = []
    runner = PathRunner(controller, nodes, dwell_s=0.5,
                         on_arrive=lambda i, x, y, label: calls.append(1))
    t = 0.0
    while not runner.done:
        t += 0.05
        runner.tick(t)
    assert sum(calls) == 1
