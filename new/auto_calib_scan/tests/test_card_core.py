"""Pure-logic tests for auto_calib_scan/card_core.py (and the
detect_card_rect/CardConfig additions to auto_calib_scan/arm_core.py) --
no pygame, no hardware, no real calib.json needed.

Self-contained fork of ../../fixed_path_scan/tests/test_path_core.py --
this whole folder duplicates rather than imports the rest of the project
(see card_core.py's module docstring for why), so this test file is its
own copy too, adapted for the one thing that's actually new here: turning
a detected AprilTag (stuck to a physical card) into a scan rectangle,
instead of two hand-taught corners.
"""

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import arm_core as core  # noqa: E402
import card_core as cc   # noqa: E402

WORLD_CORNERS = [(0.0, 150.0), (200.0, 150.0), (200.0, 0.0), (0.0, 0.0)]


def _affine_pixel(xy, scale=3.0, offset=(100.0, 50.0)):
    """Same helper as tests/test_arm_core.py -- a clean affine pixel<->mm
    mapping, so compute_homography recovers an exact H with ~0 residual,
    letting these tests check detect_card_rect's own math (not homography
    fit noise)."""
    x, y = xy
    return (scale * x + offset[0], scale * y + offset[1])


class FakeDetection:
    """Matches arm_hardware.Detection's shape (tag_id/center/corners), just
    without needing a real TagDetector to produce one."""
    def __init__(self, center, corners):
        self.center = center
        self.corners = corners


def _make_H():
    pixels = [_affine_pixel(w) for w in WORLD_CORNERS]
    H, _ = core.compute_homography(pixels, WORLD_CORNERS)
    return H


# ── detect_card_rect ──────────────────────────────────────────────────

def test_detect_card_rect_returns_none_when_tag_not_seen():
    result = core.detect_card_rect({}, card_tag_id=20, card_width_mm=85.6,
                                    card_height_mm=54.0, H=_make_H())
    assert result is None


def test_detect_card_rect_center_and_configured_size():
    H = _make_H()
    center_world = (120.0, 60.0)
    half = 5.0
    world_corners = [(center_world[0] - half, center_world[1] - half),
                      (center_world[0] + half, center_world[1] - half),
                      (center_world[0] + half, center_world[1] + half),
                      (center_world[0] - half, center_world[1] + half)]
    det = FakeDetection(center=_affine_pixel(center_world),
                         corners=[_affine_pixel(c) for c in world_corners])

    result = core.detect_card_rect({20: det}, card_tag_id=20, card_width_mm=85.6,
                                    card_height_mm=54.0, H=H)

    assert result is not None
    cx, cy, w, h, rot = result
    assert cx == pytest.approx(center_world[0], abs=1e-3)
    assert cy == pytest.approx(center_world[1], abs=1e-3)
    # width/height come from the configured card size, NOT the (much
    # smaller) tag's own extent -- the whole reason width_mm/height_mm are
    # hand-measured inputs rather than derived from the detection.
    assert w == pytest.approx(85.6)
    assert h == pytest.approx(54.0)
    assert rot == pytest.approx(0.0, abs=1e-3)  # axis-aligned tag -> 0deg


def test_detect_card_rect_rotation_follows_tag_orientation():
    H = _make_H()
    center = (100.0, 75.0)
    half = 5.0
    theta = math.radians(30.0)
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    local = [(-half, -half), (half, -half), (half, half), (-half, half)]
    world_corners = [(center[0] + lx * cos_t - ly * sin_t, center[1] + lx * sin_t + ly * cos_t)
                      for lx, ly in local]
    det = FakeDetection(center=_affine_pixel(center),
                         corners=[_affine_pixel(c) for c in world_corners])

    result = core.detect_card_rect({20: det}, card_tag_id=20, card_width_mm=85.6,
                                    card_height_mm=54.0, H=H)

    assert result is not None
    rot = result[4]
    assert rot == pytest.approx(30.0, abs=0.5)


def test_detect_card_rect_ignores_other_tag_ids():
    H = _make_H()
    det = FakeDetection(center=_affine_pixel((50.0, 50.0)),
                         corners=[_affine_pixel(c) for c in
                                  [(45.0, 45.0), (55.0, 45.0), (55.0, 55.0), (45.0, 55.0)]])
    # The card's tag is id 20; this detection dict only has id 7 (e.g. the
    # end-effector tag or a corner tag) -- must not be mistaken for the card.
    result = core.detect_card_rect({7: det}, card_tag_id=20, card_width_mm=85.6,
                                    card_height_mm=54.0, H=H)
    assert result is None


# ── CardConfig / calib_card_config ──────────────────────────────────────

def test_calib_card_config_defaults_when_section_missing():
    calib = core._default_calib()
    del calib["card"]
    cfg = core.calib_card_config(calib)
    assert cfg == core.CardConfig()


def test_calib_card_config_reads_configured_values():
    calib = core._default_calib()
    calib["card"] = {"tag_id": 42, "width_mm": 54.0, "height_mm": 85.6}
    cfg = core.calib_card_config(calib)
    assert cfg == core.CardConfig(tag_id=42, width_mm=54.0, height_mm=85.6)


def test_validate_calib_rejects_negative_card_dimensions():
    calib = core._default_calib()
    calib["card"]["width_mm"] = -10.0
    with pytest.raises(ValueError):
        core._validate_calib(calib)


def test_validate_calib_rejects_non_integer_card_tag_id():
    calib = core._default_calib()
    calib["card"]["tag_id"] = 3.5
    with pytest.raises(ValueError):
        core._validate_calib(calib)


def test_validate_calib_accepts_missing_card_section():
    calib = core._default_calib()
    del calib["card"]
    core._validate_calib(calib)  # must not raise -- optional section


# ── generate_node_path ────────────────────────────────────────────────

def test_generate_node_path_requires_at_least_2x2():
    cfg = cc.CardScanConfig(rows=1, cols=3)
    card_rect = (100.0, 75.0, 85.6, 54.0, 0.0)
    with pytest.raises(ValueError):
        cc.generate_node_path(cfg, card_rect)


def test_generate_node_path_count_and_extent():
    cfg = cc.CardScanConfig(rows=3, cols=4)
    card_rect = (100.0, 75.0, 80.0, 60.0, 0.0)
    nodes = cc.generate_node_path(cfg, card_rect)
    assert len(nodes) == 12
    xs = [n[0] for n in nodes]
    ys = [n[1] for n in nodes]
    assert min(xs) == pytest.approx(60.0)   # center_x - width/2
    assert max(xs) == pytest.approx(140.0)  # center_x + width/2
    assert min(ys) == pytest.approx(45.0)   # center_y - height/2
    assert max(ys) == pytest.approx(105.0)  # center_y + height/2


def test_generate_node_path_follows_card_rotation():
    # Wiring check: generate_node_path must forward card_rect's rotation_deg
    # to arm_core.generate_scan_path, not hardcode 0.0.
    cfg = cc.CardScanConfig(rows=2, cols=2)
    card_rect = (0.0, 0.0, 40.0, 30.0, 37.0)
    nodes = cc.generate_node_path(cfg, card_rect)
    expected = core.generate_scan_path(width_mm=40.0, height_mm=30.0, nx=2, ny=2,
                                        margin_mm=0.0, center_x_mm=0.0, center_y_mm=0.0,
                                        rotation_deg=37.0)
    assert len(nodes) == len(expected)
    for (nx_, ny_, nl_), (ex, ey, el) in zip(nodes, expected):
        assert nx_ == pytest.approx(ex)
        assert ny_ == pytest.approx(ey)
        assert nl_ == el


# ── PathRunner ──────────────────────────────────────────────────────────

class _FakeController:
    """Reaches every goal after exactly `ticks_to_arrive` tick() calls."""

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
    runner = cc.PathRunner(controller, nodes, dwell_s=0.0, on_arrive=lambda *a: None)

    assert controller.goals == [(0.0, 0.0)]
    now = 0.0
    for _ in range(20):
        runner.tick(now)
        now += 0.01
        if runner.done:
            break
    assert runner.done
    assert controller.goals == [(0.0, 0.0), (10.0, 0.0), (10.0, 5.0)]


def test_path_runner_calls_on_arrive_once_per_node_with_correct_args():
    nodes = [(1.0, 2.0, "a"), (3.0, 4.0, "b")]
    controller = _FakeController(ticks_to_arrive=0)
    calls = []
    runner = cc.PathRunner(controller, nodes, dwell_s=0.0,
                            on_arrive=lambda i, x, y, l: calls.append((i, x, y, l)))

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
    runner = cc.PathRunner(controller, nodes, dwell_s=1.0, on_arrive=lambda *a: None)

    runner.tick(now=0.0)
    assert controller.goals == [(0.0, 0.0)]
    runner.tick(now=0.5)
    assert controller.goals == [(0.0, 0.0)]
    runner.tick(now=1.5)
    assert controller.goals == [(0.0, 0.0), (1.0, 1.0)]


# ── load/save round trip ────────────────────────────────────────────────

def test_card_scan_config_round_trip(tmp_path):
    path = tmp_path / "card_scan_config.json"
    cfg = cc.CardScanConfig(rows=4, cols=6, dwell_s=2.0)
    cc.save_card_scan_config(cfg, path)
    loaded = cc.load_card_scan_config(path)
    assert loaded == cfg


def test_card_scan_config_missing_file_returns_defaults(tmp_path):
    loaded = cc.load_card_scan_config(tmp_path / "does_not_exist.json")
    assert loaded == cc.CardScanConfig()


def test_card_scan_config_ignores_stale_unknown_fields(tmp_path):
    import json
    path = tmp_path / "card_scan_config.json"
    with open(path, "w") as f:
        json.dump({"rows": 5, "cols": 4, "dwell_s": 1.5, "corner_a_mm": [1, 2]}, f)
    loaded = cc.load_card_scan_config(path)
    assert loaded == cc.CardScanConfig(rows=5, cols=4, dwell_s=1.5)


# ── camera backend config (Camera/USB webcam vs picamera2) ──────────────

def test_calib_hardware_config_defaults_to_picamera2():
    calib = core._default_calib()
    cfg = core.calib_hardware_config(calib)
    assert cfg.camera_backend == "picamera2"
    assert cfg.usb_camera_index == 0


def test_calib_hardware_config_reads_usb_backend():
    calib = core._default_calib()
    calib["hardware"]["camera_backend"] = "usb"
    calib["hardware"]["usb_camera_index"] = 2
    cfg = core.calib_hardware_config(calib)
    assert cfg.camera_backend == "usb"
    assert cfg.usb_camera_index == 2


def test_validate_calib_rejects_unknown_camera_backend():
    calib = core._default_calib()
    calib["hardware"]["camera_backend"] = "some_other_camera"
    with pytest.raises(ValueError):
        core._validate_calib(calib)


def test_validate_calib_rejects_negative_usb_camera_index():
    calib = core._default_calib()
    calib["hardware"]["camera_backend"] = "usb"
    calib["hardware"]["usb_camera_index"] = -1
    with pytest.raises(ValueError):
        core._validate_calib(calib)


def test_validate_calib_accepts_missing_hardware_section():
    calib = core._default_calib()
    del calib["hardware"]
    core._validate_calib(calib)  # must not raise -- optional section


def test_arm_hardware_wires_camera_backend_and_usb_index_through():
    # Wiring check only -- ArmHardware.connect()/Camera.connect() need
    # real hardware (picamera2 or an actual USB device) and are treated as
    # an untested black box everywhere else in this project; this just
    # confirms the constructor threads the two new fields to the right
    # place, not that a real camera opens.
    import arm_hardware as ahw
    h = ahw.ArmHardware("/dev/fake", {"joint1": 1, "joint2": 2},
                         camera_backend="usb", usb_camera_index=3)
    assert h.camera.backend == "usb"
    assert h.camera.usb_index == 3


# ── manual card-corner fallback (camera focus issues) ───────────────────

def test_calib_card_config_manual_corners_default_none():
    calib = core._default_calib()
    cfg = core.calib_card_config(calib)
    assert cfg.manual_corner_a_mm is None
    assert cfg.manual_corner_b_mm is None


def test_calib_card_config_reads_manual_corners():
    calib = core._default_calib()
    calib["card"]["manual_corner_a_mm"] = [10.0, 20.0]
    calib["card"]["manual_corner_b_mm"] = [30.0, 40.0]
    cfg = core.calib_card_config(calib)
    assert cfg.manual_corner_a_mm == (10.0, 20.0)
    assert cfg.manual_corner_b_mm == (30.0, 40.0)


def test_validate_calib_rejects_partial_manual_corners():
    calib = core._default_calib()
    calib["card"]["manual_corner_a_mm"] = [10.0, 20.0]
    # manual_corner_b_mm left unset -- a partial pair.
    with pytest.raises(ValueError):
        core._validate_calib(calib)


def test_validate_calib_rejects_malformed_manual_corner():
    calib = core._default_calib()
    calib["card"]["manual_corner_a_mm"] = [10.0]  # wrong length
    calib["card"]["manual_corner_b_mm"] = [30.0, 40.0]
    with pytest.raises(ValueError):
        core._validate_calib(calib)


def test_validate_calib_accepts_both_manual_corners_set():
    calib = core._default_calib()
    calib["card"]["manual_corner_a_mm"] = [10.0, 20.0]
    calib["card"]["manual_corner_b_mm"] = [30.0, 40.0]
    core._validate_calib(calib)  # must not raise


def test_sub_rect_from_corners_matches_axis_aligned_when_scan_area_unrotated():
    scan_area = (0.0, 0.0, 999.0, 999.0, 0.0)
    result = cc.sub_rect_from_corners(scan_area, (0.0, 0.0), (100.0, 60.0))
    assert result == pytest.approx((50.0, 30.0, 100.0, 60.0, 0.0))


def test_sub_rect_from_corners_order_independent():
    scan_area = (0.0, 0.0, 999.0, 999.0, 0.0)
    a = cc.sub_rect_from_corners(scan_area, (10.0, 80.0), (90.0, 20.0))
    b = cc.sub_rect_from_corners(scan_area, (90.0, 20.0), (10.0, 80.0))
    assert a == pytest.approx(b)
    assert a == pytest.approx((50.0, 50.0, 80.0, 60.0, 0.0))


def test_sub_rect_from_corners_inherits_scan_area_rotation():
    # Hand-computed (same worked example as fixed_path_scan's own
    # sub_rect_from_corners tests): scan_area centered at (50,50), rotated
    # 90deg. corner_a sits exactly at the scan area's own center (local
    # (0,0)); corner_b is offset by world (30,10) from it, which
    # rotate_vector(30,10,-90) maps to local (10,-30). Local bounding box
    # -> center (5,-15), w=10, h=30; converting that local center back to
    # world (rotate_vector(5,-15,90) = (15,5)) gives world center (65,55).
    scan_area = (50.0, 50.0, 999.0, 999.0, 90.0)
    result = cc.sub_rect_from_corners(scan_area, (50.0, 50.0), (80.0, 60.0))
    assert result == pytest.approx((65.0, 55.0, 10.0, 30.0, 90.0))
