"""End-to-end tests of auto_calib_scan/card_gui.py: fake hardware (no real
arm/camera) + a synthetic pygame event stream (no real display, via the
SDL dummy driver) drive the actual tool code -- detecting the card's
AprilTag, adjusting rows/cols/dwell, saving, and running the generated
path -- exactly like a person would with the keyboard.

Self-contained fork of ../../tests/test_path_gui.py (see card_gui.py's
module docstring for why this whole folder duplicates rather than
imports), adapted for the one real difference: the scan rectangle comes
from a live (faked) AprilTag detection instead of two taught corners, so
there's a FakeCamera/FakeTagDetector here instead of jog-based teaching.
"""

import importlib.util
import os
import sys
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame  # noqa: E402
import pytest  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import arm_core as ac       # noqa: E402
import arm_hardware as hw   # noqa: E402
import card_core as cc      # noqa: E402

GUI_PATH = Path(__file__).resolve().parent.parent / "card_gui.py"

WORLD_CORNERS = [(0.0, 150.0), (200.0, 150.0), (200.0, 0.0), (0.0, 0.0)]
CARD_TAG_ID = 20
CARD_WIDTH_MM = 80.0
CARD_HEIGHT_MM = 60.0
CARD_CENTER_WORLD = (100.0, 75.0)  # ~120mm from base (100,-45) -- comfortably reachable


def _affine_pixel(xy, scale=3.0, offset=(100.0, 50.0)):
    x, y = xy
    return (scale * x + offset[0], scale * y + offset[1])


def _make_H():
    pixels = [_affine_pixel(w) for w in WORLD_CORNERS]
    H, _ = ac.compute_homography(pixels, WORLD_CORNERS)
    return H.tolist()


class FakeDetection:
    def __init__(self, tag_id, center, corners):
        self.tag_id = tag_id
        self.center = center
        self.corners = corners


def _card_detection(center_world=CARD_CENTER_WORLD, tag_id=CARD_TAG_ID):
    half = 5.0
    world_corners = [(center_world[0] - half, center_world[1] - half),
                      (center_world[0] + half, center_world[1] - half),
                      (center_world[0] + half, center_world[1] + half),
                      (center_world[0] - half, center_world[1] + half)]
    return {tag_id: FakeDetection(tag_id, _affine_pixel(center_world),
                                   [_affine_pixel(c) for c in world_corners])}


class _FakeClock:
    def tick(self, fps=0):
        return 0


@pytest.fixture(autouse=True)
def _fast_and_clean(monkeypatch):
    monkeypatch.setattr(pygame.time, "Clock", _FakeClock)
    screenshot_calls = []
    monkeypatch.setattr(pygame.image, "save",
                         lambda surface, path: screenshot_calls.append(path))
    return screenshot_calls


def _load_gui_module():
    spec = importlib.util.spec_from_file_location("card_gui", GUI_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fake_calib(with_homography=True, card_tag_id=CARD_TAG_ID,
                 card_width_mm=CARD_WIDTH_MM, card_height_mm=CARD_HEIGHT_MM):
    def fake_load_calib(path=None):
        calib = ac._default_calib()
        calib["kinematics"] = {
            "L1": 125.0, "L2": 95.0, "base_x": 100.0, "base_y": -45.0,
            "servo1_offset_deg": 0.0, "servo2_offset_deg": 0.0,
            "servo1_dir": 1, "servo2_dir": 1, "elbow_offset_mm": 0.0, "fit_report": None,
        }
        calib["hardware"] = {"servo_port": "/dev/fake", "joint_ids": {"joint1": 1, "joint2": 2}}
        calib["joint_limits_deg"] = None
        calib["card"] = {"tag_id": card_tag_id, "width_mm": card_width_mm, "height_mm": card_height_mm}
        if with_homography:
            calib["homography"]["H"] = _make_H()
        return calib
    return fake_load_calib


class FakeServosStatic:
    def __init__(self, joint_ids):
        self.joint_ids = joint_ids

    def connect(self, port, baud=115200):
        pass

    def close(self):
        pass

    def set_torque_enabled(self, joint, enabled):
        raise AssertionError("auto_calib_scan/card_gui.py must never touch torque")

    def set_target_deg(self, joint, angle, speed=800, acc=0):
        pass

    def get_present_deg(self, joint):
        return 100.0 if joint == "joint1" else 175.0


class FakeCamera:
    def __init__(self, resolution=(1920, 1080)):
        pass

    def connect(self):
        pass

    def close(self):
        pass

    def capture_gray(self):
        return None  # consumed only by FakeTagDetector below, content irrelevant


class FakeTagDetector:
    """Returns a queued sequence of {tag_id: FakeDetection} dicts, one per
    call -- except the last queued response repeats forever, so a test
    doesn't need to know exactly how many times detect() gets called
    (startup + however many 'd' presses)."""

    def __init__(self, responses):
        self._responses = list(responses)

    def detect(self, frame):
        if len(self._responses) > 1:
            return self._responses.pop(0)
        return self._responses[0]


def _install_fake_hardware(monkeypatch, detections_sequence):
    monkeypatch.setattr(hw, "Servos", FakeServosStatic)
    monkeypatch.setattr(hw, "Camera", FakeCamera)
    monkeypatch.setattr(hw, "TagDetector", lambda family="tag36h11": FakeTagDetector(detections_sequence))


def _scripted_event_get(script):
    call_count = {"n": 0}

    def fake_get(*args, **kwargs):
        call_count["n"] += 1
        return script.get(call_count["n"], [])

    return fake_get


# ── Card detection at startup + PREVIEW mode ────────────────────────────

def test_card_gui_save_writes_config_and_screenshot(monkeypatch):
    monkeypatch.setattr(ac, "load_calib", _fake_calib())
    _install_fake_hardware(monkeypatch, [_card_detection()])
    monkeypatch.setattr(cc, "load_card_scan_config", lambda path=None: cc.CardScanConfig())

    saved = {}
    monkeypatch.setattr(cc, "save_card_scan_config", lambda cfg, path=None: saved.update(cfg=cfg))

    script = {
        5: [pygame.event.Event(pygame.KEYDOWN, key=pygame.K_LEFTBRACKET, mod=0)],  # cols 3->2
        6: [pygame.event.Event(pygame.KEYDOWN, key=pygame.K_QUOTE, mod=0)],        # rows 3->4
        7: [pygame.event.Event(pygame.KEYDOWN, key=pygame.K_s, mod=0)],
        10: [pygame.event.Event(pygame.QUIT)],
    }
    monkeypatch.setattr(pygame.event, "get", _scripted_event_get(script))

    _load_gui_module().main()

    assert "cfg" in saved
    assert saved["cfg"].cols == 2
    assert saved["cfg"].rows == 4


def test_card_gui_screenshot_path(monkeypatch, _fast_and_clean):
    monkeypatch.setattr(ac, "load_calib", _fake_calib())
    _install_fake_hardware(monkeypatch, [_card_detection()])
    monkeypatch.setattr(cc, "load_card_scan_config", lambda path=None: cc.CardScanConfig())
    monkeypatch.setattr(cc, "save_card_scan_config", lambda cfg, path=None: None)

    script = {
        5: [pygame.event.Event(pygame.KEYDOWN, key=pygame.K_p, mod=0)],
        8: [pygame.event.Event(pygame.QUIT)],
    }
    monkeypatch.setattr(pygame.event, "get", _scripted_event_get(script))

    _load_gui_module().main()

    assert _fast_and_clean == ["card_scan_preview.png"]


def test_card_gui_comma_period_adjust_dwell(monkeypatch):
    monkeypatch.setattr(ac, "load_calib", _fake_calib())
    _install_fake_hardware(monkeypatch, [_card_detection()])
    monkeypatch.setattr(cc, "load_card_scan_config", lambda path=None: cc.CardScanConfig(dwell_s=1.0))

    saved = {}
    monkeypatch.setattr(cc, "save_card_scan_config", lambda cfg, path=None: saved.update(cfg=cfg))

    script = {
        5: [pygame.event.Event(pygame.KEYDOWN, key=pygame.K_COMMA, mod=0)],   # -0.2
        6: [pygame.event.Event(pygame.KEYDOWN, key=pygame.K_PERIOD, mod=0)],  # +0.2
        7: [pygame.event.Event(pygame.KEYDOWN, key=pygame.K_PERIOD, mod=0)],  # +0.2
        9: [pygame.event.Event(pygame.KEYDOWN, key=pygame.K_s, mod=0)],
        12: [pygame.event.Event(pygame.QUIT)],
    }
    monkeypatch.setattr(pygame.event, "get", _scripted_event_get(script))

    _load_gui_module().main()

    assert saved["cfg"].dwell_s == pytest.approx(1.2)


# ── Card detection failure / redetect ───────────────────────────────────

def test_card_gui_refuses_to_run_when_card_never_detected(monkeypatch):
    monkeypatch.setattr(ac, "load_calib", _fake_calib())
    _install_fake_hardware(monkeypatch, [{}])  # tag never seen, at startup or on 'd'
    monkeypatch.setattr(cc, "load_card_scan_config", lambda path=None: cc.CardScanConfig())

    created = []
    real_runner = cc.PathRunner

    class SpyRunner(real_runner):
        def __init__(self, *a, **kw):
            created.append(True)
            super().__init__(*a, **kw)

    monkeypatch.setattr(cc, "PathRunner", SpyRunner)

    script = {
        5: [pygame.event.Event(pygame.KEYDOWN, key=pygame.K_g, mod=0)],
        10: [pygame.event.Event(pygame.QUIT)],
    }
    monkeypatch.setattr(pygame.event, "get", _scripted_event_get(script))

    _load_gui_module().main()

    assert created == [], "should have refused to run with no card detected"


def test_card_gui_redetect_key_finds_card_after_initial_failure(monkeypatch):
    monkeypatch.setattr(ac, "load_calib", _fake_calib())
    # Startup detection fails (empty dict); pressing 'd' succeeds.
    _install_fake_hardware(monkeypatch, [{}, _card_detection()])
    monkeypatch.setattr(cc, "load_card_scan_config", lambda path=None: cc.CardScanConfig(dwell_s=0.0))

    arrivals = []
    monkeypatch.setattr(cc, "default_on_arrive",
                         lambda i, x, y, label: arrivals.append((i, x, y, label)))

    script = {
        5: [pygame.event.Event(pygame.KEYDOWN, key=pygame.K_d, mod=0)],  # redetect -> succeeds now
        6: [pygame.event.Event(pygame.KEYDOWN, key=pygame.K_g, mod=0)],  # run
        3000: [pygame.event.Event(pygame.QUIT)],
    }
    monkeypatch.setattr(pygame.event, "get", _scripted_event_get(script))

    _load_gui_module().main()

    assert len(arrivals) == 9  # default 3x3 grid


def test_card_gui_redetect_failure_keeps_previous_card_rect(monkeypatch):
    # Startup succeeds; a later 'd' press fails to see the tag (card
    # nudged out of frame, bad lighting, ...) -- must NOT discard the
    # already-known-good rectangle (arm_core.detect_card_rect's docstring:
    # a transient miss is not fatal), so a run started right after should
    # still use the earlier detection, not refuse to run.
    monkeypatch.setattr(ac, "load_calib", _fake_calib())
    _install_fake_hardware(monkeypatch, [_card_detection(), {}])
    monkeypatch.setattr(cc, "load_card_scan_config", lambda path=None: cc.CardScanConfig(dwell_s=0.0))

    arrivals = []
    monkeypatch.setattr(cc, "default_on_arrive",
                         lambda i, x, y, label: arrivals.append((i, x, y, label)))

    script = {
        5: [pygame.event.Event(pygame.KEYDOWN, key=pygame.K_d, mod=0)],  # redetect -> fails this time
        6: [pygame.event.Event(pygame.KEYDOWN, key=pygame.K_g, mod=0)],  # run anyway
        3000: [pygame.event.Event(pygame.QUIT)],
    }
    monkeypatch.setattr(pygame.event, "get", _scripted_event_get(script))

    _load_gui_module().main()

    assert len(arrivals) == 9, "a failed redetect must not discard the previous card_rect"


# ── RUN mode ───────────────────────────────────────────────────────────

def test_card_gui_run_mode_visits_every_node_in_order(monkeypatch):
    monkeypatch.setattr(ac, "load_calib", _fake_calib())
    _install_fake_hardware(monkeypatch, [_card_detection()])
    cfg = cc.CardScanConfig(rows=2, cols=2, dwell_s=0.0)
    monkeypatch.setattr(cc, "load_card_scan_config", lambda path=None: cfg)

    card_rect = (CARD_CENTER_WORLD[0], CARD_CENTER_WORLD[1], CARD_WIDTH_MM, CARD_HEIGHT_MM, 0.0)
    expected_nodes = cc.generate_node_path(cfg, card_rect)

    arrivals = []
    monkeypatch.setattr(cc, "default_on_arrive",
                         lambda i, x, y, label: arrivals.append((i, x, y, label)))

    script = {
        5: [pygame.event.Event(pygame.KEYDOWN, key=pygame.K_g, mod=0)],
        3000: [pygame.event.Event(pygame.QUIT)],
    }
    monkeypatch.setattr(pygame.event, "get", _scripted_event_get(script))

    _load_gui_module().main()

    assert len(arrivals) == len(expected_nodes) == 4
    for (idx, x, y, label), (ex, ey, elabel) in zip(arrivals, expected_nodes):
        assert x == pytest.approx(ex)
        assert y == pytest.approx(ey)
        assert label == elabel
    assert [a[0] for a in arrivals] == list(range(4))


def test_card_gui_refuses_to_run_when_a_node_is_unreachable(monkeypatch):
    monkeypatch.setattr(ac, "load_calib", _fake_calib())
    # A card "detected" far outside the arm's reach (L1+L2=220mm from base).
    far_detection = _card_detection(center_world=(2000.0, 2000.0))
    _install_fake_hardware(monkeypatch, [far_detection])
    monkeypatch.setattr(cc, "load_card_scan_config", lambda path=None: cc.CardScanConfig(dwell_s=0.0))

    created = []
    real_runner = cc.PathRunner

    class SpyRunner(real_runner):
        def __init__(self, *a, **kw):
            created.append(True)
            super().__init__(*a, **kw)

    monkeypatch.setattr(cc, "PathRunner", SpyRunner)

    script = {
        5: [pygame.event.Event(pygame.KEYDOWN, key=pygame.K_g, mod=0)],
        10: [pygame.event.Event(pygame.QUIT)],
    }
    monkeypatch.setattr(pygame.event, "get", _scripted_event_get(script))

    _load_gui_module().main()

    assert created == [], "should have refused to run with an unreachable node"


def test_card_gui_never_touches_torque(monkeypatch):
    monkeypatch.setattr(ac, "load_calib", _fake_calib())
    _install_fake_hardware(monkeypatch, [_card_detection()])
    monkeypatch.setattr(cc, "load_card_scan_config", lambda path=None: cc.CardScanConfig(dwell_s=0.0))

    script = {
        5: [pygame.event.Event(pygame.KEYDOWN, key=pygame.K_g, mod=0)],
        3000: [pygame.event.Event(pygame.QUIT)],
    }
    monkeypatch.setattr(pygame.event, "get", _scripted_event_get(script))

    _load_gui_module().main()  # would raise via FakeServosStatic if violated
