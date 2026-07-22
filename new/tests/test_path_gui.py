"""End-to-end tests of fixed_path_scan/path_gui.py: fake hardware (no real
arm) + a synthetic pygame event stream (no real display, via the SDL dummy
driver) drive the actual tool code -- jogging, recording corners, adjusting
rows/cols, saving, and running the generated path -- exactly like a person
would with the keyboard.

Complements tests/test_path_core.py's pure-logic tests (rect_from_corners/
generate_node_path/PathRunner in isolation): this exercises the actual
teach/run wiring inside the tool itself, same pattern as
tests/test_scan_area_gui.py and tests/test_trace_boundary_gui.py.
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "fixed_path_scan"))
import path_core as pc      # noqa: E402

GUI_PATH = Path(__file__).resolve().parent.parent / "fixed_path_scan" / "path_gui.py"


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
    spec = importlib.util.spec_from_file_location("path_gui", GUI_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fake_calib(joint_limits_deg=None):
    def fake_load_calib(path=None):
        calib = ac._default_calib()
        calib["kinematics"] = {
            "L1": 125.0, "L2": 95.0, "base_x": 100.0, "base_y": -45.0,
            "servo1_offset_deg": 0.0, "servo2_offset_deg": 0.0,
            "servo1_dir": 1, "servo2_dir": 1, "elbow_offset_mm": 0.0, "fit_report": None,
        }
        calib["hardware"] = {"servo_port": "/dev/fake", "joint_ids": {"joint1": 1, "joint2": 2}}
        calib["joint_limits_deg"] = joint_limits_deg
        return calib
    return fake_load_calib


def _default_params():
    calib = ac._default_calib()
    calib["kinematics"] = {
        "L1": 125.0, "L2": 95.0, "base_x": 100.0, "base_y": -45.0,
        "servo1_offset_deg": 0.0, "servo2_offset_deg": 0.0,
        "servo1_dir": 1, "servo2_dir": 1, "elbow_offset_mm": 0.0, "fit_report": None,
    }
    return ac.calib_arm_params(calib)


class FakeServosStatic:
    """Reports a fixed encoder pose forever, regardless of what's
    commanded -- matches tests/test_scan_area_gui.py's fake: this tool's
    controller.tick() drives purely off its own precomputed segment queue
    (see jog_controller.ArmController), never reading these values back for
    correctness, only for the "real" arm drawing/display."""

    def __init__(self, joint_ids):
        self.joint_ids = joint_ids
        self.set_target_calls = []

    def connect(self, port, baud=115200):
        pass

    def close(self):
        pass

    def set_torque_enabled(self, joint, enabled):
        raise AssertionError("fixed_path_scan/path_gui.py must never touch torque")

    def set_target_deg(self, joint, angle, speed=800, acc=0):
        self.set_target_calls.append((joint, angle))

    def get_present_deg(self, joint):
        return 100.0 if joint == "joint1" else 175.0


def _scripted_event_get(script):
    call_count = {"n": 0}

    def fake_get(*args, **kwargs):
        call_count["n"] += 1
        return script.get(call_count["n"], [])

    return fake_get


# ── TEACH mode: jog + record corners + adjust grid + save ────────────────

def test_path_gui_teach_records_corners_adjusts_grid_and_saves(monkeypatch):
    monkeypatch.setattr(ac, "load_calib", _fake_calib(joint_limits_deg=None))
    monkeypatch.setattr(hw, "Servos", FakeServosStatic)
    monkeypatch.setattr(pc, "load_path_config", lambda path=None: pc.PathConfig())

    saved = {}
    monkeypatch.setattr(pc, "save_path_config", lambda cfg, path=None: saved.update(cfg=cfg))

    module = _load_gui_module()
    params = _default_params()
    initial_target = ac.fk_from_servo_angles(params, 100.0, 175.0)
    expected_corner_a = (initial_target[0], initial_target[1] + module.STEP_MM)
    expected_corner_b = (expected_corner_a[0] + module.STEP_MM, expected_corner_a[1])

    script = {
        5: [pygame.event.Event(pygame.KEYDOWN, key=pygame.K_UP, mod=0)],
        6: [pygame.event.Event(pygame.KEYDOWN, key=pygame.K_1, mod=0)],
        7: [pygame.event.Event(pygame.KEYDOWN, key=pygame.K_RIGHT, mod=0)],
        8: [pygame.event.Event(pygame.KEYDOWN, key=pygame.K_2, mod=0)],
        9: [pygame.event.Event(pygame.KEYDOWN, key=pygame.K_LEFTBRACKET, mod=0)],   # cols 3->2
        10: [pygame.event.Event(pygame.KEYDOWN, key=pygame.K_QUOTE, mod=0)],        # rows 3->4
        11: [pygame.event.Event(pygame.KEYDOWN, key=pygame.K_s, mod=0)],
        14: [pygame.event.Event(pygame.QUIT)],
    }
    monkeypatch.setattr(pygame.event, "get", _scripted_event_get(script))

    module.main()

    assert "cfg" in saved, "save_path_config was never called"
    cfg = saved["cfg"]
    assert cfg.corner_a_mm == pytest.approx(expected_corner_a)
    assert cfg.corner_b_mm == pytest.approx(expected_corner_b)
    assert cfg.cols == 2
    assert cfg.rows == 4


def test_path_gui_save_also_writes_screenshot(monkeypatch, _fast_and_clean):
    monkeypatch.setattr(ac, "load_calib", _fake_calib(joint_limits_deg=None))
    monkeypatch.setattr(hw, "Servos", FakeServosStatic)
    monkeypatch.setattr(pc, "load_path_config", lambda path=None: pc.PathConfig())
    monkeypatch.setattr(pc, "save_path_config", lambda cfg, path=None: None)

    script = {
        5: [pygame.event.Event(pygame.KEYDOWN, key=pygame.K_s, mod=0)],
        8: [pygame.event.Event(pygame.QUIT)],
    }
    monkeypatch.setattr(pygame.event, "get", _scripted_event_get(script))

    _load_gui_module().main()

    assert _fast_and_clean == ["path_preview.png"]


# ── RUN mode ───────────────────────────────────────────────────────────

def test_path_gui_run_mode_visits_every_node_in_order(monkeypatch):
    monkeypatch.setattr(ac, "load_calib", _fake_calib(joint_limits_deg=None))
    monkeypatch.setattr(hw, "Servos", FakeServosStatic)

    cfg = pc.PathConfig(corner_a_mm=(80.0, 60.0), corner_b_mm=(120.0, 90.0), rows=2, cols=2, dwell_s=0.0)
    monkeypatch.setattr(pc, "load_path_config", lambda path=None: cfg)
    expected_nodes = pc.generate_node_path(cfg)

    arrivals = []
    monkeypatch.setattr(pc, "default_on_arrive",
                         lambda i, x, y, label: arrivals.append((i, x, y, label)))

    script = {
        5: [pygame.event.Event(pygame.KEYDOWN, key=pygame.K_g, mod=0)],
        # Generous budget: 4 short jog segments need far fewer than this
        # many tick()s to fully drain, same margin trace_boundary_gui's
        # replay test uses.
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


def test_path_gui_refuses_to_run_when_a_node_is_unreachable(monkeypatch):
    monkeypatch.setattr(ac, "load_calib", _fake_calib(joint_limits_deg=None))
    monkeypatch.setattr(hw, "Servos", FakeServosStatic)

    # corner_b is far outside the arm's reach (L1+L2=220mm from base) --
    # every node in this grid should fail the reachability check.
    cfg = pc.PathConfig(corner_a_mm=(80.0, 60.0), corner_b_mm=(1000.0, 1000.0), rows=2, cols=2, dwell_s=0.0)
    monkeypatch.setattr(pc, "load_path_config", lambda path=None: cfg)

    created = []
    real_runner = pc.PathRunner

    class SpyRunner(real_runner):
        def __init__(self, *a, **kw):
            created.append(True)
            super().__init__(*a, **kw)

    monkeypatch.setattr(pc, "PathRunner", SpyRunner)

    script = {
        5: [pygame.event.Event(pygame.KEYDOWN, key=pygame.K_g, mod=0)],
        10: [pygame.event.Event(pygame.QUIT)],
    }
    monkeypatch.setattr(pygame.event, "get", _scripted_event_get(script))

    _load_gui_module().main()

    assert created == [], "should have refused to enter RUN mode with an unreachable node"


def test_path_gui_refuses_to_run_before_both_corners_are_taught(monkeypatch):
    monkeypatch.setattr(ac, "load_calib", _fake_calib(joint_limits_deg=None))
    monkeypatch.setattr(hw, "Servos", FakeServosStatic)
    monkeypatch.setattr(pc, "load_path_config", lambda path=None: pc.PathConfig())  # no corners yet

    created = []
    real_runner = pc.PathRunner

    class SpyRunner(real_runner):
        def __init__(self, *a, **kw):
            created.append(True)
            super().__init__(*a, **kw)

    monkeypatch.setattr(pc, "PathRunner", SpyRunner)

    script = {
        5: [pygame.event.Event(pygame.KEYDOWN, key=pygame.K_g, mod=0)],
        10: [pygame.event.Event(pygame.QUIT)],
    }
    monkeypatch.setattr(pygame.event, "get", _scripted_event_get(script))

    _load_gui_module().main()

    assert created == []


def test_path_gui_never_touches_torque(monkeypatch):
    # FakeServosStatic raises if set_torque_enabled is called at all --
    # jogging/running only ever calls set_target_deg (via
    # jog_controller.ArmController.tick()).
    monkeypatch.setattr(ac, "load_calib", _fake_calib(joint_limits_deg=None))
    monkeypatch.setattr(hw, "Servos", FakeServosStatic)
    cfg = pc.PathConfig(corner_a_mm=(80.0, 60.0), corner_b_mm=(120.0, 90.0), rows=2, cols=2, dwell_s=0.0)
    monkeypatch.setattr(pc, "load_path_config", lambda path=None: cfg)

    script = {
        5: [pygame.event.Event(pygame.KEYDOWN, key=pygame.K_UP, mod=0)],
        6: [pygame.event.Event(pygame.KEYDOWN, key=pygame.K_g, mod=0)],
        3000: [pygame.event.Event(pygame.QUIT)],
    }
    monkeypatch.setattr(pygame.event, "get", _scripted_event_get(script))

    _load_gui_module().main()  # would raise via FakeServosStatic if violated
