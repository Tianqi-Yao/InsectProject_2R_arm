"""End-to-end tests of manual_test/scan_area_gui.py: fake hardware (no real
arm, poll-only) + a synthetic pygame mouse/keyboard event stream (no real
display, via the SDL dummy driver) drive the actual tool code -- dragging
the scan-area rectangle's corner/rotate handles and saving -- exactly
like a person would with the mouse.

Complements test_arm_core.py's pure-logic tests for calib_scan_area()/
generate_scan_path()'s center+rotation support, and this file's own
direct tests of the pure local<->world rotation helpers: this exercises
the actual drag/save code path inside the tool itself.
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

GUI_PATH = Path(__file__).resolve().parent.parent / "manual_test" / "scan_area_gui.py"


class _FakeClock:
    def tick(self, fps=0):
        return 0


@pytest.fixture(autouse=True)
def _fast_and_clean(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["scan_area_gui.py"])
    monkeypatch.setattr(pygame.time, "Clock", _FakeClock)
    # A real pygame.image.save() would write scan_area.png into whatever
    # the test runner's cwd happens to be -- record calls instead.
    screenshot_calls = []
    monkeypatch.setattr(pygame.image, "save",
                         lambda surface, path: screenshot_calls.append(path))
    return screenshot_calls


def _load_gui_module():
    spec = importlib.util.spec_from_file_location("scan_area_gui", GUI_PATH)
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


class FakeServosStatic:
    """Reports a fixed pose forever -- this tool never moves the arm, only
    polls it for the reference drawing, so nothing needs to change over
    time for these tests."""

    def __init__(self, joint_ids):
        self.joint_ids = joint_ids

    def connect(self, port, baud=115200):
        pass

    def close(self):
        pass

    def set_torque_enabled(self, joint, enabled):
        raise AssertionError("scan_area_gui.py must never touch torque")

    def set_target_deg(self, joint, angle, speed=800, acc=0):
        raise AssertionError("scan_area_gui.py must never command a move")

    def get_present_deg(self, joint):
        return 100.0 if joint == "joint1" else 175.0


def _default_params():
    calib = ac._default_calib()
    calib["kinematics"] = {
        "L1": 125.0, "L2": 95.0, "base_x": 100.0, "base_y": -45.0,
        "servo1_offset_deg": 0.0, "servo2_offset_deg": 0.0,
        "servo1_dir": 1, "servo2_dir": 1, "elbow_offset_mm": 0.0, "fit_report": None,
    }
    return ac.calib_arm_params(calib)


def _scripted_event_get(script):
    call_count = {"n": 0}

    def fake_get(*args, **kwargs):
        call_count["n"] += 1
        return script.get(call_count["n"], [])

    return fake_get


# ── Pure rotation/local-frame geometry (no pygame event loop needed) ──────

def test_local_world_round_trip():
    module = _load_gui_module()
    rect = (100.0, 75.0, 200.0, 150.0, 37.0)
    wx, wy = 150.0, 100.0
    lx, ly = module._world_to_local(rect, wx, wy)
    wx2, wy2 = module._local_to_world(rect, lx, ly)
    assert wx2 == pytest.approx(wx)
    assert wy2 == pytest.approx(wy)


def test_apply_handle_drag_resizes_symmetrically_about_center():
    module = _load_gui_module()
    rect = (100.0, 75.0, 200.0, 150.0, 0.0)
    # drag the top-right corner to local (150, 100) from center
    new_rect = module._apply_handle_drag(rect, "tr", 100.0 + 150.0, 75.0 + 100.0)
    assert new_rect[0:2] == rect[0:2]  # center unchanged
    assert new_rect[2] == pytest.approx(300.0)  # width = 2*150
    assert new_rect[3] == pytest.approx(200.0)  # height = 2*100
    assert new_rect[4] == rect[4]


def test_apply_handle_drag_respects_rotation():
    module = _load_gui_module()
    rect = (100.0, 75.0, 200.0, 150.0, 90.0)
    tr_world = module._corner_positions(rect)["tr"]
    # dragging a handle back to its own current position should be a no-op
    same_rect = module._apply_handle_drag(rect, "tr", *tr_world)
    assert same_rect[2] == pytest.approx(rect[2])
    assert same_rect[3] == pytest.approx(rect[3])


def test_apply_rotate_drag_sets_angle_from_center_to_mouse():
    module = _load_gui_module()
    rect = (100.0, 75.0, 200.0, 150.0, 0.0)
    # dragging the handle to directly "above" center (local +y) -> rotation 0
    straight_up = module._apply_rotate_drag(rect, 100.0, 75.0 + 100.0)
    assert straight_up[4] == pytest.approx(0.0, abs=1e-6)
    # dragging to directly right of center -> -90 (local "up" now points +x)
    to_the_right = module._apply_rotate_drag(rect, 100.0 + 100.0, 75.0)
    assert to_the_right[4] == pytest.approx(-90.0, abs=1e-6)


# ── Full GUI event loop (headless) ────────────────────────────────────────

def test_scan_area_gui_dragging_top_left_handle_resizes_rect(monkeypatch):
    calib_holder = {}
    monkeypatch.setattr(ac, "load_calib", _fake_calib(joint_limits_deg=None))
    monkeypatch.setattr(ac, "save_calib", lambda calib, path=None: calib_holder.update(calib=calib))
    monkeypatch.setattr(hw, "Servos", FakeServosStatic)

    module = _load_gui_module()
    params = _default_params()
    layout = module.Layout(base_x=params.base_x, base_y=params.base_y, max_r_mm=params.L1 + params.L2)
    default_rect = (100.0, 75.0, 200.0, 150.0, 0.0)
    tl_screen = layout.ws2s(*module._corner_positions(default_rect)["tl"])
    dragged_to_screen = layout.ws2s(60.0, 115.0)  # local (-40,40) from center -> half-extents 40/40

    script = {
        5: [pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=tl_screen)],
        6: [pygame.event.Event(pygame.MOUSEMOTION, pos=dragged_to_screen)],
        7: [pygame.event.Event(pygame.MOUSEBUTTONUP, button=1)],
        10: [pygame.event.Event(pygame.KEYDOWN, key=pygame.K_s, mod=0)],
        12: [pygame.event.Event(pygame.QUIT)],
    }
    monkeypatch.setattr(pygame.event, "get", _scripted_event_get(script))

    module.main()

    assert "calib" in calib_holder, "save_calib was never called"
    motion = calib_holder["calib"]["motion"]
    # center stays fixed for a corner-handle drag; size grows/shrinks
    # symmetrically to match the dragged corner's new local extent.
    assert motion["scan_center_x_mm"] == pytest.approx(100.0, abs=0.5)
    assert motion["scan_center_y_mm"] == pytest.approx(75.0, abs=0.5)
    assert motion["scan_width_mm"] == pytest.approx(80.0, abs=3.0)
    assert motion["scan_height_mm"] == pytest.approx(80.0, abs=3.0)
    assert motion["scan_rotation_deg"] == pytest.approx(0.0, abs=0.5)


def test_scan_area_gui_dragging_rotate_handle_sets_rotation(monkeypatch):
    calib_holder = {}
    monkeypatch.setattr(ac, "load_calib", _fake_calib(joint_limits_deg=None))
    monkeypatch.setattr(ac, "save_calib", lambda calib, path=None: calib_holder.update(calib=calib))
    monkeypatch.setattr(hw, "Servos", FakeServosStatic)

    module = _load_gui_module()
    params = _default_params()
    layout = module.Layout(base_x=params.base_x, base_y=params.base_y, max_r_mm=params.L1 + params.L2)
    default_rect = (100.0, 75.0, 200.0, 150.0, 0.0)
    rotate_screen = layout.ws2s(*module._rotate_handle_position(default_rect))
    dragged_to_screen = layout.ws2s(100.0 + 100.0, 75.0)  # right of center -> -90deg

    script = {
        5: [pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=rotate_screen)],
        6: [pygame.event.Event(pygame.MOUSEMOTION, pos=dragged_to_screen)],
        7: [pygame.event.Event(pygame.MOUSEBUTTONUP, button=1)],
        10: [pygame.event.Event(pygame.KEYDOWN, key=pygame.K_s, mod=0)],
        12: [pygame.event.Event(pygame.QUIT)],
    }
    monkeypatch.setattr(pygame.event, "get", _scripted_event_get(script))

    module.main()

    motion = calib_holder["calib"]["motion"]
    assert motion["scan_rotation_deg"] == pytest.approx(-90.0, abs=2.0)
    assert motion["scan_width_mm"] == pytest.approx(200.0, abs=0.5)
    assert motion["scan_height_mm"] == pytest.approx(150.0, abs=0.5)


def test_scan_area_gui_save_also_writes_screenshot(monkeypatch, _fast_and_clean):
    calib_holder = {}
    monkeypatch.setattr(ac, "load_calib", _fake_calib(joint_limits_deg=None))
    monkeypatch.setattr(ac, "save_calib", lambda calib, path=None: calib_holder.update(calib=calib))
    monkeypatch.setattr(hw, "Servos", FakeServosStatic)

    script = {
        5: [pygame.event.Event(pygame.KEYDOWN, key=pygame.K_s, mod=0)],
        8: [pygame.event.Event(pygame.QUIT)],
    }
    monkeypatch.setattr(pygame.event, "get", _scripted_event_get(script))

    _load_gui_module().main()

    assert "calib" in calib_holder
    assert _fast_and_clean == ["scan_area.png"]


def test_scan_area_gui_reset_key_restores_full_sheet(monkeypatch):
    calib_holder = {}
    monkeypatch.setattr(ac, "load_calib", _fake_calib(joint_limits_deg=None))
    monkeypatch.setattr(ac, "save_calib", lambda calib, path=None: calib_holder.update(calib=calib))
    monkeypatch.setattr(hw, "Servos", FakeServosStatic)

    module = _load_gui_module()
    params = _default_params()
    layout = module.Layout(base_x=params.base_x, base_y=params.base_y, max_r_mm=params.L1 + params.L2)
    default_rect = (100.0, 75.0, 200.0, 150.0, 0.0)
    rotate_screen = layout.ws2s(*module._rotate_handle_position(default_rect))
    dragged_to_screen = layout.ws2s(100.0 + 100.0, 75.0)

    script = {
        5: [pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=rotate_screen)],
        6: [pygame.event.Event(pygame.MOUSEMOTION, pos=dragged_to_screen)],
        7: [pygame.event.Event(pygame.MOUSEBUTTONUP, button=1)],
        9: [pygame.event.Event(pygame.KEYDOWN, key=pygame.K_f, mod=0)],  # reset before saving
        10: [pygame.event.Event(pygame.KEYDOWN, key=pygame.K_s, mod=0)],
        12: [pygame.event.Event(pygame.QUIT)],
    }
    monkeypatch.setattr(pygame.event, "get", _scripted_event_get(script))

    module.main()

    motion = calib_holder["calib"]["motion"]
    assert motion["scan_center_x_mm"] == pytest.approx(100.0)
    assert motion["scan_center_y_mm"] == pytest.approx(75.0)
    assert motion["scan_width_mm"] == pytest.approx(200.0)
    assert motion["scan_height_mm"] == pytest.approx(150.0)
    assert motion["scan_rotation_deg"] == pytest.approx(0.0)


def test_scan_area_gui_never_touches_torque_or_commands_motion(monkeypatch):
    # FakeServosStatic raises if set_torque_enabled/set_target_deg are
    # called at all -- this tool is read-only with respect to the arm.
    monkeypatch.setattr(ac, "load_calib", _fake_calib(joint_limits_deg=None))
    monkeypatch.setattr(ac, "save_calib", lambda calib, path=None: None)
    monkeypatch.setattr(hw, "Servos", FakeServosStatic)

    script = {20: [pygame.event.Event(pygame.QUIT)]}
    monkeypatch.setattr(pygame.event, "get", _scripted_event_get(script))

    _load_gui_module().main()  # would raise via FakeServosStatic if violated
