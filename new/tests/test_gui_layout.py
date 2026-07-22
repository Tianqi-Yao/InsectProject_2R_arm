"""Pure-logic tests for manual_test/gui.py's Layout.fit() -- the bounding
box computation that sizes the window to show BOTH the AprilTag
calibration sheet and the (possibly rotated, possibly-outside-the-sheet)
jog/scan area (see arm_core.calib_scan_area/scan_area_corners).

This regression-tests a real bug: gui.py's window used to be sized to
just the calibration sheet's own dimensions, so a scan area fitted (via
manual_test/scan_area_gui.py) to extend outside the sheet -- exactly the
point of decoupling the two -- got silently clipped at the window edge,
even though it had saved correctly.

No pygame display/hardware needed -- Layout.fit is pure arithmetic.
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
import arm_core as core       # noqa: E402
import arm_hardware as hw     # noqa: E402

GUI_PATH = Path(__file__).resolve().parent.parent / "manual_test" / "gui.py"


def _load_gui_module():
    spec = importlib.util.spec_from_file_location("gui", GUI_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_layout_fit_matches_old_sheet_only_sizing_when_unconfigured():
    # Not configured (calib_scan_area's fallback) -> scan area == the
    # sheet itself, so the bounding box is just the sheet plus margin,
    # same as before this concept existed.
    module = _load_gui_module()
    sheet_w, sheet_h = 200.0, 150.0
    scan_area = (sheet_w / 2.0, sheet_h / 2.0, sheet_w, sheet_h, 0.0)
    layout = module.Layout.fit(sheet_w, sheet_h, scan_area, margin_mm=15.0)
    assert layout.origin_x_mm == pytest.approx(-15.0)
    assert layout.origin_y_mm == pytest.approx(-15.0)
    assert layout.span_width_mm == pytest.approx(sheet_w + 30.0)
    assert layout.span_height_mm == pytest.approx(sheet_h + 30.0)


def test_layout_fit_expands_to_cover_a_scan_area_outside_the_sheet():
    module = _load_gui_module()
    sheet_w, sheet_h = 200.0, 150.0
    # Well outside the sheet's own (0,0)-(200,150) extent, and rotated --
    # exactly the case that used to get clipped.
    scan_area = (250.0, 100.0, 150.0, 100.0, 30.0)
    layout = module.Layout.fit(sheet_w, sheet_h, scan_area, margin_mm=15.0)

    for wx, wy in core.scan_area_corners(*scan_area):
        px, py = layout.ws2s(wx, wy)
        assert 0 <= px <= layout.win_w, f"corner ({wx},{wy}) -> x={px} outside [0,{layout.win_w}]"
        assert 0 <= py <= layout.win_h, f"corner ({wx},{wy}) -> y={py} outside [0,{layout.win_h}]"

    # the sheet's own corners must still fit too
    for wx, wy in [(0.0, 0.0), (sheet_w, 0.0), (sheet_w, sheet_h), (0.0, sheet_h)]:
        px, py = layout.ws2s(wx, wy)
        assert 0 <= px <= layout.win_w
        assert 0 <= py <= layout.win_h


def test_layout_ws2s_places_origin_at_the_margin_corner():
    module = _load_gui_module()
    scan_area = (100.0, 75.0, 200.0, 150.0, 0.0)
    layout = module.Layout.fit(200.0, 150.0, scan_area, margin_mm=15.0, ws_ox=0, ws_oy=0)
    # the bounding box's own bottom-left corner (origin_x_mm, origin_y_mm)
    # should map to screen (0, span_height*scale) -- i.e. the visual
    # bottom-left of the drawable area, since workspace +y is "up".
    px, py = layout.ws2s(layout.origin_x_mm, layout.origin_y_mm)
    assert px == pytest.approx(0, abs=1)
    assert py == pytest.approx(layout.span_height_mm * layout.scale, abs=1)


class _FakeClock:
    def tick(self, fps=0):
        return 0


class FakeServosStatic:
    def __init__(self, joint_ids):
        self.joint_ids = joint_ids

    def connect(self, port, baud=115200):
        pass

    def close(self):
        pass

    def set_torque_enabled(self, joint, enabled):
        pass

    def set_target_deg(self, joint, angle, speed=800, acc=0):
        pass

    def get_present_deg(self, joint):
        return 100.0 if joint == "joint1" else 175.0


def test_gui_main_runs_end_to_end_with_a_rotated_offsheet_scan_area(monkeypatch):
    # Regression test for the actual crash this was built to catch: a
    # full main() run (not just Layout.fit in isolation) with a scan area
    # that's rotated and extends outside the calibration sheet -- the
    # exact configuration a real fitted-and-saved scan_area_gui.py session
    # produces. Layout.fit alone passing doesn't guarantee every draw_*/
    # panel call site was updated to match its new field names.
    def fake_load_calib(path=None):
        calib = core._default_calib()
        calib["kinematics"] = {
            "L1": 125.0, "L2": 95.0, "base_x": 100.0, "base_y": -45.0,
            "servo1_offset_deg": 0.0, "servo2_offset_deg": 0.0,
            "servo1_dir": 1, "servo2_dir": 1, "elbow_offset_mm": 0.0, "fit_report": None,
        }
        calib["hardware"] = {"servo_port": "/dev/fake", "joint_ids": {"joint1": 1, "joint2": 2}}
        calib["motion"]["scan_center_x_mm"] = 250.0
        calib["motion"]["scan_center_y_mm"] = 100.0
        calib["motion"]["scan_width_mm"] = 150.0
        calib["motion"]["scan_height_mm"] = 100.0
        calib["motion"]["scan_rotation_deg"] = 30.0
        return calib

    monkeypatch.setattr(core, "load_calib", fake_load_calib)
    monkeypatch.setattr(hw, "Servos", FakeServosStatic)
    monkeypatch.setattr(pygame.time, "Clock", _FakeClock)

    call_count = {"n": 0}

    def fake_event_get(*args, **kwargs):
        call_count["n"] += 1
        if call_count["n"] == 10:
            return [pygame.event.Event(pygame.QUIT)]
        return []

    monkeypatch.setattr(pygame.event, "get", fake_event_get)

    _load_gui_module().main()  # would raise (as it did) if any draw_*/panel call site was stale
