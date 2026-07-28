"""Tests for control_gui.py's pure-logic helpers -- the PARAMS-mode field
model, text-input widget, and dataclass copy helper. None of this needs a
real display: pygame.init()/display.set_mode() only happen inside main(),
never at import time or in these helpers."""

import sys
from pathlib import Path
from types import SimpleNamespace

import pygame
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import arm_core as core  # noqa: E402
import control_gui as cg  # noqa: E402


# ── Field / _attr_field / _adjust_field / _format_field ──────────────────

def test_attr_field_get_set_round_trip():
    params = core.ArmParams.nominal()
    f = cg._attr_field(params, "L1", "kinematics", "L1 (mm)", step=1.0)
    assert f.get() == params.L1
    f.set(150.0)
    assert params.L1 == 150.0


def test_adjust_field_float_and_int_and_sign():
    params = core.ArmParams.nominal()
    f_float = cg._attr_field(params, "L1", "kinematics", "L1", step=2.0)
    cg._adjust_field(f_float, +1)
    assert params.L1 == pytest.approx(core.ArmParams.nominal().L1 + 2.0)
    cg._adjust_field(f_float, -1)
    assert params.L1 == pytest.approx(core.ArmParams.nominal().L1)

    f_sign = cg.Field("kinematics", "servo1_dir", "sign",
                       get=lambda: params.servo1_dir, set=lambda v: setattr(params, "servo1_dir", v))
    assert params.servo1_dir == 1
    cg._adjust_field(f_sign, +1)  # direction is ignored for "sign" -- it always flips
    assert params.servo1_dir == -1
    cg._adjust_field(f_sign, -1)
    assert params.servo1_dir == 1

    hw_cfg = core.HardwareConfig()
    f_int = cg.Field("hardware*", "joint1 id", "int", step=1,
                      get=lambda: hw_cfg.joint_ids["joint1"],
                      set=lambda v: hw_cfg.joint_ids.__setitem__("joint1", int(v)))
    cg._adjust_field(f_int, +1)
    assert hw_cfg.joint_ids["joint1"] == 2


def test_adjust_field_text_is_a_noop():
    photo_cfg = core.PhotoConfig()
    f = cg.Field("photo", "photo_dir", "text",
                 get=lambda: photo_cfg.photo_dir, set=lambda v: setattr(photo_cfg, "photo_dir", v))
    cg._adjust_field(f, +1)
    assert photo_cfg.photo_dir == "photos"  # unchanged


def test_format_field_variants():
    params = core.ArmParams.nominal()
    f_float = cg._attr_field(params, "L1", "kinematics", "L1", step=1.0, fmt="{:.2f}")
    assert cg._format_field(f_float) == f"{params.L1:.2f}"

    f_sign_pos = cg.Field("k", "dir", "sign", get=lambda: 1, set=lambda v: None)
    f_sign_neg = cg.Field("k", "dir", "sign", get=lambda: -1, set=lambda v: None)
    assert cg._format_field(f_sign_pos) == "+1"
    assert cg._format_field(f_sign_neg) == "-1"

    f_text = cg.Field("p", "photo_dir", "text", get=lambda: "photos", set=lambda v: None)
    assert cg._format_field(f_text) == "photos"


def test_build_fields_includes_joint_limits_only_when_configured():
    params = core.ArmParams.nominal()
    motion_cfg = core.MotionConfig()
    photo_cfg = core.PhotoConfig()
    hw_cfg = core.HardwareConfig()

    calib_no_limits = core._default_calib()
    fields = cg.build_fields(calib_no_limits, params, motion_cfg, photo_cfg, hw_cfg)
    assert not any(f.section == "joint_limits" for f in fields)

    calib_with_limits = core._default_calib()
    calib_with_limits["joint_limits_deg"] = {"joint1": [0.0, 180.0], "joint2": [0.0, 220.0],
                                              "coupled_boundary": []}
    fields2 = cg.build_fields(calib_with_limits, params, motion_cfg, photo_cfg, hw_cfg)
    jl_fields = [f for f in fields2 if f.section == "joint_limits"]
    assert len(jl_fields) == 4
    # Editing a joint_limits field mutates the underlying calib dict directly.
    lo_field = next(f for f in jl_fields if f.label == "joint1 lo (deg)")
    lo_field.set(10.0)
    assert calib_with_limits["joint_limits_deg"]["joint1"][0] == 10.0


def test_build_fields_editing_does_not_affect_other_joint_independently():
    calib = core._default_calib()
    calib["joint_limits_deg"] = {"joint1": [0.0, 180.0], "joint2": [0.0, 220.0],
                                  "coupled_boundary": []}
    fields = cg.build_fields(calib, core.ArmParams.nominal(), core.MotionConfig(),
                              core.PhotoConfig(), core.HardwareConfig())
    j1_hi = next(f for f in fields if f.label == "joint1 hi (deg)")
    j1_hi.set(170.0)
    assert calib["joint_limits_deg"]["joint1"] == [0.0, 170.0]
    assert calib["joint_limits_deg"]["joint2"] == [0.0, 220.0]


# ── _copy_dataclass_fields ────────────────────────────────────────────────

def test_copy_dataclass_fields():
    src = core.ArmParams(L1=99.0, L2=88.0, base_x=1.0, base_y=2.0,
                          servo1_offset_deg=3.0, servo2_offset_deg=4.0,
                          servo1_dir=-1, servo2_dir=-1, elbow_offset_mm=5.0)
    dst = core.ArmParams.nominal()
    cg._copy_dataclass_fields(dst, src)
    assert dst == src
    assert dst is not src  # copied field-by-field, not the same object


# ── TextInput ──────────────────────────────────────────────────────────

def _key_event(key, unicode=""):
    return SimpleNamespace(key=key, unicode=unicode)


def test_text_input_types_and_confirms():
    ti = cg.TextInput("name", initial="")
    for ch in "abc":
        ti.handle_key(_key_event(pygame.K_a, unicode=ch))
    assert ti.buffer == "abc"
    assert ti.active
    ti.handle_key(_key_event(pygame.K_RETURN))
    assert not ti.active
    assert ti.result == "abc"


def test_text_input_backspace():
    ti = cg.TextInput("name", initial="abc")
    ti.handle_key(_key_event(pygame.K_BACKSPACE))
    assert ti.buffer == "ab"


def test_text_input_escape_cancels_with_none_result():
    ti = cg.TextInput("name", initial="abc")
    ti.handle_key(_key_event(pygame.K_ESCAPE))
    assert not ti.active
    assert ti.result is None


# ── _toggle_torque: resync-on-lock, reset-jog-target-on-any-transition ───

class FakeTorqueServos:
    def __init__(self):
        self.enabled = {"joint1": True, "joint2": True}
        self.present = {"joint1": 10.0, "joint2": 20.0}
        self.target_calls = []

    def get_present_deg(self, joint):
        return self.present[joint]

    def set_target_deg(self, joint, angle_deg, speed=800, acc=0):
        self.target_calls.append((joint, angle_deg))

    def set_torque_enabled(self, joint, enabled):
        self.enabled[joint] = enabled


class FakeResyncController:
    def __init__(self):
        self.resync_calls = []

    def resync(self, j1_deg, j2_deg):
        self.resync_calls.append((j1_deg, j2_deg))


def test_toggle_torque_resyncs_controller_when_locking():
    servos = FakeTorqueServos()
    controller = FakeResyncController()
    torque = {"joint1": False, "joint2": True}
    state = cg.AppState()
    state.jog_target = (1.0, 2.0)  # stale, from before the transition

    cg._toggle_torque(servos, controller, torque, state, "joint1")

    assert torque["joint1"] is True
    assert servos.enabled["joint1"] is True
    # resync's whole point is closing the gap between "the controller's
    # remembered position" and "reality" -- verify it's called with the
    # servo's REAL present angles, not anything derived from jog_target.
    assert controller.resync_calls == [(10.0, 20.0)]
    assert state.jog_target is None


def test_toggle_torque_does_not_resync_when_releasing():
    servos = FakeTorqueServos()
    controller = FakeResyncController()
    torque = {"joint1": True, "joint2": True}
    state = cg.AppState()

    cg._toggle_torque(servos, controller, torque, state, "joint1")

    assert torque["joint1"] is False
    assert servos.enabled["joint1"] is False
    assert controller.resync_calls == []
    assert state.jog_target is None  # still reset, even on release


# ── _sync_calib_from_live ─────────────────────────────────────────────────

def test_sync_calib_from_live_overwrites_stale_disk_values():
    calib = core._default_calib()
    calib["kinematics"]["L1"] = 1.0  # stale -- as if loaded before a live edit
    params = core.ArmParams.nominal()
    params.L1 = 999.0
    ctx = cg.Ctx(calib=calib, params=params, motion_cfg=core.MotionConfig(),
                 photo_cfg=core.PhotoConfig(), hw_cfg=core.HardwareConfig(),
                 servos=None, camera=None, controller=None, torque={}, paths={})

    cg._sync_calib_from_live(ctx)

    assert calib["kinematics"]["L1"] == 999.0


# ── _poll_held_repeat: continuous jog/nav without pygame's global repeat ──

class _FakePressed:
    def __init__(self, held_key=None):
        self.held_key = held_key

    def __getitem__(self, key):
        return key == self.held_key


def test_poll_held_repeat_fires_after_delay_then_at_interval(monkeypatch):
    calls = []
    monkeypatch.setattr(cg, "_handle_jog_key", lambda event, ctx, state: calls.append(event.key))
    monkeypatch.setattr(pygame.key, "get_pressed", lambda: _FakePressed(pygame.K_UP))

    state = cg.AppState(mode=cg.MODE_JOG)
    ctx = object()  # never touched -- _handle_jog_key is stubbed above

    cg._poll_held_repeat(ctx, state, now=0.0)
    assert calls == []  # first frame only registers the hold
    assert state.repeat_key == pygame.K_UP

    cg._poll_held_repeat(ctx, state, now=0.1)  # before REPEAT_DELAY_S
    assert calls == []

    cg._poll_held_repeat(ctx, state, now=cg.REPEAT_DELAY_S + 0.01)
    assert calls == [pygame.K_UP]

    cg._poll_held_repeat(ctx, state, now=cg.REPEAT_DELAY_S + 0.02)  # interval not elapsed
    assert calls == [pygame.K_UP]

    cg._poll_held_repeat(ctx, state, now=cg.REPEAT_DELAY_S + cg.REPEAT_INTERVAL_S + 0.02)
    assert calls == [pygame.K_UP, pygame.K_UP]


def test_poll_held_repeat_resets_when_key_released(monkeypatch):
    monkeypatch.setattr(cg, "_handle_jog_key", lambda event, ctx, state: None)
    monkeypatch.setattr(pygame.key, "get_pressed", lambda: _FakePressed(None))

    state = cg.AppState(mode=cg.MODE_JOG, repeat_key=pygame.K_UP, repeat_next_at=999.0)
    cg._poll_held_repeat(object(), state, now=0.0)
    assert state.repeat_key is None


def test_poll_held_repeat_ignores_non_repeatable_modes(monkeypatch):
    fired = []
    monkeypatch.setattr(cg, "_handle_jog_key", lambda event, ctx, state: fired.append(True))
    monkeypatch.setattr(pygame.key, "get_pressed", lambda: _FakePressed(pygame.K_UP))

    state = cg.AppState(mode=cg.MODE_TEACH)  # not JOG or PARAMS
    cg._poll_held_repeat(object(), state, now=0.0)
    assert state.repeat_key is None
    assert fired == []


def test_poll_held_repeat_routes_backspace_while_text_input_active():
    state = cg.AppState(mode=cg.MODE_JOG)
    state.text_input = cg.TextInput("name", initial="abcd")
    import pygame as _pg

    class _FakePressedBackspace:
        def __getitem__(self, key):
            return key == _pg.K_BACKSPACE

    orig = _pg.key.get_pressed
    _pg.key.get_pressed = lambda: _FakePressedBackspace()
    try:
        cg._poll_held_repeat(object(), state, now=0.0)
        assert state.text_input.buffer == "abcd"  # first frame only registers the hold
        cg._poll_held_repeat(object(), state, now=cg.REPEAT_DELAY_S + 0.01)
        assert state.text_input.buffer == "abc"
        cg._poll_held_repeat(object(), state, now=cg.REPEAT_DELAY_S + cg.REPEAT_INTERVAL_S + 0.02)
        assert state.text_input.buffer == "ab"
    finally:
        _pg.key.get_pressed = orig


# ── PATH mode rename (via _apply_text_input) ──────────────────────────────

def test_rename_path_via_text_input(monkeypatch, tmp_path):
    import path_core as pc
    # _apply_text_input's rename branch persists via pc.save_paths(), which
    # defaults to the real v3/paths.json -- redirect it so this test can't
    # write into the actual project directory.
    monkeypatch.setattr(pc, "DEFAULT_PATHS_PATH", tmp_path / "paths.json")

    ctx = cg.Ctx(calib=core._default_calib(), params=core.ArmParams.nominal(),
                 motion_cfg=core.MotionConfig(), photo_cfg=core.PhotoConfig(),
                 hw_cfg=core.HardwareConfig(), servos=None, camera=None, controller=None,
                 torque={}, paths={"old": [(1.0, 2.0)]})
    state = cg.AppState()
    state.path_names = ["old"]
    state.text_input = cg.TextInput("rename to", initial="old")
    state.text_input.buffer = "new"
    state.text_input.result = "new"
    state.text_input.active = False
    state.text_input_purpose = "rename_path"
    state.rename_old_name = "old"

    cg._apply_text_input(ctx, state)

    assert "old" not in ctx.paths
    assert ctx.paths["new"] == [(1.0, 2.0)]
    assert state.path_names == ["new"]
    assert state.selected_path_idx == 0
    assert state.text_input is None
    assert state.rename_old_name is None
