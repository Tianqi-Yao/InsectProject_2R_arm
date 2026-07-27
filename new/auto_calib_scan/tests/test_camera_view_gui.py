"""Tests for auto_calib_scan/camera_view_gui.py: a mostly-pure conversion
function (_convert_tag_positions) tested directly with known affine
transforms, plus headless pygame end-to-end tests (fake hardware, SDL
dummy driver) confirming the tool runs, screenshots, and -- since this
tool fuses in manual_test/trace_boundary_gui.py's boundary trace/replay/
servo2_offset quick-fix -- releases torque at startup, records/saves/
replays a coupled boundary correctly, disables 'b'/'r'/'s' without
joint_limits_deg configured, and re-syncs+re-locks on exit. Same pattern
as tests/test_trace_boundary_gui.py and tests/test_card_gui.py.

Runs fast (no real per-frame delay) via the same two speedups
tests/test_trace_boundary_gui.py uses: pygame.time.Clock replaced with a
no-op, and time.monotonic() replaced with a synthetic clock that advances
a fixed step every call, so ENCODER_POLL_INTERVAL_S/CAMERA_POLL_INTERVAL_S
-gated code still fires deterministically without a real wall-clock wait.
"""

import importlib.util
import os
import sys
import time as time_module
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import numpy as np    # noqa: E402
import pygame          # noqa: E402
import pytest           # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import arm_core as ac       # noqa: E402
import arm_hardware as hw   # noqa: E402

GUI_PATH = Path(__file__).resolve().parent.parent / "camera_view_gui.py"

WORLD_CORNERS = [(0.0, 150.0), (200.0, 150.0), (200.0, 0.0), (0.0, 0.0)]
# Same 4-vertex loop tests/test_trace_boundary_gui.py uses for its own
# FakeServosWalkingLoop -- a square in (joint1, joint2) space.
LOOP = [(100.0, 150.0), (200.0, 150.0), (200.0, 200.0), (100.0, 200.0)]


def _affine_pixel(xy, scale=3.0, offset=(100.0, 50.0)):
    x, y = xy
    return (scale * x + offset[0], scale * y + offset[1])


def _make_H():
    pixels = [_affine_pixel(w) for w in WORLD_CORNERS]
    H, _ = ac.compute_homography(pixels, WORLD_CORNERS)
    return H


def _load_gui_module():
    spec = importlib.util.spec_from_file_location("camera_view_gui", GUI_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeDetection:
    def __init__(self, tag_id, center, corners=None):
        self.tag_id = tag_id
        self.center = center
        self.corners = corners or [center, center, center, center]


# ── _convert_tag_positions: pure logic, no pygame/hardware ──────────────

def test_convert_tag_positions_empty_without_homography():
    module = _load_gui_module()
    detections = {20: FakeDetection(20, (150.0, 200.0))}
    assert module._convert_tag_positions(detections, None) == {}


def test_convert_tag_positions_converts_every_tag_correctly():
    module = _load_gui_module()
    H = _make_H()
    world_points = {0: (0.0, 150.0), 10: (100.0, 75.0), 20: (150.0, 40.0)}
    detections = {tid: FakeDetection(tid, _affine_pixel(xy)) for tid, xy in world_points.items()}

    result = module._convert_tag_positions(detections, H)

    assert set(result.keys()) == set(world_points.keys())
    for tid, (ex, ey) in world_points.items():
        gx, gy = result[tid]
        assert gx == pytest.approx(ex, abs=1e-3)
        assert gy == pytest.approx(ey, abs=1e-3)


def test_convert_tag_positions_does_not_filter_unknown_tag_ids():
    # Tag ids that don't match any of calib.json's named tags (corner/ee/
    # card) must still come through -- this tool deliberately shows
    # everything detected, not just the ones calib.json happens to name.
    module = _load_gui_module()
    H = _make_H()
    detections = {
        0: FakeDetection(0, _affine_pixel((0.0, 150.0))),     # a known corner id
        999: FakeDetection(999, _affine_pixel((50.0, 50.0))),  # an unrecognized/stray id
    }
    result = module._convert_tag_positions(detections, H)
    assert set(result.keys()) == {0, 999}


# ── headless end-to-end (fake hardware) ─────────────────────────────────

class _FakeClock:
    def tick(self, fps=0):
        return 0


@pytest.fixture(autouse=True)
def _fast_and_clean(monkeypatch):
    # main() parses sys.argv itself (--elbow-ref-deg) -- without this it
    # chokes on pytest's own CLI args.
    monkeypatch.setattr(sys, "argv", ["camera_view_gui.py"])
    monkeypatch.setattr(pygame.time, "Clock", _FakeClock)
    fake_now = {"t": 0.0}

    def fake_monotonic():
        fake_now["t"] += 0.01
        return fake_now["t"]

    monkeypatch.setattr(time_module, "monotonic", fake_monotonic)
    screenshot_calls = []
    monkeypatch.setattr(pygame.image, "save",
                         lambda surface, path: screenshot_calls.append(path))
    return screenshot_calls


def _fake_calib(with_homography=True, joint_limits_deg=None, scan_area=None):
    """scan_area, if given, is (center_x, center_y, width, height,
    rotation_deg) written into calib["motion"] -- otherwise
    arm_core.calib_scan_area() falls back to the full (unrotated)
    calibration sheet, (100.0, 75.0, 200.0, 150.0, 0.0)."""
    def fake_load_calib(path=None):
        calib = ac._default_calib()
        calib["kinematics"] = {
            "L1": 125.0, "L2": 95.0, "base_x": 100.0, "base_y": -45.0,
            "servo1_offset_deg": 0.0, "servo2_offset_deg": 0.0,
            "servo1_dir": 1, "servo2_dir": 1, "elbow_offset_mm": 0.0, "fit_report": None,
        }
        calib["hardware"] = {"servo_port": "/dev/fake", "joint_ids": {"joint1": 1, "joint2": 2}}
        calib["joint_limits_deg"] = joint_limits_deg
        calib["card"] = {"tag_id": 20, "width_mm": 85.6, "height_mm": 54.0}
        if with_homography:
            calib["homography"]["H"] = _make_H().tolist()
        if scan_area is not None:
            cx, cy, w, h, rot = scan_area
            calib["motion"]["scan_center_x_mm"] = cx
            calib["motion"]["scan_center_y_mm"] = cy
            calib["motion"]["scan_width_mm"] = w
            calib["motion"]["scan_height_mm"] = h
            calib["motion"]["scan_rotation_deg"] = rot
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


class FakeServos:
    """Tracks every set_torque_enabled/set_target_deg call (in order) so
    tests can check startup/shutdown torque handling and replay's final
    commanded position. get_present_deg either reports a fixed pose
    (loop=None) or cycles through `loop`'s vertices one at a time -- joint1
    read first then joint2, matching camera_view_gui.py's own polling
    order, so both reads within one poll correspond to the SAME vertex --
    same convention tests/test_trace_boundary_gui.py's FakeServosWalkingLoop
    uses. ArmController.tick() drives purely off its own precomputed
    segment queue (never re-reading the servo), so what get_present_deg
    reports doesn't affect replay completion, only what gets *recorded*
    while tracing.

    IMPORTANT: when `loop` isn't given, get_present_deg mirrors whatever
    was last commanded (a well-behaved instant-tracking servo), NOT a
    fixed constant -- main()'s `finally` block calls _resync_and_relock,
    which reads get_present_deg and immediately re-commands that same
    value; if get_present_deg instead returned something unrelated to the
    real commanded position (e.g. a `loop` still cycling from unrelated
    display polling elsewhere in the run), that resync would silently
    clobber `commanded` with a bogus value right before the test asserts
    on it. `loop` is only for the boundary-recording test, which needs
    get_present_deg to visit distinct vertices over time regardless of
    anything commanded (torque is off during recording -- nothing is
    tracking a commanded target at all)."""

    def __init__(self, joint_ids, loop=None):
        self.joint_ids = joint_ids
        self.torque_calls = []
        self.commanded = {"joint1": 100.0, "joint2": 175.0}
        self._loop = loop
        self._idx = 0
        self._cur = None

    def connect(self, port, baud=115200):
        pass

    def close(self):
        pass

    def set_torque_enabled(self, joint, enabled):
        self.torque_calls.append((joint, enabled))

    def set_target_deg(self, joint, angle, speed=800, acc=0):
        self.commanded[joint] = angle

    def get_present_deg(self, joint):
        if self._loop is None:
            return self.commanded[joint]
        if self._cur is None:
            self._cur = self._loop[self._idx % len(self._loop)]
        val = self._cur[0] if joint == "joint1" else self._cur[1]
        if joint == "joint2":
            self._idx += 1
            self._cur = None
        return val


class FakeCamera:
    def __init__(self, resolution=(1920, 1080), backend="usb", usb_index=0):
        pass

    def connect(self):
        pass

    def close(self):
        pass

    def capture_gray(self):
        # Small real grayscale frame -- exercises the actual numpy->pygame
        # surface conversion path with real data, not a mock return value.
        return (np.arange(48 * 64, dtype=np.uint8).reshape(48, 64) % 255)


class FakeTagDetector:
    def __init__(self, detections):
        self._detections = detections

    def detect(self, frame):
        return self._detections


def _install_fake_hardware(monkeypatch, detections, loop=None):
    servos_holder = {}

    def make_servos(joint_ids):
        s = FakeServos(joint_ids, loop=loop)
        servos_holder["servos"] = s
        return s

    monkeypatch.setattr(hw, "Servos", make_servos)
    monkeypatch.setattr(hw, "Camera", FakeCamera)
    monkeypatch.setattr(hw, "TagDetector", lambda family="tag36h11": FakeTagDetector(detections))
    return servos_holder


def _scripted_event_get(script):
    call_count = {"n": 0}

    def fake_get(*args, **kwargs):
        call_count["n"] += 1
        return script.get(call_count["n"], [])

    return fake_get


def test_camera_view_gui_runs_without_homography(monkeypatch):
    monkeypatch.setattr(ac, "load_calib", _fake_calib(with_homography=False))
    detections = {10: FakeDetection(10, (200.0, 150.0))}
    _install_fake_hardware(monkeypatch, detections)

    script = {5: [pygame.event.Event(pygame.QUIT)]}
    monkeypatch.setattr(pygame.event, "get", _scripted_event_get(script))

    _load_gui_module().main()  # must not crash even with H=None


def test_camera_view_gui_screenshot(monkeypatch, _fast_and_clean):
    monkeypatch.setattr(ac, "load_calib", _fake_calib())
    _install_fake_hardware(monkeypatch, {})

    script = {
        5: [pygame.event.Event(pygame.KEYDOWN, key=pygame.K_p, mod=0)],
        8: [pygame.event.Event(pygame.QUIT)],
    }
    monkeypatch.setattr(pygame.event, "get", _scripted_event_get(script))

    _load_gui_module().main()

    assert "camera_view.png" in _fast_and_clean


# ── torque handling (this tool is no longer read-only) ──────────────────

def test_camera_view_gui_releases_torque_at_startup(monkeypatch):
    monkeypatch.setattr(ac, "load_calib", _fake_calib())
    holder = _install_fake_hardware(monkeypatch, {})

    script = {5: [pygame.event.Event(pygame.QUIT)]}
    monkeypatch.setattr(pygame.event, "get", _scripted_event_get(script))

    _load_gui_module().main()

    # The first two torque calls (before anything else happens) must be
    # both joints being released, ready for hand-tracing immediately --
    # even though _resync_and_relock re-enables torque again at exit
    # (checked separately below), so we look at the PREFIX of the call
    # history, not its final state.
    torque_calls = holder["servos"].torque_calls
    assert torque_calls[:2] == [("joint1", False), ("joint2", False)]


def test_camera_view_gui_resyncs_and_relocks_on_exit(monkeypatch):
    monkeypatch.setattr(ac, "load_calib", _fake_calib())
    holder = _install_fake_hardware(monkeypatch, {})

    script = {5: [pygame.event.Event(pygame.QUIT)]}
    monkeypatch.setattr(pygame.event, "get", _scripted_event_get(script))

    _load_gui_module().main()

    servos = holder["servos"]
    assert servos.torque_calls[-2:] == [("joint1", True), ("joint2", True)]
    assert servos.commanded["joint1"] == pytest.approx(100.0)
    assert servos.commanded["joint2"] == pytest.approx(175.0)


# ── boundary trace / save / replay ───────────────────────────────────────

def test_camera_view_gui_records_and_saves_boundary(monkeypatch):
    calib_holder = {}
    monkeypatch.setattr(ac, "load_calib", _fake_calib(
        joint_limits_deg={"joint1": [30.0, 239.0], "joint2": [32.0, 227.0], "coupled_boundary": []}))
    monkeypatch.setattr(ac, "save_calib", lambda calib, path=None: calib_holder.update(calib=calib))
    _install_fake_hardware(monkeypatch, {}, loop=LOOP)

    # Frame 5: start recording. Frame 60: stop. Frame 65: save.
    script = {
        5: [pygame.event.Event(pygame.KEYDOWN, key=pygame.K_b, mod=0)],
        60: [pygame.event.Event(pygame.KEYDOWN, key=pygame.K_b, mod=0)],
        65: [pygame.event.Event(pygame.KEYDOWN, key=pygame.K_s, mod=0)],
        70: [pygame.event.Event(pygame.QUIT)],
    }
    monkeypatch.setattr(pygame.event, "get", _scripted_event_get(script))

    _load_gui_module().main()

    assert "calib" in calib_holder, "save_calib was never called"
    boundary = calib_holder["calib"]["joint_limits_deg"]["coupled_boundary"]
    assert len(boundary) >= 3
    for vertex in boundary:
        assert (vertex["joint1"], vertex["joint2"]) in LOOP


def test_camera_view_gui_replay_drives_arm_back_to_first_vertex(monkeypatch):
    monkeypatch.setattr(ac, "load_calib", _fake_calib(
        joint_limits_deg={"joint1": [30.0, 239.0], "joint2": [32.0, 227.0],
                          "coupled_boundary": [{"joint1": j1, "joint2": j2} for j1, j2 in LOOP]}))
    # No `loop=` here -- get_present_deg mirrors whatever's commanded
    # (see FakeServos' docstring for why the recording-only cycling fake
    # would corrupt this test's final assertion via _resync_and_relock).
    holder = _install_fake_hardware(monkeypatch, {})

    script = {
        5: [pygame.event.Event(pygame.KEYDOWN, key=pygame.K_r, mod=0)],
        # Generous budget -- a 4-vertex boundary at jog speed needs several
        # hundred tick()s to fully drain every segment's planned queue.
        3000: [pygame.event.Event(pygame.QUIT)],
    }
    monkeypatch.setattr(pygame.event, "get", _scripted_event_get(script))

    _load_gui_module().main()

    servos = holder["servos"]
    assert servos.commanded["joint1"] == pytest.approx(LOOP[0][0])
    assert servos.commanded["joint2"] == pytest.approx(LOOP[0][1])


def test_camera_view_gui_boundary_screenshot_path(monkeypatch, _fast_and_clean):
    monkeypatch.setattr(ac, "load_calib", _fake_calib(
        joint_limits_deg={"joint1": [30.0, 239.0], "joint2": [32.0, 227.0], "coupled_boundary": []}))
    monkeypatch.setattr(ac, "save_calib", lambda calib, path=None: None)
    _install_fake_hardware(monkeypatch, {}, loop=LOOP)

    script = {
        5: [pygame.event.Event(pygame.KEYDOWN, key=pygame.K_b, mod=0)],
        60: [pygame.event.Event(pygame.KEYDOWN, key=pygame.K_b, mod=0)],
        65: [pygame.event.Event(pygame.KEYDOWN, key=pygame.K_s, mod=0)],
        70: [pygame.event.Event(pygame.QUIT)],
    }
    monkeypatch.setattr(pygame.event, "get", _scripted_event_get(script))

    _load_gui_module().main()

    assert "joint_limits_trace.png" in _fast_and_clean


def test_camera_view_gui_boundary_keys_disabled_without_joint_limits(monkeypatch):
    calib_holder = {}
    monkeypatch.setattr(ac, "load_calib", _fake_calib(joint_limits_deg=None))
    monkeypatch.setattr(ac, "save_calib", lambda calib, path=None: calib_holder.update(calib=calib))
    _install_fake_hardware(monkeypatch, {}, loop=LOOP)

    script = {
        5: [pygame.event.Event(pygame.KEYDOWN, key=pygame.K_b, mod=0)],
        60: [pygame.event.Event(pygame.KEYDOWN, key=pygame.K_b, mod=0)],
        65: [pygame.event.Event(pygame.KEYDOWN, key=pygame.K_s, mod=0)],
        70: [pygame.event.Event(pygame.KEYDOWN, key=pygame.K_r, mod=0)],
        75: [pygame.event.Event(pygame.QUIT)],
    }
    monkeypatch.setattr(pygame.event, "get", _scripted_event_get(script))

    _load_gui_module().main()

    assert "calib" not in calib_holder, \
        "b/s must be refused (no save) when joint_limits_deg isn't configured"


def test_camera_view_gui_k_key_fixes_servo2_offset_without_joint_limits(monkeypatch):
    calib_holder = {}
    monkeypatch.setattr(ac, "load_calib", _fake_calib(joint_limits_deg=None))
    monkeypatch.setattr(ac, "save_calib", lambda calib, path=None: calib_holder.update(calib=calib))
    _install_fake_hardware(monkeypatch, {})

    script = {
        5: [pygame.event.Event(pygame.KEYDOWN, key=pygame.K_k, mod=0)],
        10: [pygame.event.Event(pygame.QUIT)],
    }
    monkeypatch.setattr(pygame.event, "get", _scripted_event_get(script))

    _load_gui_module().main()

    assert "calib" in calib_holder, "'k' must work even without joint_limits_deg configured"
    assert calib_holder["calib"]["kinematics"]["servo2_offset_deg"] != 0.0


# ── lock/jog mode + manual card-corner teaching ─────────────────────────

def test_camera_view_gui_j_toggles_lock_state(monkeypatch):
    monkeypatch.setattr(ac, "load_calib", _fake_calib())
    holder = _install_fake_hardware(monkeypatch, {})

    script = {
        5: [pygame.event.Event(pygame.KEYDOWN, key=pygame.K_j, mod=0)],   # lock
        10: [pygame.event.Event(pygame.KEYDOWN, key=pygame.K_j, mod=0)],  # unlock
        15: [pygame.event.Event(pygame.QUIT)],
    }
    monkeypatch.setattr(pygame.event, "get", _scripted_event_get(script))

    _load_gui_module().main()

    # startup unlock -> lock -> unlock -> exit resync's re-lock.
    assert holder["servos"].torque_calls == [
        ("joint1", False), ("joint2", False),
        ("joint1", True), ("joint2", True),
        ("joint1", False), ("joint2", False),
        ("joint1", True), ("joint2", True),
    ]


def test_camera_view_gui_j_disabled_while_recording(monkeypatch):
    monkeypatch.setattr(ac, "load_calib", _fake_calib(
        joint_limits_deg={"joint1": [30.0, 239.0], "joint2": [32.0, 227.0], "coupled_boundary": []}))
    holder = _install_fake_hardware(monkeypatch, {}, loop=LOOP)

    script = {
        5: [pygame.event.Event(pygame.KEYDOWN, key=pygame.K_b, mod=0)],   # start recording
        10: [pygame.event.Event(pygame.KEYDOWN, key=pygame.K_j, mod=0)],  # refused: recording
        15: [pygame.event.Event(pygame.QUIT)],
    }
    monkeypatch.setattr(pygame.event, "get", _scripted_event_get(script))

    _load_gui_module().main()

    # startup unlock, 'b' re-affirming unlock, exit resync's re-lock --
    # no extra True/True pair from 'j' in between.
    assert holder["servos"].torque_calls == [
        ("joint1", False), ("joint2", False),
        ("joint1", False), ("joint2", False),
        ("joint1", True), ("joint2", True),
    ]


def test_camera_view_gui_arrow_keys_only_jog_while_locked(monkeypatch):
    import jog_controller as jc
    monkeypatch.setattr(ac, "load_calib", _fake_calib())
    _install_fake_hardware(monkeypatch, {})

    nudges = []
    original_nudge = jc.ArmController.nudge_workspace

    def traced_nudge(self, dx, dy, base):
        nudges.append((dx, dy))
        return original_nudge(self, dx, dy, base)

    monkeypatch.setattr(jc.ArmController, "nudge_workspace", traced_nudge)

    script = {
        5: [pygame.event.Event(pygame.KEYDOWN, key=pygame.K_UP, mod=0)],   # unlocked -- ignored
        10: [pygame.event.Event(pygame.KEYDOWN, key=pygame.K_j, mod=0)],   # lock
        15: [pygame.event.Event(pygame.KEYDOWN, key=pygame.K_UP, mod=0)],  # locked -- should nudge
        20: [pygame.event.Event(pygame.QUIT)],
    }
    monkeypatch.setattr(pygame.event, "get", _scripted_event_get(script))

    _load_gui_module().main()

    assert len(nudges) == 1, "arrow keys must only jog while locked"
    dx, dy = nudges[0]
    assert dx == pytest.approx(0.0, abs=1e-6)
    assert dy > 0


def test_camera_view_gui_jog_actually_moves_the_servos(monkeypatch):
    # nudge_workspace only PLANS a segment (jog_controller.ArmController.
    # set_joint_goal) -- something still has to call controller.tick() every
    # frame to actually drain that plan into set_target_deg calls. Catches
    # the bug where tick() was only invoked `if replaying:`, so locking +
    # jogging queued a plan that never executed and the arm never moved.
    monkeypatch.setattr(ac, "load_calib", _fake_calib())
    servos_holder = _install_fake_hardware(monkeypatch, {})

    script = {
        5: [pygame.event.Event(pygame.KEYDOWN, key=pygame.K_j, mod=0)],   # lock
        10: [pygame.event.Event(pygame.KEYDOWN, key=pygame.K_UP, mod=0)],  # nudge +y
        300: [pygame.event.Event(pygame.QUIT)],  # plenty of frames for tick() to drain the plan
    }
    monkeypatch.setattr(pygame.event, "get", _scripted_event_get(script))

    _load_gui_module().main()

    servos = servos_holder["servos"]
    assert servos.commanded["joint1"] != pytest.approx(100.0) or \
        servos.commanded["joint2"] != pytest.approx(175.0), \
        "jogging while locked must actually command the servos, not just plan a move"


def test_camera_view_gui_jog_follows_scan_area_rotation(monkeypatch):
    import jog_controller as jc
    rotated_scan_area = (100.0, -45.0, 400.0, 400.0, 90.0)
    monkeypatch.setattr(ac, "load_calib", _fake_calib(scan_area=rotated_scan_area))
    _install_fake_hardware(monkeypatch, {})

    nudges = []
    original_nudge = jc.ArmController.nudge_workspace

    def traced_nudge(self, dx, dy, base):
        nudges.append((dx, dy))
        return original_nudge(self, dx, dy, base)

    monkeypatch.setattr(jc.ArmController, "nudge_workspace", traced_nudge)

    script = {
        5: [pygame.event.Event(pygame.KEYDOWN, key=pygame.K_j, mod=0)],   # lock
        10: [pygame.event.Event(pygame.KEYDOWN, key=pygame.K_UP, mod=0)],
        15: [pygame.event.Event(pygame.QUIT)],
    }
    monkeypatch.setattr(pygame.event, "get", _scripted_event_get(script))

    _load_gui_module().main()

    assert len(nudges) == 1
    dx, dy = nudges[0]
    # A 90deg-rotated scan area turns "up" into world -x (same convention
    # fixed_path_scan/path_gui.py's arrow keys already follow).
    assert dx < -1e-6
    assert dy == pytest.approx(0.0, abs=1e-6)


def test_camera_view_gui_1_2_ignored_when_not_locked(monkeypatch):
    calib_holder = {}
    monkeypatch.setattr(ac, "load_calib", _fake_calib())
    monkeypatch.setattr(ac, "save_calib", lambda calib, path=None: calib_holder.update(calib=calib))
    _install_fake_hardware(monkeypatch, {})

    script = {
        5: [pygame.event.Event(pygame.KEYDOWN, key=pygame.K_1, mod=0)],  # unlocked -- ignored
        10: [pygame.event.Event(pygame.KEYDOWN, key=pygame.K_2, mod=0)],  # unlocked -- ignored
        15: [pygame.event.Event(pygame.KEYDOWN, key=pygame.K_m, mod=0)],  # nothing recorded -> no save
        20: [pygame.event.Event(pygame.QUIT)],
    }
    monkeypatch.setattr(pygame.event, "get", _scripted_event_get(script))

    _load_gui_module().main()

    assert "calib" not in calib_holder, "'1'/'2' must be no-ops while unlocked"


def test_camera_view_gui_m_saves_manual_corners_recorded_while_locked(monkeypatch):
    calib_holder = {}
    monkeypatch.setattr(ac, "load_calib", _fake_calib())
    monkeypatch.setattr(ac, "save_calib", lambda calib, path=None: calib_holder.update(calib=calib))
    _install_fake_hardware(monkeypatch, {})

    module = _load_gui_module()
    params = _default_params()
    initial_target = ac.fk_from_servo_angles(params, 100.0, 175.0)
    expected_a = (initial_target[0], initial_target[1] + module.STEP_MM)
    expected_b = (expected_a[0] + module.STEP_MM, expected_a[1])

    script = {
        5: [pygame.event.Event(pygame.KEYDOWN, key=pygame.K_j, mod=0)],      # lock
        6: [pygame.event.Event(pygame.KEYDOWN, key=pygame.K_UP, mod=0)],     # nudge +y
        7: [pygame.event.Event(pygame.KEYDOWN, key=pygame.K_1, mod=0)],      # record corner A
        8: [pygame.event.Event(pygame.KEYDOWN, key=pygame.K_RIGHT, mod=0)],  # nudge +x
        9: [pygame.event.Event(pygame.KEYDOWN, key=pygame.K_2, mod=0)],      # record corner B
        10: [pygame.event.Event(pygame.KEYDOWN, key=pygame.K_m, mod=0)],     # save
        13: [pygame.event.Event(pygame.QUIT)],
    }
    monkeypatch.setattr(pygame.event, "get", _scripted_event_get(script))

    module.main()

    assert "calib" in calib_holder, "save_calib was never called"
    a = calib_holder["calib"]["card"]["manual_corner_a_mm"]
    b = calib_holder["calib"]["card"]["manual_corner_b_mm"]
    assert a == pytest.approx(list(expected_a))
    assert b == pytest.approx(list(expected_b))


def test_camera_view_gui_m_refuses_to_save_incomplete_corners(monkeypatch):
    calib_holder = {}
    monkeypatch.setattr(ac, "load_calib", _fake_calib())
    monkeypatch.setattr(ac, "save_calib", lambda calib, path=None: calib_holder.update(calib=calib))
    _install_fake_hardware(monkeypatch, {})

    script = {
        5: [pygame.event.Event(pygame.KEYDOWN, key=pygame.K_j, mod=0)],  # lock
        6: [pygame.event.Event(pygame.KEYDOWN, key=pygame.K_1, mod=0)],  # only corner A recorded
        7: [pygame.event.Event(pygame.KEYDOWN, key=pygame.K_m, mod=0)],  # refused: B missing
        10: [pygame.event.Event(pygame.QUIT)],
    }
    monkeypatch.setattr(pygame.event, "get", _scripted_event_get(script))

    _load_gui_module().main()

    assert "calib" not in calib_holder, "'m' must refuse to save with only one corner recorded"
