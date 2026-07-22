"""End-to-end tests of manual_test/trace_boundary_gui.py: fake hardware (no
real arm) + a synthetic pygame event stream (no real display, via the SDL
dummy driver) drive the actual tool code -- recording a boundary by
"walking" a known square in (joint1, joint2) space, saving it, and
replaying it back -- exactly like a person would with the keyboard.

Complements test_arm_core.py's pure-logic tests: those check
_point_in_polygon/within_joint_limits/servo2_offset_from_known_elbow_angle
in isolation, this exercises the actual capture/save/replay code path
inside the tool itself -- the same one a real bring-up session runs.

Runs fast (no real per-frame delay) via two speedups that don't change
what's being tested, only how long it takes:
  - pygame.time.Clock is replaced with a no-op version, since the FPS cap
    is a real-time wall-clock throttle with nothing to do with logic
    correctness.
  - time.monotonic() is replaced with a synthetic clock that advances a
    fixed step every call, so POLL_INTERVAL_S-gated code (encoder
    polling, save-message timeouts) still fires deterministically, just
    without actually waiting on a real clock.
"""

import importlib.util
import os
import sys
import time as time_module
from pathlib import Path

# Must be set before pygame is imported/initialized -- runs the whole
# test headless, no real window needed.
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame  # noqa: E402
import pytest  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import arm_core as ac       # noqa: E402
import arm_hardware as hw   # noqa: E402

GUI_PATH = Path(__file__).resolve().parent.parent / "manual_test" / "trace_boundary_gui.py"

# A simple closed square loop in (joint1, joint2) space -- the shape being
# "traced". Center (150, 175) is inside it, (50, 175) is outside.
LOOP = [(100.0, 150.0), (200.0, 150.0), (200.0, 200.0), (100.0, 200.0)]
INSIDE_POINT = (150.0, 175.0)
OUTSIDE_POINT = (50.0, 175.0)


class _FakeClock:
    def tick(self, fps=0):
        return 0


@pytest.fixture(autouse=True)
def _fast_and_clean(monkeypatch):
    # main() parses sys.argv itself (it's a CLI script) -- without this,
    # it would choke on pytest's own CLI args (file paths, -v, ...).
    monkeypatch.setattr(sys, "argv", ["trace_boundary_gui.py"])
    monkeypatch.setattr(pygame.time, "Clock", _FakeClock)
    fake_now = {"t": 0.0}

    def fake_monotonic():
        fake_now["t"] += 0.01
        return fake_now["t"]

    monkeypatch.setattr(time_module, "monotonic", fake_monotonic)

    # A real pygame.image.save() would write joint_limits_trace.png into
    # whatever the test runner's cwd happens to be (this repo, normally --
    # exactly the kind of test pollution this project avoids elsewhere).
    # Record calls instead of touching disk; tests that care about this can
    # inspect `screenshot_calls`.
    screenshot_calls = []
    monkeypatch.setattr(pygame.image, "save",
                         lambda surface, path: screenshot_calls.append(path))
    return screenshot_calls


def _load_gui_module():
    spec = importlib.util.spec_from_file_location("trace_boundary_gui", GUI_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fake_calib(joint1_range, joint2_range):
    def fake_load_calib(path=None):
        calib = ac._default_calib()
        calib["kinematics"] = {
            "L1": 125.0, "L2": 95.0, "base_x": 100.0, "base_y": -45.0,
            "servo1_offset_deg": 23.08, "servo2_offset_deg": 0.0,
            "servo1_dir": 1, "servo2_dir": 1, "elbow_offset_mm": 0.0, "fit_report": None,
        }
        calib["hardware"] = {"servo_port": "/dev/fake", "joint_ids": {"joint1": 1, "joint2": 2}}
        calib["joint_limits_deg"] = {
            "joint1": list(joint1_range), "joint2": list(joint2_range), "coupled_boundary": [],
        }
        return calib
    return fake_load_calib


class FakeServosWalkingLoop:
    """Cycles through `loop`'s vertices, one vertex per (joint1, joint2)
    poll pair -- joint1 is always read first then joint2 (matching
    trace_boundary_gui.py's polling order), so both reads within one poll
    correspond to the SAME vertex, only advancing to the next vertex once
    both have been read. Also records every set_target_deg call, so a
    replay's final commanded position can be checked.

    get_present_deg() cycling through the loop during replay is harmless
    (unused for correctness there): ArmController.tick() drives purely off
    its own precomputed segment queue, never re-reading the servo, so
    replay completion depends only on how many frames elapse, not on what
    this fake reports as the "real" position.
    """

    def __init__(self, joint_ids, loop):
        self.joint_ids = joint_ids
        self._loop = loop
        self._idx = 0
        self._cur = None
        self.commanded = {"joint1": loop[0][0], "joint2": loop[0][1]}

    def connect(self, port, baud=115200):
        pass

    def close(self):
        pass

    def set_torque_enabled(self, joint, enabled):
        pass

    def set_target_deg(self, joint, angle, speed=800, acc=0):
        self.commanded[joint] = angle

    def get_present_deg(self, joint):
        if self._cur is None:
            self._cur = self._loop[self._idx % len(self._loop)]
        val = self._cur[0] if joint == "joint1" else self._cur[1]
        if joint == "joint2":
            self._idx += 1
            self._cur = None
        return val


def _scripted_event_get(script):
    """Returns a fake pygame.event.get() that yields `script[frame]` (a
    list of events, possibly empty) for each successive call, indexed by
    call count starting at 1."""
    call_count = {"n": 0}

    def fake_get(*args, **kwargs):
        call_count["n"] += 1
        return script.get(call_count["n"], [])

    return fake_get


def test_trace_boundary_gui_records_one_loop_and_saves_usable_boundary(monkeypatch, _fast_and_clean):
    calib_holder = {}
    monkeypatch.setattr(ac, "load_calib", _fake_calib((30.0, 239.0), (32.0, 227.0)))
    monkeypatch.setattr(ac, "save_calib", lambda calib, path=None: calib_holder.update(calib=calib))
    monkeypatch.setattr(hw, "Servos", lambda joint_ids: FakeServosWalkingLoop(joint_ids, LOOP))

    # Frame 5: start recording ('b'). Frame 60: stop ('b' again) -- by then
    # the fake arm has cycled through the 4-vertex loop many times over.
    # Frame 65: save ('s'). Frame 70: quit.
    script = {
        5: [pygame.event.Event(pygame.KEYDOWN, key=pygame.K_b, mod=0)],
        60: [pygame.event.Event(pygame.KEYDOWN, key=pygame.K_b, mod=0)],
        65: [pygame.event.Event(pygame.KEYDOWN, key=pygame.K_s, mod=0)],
        70: [pygame.event.Event(pygame.QUIT)],
    }
    monkeypatch.setattr(pygame.event, "get", _scripted_event_get(script))

    _load_gui_module().main()

    assert "calib" in calib_holder, "save_calib was never called"
    joint_limits = calib_holder["calib"]["joint_limits_deg"]
    boundary = joint_limits["coupled_boundary"]

    # Saving also writes a screenshot of the traced boundary (see
    # trace_boundary_gui.SCREENSHOT_PATH).
    assert _fast_and_clean == ["joint_limits_trace.png"]

    # Recorded and saved verbatim: every vertex is one of the loop's own
    # corners, no derived/averaged values snuck in.
    assert len(boundary) >= 3
    for vertex in boundary:
        assert (vertex["joint1"], vertex["joint2"]) in LOOP

    # The independent joint1/joint2 ranges from set-joint-limits already
    # cover the whole loop here, so they shouldn't have needed widening.
    assert joint_limits["joint1"] == [30.0, 239.0]
    assert joint_limits["joint2"] == [32.0, 227.0]

    # And the saved data is directly usable by within_joint_limits.
    parsed = ac.calib_joint_limits(calib_holder["calib"])
    assert ac.within_joint_limits(*INSIDE_POINT, parsed)
    assert not ac.within_joint_limits(*OUTSIDE_POINT, parsed)


def test_trace_boundary_gui_widens_independent_ranges_to_include_loop(monkeypatch):
    # This time the independent joint1/joint2 ranges measured earlier are
    # narrower than the traced loop -- the tool must widen them to the
    # union, since the hardware registers can only hold one fixed range.
    calib_holder = {}
    # Narrower than LOOP's own extent (100-200, 150-200).
    monkeypatch.setattr(ac, "load_calib", _fake_calib((120.0, 180.0), (160.0, 190.0)))
    monkeypatch.setattr(ac, "save_calib", lambda calib, path=None: calib_holder.update(calib=calib))
    monkeypatch.setattr(hw, "Servos", lambda joint_ids: FakeServosWalkingLoop(joint_ids, LOOP))

    script = {
        5: [pygame.event.Event(pygame.KEYDOWN, key=pygame.K_b, mod=0)],
        60: [pygame.event.Event(pygame.KEYDOWN, key=pygame.K_b, mod=0)],
        65: [pygame.event.Event(pygame.KEYDOWN, key=pygame.K_s, mod=0)],
        70: [pygame.event.Event(pygame.QUIT)],
    }
    monkeypatch.setattr(pygame.event, "get", _scripted_event_get(script))

    _load_gui_module().main()

    joint_limits = calib_holder["calib"]["joint_limits_deg"]
    assert joint_limits["joint1"] == [100.0, 200.0]
    assert joint_limits["joint2"] == [150.0, 200.0]


def test_trace_boundary_gui_refuses_to_save_fewer_than_3_vertices(monkeypatch):
    calib_holder = {}
    monkeypatch.setattr(ac, "load_calib", _fake_calib((30.0, 239.0), (32.0, 227.0)))
    monkeypatch.setattr(ac, "save_calib", lambda calib, path=None: calib_holder.update(calib=calib))
    monkeypatch.setattr(hw, "Servos", lambda joint_ids: FakeServosWalkingLoop(joint_ids, LOOP))

    # Start and stop recording almost immediately -- only 1-2 vertices
    # get recorded, not enough for a closed polygon.
    script = {
        2: [pygame.event.Event(pygame.KEYDOWN, key=pygame.K_b, mod=0)],
        3: [pygame.event.Event(pygame.KEYDOWN, key=pygame.K_b, mod=0)],
        5: [pygame.event.Event(pygame.KEYDOWN, key=pygame.K_s, mod=0)],
        10: [pygame.event.Event(pygame.QUIT)],
    }
    monkeypatch.setattr(pygame.event, "get", _scripted_event_get(script))

    _load_gui_module().main()

    assert "calib" not in calib_holder, "should have refused to save too few vertices"


def test_trace_boundary_gui_replay_drives_arm_back_to_first_vertex(monkeypatch):
    # Record a loop, save it, then replay it ('r') -- the replay should
    # actively command the servos (via jog_controller.ArmController, the
    # same smooth planner every other tool here uses) through every saved
    # vertex and back to the first one, ending there.
    calib_holder = {}
    monkeypatch.setattr(ac, "load_calib", _fake_calib((30.0, 239.0), (32.0, 227.0)))
    monkeypatch.setattr(ac, "save_calib", lambda calib, path=None: calib_holder.update(calib=calib))
    fake_servos_holder = {}

    def make_fake_servos(joint_ids):
        fs = FakeServosWalkingLoop(joint_ids, LOOP)
        fake_servos_holder["servos"] = fs
        return fs

    monkeypatch.setattr(hw, "Servos", make_fake_servos)

    script = {
        5: [pygame.event.Event(pygame.KEYDOWN, key=pygame.K_b, mod=0)],    # start recording
        60: [pygame.event.Event(pygame.KEYDOWN, key=pygame.K_b, mod=0)],   # stop recording
        65: [pygame.event.Event(pygame.KEYDOWN, key=pygame.K_s, mod=0)],   # save
        70: [pygame.event.Event(pygame.KEYDOWN, key=pygame.K_r, mod=0)],   # start replay
        # Generous budget: a ~10-vertex boundary at jog speed needs several
        # hundred tick()s to fully drain every segment's planned queue.
        5000: [pygame.event.Event(pygame.QUIT)],
    }
    monkeypatch.setattr(pygame.event, "get", _scripted_event_get(script))

    _load_gui_module().main()

    assert "calib" in calib_holder
    boundary = calib_holder["calib"]["joint_limits_deg"]["coupled_boundary"]
    assert len(boundary) >= 3

    final = fake_servos_holder["servos"].commanded
    first_vertex = boundary[0]
    assert final["joint1"] == pytest.approx(first_vertex["joint1"])
    assert final["joint2"] == pytest.approx(first_vertex["joint2"])
