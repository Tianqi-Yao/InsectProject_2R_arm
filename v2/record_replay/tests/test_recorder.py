import pytest

from record_replay.config import RecordedPoint
from record_replay.controller import ArmController, MotionParams
from record_replay.recorder import Player, Recorder


class FakeServos:
    def __init__(self, start=(0.0, 0.0)):
        self.present = {"joint1": start[0], "joint2": start[1]}
        self.torque = {"joint1": True, "joint2": True}

    def get_present_deg(self, joint):
        return self.present[joint]

    def set_target_deg(self, joint, angle_deg, speed=800, acc=0):
        self.present[joint] = angle_deg

    def set_torque_enabled(self, joint, enabled):
        self.torque[joint] = enabled


class FakeCamera:
    def __init__(self):
        self.saved_frames = []

    def capture_gray(self):
        return "a-frame"


# ── Recorder ─────────────────────────────────────────────────────────────

def test_recorder_start_releases_torque():
    servos = FakeServos()
    Recorder(servos).start()
    assert servos.torque == {"joint1": False, "joint2": False}


def test_recorder_mark_reads_present_position():
    servos = FakeServos(start=(12.0, -34.0))
    point = Recorder(servos).mark()
    assert point == RecordedPoint(12.0, -34.0)


def test_recorder_stop_resyncs_goal_register_before_reenabling_torque():
    servos = FakeServos(start=(5.0, 6.0))
    recorder = Recorder(servos)
    recorder.start()
    servos.present = {"joint1": 77.0, "joint2": 88.0}  # simulate a hand-drag
    recorder.stop()
    assert servos.torque == {"joint1": True, "joint2": True}
    assert servos.present == {"joint1": 77.0, "joint2": 88.0}, (
        "stop() must sync the goal register to the ACTUAL hand-moved position, "
        "not snap back toward wherever torque was released from"
    )


# ── Player ───────────────────────────────────────────────────────────────

def _player(start=(0.0, 0.0), camera=None):
    servos = FakeServos(start)
    controller = ArmController(servos, MotionParams(vmax_deg_s=90.0, amax_deg_s2=180.0))
    return Player(servos, controller, camera=camera), servos


def test_prepare_resyncs_controller_and_reenables_torque_after_a_hand_move():
    player, servos = _player(start=(0.0, 0.0))
    servos.torque = {"joint1": False, "joint2": False}
    servos.present = {"joint1": 15.0, "joint2": 25.0}  # hand-moved since recording

    player.prepare()

    assert servos.torque == {"joint1": True, "joint2": True}
    assert player.controller.commanded_deg == pytest.approx((15.0, 25.0), abs=1e-6)


def test_goto_and_dwell_moves_the_arm_and_sleeps_full_dwell_without_camera():
    player, servos = _player()
    player.prepare()
    sleeps = []
    player.goto_and_dwell(RecordedPoint(20.0, 30.0), dwell_s=5.0, sleep=sleeps.append)
    assert servos.present == {"joint1": 20.0, "joint2": 30.0}
    assert sleeps == [5.0]


def test_goto_and_dwell_with_camera_splits_sleep_around_the_photo(monkeypatch):
    camera = FakeCamera()
    player, servos = _player(camera=camera)
    player.prepare()
    sleeps = []
    saved = {}

    def fake_save_photo(self, path):
        saved["path"] = path
        saved["frame"] = self.camera.capture_gray()

    monkeypatch.setattr(Player, "_save_photo", fake_save_photo)
    player.goto_and_dwell(RecordedPoint(1.0, 1.0), dwell_s=5.0, photo_path="/tmp/x.jpg",
                           photo_delay_s=2.0, sleep=sleeps.append)

    assert sleeps == [2.0, 3.0]  # photo_delay_s, then dwell_s - photo_delay_s
    assert saved == {"path": "/tmp/x.jpg", "frame": "a-frame"}
