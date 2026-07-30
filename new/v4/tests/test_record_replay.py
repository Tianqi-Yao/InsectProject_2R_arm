"""Tests for record_replay.replay()'s dwell/photo timing, using a fake
Servos handle (same idea as test_jog_controller.py's FakeServos) plus a
fake Camera standing in for arm_hardware.Camera (which needs picamera2/cv2,
Pi-only) -- no real hardware needed."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import record_replay as rr  # noqa: E402


class FakeServos:
    def __init__(self, start=(0.0, 0.0)):
        self._pos = {"joint1": start[0], "joint2": start[1]}

    def get_present_deg(self, joint):
        return self._pos[joint]

    def set_target_deg(self, joint, angle_deg, speed=800, acc=0):
        self._pos[joint] = angle_deg

    def set_torque_enabled(self, joint, enabled):
        pass

    def close(self):
        pass


class FakeCamera:
    def __init__(self, resolution):
        self.resolution = resolution
        self.connected = False
        self.closed = False
        self.captured = []

    def connect(self):
        self.connected = True

    def capture_and_save(self, path):
        self.captured.append(path)

    def close(self):
        self.closed = True


@pytest.fixture
def two_points_path(tmp_path):
    path = tmp_path / "recorded_path.json"
    path.write_text(json.dumps([[10.0, 20.0], [30.0, 40.0]]))
    return path


def _patch_hardware(monkeypatch, camera_holder=None):
    monkeypatch.setattr(rr, "_connect", lambda servo_port, joint_ids: FakeServos())
    monkeypatch.setattr(rr, "_resync_and_relock", lambda servos: None)
    sleeps = []
    monkeypatch.setattr(rr.time, "sleep", lambda s: sleeps.append(s))
    if camera_holder is not None:
        def fake_camera_ctor(resolution):
            cam = FakeCamera(resolution)
            camera_holder.append(cam)
            return cam
        monkeypatch.setattr(rr.hw, "Camera", fake_camera_ctor)
    return sleeps


def test_replay_without_photos_just_dwells(monkeypatch, two_points_path):
    sleeps = _patch_hardware(monkeypatch)
    rr.replay(two_points_path, dwell_s=4.0)
    # one dwell_s sleep per point, nothing else photo-related
    dwell_sleeps = [s for s in sleeps if s == 4.0]
    assert len(dwell_sleeps) == 2


def test_replay_with_photos_splits_the_dwell_around_one_shot(monkeypatch, two_points_path, tmp_path):
    cameras = []
    sleeps = _patch_hardware(monkeypatch, camera_holder=cameras)
    photo_dir = tmp_path / "photos"

    rr.replay(two_points_path, dwell_s=4.0, photo_dir=photo_dir, photo_delay_s=2.0)

    assert len(cameras) == 1
    cam = cameras[0]
    assert cam.connected and cam.closed
    assert cam.resolution == rr.DEFAULT_PHOTO_RESOLUTION
    # 2s wait, shutter, 2s wait per point (the many other tiny sleeps are the
    # motion controller's own per-tick dt, unrelated to the dwell/photo timing)
    assert sleeps.count(2.0) == 4
    assert sum(1 for s in sleeps if s != 2.0) == len(sleeps) - 4
    assert [p.name for p in cam.captured] == ["point_001.jpg", "point_002.jpg"]
    run_dirs = list(photo_dir.iterdir())
    assert len(run_dirs) == 1  # one timestamped subfolder for this run
    assert cam.captured[0].parent == run_dirs[0]


def test_replay_rejects_photo_delay_not_inside_the_dwell(monkeypatch, two_points_path, tmp_path):
    _patch_hardware(monkeypatch)
    with pytest.raises(ValueError):
        rr.replay(two_points_path, dwell_s=4.0, photo_dir=tmp_path / "photos", photo_delay_s=4.0)
    with pytest.raises(ValueError):
        rr.replay(two_points_path, dwell_s=4.0, photo_dir=tmp_path / "photos", photo_delay_s=0.0)
