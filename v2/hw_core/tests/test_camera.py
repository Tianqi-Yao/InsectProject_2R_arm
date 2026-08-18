import sys
import types

import pytest

from arm_hw_core.camera import Camera, build_camera


class _FakeVideoCapture:
    instances: list["_FakeVideoCapture"] = []

    def __init__(self, index):
        self.index = index
        self.opened = True
        self.props: dict[int, float] = {}
        self.read_results: list[tuple[bool, object]] = []
        _FakeVideoCapture.instances.append(self)

    def isOpened(self):
        return self.opened

    def set(self, prop, value):
        self.props[prop] = value

    def read(self):
        if self.read_results:
            return self.read_results.pop(0)
        return True, "frame"

    def release(self):
        self.opened = False


@pytest.fixture
def fake_cv2(monkeypatch):
    _FakeVideoCapture.instances.clear()
    fake = types.ModuleType("cv2")
    fake.VideoCapture = _FakeVideoCapture
    fake.CAP_PROP_FRAME_WIDTH = 3
    fake.CAP_PROP_FRAME_HEIGHT = 4
    fake.COLOR_BGR2GRAY = 6
    fake.COLOR_RGB2GRAY = 7
    fake.cvtColor = lambda frame, code: (frame, code)
    monkeypatch.setitem(sys.modules, "cv2", fake)
    return fake


def test_build_camera_selects_backend():
    cam = build_camera("usb", resolution=(640, 480), usb_index=2)
    assert cam.backend == "usb"
    assert cam.usb_index == 2
    assert cam.resolution == (640, 480)


def test_unknown_backend_raises_on_connect():
    cam = Camera(backend="not-a-real-backend")
    with pytest.raises(ValueError, match="unknown camera backend"):
        cam.connect()


def test_usb_backend_opens_capture_at_configured_index_and_resolution(fake_cv2, monkeypatch):
    monkeypatch.setattr("time.sleep", lambda *_: None)
    cam = Camera(backend="usb", resolution=(1280, 720), usb_index=3)
    cam.connect()
    fake_cap = _FakeVideoCapture.instances[0]
    assert fake_cap.index == 3
    assert fake_cap.props[fake_cv2.CAP_PROP_FRAME_WIDTH] == 1280
    assert fake_cap.props[fake_cv2.CAP_PROP_FRAME_HEIGHT] == 720


def test_usb_backend_raises_if_device_does_not_open(fake_cv2, monkeypatch):
    monkeypatch.setattr("time.sleep", lambda *_: None)

    class NeverOpens(_FakeVideoCapture):
        def isOpened(self):
            return False

    fake_cv2.VideoCapture = NeverOpens
    cam = Camera(backend="usb")
    with pytest.raises(IOError, match="could not open USB camera"):
        cam.connect()


def test_usb_capture_gray_retries_flaky_first_reads(fake_cv2, monkeypatch):
    monkeypatch.setattr("time.sleep", lambda *_: None)
    cam = Camera(backend="usb")
    cam.connect()
    fake_cap = _FakeVideoCapture.instances[0]
    # Simulate the first two reads dropping (common on real UVC drivers
    # right after VideoCapture opens), succeeding on the third.
    fake_cap.read_results = [(False, None), (False, None), (True, "real_frame")]
    frame, code = cam.capture_gray()
    assert frame == "real_frame"
    assert code == fake_cv2.COLOR_BGR2GRAY


def test_usb_capture_gray_raises_after_exhausting_retries(fake_cv2, monkeypatch):
    monkeypatch.setattr("time.sleep", lambda *_: None)
    cam = Camera(backend="usb")
    cam.connect()
    fake_cap = _FakeVideoCapture.instances[0]
    fake_cap.read_results = [(False, None)] * 10
    with pytest.raises(IOError, match="failed to read a frame"):
        cam.capture_gray()


def test_close_releases_usb_capture(fake_cv2, monkeypatch):
    monkeypatch.setattr("time.sleep", lambda *_: None)
    cam = Camera(backend="usb")
    cam.connect()
    fake_cap = _FakeVideoCapture.instances[0]
    cam.close()
    assert fake_cap.opened is False
