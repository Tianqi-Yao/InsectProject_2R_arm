"""Dual-backend camera: either picamera2 (Raspberry Pi CSI camera, the
eventual deployment target) or a plain UVC/USB webcam via OpenCV's
cv2.VideoCapture (for bench debugging on a dev machine before moving to the
real Pi) behind the same connect()/close()/capture_gray() interface --
selected by hw_state.json's `camera_backend` field, not a code change, so
switching from desktop debugging to the Pi later is a one-line config edit."""

from __future__ import annotations

import time


class Camera:
    def __init__(self, resolution: tuple[int, int] = (1920, 1080),
                 backend: str = "picamera2", usb_index: int = 0):
        self.resolution = resolution
        self.backend = backend
        self.usb_index = usb_index
        self._picam = None
        self._cap = None

    def connect(self) -> None:
        if self.backend == "usb":
            import cv2

            self._cap = cv2.VideoCapture(self.usb_index)
            if not self._cap.isOpened():
                raise IOError(f"could not open USB camera at index {self.usb_index}")
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
            time.sleep(0.5)  # let auto-exposure settle before first capture
        elif self.backend == "picamera2":
            from picamera2 import Picamera2  # deferred: Pi-only, install via apt

            self._picam = Picamera2()
            config = self._picam.create_still_configuration(
                main={"size": self.resolution, "format": "RGB888"})
            self._picam.configure(config)
            self._picam.start()
            time.sleep(1.0)  # let auto-exposure/focus settle before first capture
        else:
            raise ValueError(f"unknown camera backend {self.backend!r}, expected 'usb' or 'picamera2'")

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
        if self._picam is not None:
            self._picam.stop()

    def capture_gray(self):
        import cv2

        if self.backend == "usb":
            # A handful of UVC drivers (macOS included) drop the first
            # read() or two right after VideoCapture opens -- frames
            # haven't started streaming yet even after connect()'s warm-up
            # sleep -- so a bare single read() is flaky. Retry a few times.
            ok, frame = False, None
            for _ in range(5):
                ok, frame = self._cap.read()
                if ok:
                    break
                time.sleep(0.1)
            if not ok:
                raise IOError("failed to read a frame from the USB camera")
            return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # VideoCapture gives BGR, not RGB

        frame = self._picam.capture_array()
        return cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)


def build_camera(backend: str, resolution: tuple[int, int] = (1920, 1080),
                  usb_index: int = 0) -> Camera:
    return Camera(resolution=resolution, backend=backend, usb_index=usb_index)
