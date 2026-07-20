"""Hardware I/O black box: servo bus, camera, and AprilTag detection.

Everything in this file is plumbing around third-party libraries/protocols --
there is no decision logic here. arm_core.py and main.py only ever call the
small set of public methods on Servos / Camera / TagDetector below (connect,
set_target_deg, get_present_deg, move_and_wait, capture_gray, detect); what
happens inside each is not meant to require close reading.

NOTE (read this once when real hardware arrives, then forget it): written
against the SO-ARM100/lerobot-style `scservo_sdk` package for Feetech
STS3215 servos over a Waveshare bus-servo driver board, plus a Raspberry Pi
`picamera2` camera and `pupil_apriltags` detector. Exact method names on
`scservo_sdk` vary slightly between forks/versions -- before trusting this
file, run `scripts: main.py test-servo` (see main.py) against one real
servo and confirm ping/read/write behave as expected; adjust the few calls
inside Servos if your installed SDK names things differently
(`python -c "import scservo_sdk as s; print([n for n in dir(s) if not n.startswith('_')])"`
is the fastest way to check).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

logger = logging.getLogger("arm_hardware")

TICKS_PER_REV = 4096
DEG_PER_TICK = 360.0 / TICKS_PER_REV


class Servos:
    """Talks to the two STS3215 bus servos through a Waveshare serial driver
    board. `joint1`/`joint2` are logical names mapped to bus servo IDs."""

    def __init__(self, joint_ids: dict):
        self.joint_ids = joint_ids
        self._port_handler = None
        self._packet_handler = None

    def connect(self, port: str, baud: int = 1_000_000) -> None:
        import scservo_sdk as scs  # deferred: only needed on the real robot

        self._port_handler = scs.PortHandler(port)
        self._packet_handler = scs.sms_sts(self._port_handler)
        if not self._port_handler.openPort():
            raise IOError(f"failed to open servo port {port}")
        if not self._port_handler.setBaudRate(baud):
            raise IOError(f"failed to set servo baud rate {baud}")
        for name, sid in self.joint_ids.items():
            _pos, _comm, err = self._packet_handler.ReadPos(sid)
            if err != 0:
                raise IOError(f"servo '{name}' (id={sid}) did not respond to ping")

    def close(self) -> None:
        if self._port_handler is not None:
            self._port_handler.closePort()

    def set_target_deg(self, joint: str, angle_deg: float, speed: int = 800) -> None:
        sid = self.joint_ids[joint]
        ticks = int(round(angle_deg / DEG_PER_TICK)) % TICKS_PER_REV
        self._packet_handler.WritePosEx(sid, ticks, speed, 0)

    def get_present_deg(self, joint: str) -> float:
        """Read the servo's real magnetic-encoder angle -- never trust the
        last commanded value, that defeats the point of using feedback
        servos over the old open-loop PWM ones."""
        sid = self.joint_ids[joint]
        ticks, _comm, _err = self._packet_handler.ReadPos(sid)
        return ticks * DEG_PER_TICK

    def move_and_wait(self, targets_deg: dict, timeout_s: float = 4.0,
                       tol_deg: float = 0.5, poll_hz: float = 20.0) -> dict:
        """Command target angles, then poll Present Position until every
        joint settles within tol_deg (or timeout). Returns the angles
        actually reached, read back from the encoders."""
        for joint, angle in targets_deg.items():
            self.set_target_deg(joint, angle)

        deadline = time.monotonic() + timeout_s
        reached = {joint: None for joint in targets_deg}
        settled = False
        while time.monotonic() < deadline:
            time.sleep(1.0 / poll_hz)
            settled = True
            for joint, target in targets_deg.items():
                current = self.get_present_deg(joint)
                reached[joint] = current
                if abs(current - target) > tol_deg:
                    settled = False
            if settled:
                break
        if not settled:
            logger.warning("move_and_wait timed out before settling: target=%s reached=%s",
                            targets_deg, reached)
        return {j: (v if v is not None else targets_deg[j]) for j, v in reached.items()}


@dataclass
class Detection:
    tag_id: int
    center: tuple
    corners: list


class Camera:
    """Wraps picamera2 to grab a single grayscale frame on demand."""

    def __init__(self, resolution: tuple = (1920, 1080)):
        self.resolution = resolution
        self._picam = None

    def connect(self) -> None:
        from picamera2 import Picamera2  # deferred: Pi-only, install via apt

        self._picam = Picamera2()
        config = self._picam.create_still_configuration(
            main={"size": self.resolution, "format": "RGB888"})
        self._picam.configure(config)
        self._picam.start()
        time.sleep(1.0)  # let auto-exposure/focus settle before first capture

    def close(self) -> None:
        if self._picam is not None:
            self._picam.stop()

    def capture_gray(self):
        import cv2

        frame = self._picam.capture_array()
        return cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)


class TagDetector:
    """Wraps pupil_apriltags.Detector, keyed by tag_id for easy lookup."""

    def __init__(self, family: str = "tag36h11"):
        from pupil_apriltags import Detector  # deferred import

        self._detector = Detector(families=family, nthreads=2, quad_decimate=1.0)

    def detect(self, frame) -> dict:
        results = self._detector.detect(frame)
        return {
            r.tag_id: Detection(tag_id=r.tag_id, center=tuple(r.center), corners=list(r.corners))
            for r in results
        }


class ArmHardware:
    """Bundles the three black-box handles behind one object, matching the
    `hw` parameter arm_core.run_selfcheck() expects (hw.camera, hw.detector,
    hw.servos)."""

    def __init__(self, servo_port: str, joint_ids: dict, camera_resolution: tuple = (1920, 1080)):
        self.servos = Servos(joint_ids)
        self.camera = Camera(camera_resolution)
        self.detector = TagDetector()
        self._servo_port = servo_port

    def connect(self) -> None:
        self.servos.connect(self._servo_port)
        self.camera.connect()

    def close(self) -> None:
        self.servos.close()
        self.camera.close()
