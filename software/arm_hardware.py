"""Hardware I/O black box: servo bus, camera, and AprilTag detection.

Everything in this file is plumbing around third-party libraries/protocols --
there is no decision logic here. arm_core.py and main.py only ever call the
small set of public methods on Servos / Camera / TagDetector below (connect,
set_target_deg, get_present_deg, move_and_wait, capture_gray, detect); what
happens inside each is not meant to require close reading.

NOTE on the servo SDK: the `feetech-servo-sdk` PyPI package (import name
`scservo_sdk`, from Adam-Software/FEETECH-Servo-Python-SDK) only ships a
low-level, Dynamixel-SDK-style packet handler (`PortHandler` + a
`PacketHandler(protocol_end)` factory with `read/write<N>ByteTxRx(port, id,
address, ...)`) -- there's no high-level `sms_sts`-style class with
convenience methods. So Servos below talks the STS3215's control table
directly via its documented register addresses (confirmed against Waveshare's
own working Arduino firmware, which uses the same addresses through their
C++ SCServo library):

    40  Torque Enable        (1 byte)
    42  Goal Position         (2 bytes)
    46  Goal Speed            (2 bytes)
    56  Present Position      (2 bytes)

protocol_end=0 (little-endian) is the standard setting for STS/SMS-series
servos. If you're on a different SDK fork that *does* expose a higher-level
class, this file is still the one place to change -- nothing else depends
on these details.
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

    ADDR_TORQUE_ENABLE = 40
    ADDR_GOAL_ACC = 41         # 1 byte
    ADDR_GOAL_POSITION = 42    # 2 bytes
    ADDR_GOAL_SPEED = 46       # 2 bytes
    ADDR_PRESENT_POSITION = 56  # 2 bytes

    def __init__(self, joint_ids: dict):
        self.joint_ids = joint_ids
        self._port = None
        self._packet = None
        self._last_speed = {}
        self._last_acc = {}

    def connect(self, port: str, baud: int = 115_200) -> None:
        # This is the USB<->ESP32 baud, not the servo bus's -- SerialBridge.ino
        # bridges it to a fixed 1,000,000 on the Serial1/servo-bus side, since
        # 1,000,000 on the USB hop itself proved unreliable on macOS with this
        # board's USB-serial chip (see SerialBridge.ino's comment).
        import scservo_sdk as scs  # deferred: only needed on the real robot

        self._port = scs.PortHandler(port)
        self._packet = scs.PacketHandler(0)  # protocol_end=0: little-endian, STS/SMS servos
        if not self._port.setBaudRate(baud):
            raise IOError(f"failed to open servo port {port} at {baud} baud")
        # Opening the USB serial port resets many ESP32/Arduino boards (DTR
        # toggle); give SerialBridge.ino time to reboot and start relaying
        # bytes before the first ping, or it gets sent into the void.
        time.sleep(2.0)
        for name, sid in self.joint_ids.items():
            _model, comm, _err = self._packet.ping(self._port, sid)
            if comm != scs.COMM_SUCCESS:
                raise IOError(f"servo '{name}' (id={sid}) did not respond to ping")
            self._packet.write1ByteTxRx(self._port, sid, self.ADDR_TORQUE_ENABLE, 1)

    def close(self) -> None:
        if self._port is not None:
            self._port.closePort()

    def set_target_deg(self, joint: str, angle_deg: float, speed: int = 800, acc: int = 0) -> None:
        """acc=0 (the default) matches the servo's original snap-to-speed
        behaviour. A small nonzero acc (e.g. 20-40) makes it ramp up to
        speed and ramp down into arrival instead of starting/stopping
        abruptly -- worth setting for jogging, where target changes are
        frequent and small; not needed for the calibration sweep's larger,
        already-paced moves."""
        sid = self.joint_ids[joint]
        ticks = int(round(angle_deg / DEG_PER_TICK)) % TICKS_PER_REV
        # Skip the ACC/GOAL_SPEED writes when unchanged -- fewer serial
        # round trips for the common case (jogging/streaming at a fixed
        # speed+acc), which matters when position updates are sent
        # frequently: less variable per-update latency means a steadier
        # cadence instead of jitter.
        if self._last_acc.get(joint) != acc:
            self._packet.write1ByteTxRx(self._port, sid, self.ADDR_GOAL_ACC, acc)
            self._last_acc[joint] = acc
        if self._last_speed.get(joint) != speed:
            self._packet.write2ByteTxRx(self._port, sid, self.ADDR_GOAL_SPEED, speed)
            self._last_speed[joint] = speed
        self._packet.write2ByteTxRx(self._port, sid, self.ADDR_GOAL_POSITION, ticks)

    def get_present_deg(self, joint: str) -> float:
        """Read the servo's real magnetic-encoder angle -- never trust the
        last commanded value, that defeats the point of using feedback
        servos over the old open-loop PWM ones."""
        sid = self.joint_ids[joint]
        ticks, _comm, _err = self._packet.read2ByteTxRx(self._port, sid, self.ADDR_PRESENT_POSITION)
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
