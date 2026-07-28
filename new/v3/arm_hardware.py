"""Hardware I/O black box: servo bus + camera. Everything here is plumbing
around third-party libraries/protocols -- there is no decision logic here.
arm_core.py/path_core.py/main.py only ever call the small set of public
methods on Servos / Camera below (connect, set_target_deg, get_present_deg,
move_and_wait, capture_and_save); what happens inside each is not meant to
require close reading.

`Servos` is copied verbatim from ../arm_hardware.py (see that file's
docstring for the STS3215 register-map details and STREAMING_SPEED/
STREAMING_ACC rationale -- nothing about the servo protocol changes for v3).
`Camera` drops AprilTag support entirely (v3 doesn't do vision-based
calibration): `capture_gray()`/`TagDetector`/`Detection` are gone, replaced
by `capture_and_save()`, which just writes a real color photo to disk.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

logger = logging.getLogger("arm_hardware")

TICKS_PER_REV = 4096
DEG_PER_TICK = 360.0 / TICKS_PER_REV

# See the module docstring's STREAMING_SPEED/STREAMING_ACC note. 4000 is
# near the STS3215's practical max speed setting (matches the value
# Waveshare's own stock firmware uses); acc=0 disables the servo's own
# ramping so software-side setpoint spacing is the only thing shaping motion.
STREAMING_SPEED = 4000
STREAMING_ACC = 0


class Servos:
    """Talks to the two STS3215 bus servos through a Waveshare serial driver
    board. `joint1`/`joint2` are logical names mapped to bus servo IDs."""

    ADDR_MIN_ANGLE_LIMIT = 9   # 2 bytes, EEPROM
    ADDR_MAX_ANGLE_LIMIT = 11  # 2 bytes, EEPROM
    ADDR_TORQUE_ENABLE = 40
    ADDR_GOAL_ACC = 41         # 1 byte
    ADDR_GOAL_POSITION = 42    # 2 bytes
    ADDR_GOAL_SPEED = 46       # 2 bytes
    ADDR_LOCK = 55             # 1 byte, EEPROM write-protect flag
    ADDR_PRESENT_POSITION = 56  # 2 bytes

    def __init__(self, joint_ids: dict):
        self.joint_ids = joint_ids
        self._port = None
        self._packet = None
        self._last_speed = {}
        self._last_acc = {}

    def connect(self, port: str, baud: int = 115_200) -> None:
        import scservo_sdk as scs  # deferred: only needed on the real robot

        self._port = scs.PortHandler(port)
        self._packet = scs.PacketHandler(0)  # protocol_end=0: little-endian, STS/SMS servos
        if not self._port.setBaudRate(baud):
            raise IOError(f"failed to open servo port {port} at {baud} baud")
        time.sleep(2.0)
        for name, sid in self.joint_ids.items():
            _model, comm, _err = self._packet.ping(self._port, sid)
            if comm != scs.COMM_SUCCESS:
                raise IOError(f"servo '{name}' (id={sid}) did not respond to ping")
            self._packet.write1ByteTxRx(self._port, sid, self.ADDR_TORQUE_ENABLE, 1)

    def close(self) -> None:
        if self._port is not None:
            self._port.closePort()

    def set_torque_enabled(self, joint: str, enabled: bool) -> None:
        """Disable to let a joint be moved freely by hand (teach-in) then
        re-enable before commanding motion again. Safe on this arm: it
        moves in a horizontal plane, so a joint won't fall/drift under
        gravity while torque is off."""
        sid = self.joint_ids[joint]
        self._packet.write1ByteTxRx(self._port, sid, self.ADDR_TORQUE_ENABLE, 1 if enabled else 0)

    def _write_checked(self, sid: int, addr: int, value: int, nbytes: int) -> None:
        import scservo_sdk as scs

        writer = self._packet.write1ByteTxRx if nbytes == 1 else self._packet.write2ByteTxRx
        try:
            comm, err = writer(self._port, sid, addr, value)
        except Exception as e:  # noqa: BLE001 -- deliberately broad, see docstring
            raise IOError(f"servo id={sid}: write to register {addr} raised {e!r} "
                          f"(likely a dropped/corrupted byte on the bus)") from e
        if comm != scs.COMM_SUCCESS:
            raise IOError(f"servo id={sid}: write to register {addr} failed "
                          f"(comm={comm}, err={err})")

    def _read_checked(self, sid: int, addr: int, nbytes: int) -> int:
        import scservo_sdk as scs

        reader = self._packet.read1ByteTxRx if nbytes == 1 else self._packet.read2ByteTxRx
        try:
            value, comm, err = reader(self._port, sid, addr)
        except Exception as e:  # noqa: BLE001
            raise IOError(f"servo id={sid}: read of register {addr} raised {e!r} "
                          f"(likely a dropped/corrupted byte on the bus)") from e
        if comm != scs.COMM_SUCCESS:
            raise IOError(f"servo id={sid}: read of register {addr} failed "
                          f"(comm={comm}, err={err})")
        return value

    def set_hardware_angle_limits(self, joint: str, min_deg: float, max_deg: float) -> None:
        """Write the servo's own EEPROM-resident Min/Max Angle Limit
        registers -- the outermost, most trustworthy safety layer against
        driving a joint into a mechanical dead zone."""
        if not (0.0 <= min_deg < max_deg <= 360.0):
            raise ValueError(f"angle limits [{min_deg}, {max_deg}] must satisfy "
                              f"0 <= min < max <= 360 (wrapping safe ranges aren't supported)")
        sid = self.joint_ids[joint]
        min_ticks = int(round(min_deg / DEG_PER_TICK))
        max_ticks = int(round(max_deg / DEG_PER_TICK))
        self._write_checked(sid, self.ADDR_LOCK, 0, nbytes=1)  # unlock EEPROM
        try:
            self._write_checked(sid, self.ADDR_MIN_ANGLE_LIMIT, min_ticks, nbytes=2)
            self._write_checked(sid, self.ADDR_MAX_ANGLE_LIMIT, max_ticks, nbytes=2)
        finally:
            self._write_checked(sid, self.ADDR_LOCK, 1, nbytes=1)  # always re-lock

    def get_hardware_angle_limits(self, joint: str) -> tuple[float, float]:
        sid = self.joint_ids[joint]
        min_ticks = self._read_checked(sid, self.ADDR_MIN_ANGLE_LIMIT, nbytes=2)
        max_ticks = self._read_checked(sid, self.ADDR_MAX_ANGLE_LIMIT, nbytes=2)
        return min_ticks * DEG_PER_TICK, max_ticks * DEG_PER_TICK

    def set_target_deg(self, joint: str, angle_deg: float, speed: int = 800, acc: int = 0) -> None:
        """acc=0 (the default) matches the servo's original snap-to-speed
        behaviour. For frontends driven by motion_planning/
        (jog_controller.py), pass speed=STREAMING_SPEED, acc=STREAMING_ACC
        instead -- see this module's docstring."""
        sid = self.joint_ids[joint]
        ticks = int(round(angle_deg / DEG_PER_TICK)) % TICKS_PER_REV
        if self._last_acc.get(joint) != acc:
            self._packet.write1ByteTxRx(self._port, sid, self.ADDR_GOAL_ACC, acc)
            self._last_acc[joint] = acc
        if self._last_speed.get(joint) != speed:
            self._packet.write2ByteTxRx(self._port, sid, self.ADDR_GOAL_SPEED, speed)
            self._last_speed[joint] = speed
        self._packet.write2ByteTxRx(self._port, sid, self.ADDR_GOAL_POSITION, ticks)

    def get_present_deg(self, joint: str) -> float:
        """Read the servo's real magnetic-encoder angle -- never trust the
        last commanded value."""
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


class Camera:
    """Wraps picamera2 to grab and save a single still photo on demand --
    no AprilTag detection, no grayscale conversion: v3 doesn't do
    vision-based calibration, it only archives what the camera sees at
    each photo stop."""

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

    def capture_and_save(self, path: Path) -> None:
        """Grab one still frame and write it to `path` (any extension
        cv2.imwrite supports, e.g. .jpg/.png). RGB888 (picamera2's own
        convention) needs a channel swap for cv2.imwrite, which expects
        BGR."""
        import cv2

        frame = self._picam.capture_array()
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(path), bgr):
            raise IOError(f"cv2.imwrite failed to write {path}")


class ArmHardware:
    """Bundles the two black-box handles behind one object."""

    def __init__(self, servo_port: str, joint_ids: dict, camera_resolution: tuple = (1920, 1080)):
        self.servos = Servos(joint_ids)
        self.camera = Camera(camera_resolution)
        self._servo_port = servo_port

    def connect(self, with_camera: bool = True) -> None:
        self.servos.connect(self._servo_port)
        if with_camera:
            self.camera.connect()

    def close(self) -> None:
        self.servos.close()
        self.camera.close()
