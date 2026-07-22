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

     9  Min Angle Limit      (2 bytes, EEPROM)
    11  Max Angle Limit      (2 bytes, EEPROM)
    40  Torque Enable        (1 byte)
    41  Goal Acceleration     (1 byte)
    42  Goal Position         (2 bytes)
    46  Goal Speed            (2 bytes)
    55  Lock                 (1 byte, EEPROM write-protect: 0=unlocked, 1=locked)
    56  Present Position      (2 bytes)

protocol_end=0 (little-endian) is the standard setting for STS/SMS-series
servos. If you're on a different SDK fork that *does* expose a higher-level
class, this file is still the one place to change -- nothing else depends
on these details.

NOTE on Min/Max Angle Limit (registers 9/11): these are the servo's OWN
hardware-enforced position bounds -- once set, the servo firmware itself
refuses to move past them, regardless of what any software (including a
bug in this project's IK/motion code) commands. This is the outermost,
most trustworthy layer of protection against driving a joint into a
mechanical dead zone; arm_core.py's joint_limits_deg is a *software*
soft-limit that complements but does not replace this. Being EEPROM-
resident, writing them requires clearing the Lock register (55) first and
setting it back after -- see set_hardware_angle_limits(). `main.py
set-joint-limits` drives this interactively. Register addresses confirmed
against Feetech's published STS3215 control table (matches address 11
already used in the reference ServoDriverST/STSCTRL.h's own mode-switching
code, and address 55 as the documented EEPROM lock flag).

NOTE on STREAMING_SPEED/STREAMING_ACC: motion smoothing used to be split
across two uncoordinated layers -- the servo's own GOAL_SPEED/GOAL_ACC
registers doing one kind of limiting, and callers separately picking
"jog" vs. "scan" speed constants by hand. Now all smoothing (accel ramps,
synchronized dual-joint arrival, corner blending) happens once, in
software, via motion_planning/ -- jog_controller.py streams closely-spaced
setpoints from a planned trajectory. For that to work without the servo's
own limiting fighting the planner's, the servo needs to just track
setpoints as fast as it can: STREAMING_SPEED/STREAMING_ACC below are what
jog_controller.py passes to set_target_deg for that purpose. The
speed/acc parameters on set_target_deg stay general-purpose for the few
callers that still want single-shot, servo-limited moves (e.g.
calibrate's/selfcheck's coarse moves via move_and_wait).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

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

    def set_torque_enabled(self, joint: str, enabled: bool) -> None:
        """Disable to let a joint be moved freely by hand -- e.g. to find
        safe angle-limit boundaries during bring-up (see
        set_hardware_angle_limits) -- then re-enable before commanding
        motion again. Safe to do on this arm: it moves in a horizontal
        plane, so a joint won't fall/drift under gravity while torque is off."""
        sid = self.joint_ids[joint]
        self._packet.write1ByteTxRx(self._port, sid, self.ADDR_TORQUE_ENABLE, 1 if enabled else 0)

    def _write_checked(self, sid: int, addr: int, value: int, nbytes: int) -> None:
        """write1/2ByteTxRx, but actually verified: the underlying SDK
        (feetech-servo-sdk) has been observed to occasionally raise a bare
        IndexError from inside its own response-parsing on a flaky/dropped
        byte, rather than cleanly reporting a non-success result code --
        catch that here and convert it (and an explicit non-success comm
        result) into one clear IOError, instead of letting either crash
        the caller with a confusing low-level traceback."""
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
        except Exception as e:  # noqa: BLE001 -- see _write_checked's docstring
            raise IOError(f"servo id={sid}: read of register {addr} raised {e!r} "
                          f"(likely a dropped/corrupted byte on the bus)") from e
        if comm != scs.COMM_SUCCESS:
            raise IOError(f"servo id={sid}: read of register {addr} failed "
                          f"(comm={comm}, err={err})")
        return value

    def set_hardware_angle_limits(self, joint: str, min_deg: float, max_deg: float) -> None:
        """Write the servo's own EEPROM-resident Min/Max Angle Limit
        registers -- see this module's docstring for why this is the
        outermost, most trustworthy safety layer against driving a joint
        into a mechanical dead zone. `min_deg`/`max_deg` are in raw servo-
        degree space (same convention as get_present_deg()), and must
        satisfy 0 <= min_deg < max_deg <= 360: a dead zone that straddles
        the 0/360 wraparound point (leaving a single contiguous safe arc)
        is supported; a safe range that itself wraps through 0/360 is not
        -- see arm_core.within_joint_limits's docstring for the same
        assumption on the software side, and prefer mounting the servo
        horn so the dead zone (not the safe range) straddles the wrap.

        Raises IOError (not silently partial) if any step of the
        unlock/write/lock sequence fails -- in particular, a failed final
        lock-write would leave the servo's EEPROM unprotected, so that
        failure is not swallowed."""
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
        """Read back the servo's own Min/Max Angle Limit registers -- use
        this to confirm a set_hardware_angle_limits() write actually took.
        Raises IOError on a communication failure (see _read_checked) --
        note that failing to *verify* doesn't necessarily mean the write
        itself failed, just that this readback couldn't confirm it."""
        sid = self.joint_ids[joint]
        min_ticks = self._read_checked(sid, self.ADDR_MIN_ANGLE_LIMIT, nbytes=2)
        max_ticks = self._read_checked(sid, self.ADDR_MAX_ANGLE_LIMIT, nbytes=2)
        return min_ticks * DEG_PER_TICK, max_ticks * DEG_PER_TICK

    def set_target_deg(self, joint: str, angle_deg: float, speed: int = 800, acc: int = 0) -> None:
        """acc=0 (the default) matches the servo's original snap-to-speed
        behaviour. A small nonzero acc (e.g. 20-40) makes it ramp up to
        speed and ramp down into arrival instead of starting/stopping
        abruptly. For frontends driven by motion_planning/ (jog_controller.py),
        pass speed=STREAMING_SPEED, acc=STREAMING_ACC instead -- see this
        module's docstring for why."""
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
