"""STS3215 bus servo I/O, talking directly to the servo's raw control-table
registers (there is no higher-level convenience class in the `scservo_sdk`
package this project uses -- it only exposes a low-level, Dynamixel-SDK-style
packet handler). Register addresses confirmed against Feetech's published
STS3215 control table and Waveshare's own reference firmware:

     9  Min Angle Limit      (2 bytes, EEPROM)
    11  Max Angle Limit      (2 bytes, EEPROM)
    40  Torque Enable        (1 byte)
    41  Goal Acceleration     (1 byte)
    42  Goal Position         (2 bytes)
    46  Goal Speed            (2 bytes)
    55  Lock                 (1 byte, EEPROM write-protect: 0=unlocked, 1=locked)
    56  Present Position      (2 bytes)

protocol_end=0 (little-endian) is the standard setting for STS/SMS-series
servos.

Min/Max Angle Limit (registers 9/11) are the servo's OWN hardware-enforced
position bounds: once set, the servo firmware refuses to move past them no
matter what any software commands. This is the outermost, most trustworthy
layer of protection against driving a joint into a mechanical dead zone --
see limits.py for the software soft-limit layer that complements (does not
replace) this.

Servo native smoothing (Goal Speed/Goal Acc) is deliberately not used for
real-time motion: every feature package's own trajectory planner streams
closely-spaced setpoints, so the servo just needs to track them as fast as
it can. STREAMING_SPEED/STREAMING_ACC below are what a planner-driven
controller should pass to set_target_deg for that purpose; the speed/acc
parameters stay general-purpose for single-shot, servo-limited moves (e.g.
move_and_wait for calibration sampling or self-check spot-checks).
"""

from __future__ import annotations

import logging
import time

logger = logging.getLogger("arm_hw_core.servos")

TICKS_PER_REV = 4096
DEG_PER_TICK = 360.0 / TICKS_PER_REV

# Near the STS3215's practical max speed setting; acc=0 disables the servo's
# own ramping so a planner's setpoint spacing is the only thing shaping motion.
STREAMING_SPEED = 4000
STREAMING_ACC = 0


class Servos:
    """Talks to the two STS3215 bus servos through a Waveshare serial driver
    board / ESP32 USB transparent bridge. `joint1`/`joint2` are logical
    names mapped to bus servo IDs."""

    ADDR_MIN_ANGLE_LIMIT = 9    # 2 bytes, EEPROM
    ADDR_MAX_ANGLE_LIMIT = 11   # 2 bytes, EEPROM
    ADDR_TORQUE_ENABLE = 40
    ADDR_GOAL_ACC = 41          # 1 byte
    ADDR_GOAL_POSITION = 42     # 2 bytes
    ADDR_GOAL_SPEED = 46        # 2 bytes
    ADDR_LOCK = 55              # 1 byte, EEPROM write-protect flag
    ADDR_PRESENT_POSITION = 56  # 2 bytes

    def __init__(self, joint_ids: dict[str, int]):
        self.joint_ids = joint_ids
        self._port = None
        self._packet = None
        self._last_speed: dict[str, int] = {}
        self._last_acc: dict[str, int] = {}

    def connect(self, port: str, baud: int = 115_200) -> None:
        import scservo_sdk as scs  # deferred: only needed on the real robot

        self._port = scs.PortHandler(port)
        self._packet = scs.PacketHandler(0)  # protocol_end=0: little-endian, STS/SMS servos
        if not self._port.setBaudRate(baud):
            raise IOError(f"failed to open servo port {port} at {baud} baud")
        # Opening the USB serial port resets many ESP32/Arduino boards (DTR
        # toggle); give the bridge firmware time to reboot and start
        # relaying bytes before the first ping, or it gets sent into the void.
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
        """Disable to let a joint be moved freely by hand (e.g. to find safe
        angle-limit boundaries, or to hand-teach a pose), then re-enable
        before commanding motion again. Safe on this arm: it moves in a
        horizontal plane, so a joint won't fall/drift under gravity while
        torque is off."""
        sid = self.joint_ids[joint]
        self._packet.write1ByteTxRx(self._port, sid, self.ADDR_TORQUE_ENABLE, 1 if enabled else 0)

    def _write_checked(self, sid: int, addr: int, value: int, nbytes: int) -> None:
        """write1/2ByteTxRx, but actually verified: the underlying SDK has
        been observed to occasionally raise a bare IndexError from inside
        its own response-parsing on a flaky/dropped byte, rather than
        cleanly reporting a non-success result code -- catch that here and
        convert it (and an explicit non-success comm result) into one clear
        IOError, instead of letting either crash the caller with a
        confusing low-level traceback."""
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
        outermost, most trustworthy safety layer. `min_deg`/`max_deg` are
        in raw servo-degree space (same convention as get_present_deg()),
        and must satisfy 0 <= min_deg < max_deg <= 360: a dead zone that
        straddles the 0/360 wraparound point is fine; a safe range that
        itself wraps through 0/360 is not -- see limits.within_joint_limits
        for the same assumption on the software side.

        Raises IOError (not silently partial) if any step of the
        unlock/write/lock sequence fails -- in particular, a failed final
        lock-write would leave the servo's EEPROM unprotected, so that
        failure is never swallowed."""
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
        this to confirm a set_hardware_angle_limits() write actually took."""
        sid = self.joint_ids[joint]
        min_ticks = self._read_checked(sid, self.ADDR_MIN_ANGLE_LIMIT, nbytes=2)
        max_ticks = self._read_checked(sid, self.ADDR_MAX_ANGLE_LIMIT, nbytes=2)
        return min_ticks * DEG_PER_TICK, max_ticks * DEG_PER_TICK

    def set_target_deg(self, joint: str, angle_deg: float, speed: int = 800, acc: int = 0) -> None:
        """acc=0 (the default) matches the servo's original snap-to-speed
        behaviour. Callers driving the arm through a trajectory planner
        should pass speed=STREAMING_SPEED, acc=STREAMING_ACC instead, so
        the servo's own limiting doesn't fight the planner's -- see this
        module's docstring."""
        sid = self.joint_ids[joint]
        ticks = int(round(angle_deg / DEG_PER_TICK)) % TICKS_PER_REV
        # Skip the ACC/GOAL_SPEED writes when unchanged -- fewer serial
        # round trips for the common case (streaming at a fixed speed+acc),
        # which matters for a steady control-loop cadence.
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

    def move_and_wait(self, targets_deg: dict[str, float], timeout_s: float = 4.0,
                       tol_deg: float = 0.5, poll_hz: float = 20.0) -> dict[str, float]:
        """Command target angles, then poll Present Position until every
        joint settles within tol_deg (or timeout). Returns the angles
        actually reached, read back from the encoders -- never the
        commanded values."""
        for joint, angle in targets_deg.items():
            self.set_target_deg(joint, angle)

        deadline = time.monotonic() + timeout_s
        reached: dict[str, float | None] = {joint: None for joint in targets_deg}
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
