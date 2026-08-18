import pytest

from arm_hw_core.servos import DEG_PER_TICK, Servos


def _connected_servos(fake_scservo_sdk) -> Servos:
    servos = Servos({"joint1": 1, "joint2": 2})
    servos.connect("/dev/fake")
    return servos


def test_connect_pings_every_joint_and_enables_torque(fake_scservo_sdk):
    servos = _connected_servos(fake_scservo_sdk)
    torque_writes = [w for w in servos._packet.writes if w[1] == Servos.ADDR_TORQUE_ENABLE]
    assert (1, Servos.ADDR_TORQUE_ENABLE, 1) in torque_writes
    assert (2, Servos.ADDR_TORQUE_ENABLE, 1) in torque_writes


def test_write_checked_converts_bare_index_error_to_ioerror(fake_scservo_sdk):
    servos = _connected_servos(fake_scservo_sdk)
    servos._packet.write_behavior[Servos.ADDR_GOAL_POSITION] = IndexError("dropped byte")
    with pytest.raises(IOError, match="dropped/corrupted byte"):
        servos._write_checked(1, Servos.ADDR_GOAL_POSITION, 100, nbytes=2)


def test_write_checked_converts_non_success_comm_to_ioerror(fake_scservo_sdk):
    servos = _connected_servos(fake_scservo_sdk)
    servos._packet.write_behavior[Servos.ADDR_GOAL_POSITION] = (1, 5)  # comm != COMM_SUCCESS(0)
    with pytest.raises(IOError, match="failed"):
        servos._write_checked(1, Servos.ADDR_GOAL_POSITION, 100, nbytes=2)


def test_read_checked_converts_bare_index_error_to_ioerror(fake_scservo_sdk):
    servos = _connected_servos(fake_scservo_sdk)

    def boom(port, sid, addr):
        raise IndexError("dropped byte")

    servos._packet.read2ByteTxRx = boom
    with pytest.raises(IOError, match="dropped/corrupted byte"):
        servos._read_checked(1, Servos.ADDR_PRESENT_POSITION, nbytes=2)


def test_set_hardware_angle_limits_rejects_invalid_ranges(fake_scservo_sdk):
    servos = _connected_servos(fake_scservo_sdk)
    with pytest.raises(ValueError):
        servos.set_hardware_angle_limits("joint1", 100.0, 50.0)  # min >= max
    with pytest.raises(ValueError):
        servos.set_hardware_angle_limits("joint1", -1.0, 50.0)  # out of [0, 360]
    with pytest.raises(ValueError):
        servos.set_hardware_angle_limits("joint1", 0.0, 361.0)


def test_set_hardware_angle_limits_writes_correct_ticks_and_locks(fake_scservo_sdk):
    servos = _connected_servos(fake_scservo_sdk)
    servos.set_hardware_angle_limits("joint1", 10.0, 350.0)

    lock_writes = [w for w in servos._packet.writes if w[1] == Servos.ADDR_LOCK]
    assert lock_writes == [(1, Servos.ADDR_LOCK, 0), (1, Servos.ADDR_LOCK, 1)]

    min_ticks = round(10.0 / DEG_PER_TICK)
    max_ticks = round(350.0 / DEG_PER_TICK)
    assert servos._packet.registers[(1, Servos.ADDR_MIN_ANGLE_LIMIT)] == min_ticks
    assert servos._packet.registers[(1, Servos.ADDR_MAX_ANGLE_LIMIT)] == max_ticks


def test_set_hardware_angle_limits_always_relocks_even_if_a_write_fails(fake_scservo_sdk):
    """The single highest-cost mistake this layer can make is leaving the
    servo's EEPROM unlocked after a failed write -- the `finally` around
    the lock-write must run regardless."""
    servos = _connected_servos(fake_scservo_sdk)
    servos._packet.write_behavior[Servos.ADDR_MAX_ANGLE_LIMIT] = IndexError("bus glitch")

    with pytest.raises(IOError):
        servos.set_hardware_angle_limits("joint1", 10.0, 350.0)

    lock_writes = [w for w in servos._packet.writes if w[1] == Servos.ADDR_LOCK]
    assert lock_writes == [(1, Servos.ADDR_LOCK, 0), (1, Servos.ADDR_LOCK, 1)], (
        "must unlock then re-lock even though the min/max write in between failed"
    )


def test_get_hardware_angle_limits_round_trips(fake_scservo_sdk):
    servos = _connected_servos(fake_scservo_sdk)
    servos.set_hardware_angle_limits("joint2", 20.0, 300.0)
    lo, hi = servos.get_hardware_angle_limits("joint2")
    assert lo == pytest.approx(20.0, abs=DEG_PER_TICK)
    assert hi == pytest.approx(300.0, abs=DEG_PER_TICK)


def test_set_target_deg_skips_redundant_speed_acc_writes(fake_scservo_sdk):
    servos = _connected_servos(fake_scservo_sdk)
    servos.set_target_deg("joint1", 10.0, speed=800, acc=0)
    servos.set_target_deg("joint1", 20.0, speed=800, acc=0)

    acc_writes = [w for w in servos._packet.writes if w[1] == Servos.ADDR_GOAL_ACC]
    speed_writes = [w for w in servos._packet.writes if w[1] == Servos.ADDR_GOAL_SPEED]
    assert len(acc_writes) == 1
    assert len(speed_writes) == 1


def test_move_and_wait_returns_actually_reached_not_commanded(fake_scservo_sdk):
    servos = _connected_servos(fake_scservo_sdk)
    # Pre-seed present position registers so the settle check reads "already there".
    servos._packet.registers[(1, Servos.ADDR_PRESENT_POSITION)] = round(90.0 / DEG_PER_TICK)
    reached = servos.move_and_wait({"joint1": 90.0}, timeout_s=0.5, tol_deg=1.0, poll_hz=50)
    assert reached["joint1"] == pytest.approx(90.0, abs=DEG_PER_TICK)
