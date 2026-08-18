"""Shared test fixtures: a fake `scservo_sdk` module so servos.py's tests
never need a real serial port or real STS3215 hardware."""

from __future__ import annotations

import sys
import types

import pytest


class FakePacketHandler:
    """Records every write, and lets a test script specific
    (comm, err)/exception outcomes per (address, call_index)."""

    def __init__(self, protocol_end):
        self.protocol_end = protocol_end
        self.writes: list[tuple[int, int, int]] = []  # (sid, addr, value)
        self.registers: dict[tuple[int, int], int] = {}
        # addr -> either an exception to raise, or a (comm, err) tuple to return
        self.write_behavior: dict[int, object] = {}

    def ping(self, port, sid):
        return (0, 0, 0)  # (model, comm=COMM_SUCCESS, err)

    def _do_write(self, sid, addr, value):
        behavior = self.write_behavior.get(addr)
        if isinstance(behavior, BaseException):
            raise behavior
        self.writes.append((sid, addr, value))
        self.registers[(sid, addr)] = value
        if behavior is not None:
            return behavior  # explicit (comm, err) override
        return (0, 0)  # COMM_SUCCESS

    def write1ByteTxRx(self, port, sid, addr, value):
        return self._do_write(sid, addr, value)

    def write2ByteTxRx(self, port, sid, addr, value):
        return self._do_write(sid, addr, value)

    def read1ByteTxRx(self, port, sid, addr):
        return self.registers.get((sid, addr), 0), 0, 0

    def read2ByteTxRx(self, port, sid, addr):
        return self.registers.get((sid, addr), 0), 0, 0


class FakePortHandler:
    def __init__(self, port):
        self.port = port

    def setBaudRate(self, baud):
        return True

    def closePort(self):
        pass


@pytest.fixture
def fake_scservo_sdk(monkeypatch):
    fake = types.ModuleType("scservo_sdk")
    fake.COMM_SUCCESS = 0
    fake.PortHandler = FakePortHandler
    fake.PacketHandler = FakePacketHandler
    monkeypatch.setitem(sys.modules, "scservo_sdk", fake)
    return fake
