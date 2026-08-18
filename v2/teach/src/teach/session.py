"""Core hand-teach workflow, factored out of cli.py so it's testable
without a real interactive REPL or real hardware (see tests/test_session.py,
which drives this against a fake Servos).

Session invariant: torque is OFF by default (the arm is free to hand-move,
which is the whole point of this mode) -- `goto()` is the one place torque
is briefly re-enabled, and it always re-syncs the controller to the arm's
actual current position first (see controller.ArmController.resync's
docstring for why skipping that is a real hazard, not a nicety)."""

from __future__ import annotations

from typing import Optional

from arm_hw_core.servos import Servos

from .config import TaughtPoint, TeachConfig
from .controller import ArmController
from .kinematics import ArmParams, fk_from_servo_angles


class TeachSession:
    def __init__(self, servos: Servos, controller: ArmController,
                 teach_config: TeachConfig, params: Optional[ArmParams] = None):
        self.servos = servos
        self.controller = controller
        self.teach_config = teach_config
        self.params = params

    def start(self) -> None:
        """Release both joints' torque so the arm can be moved by hand --
        the default state for the rest of the session."""
        self.servos.set_torque_enabled("joint1", False)
        self.servos.set_torque_enabled("joint2", False)

    def stop(self) -> None:
        """Re-enable torque before exiting, holding wherever the arm
        currently is (not snapping toward a stale prior target) -- same
        resync-before-relock safety as arm_hw_core's bring-up procedure."""
        s1 = self.servos.get_present_deg("joint1")
        s2 = self.servos.get_present_deg("joint2")
        self.controller.resync(s1, s2)
        self.servos.set_target_deg("joint1", s1)
        self.servos.set_target_deg("joint2", s2)
        self.servos.set_torque_enabled("joint1", True)
        self.servos.set_torque_enabled("joint2", True)

    def mark(self, label: str) -> TaughtPoint:
        """Record wherever the arm physically is right now (torque
        assumed off, positioned by hand) under `label`, overwriting any
        existing point with the same label."""
        s1 = self.servos.get_present_deg("joint1")
        s2 = self.servos.get_present_deg("joint2")
        x_mm = y_mm = None
        if self.params is not None:
            x_mm, y_mm = fk_from_servo_angles(self.params, s1, s2)
        point = TaughtPoint(label=label, joint1_deg=s1, joint2_deg=s2, x_mm=x_mm, y_mm=y_mm)
        self.teach_config.upsert(point)
        return point

    def goto(self, label: str, timeout_s: float = 8.0) -> tuple[float, float]:
        """Move to a previously taught point under torque, then release
        torque again -- hand-teaching remains the default state between
        commands, not an explicit mode switch the operator must remember."""
        point = self.teach_config.get(label)
        if point is None:
            raise KeyError(f"no taught point named {label!r}")

        s1 = self.servos.get_present_deg("joint1")
        s2 = self.servos.get_present_deg("joint2")
        self.controller.resync(s1, s2)  # torque was off -- arm may have been hand-moved since
        self.servos.set_torque_enabled("joint1", True)
        self.servos.set_torque_enabled("joint2", True)
        try:
            reached = self.controller.run_to_completion(point.joint1_deg, point.joint2_deg,
                                                          timeout_s=timeout_s)
        finally:
            self.servos.set_torque_enabled("joint1", False)
            self.servos.set_torque_enabled("joint2", False)
        return reached
