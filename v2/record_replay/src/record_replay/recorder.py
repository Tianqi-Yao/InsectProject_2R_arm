"""Recording and replay workflow, factored out of cli.py so it's testable
without a real interactive REPL, real hardware, or real sleeps (see
tests/test_recorder.py, which drives this against fakes and an injectable
sleep function)."""

from __future__ import annotations

import time
from typing import Callable, Optional

from arm_hw_core.camera import Camera
from arm_hw_core.servos import Servos

from .config import RecordedPoint
from .controller import ArmController


class Recorder:
    """Torque off; the operator hand-drags the arm and calls mark() at
    each position they want recorded."""

    def __init__(self, servos: Servos):
        self.servos = servos

    def start(self) -> None:
        for joint in ("joint1", "joint2"):
            self.servos.set_torque_enabled(joint, False)

    def mark(self) -> RecordedPoint:
        s1 = self.servos.get_present_deg("joint1")
        s2 = self.servos.get_present_deg("joint2")
        return RecordedPoint(s1, s2)

    def stop(self) -> None:
        """Re-sync each joint's goal register to its actual position
        before re-enabling torque, so it doesn't snap toward a stale old
        target the instant torque re-engages -- same hazard as every other
        v2 package's bring-up/teach flow."""
        for joint in ("joint1", "joint2"):
            angle = self.servos.get_present_deg(joint)
            self.servos.set_target_deg(joint, angle)
        for joint in ("joint1", "joint2"):
            self.servos.set_torque_enabled(joint, True)


class Player:
    """Replays a recorded path in order: move to each point, wait for a
    full stop, dwell, optionally snap a photo partway through the dwell."""

    def __init__(self, servos: Servos, controller: ArmController,
                 camera: Optional[Camera] = None):
        self.servos = servos
        self.controller = controller
        self.camera = camera

    def prepare(self) -> None:
        """Must be called once before the first goto_and_dwell(): syncs
        both the servo's own goal register AND the controller's internal
        state to the arm's actual current position, then re-engages
        torque. Handles both possible starting conditions -- fresh boot
        (torque already on, controller's constructor-time snapshot may be
        stale if anything moved it since) and right after a recording
        session (torque off, arm hand-moved)."""
        s1 = self.servos.get_present_deg("joint1")
        s2 = self.servos.get_present_deg("joint2")
        self.servos.set_target_deg("joint1", s1)
        self.servos.set_target_deg("joint2", s2)
        for joint in ("joint1", "joint2"):
            self.servos.set_torque_enabled(joint, True)
        self.controller.resync(s1, s2)

    def goto_and_dwell(self, point: RecordedPoint, dwell_s: float,
                        photo_path=None, photo_delay_s: float = 0.0,
                        sleep: Callable[[float], None] = time.sleep) -> None:
        self.controller.run_to_completion(point.joint1_deg, point.joint2_deg)
        if self.camera is not None and photo_path is not None:
            sleep(photo_delay_s)
            self._save_photo(photo_path)
            sleep(dwell_s - photo_delay_s)
        else:
            sleep(dwell_s)

    def _save_photo(self, photo_path) -> None:
        import cv2

        frame = self.camera.capture_gray()
        cv2.imwrite(str(photo_path), frame)
