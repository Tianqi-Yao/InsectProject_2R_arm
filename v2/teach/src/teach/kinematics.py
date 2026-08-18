"""2R planar-arm forward/inverse kinematics -- see kinematics_fit/kinematics.py
for the full explanation of ArmParams' fields (in particular why there is
no base_rotation_deg field, and why elbow_offset_mm is a fixed input, never
fit). teach/ only ever uses a hand-entered or previously-fitted ArmParams
(via kinematics_read.py) for optional workspace-mm display/goto -- it never
fits kinematics itself."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from arm_hw_core.limits import JointLimits, within_joint_limits


@dataclass
class ArmParams:
    L1: float
    L2: float
    base_x: float
    base_y: float
    servo1_offset_deg: float
    servo2_offset_deg: float
    servo1_dir: int = 1
    servo2_dir: int = 1
    elbow_offset_mm: float = 0.0

    @classmethod
    def nominal(cls) -> "ArmParams":
        return cls(L1=125.0, L2=95.0, base_x=100.0, base_y=-45.0,
                    servo1_offset_deg=23.08, servo2_offset_deg=0.0)


@dataclass
class IKResult:
    theta1_deg: float = 0.0
    theta2_deg: float = 0.0
    servo1_deg: float = 0.0
    servo2_deg: float = 0.0
    reachable: bool = False


def _elbow_reach_and_angle(p: ArmParams) -> tuple[float, float]:
    reach = math.hypot(p.L1, p.elbow_offset_mm)
    angle_offset_deg = math.degrees(math.atan2(p.elbow_offset_mm, p.L1))
    return reach, angle_offset_deg


def ik_solve(p: ArmParams, x_ws: float, y_ws: float,
             joint_limits: Optional[JointLimits] = None) -> IKResult:
    ax, ay = x_ws - p.base_x, y_ws - p.base_y
    reach, angle_offset_deg = _elbow_reach_and_angle(p)

    d2 = ax * ax + ay * ay
    c2 = (d2 - reach ** 2 - p.L2 ** 2) / (2.0 * reach * p.L2)
    if c2 < -1.0 or c2 > 1.0:
        return IKResult(reachable=False)

    s2 = math.sqrt(1.0 - c2 * c2)
    theta2_std = math.degrees(math.atan2(s2, c2))
    alpha = math.degrees(math.atan2(ay, ax))
    beta = math.degrees(math.atan2(p.L2 * s2, reach + p.L2 * c2))
    theta1_std = alpha - beta

    theta1 = theta1_std - angle_offset_deg
    theta2 = theta2_std + angle_offset_deg

    servo1 = p.servo1_offset_deg + p.servo1_dir * theta1
    servo2 = p.servo2_offset_deg + p.servo2_dir * theta2
    if not within_joint_limits(servo1, servo2, joint_limits):
        return IKResult(reachable=False)
    return IKResult(theta1, theta2, servo1, servo2, reachable=True)


def fk_from_servo_angles(p: ArmParams, servo1_deg: float, servo2_deg: float) -> tuple[float, float]:
    theta1 = math.radians(p.servo1_dir * (servo1_deg - p.servo1_offset_deg))
    theta2 = math.radians(p.servo2_dir * (servo2_deg - p.servo2_offset_deg))
    reach, angle_offset_deg = _elbow_reach_and_angle(p)
    angle_offset = math.radians(angle_offset_deg)
    ex = reach * math.cos(theta1 + angle_offset) + p.L2 * math.cos(theta1 + theta2)
    ey = reach * math.sin(theta1 + angle_offset) + p.L2 * math.sin(theta1 + theta2)
    return p.base_x + ex, p.base_y + ey
