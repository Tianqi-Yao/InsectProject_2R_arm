"""2R planar-arm forward/inverse kinematics.

NOTE on a parameter deliberately NOT here: a "base mounting rotation" (the
arm's shoulder axis not aligned with the workspace sheet's x-axis) looks
like a natural 7th unknown, but it is mathematically indistinguishable from
servo1_offset_deg when only end-effector *position* is measured (no
orientation sensing): rotating the whole base by delta has the exact same
effect on (ex, ey) as shifting theta1 by delta for every sample, because
both angles feeding the FK (theta1 and theta1+theta2) shift by the same
delta either way. Confirmed empirically on synthetic data: least_squares
converges but arbitrarily splits the true rotation between the two
parameters instead of recovering either one. So: any base-mounting
misalignment is simply absorbed into servo1_offset_deg -- no separate field.
"""

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
    # +1 or -1: which way a joint's raw servo angle increases relative to
    # our math convention (theta measured CCW). A fixed fact about how a
    # servo happens to be mounted/wired -- not something to fit numerically
    # (a sign flip is a reflection, no amount of offset/L1/L2 tuning can
    # reproduce it) -- excluded from as_vector/from_vector.
    servo1_dir: int = 1
    servo2_dir: int = 1
    # The servo2/joint2 mounting offset: L1 ends where servo2's *body* is
    # bolted on, but servo2's *rotation axis* (where L2 actually starts)
    # sits this many mm to the side of L1's own line -- a rigid mechanical
    # fact, 0.0 for a build where the two are colinear.
    #
    # MUST be a fixed, independently-measured constant (CAD/calipers) --
    # NOT something fit_kinematics can solve for. Proven by direct
    # substitution: reach=hypot(L1, elbow_offset_mm) and
    # angle_offset=atan2(elbow_offset_mm, L1) are the only things vision
    # data can ever pin down (see _elbow_reach_and_angle) -- the *split*
    # between L1 and elbow_offset_mm for a given reach is exactly
    # absorbable by trading servo1_offset_deg/servo2_offset_deg against
    # each other. Confirmed empirically: fitting synthetic data generated
    # with elbow_offset_mm=28 recovered elbow_offset_mm=4 with a
    # correspondingly shifted L1, at statistically perfect RMS error
    # regardless of the fit's starting guess or how wide/dense the sampled
    # joint angles were -- more or better vision data cannot resolve this.
    elbow_offset_mm: float = 0.0

    @classmethod
    def nominal(cls) -> "ArmParams":
        """CAD-theoretical starting values -- used as the least-squares
        initial guess, not as a value to trust."""
        return cls(L1=125.0, L2=95.0, base_x=100.0, base_y=-45.0,
                    servo1_offset_deg=23.08, servo2_offset_deg=0.0)

    def as_vector(self) -> list[float]:
        return [self.L1, self.L2, self.base_x, self.base_y,
                self.servo1_offset_deg, self.servo2_offset_deg]

    @classmethod
    def from_vector(cls, vec, servo1_dir: int = 1, servo2_dir: int = 1,
                     elbow_offset_mm: float = 0.0) -> "ArmParams":
        return cls(*vec, servo1_dir=servo1_dir, servo2_dir=servo2_dir,
                    elbow_offset_mm=elbow_offset_mm)


@dataclass
class IKResult:
    theta1_deg: float = 0.0
    theta2_deg: float = 0.0
    servo1_deg: float = 0.0
    servo2_deg: float = 0.0
    reachable: bool = False


def _elbow_reach_and_angle(p: ArmParams) -> tuple[float, float]:
    """The servo2 mounting offset turns the joint1-to-joint2 "link" into a
    right triangle instead of a straight line: the true distance from
    joint1's axis to joint2's axis is hypot(L1, elbow_offset_mm), at a
    fixed angle atan2(elbow_offset_mm, L1) off from the direction theta1
    alone would give. Returns (reach_mm, angle_offset_deg); angle_offset_deg
    is exactly 0 when elbow_offset_mm is 0, so every caller reduces to the
    plain-colinear formula for arms without this offset."""
    reach = math.hypot(p.L1, p.elbow_offset_mm)
    angle_offset_deg = math.degrees(math.atan2(p.elbow_offset_mm, p.L1))
    return reach, angle_offset_deg


def ik_solve(p: ArmParams, x_ws: float, y_ws: float,
             joint_limits: Optional[JointLimits] = None) -> IKResult:
    """Inverse kinematics: workspace (x, y) in mm -> joint/servo angles.

    `joint_limits`: when given, a geometrically reachable point whose
    computed servo angle would fall inside a configured dead zone / outside
    the mechanically safe range is also reported reachable=False --
    callers shouldn't need to separately re-check this after calling
    ik_solve."""
    ax, ay = x_ws - p.base_x, y_ws - p.base_y
    reach, angle_offset_deg = _elbow_reach_and_angle(p)

    d2 = ax * ax + ay * ay
    c2 = (d2 - reach ** 2 - p.L2 ** 2) / (2.0 * reach * p.L2)
    if c2 < -1.0 or c2 > 1.0:
        return IKResult(reachable=False)

    s2 = math.sqrt(1.0 - c2 * c2)  # elbow-up: theta2_std > 0
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
    """Forward kinematics from *measured* servo angles -> workspace (x, y)
    mm. Shared by both the kinematic-parameter fit and the self-check spot
    check: both boil down to "take a real encoder angle pair, predict where
    the end effector should be, compare against what the camera measured."""
    theta1 = math.radians(p.servo1_dir * (servo1_deg - p.servo1_offset_deg))
    theta2 = math.radians(p.servo2_dir * (servo2_deg - p.servo2_offset_deg))
    reach, angle_offset_deg = _elbow_reach_and_angle(p)
    angle_offset = math.radians(angle_offset_deg)
    ex = reach * math.cos(theta1 + angle_offset) + p.L2 * math.cos(theta1 + theta2)
    ey = reach * math.sin(theta1 + angle_offset) + p.L2 * math.sin(theta1 + theta2)
    return p.base_x + ex, p.base_y + ey
