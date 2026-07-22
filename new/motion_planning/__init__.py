"""Pluggable joint-space trajectory planners.

This package exists so the answer to "how do we make the arm's motion
smooth" is a swappable algorithm, not a pile of hand-tuned speed/acc
constants scattered across every frontend (which is what it used to be --
see jog_controller.py's docstring for the before/after).

To add a new planner (e.g. a jerk-limited S-curve): create a new module
here implementing TrajectoryPlanner, decorate the class with
@register("your_name"), and import that module at the bottom of this file
so the decorator runs. Nothing else needs to change -- callers select a
planner by name (see arm_core.MotionConfig.planner_name) via get_planner().
"""

from __future__ import annotations

from abc import ABC, abstractmethod

TWO_TUPLE = tuple  # (joint1_value, joint2_value) -- not worth a NamedTuple for two floats


class TrajectoryPlanner(ABC):
    """A planner turns "move from A to B, possibly already moving" into a
    time-sampled sequence of intermediate joint targets, both joints
    synchronized to arrive together."""

    @abstractmethod
    def plan_segment(
        self,
        start_deg: TWO_TUPLE,
        goal_deg: TWO_TUPLE,
        v_start_deg_s: TWO_TUPLE,
        v_end_deg_s: TWO_TUPLE,
        vmax_deg_s: TWO_TUPLE,
        amax_deg_s2: TWO_TUPLE,
        dt_s: float,
    ) -> list[TWO_TUPLE]:
        """Plan one point-to-point segment for both joints at once.

        v_start_deg_s / v_end_deg_s: per-joint velocity this segment should
        begin/end at (0, 0) = come to a full stop, matching how the joint
        was actually moving at the hand-off. Nonzero values are how corner
        blending avoids a full stop between consecutive scan waypoints --
        see jog_controller.py.

        Returns a list of (joint1_deg, joint2_deg) samples spaced dt_s
        apart in time, starting one step after `start_deg` and ending
        exactly at `goal_deg` -- i.e. len(start) is NOT included, so
        callers can just concatenate consecutive segments' outputs.
        """
        raise NotImplementedError


_REGISTRY: dict[str, type[TrajectoryPlanner]] = {}


def register(name: str):
    """Class decorator: makes a TrajectoryPlanner subclass available to
    get_planner(name)."""
    def deco(cls: type[TrajectoryPlanner]) -> type[TrajectoryPlanner]:
        _REGISTRY[name] = cls
        return cls
    return deco


def get_planner(name: str) -> TrajectoryPlanner:
    if name not in _REGISTRY:
        raise ValueError(f"unknown motion planner {name!r}, available: {sorted(_REGISTRY)}")
    return _REGISTRY[name]()


def available_planners() -> list[str]:
    return sorted(_REGISTRY)


from . import trapezoidal  # noqa: E402,F401  -- import triggers @register
