"""Joint-space, two-joint-synchronized trapezoidal velocity profile.

This is the standard baseline motion-planning method used across
robotics/CNC controllers: bounded acceleration ramps up to a bounded
cruise velocity, then ramps back down -- as opposed to the old approach
(register-level servo speed/acc limits plus blindly streaming waypoints at
a fixed wall-clock interval, with no real notion of "how far are we and
how fast should we be going").

Math, single axis, given distance D, max velocity Vmax, max acceleration
Amax, and boundary velocities v_entry/v_exit (0 = start/end at rest):

    v_peak = sqrt(Amax*D + (v_entry^2 + v_exit^2)/2)   # peak if no cruise phase fits

    if v_peak <= Vmax:                      # triangular: never reaches Vmax
        accelerate v_entry -> v_peak, decelerate v_peak -> v_exit, no cruise
    else:                                    # trapezoidal: cruise at Vmax
        accelerate v_entry -> Vmax, cruise, decelerate Vmax -> v_exit

Two joints are synchronized by solving this independently for each joint,
taking the longer of the two times T as the shared segment duration, and
driving BOTH joints from a single normalized progress curve s(t) built
from the *slower* (time-dominant) joint's own profile shape:

    angle_i(t) = start_i + s(t) * (goal_i - start_i)

This keeps the path a straight line in joint space with an exact
simultaneous arrival. The non-dominant joint's actual velocity/accel end
up scaled down proportionally -- safe as long as it isn't itself more
constrained for its own distance (true by construction: it wasn't the one
that took longer). If link/servo limits ever diverge enough for that
assumption to break, the fix is solving each joint's own accel phase
against the shared T instead of reusing the dominant joint's shape -- not
needed at this project's parameter scale.
"""

from __future__ import annotations

import math

from arm_core import wrap_angle_near

from . import TrajectoryPlanner, register


def _solve_single_axis(distance: float, v_entry: float, v_exit: float,
                        vmax: float, amax: float) -> tuple[float, float, float, float, float]:
    """Returns (t1, t2, t3, v_peak, T) for one joint's accel/cruise/decel
    phases covering `distance` (>=0), starting at v_entry and ending at
    v_exit (both clamped to vmax)."""
    v_entry = max(0.0, min(v_entry, vmax))
    v_exit = max(0.0, min(v_exit, vmax))

    v_peak_sq = amax * distance + 0.5 * (v_entry ** 2 + v_exit ** 2)
    v_peak = math.sqrt(max(v_peak_sq, 0.0))

    if v_peak <= vmax:
        # Triangular profile: distance too short to reach vmax.
        t1 = (v_peak - v_entry) / amax if v_peak > v_entry else 0.0
        t3 = (v_peak - v_exit) / amax if v_peak > v_exit else 0.0
        t2 = 0.0
    else:
        v_peak = vmax
        t1 = (vmax - v_entry) / amax
        t3 = (vmax - v_exit) / amax
        d1 = (v_entry + vmax) / 2.0 * t1
        d3 = (vmax + v_exit) / 2.0 * t3
        d2 = max(distance - d1 - d3, 0.0)
        t2 = d2 / vmax if vmax > 0 else 0.0

    return t1, t2, t3, v_peak, t1 + t2 + t3


def _progress_at(t: float, t1: float, t2: float, v_entry: float,
                  v_peak: float, amax: float, distance: float) -> float:
    """Fraction of `distance` covered by the dominant axis at time t,
    given its own (t1, t2, v_entry, v_peak) phase boundaries."""
    if distance <= 0.0:
        return 1.0
    if t <= t1:
        pos = v_entry * t + 0.5 * amax * t * t
    elif t <= t1 + t2:
        d1 = v_entry * t1 + 0.5 * amax * t1 * t1
        pos = d1 + v_peak * (t - t1)
    else:
        d1 = v_entry * t1 + 0.5 * amax * t1 * t1
        d2 = v_peak * t2
        td = t - t1 - t2
        pos = d1 + d2 + v_peak * td - 0.5 * amax * td * td
    return min(pos / distance, 1.0)


@register("trapezoidal")
class TrapezoidalPlanner(TrajectoryPlanner):
    def plan_segment(self, start_deg, goal_deg, v_start_deg_s, v_end_deg_s,
                      vmax_deg_s, amax_deg_s2, dt_s):
        # Re-express each goal as whichever angle congruent to it (mod
        # 360) is nearest start_deg, BEFORE computing a distance to
        # travel: goal_deg comes from ik_solve()'s atan2-based math (or a
        # raw operator-typed angle), with no reason to land anywhere near
        # the arm's current position numerically, even when it's only a
        # couple degrees away physically (e.g. start=359, goal=1 is a
        # 2deg move, not the 358deg a raw subtraction would compute) --
        # see arm_core.wrap_angle_near.
        goal_deg = tuple(wrap_angle_near(goal_deg[i], start_deg[i]) for i in (0, 1))
        D = [goal_deg[i] - start_deg[i] for i in (0, 1)]
        absD = [abs(d) for d in D]

        if absD[0] == 0.0 and absD[1] == 0.0:
            return [tuple(goal_deg)]

        axes = []
        for i in (0, 1):
            t1, t2, t3, v_peak, T = _solve_single_axis(
                absD[i], abs(v_start_deg_s[i]), abs(v_end_deg_s[i]),
                vmax_deg_s[i], amax_deg_s2[i])
            axes.append((t1, t2, t3, v_peak, T))

        dominant = 0 if axes[0][4] >= axes[1][4] else 1
        t1, t2, t3, v_peak, T = axes[dominant]
        v_entry_dom = max(0.0, min(abs(v_start_deg_s[dominant]), vmax_deg_s[dominant]))
        amax_dom = amax_deg_s2[dominant]
        D_dom = absD[dominant]

        # Space samples evenly across [0, T] rather than at fixed multiples
        # of dt_s: if T isn't an exact multiple of dt_s, a fixed-dt_s grid
        # clamped to T leaves a short final interval, which understates the
        # true exit velocity when computed as (last_sample - prev_sample)/dt_s
        # -- exactly the case corner blending depends on getting right.
        # Using an adjusted per-segment step (T/n_steps, close to but not
        # exactly dt_s) keeps every interval uniform and lands exactly on T.
        n_steps = max(1, round(T / dt_s)) if T > 0 else 1
        step = T / n_steps if T > 0 else dt_s
        samples = []
        for k in range(1, n_steps + 1):
            t = min(k * step, T)
            s = _progress_at(t, t1, t2, v_entry_dom, v_peak, amax_dom, D_dom)
            samples.append((start_deg[0] + s * D[0], start_deg[1] + s * D[1]))
        samples[-1] = tuple(goal_deg)  # exact landing, guards against fp roundoff
        return samples
