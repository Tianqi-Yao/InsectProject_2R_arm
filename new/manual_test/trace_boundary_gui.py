"""Live pygame tool for capturing the joint2-vs-joint1 coupled dead zone by
hand, with real-time visual feedback.

Press 'b' to start recording, then walk the arm by hand (both joints'
torque released) around the FULL PERIMETER of the safe region -- one
closed loop, back to about where you started -- while the window plots
every sample (converted to real workspace mm via forward kinematics) as a
connected line, live. Press 'b' again to stop. The traced polygon IS the
boundary, exactly as drawn: no binning, smoothing, or min/max derivation
applied to it (see arm_core._point_in_polygon, which decides in/out
later). This replaced an earlier automatic-derivation approach (sweep the
interior, bin by joint1, take min/max per bucket) that produced visibly
wrong results on real hardware -- tracing the boundary by hand and using
it as-is puts you in full control of exactly what gets enforced.

's'/Enter saves to calib.json AND writes a screenshot of the window at
that moment (SCREENSHOT_PATH, "joint_limits_trace.png") -- a plain
pygame.image.save of whatever's currently drawn, not a separately
generated plot, so what you see saved to calib.json is exactly what's in
the picture.

This assumes main.py's `set-joint-limits` has already been run at least
once, to establish each joint's own independent safe range (joint1/joint2
in calib.json's joint_limits_deg) -- this tool only adds/replaces the
coupled_boundary on top of that, it doesn't replace the per-joint sweep.

This same live window also doubles as a quick sanity check on whether
calib.json's kinematics are any good at all: the drawn arm is the
simulation's belief (real encoder angles run through calib.json's L1/L2/
offsets), so if it visibly doesn't match the physical arm's pose (e.g. the
real L1-L2 angle is 90deg but the drawing shows something else), that's a
calibration problem, not a bug in this viewer. If you haven't run
`main.py calibrate` yet (or just want a fast fix for servo2_offset_deg
specifically without the camera), fold the arm by hand to a known L1-L2
angle -- read with a protractor/set-square at the elbow -- and press 'k'
(see --elbow-ref-deg below and arm_core.servo2_offset_from_known_elbow_angle).

Press 'r' to REPLAY the saved coupled_boundary: torque re-engages and the
arm is actively driven (via jog_controller.ArmController, the same smooth
trapezoidal planner every other tool here uses) through every saved
vertex in order, back to the first one -- so you can watch both the
simulated and the real arm trace the exact boundary you saved, as a
sanity check that it's really where you think it is. This drives the arm
RIGHT AT the edge of the configured dead zone by definition -- there's no
inward safety margin applied -- so watch closely and be ready to press
'r' again (or q/ESC) to stop early if anything looks wrong. Moves at
calib.json's jog speed/accel; the coupled-boundary polygon check itself is
deliberately not re-applied during replay (a saved vertex sits exactly ON
that polygon's own edge, which is a genuinely ambiguous case for
point-in-polygon -- see arm_core._point_in_polygon -- so re-checking a
boundary point against its own boundary could reject it by a coin flip);
the independent joint1/joint2 hardware-backed ranges are still enforced.
"""

import sys
import time
from dataclasses import dataclass
from pathlib import Path

import pygame

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import arm_core as core          # noqa: E402
import arm_hardware as hw         # noqa: E402
import jog_controller as jc       # noqa: E402

POLL_INTERVAL_S = 0.05
FPS = 60
SCREENSHOT_PATH = "joint_limits_trace.png"

BG = (28, 30, 40)
GRID = (48, 54, 72)
LINK1_C = (80, 160, 255)
LINK2_C = (255, 120, 55)
JOINT_C = (220, 220, 220)
EE_C = (55, 215, 95)
BASE_C = (200, 100, 55)
BOUNDARY_C = (255, 210, 60)
GHOST_C = (170, 90, 220)
TEXT_C = (200, 210, 225)
LABEL_C = (110, 130, 155)
HIGHLIGHT = (255, 220, 80)
WARN_C = (220, 80, 80)
OK_C = (80, 220, 130)


@dataclass
class Layout:
    """Screen-space layout centered on the base, sized to the arm's full
    reach circle -- the dead zone can extend well outside the (much
    smaller) camera-calibrated work sheet, so this doesn't reuse
    manual_test/gui.py's workspace-rect layout."""
    base_x: float
    base_y: float
    max_r_mm: float
    scale: float = 1.0
    margin_px: int = 40
    panel_w: int = 260

    def __post_init__(self):
        canvas_px = 560
        self.scale = canvas_px / (2 * self.max_r_mm)

    @property
    def win_w(self) -> int:
        return int(2 * self.max_r_mm * self.scale) + 2 * self.margin_px + self.panel_w

    @property
    def win_h(self) -> int:
        return int(2 * self.max_r_mm * self.scale) + 2 * self.margin_px

    def ws2s(self, wx: float, wy: float) -> tuple:
        cx = self.margin_px + self.max_r_mm * self.scale
        cy = self.margin_px + self.max_r_mm * self.scale
        return (int(cx + (wx - self.base_x) * self.scale),
                int(cy - (wy - self.base_y) * self.scale))


def _resync_and_relock(servos):
    """Sync goal to current position before re-enabling torque, so the
    servo doesn't snap toward a stale old target -- same convention as
    main.py's hand-capture helpers."""
    for joint in ("joint1", "joint2"):
        try:
            angle = servos.get_present_deg(joint)
            servos.set_target_deg(joint, angle)
        except Exception:
            pass
    for joint in ("joint1", "joint2"):
        servos.set_torque_enabled(joint, True)


def _build_replay_controller(servos, calib):
    """Same ArmController every other tool here uses (smooth trapezoidal
    planning, not a single raw hardware move), but with the coupled-
    boundary polygon check dropped from what it enforces: replay's whole
    job is to visit that polygon's own vertices, which sit exactly ON its
    edge -- a case _point_in_polygon documents as genuinely ambiguous, so
    re-checking a boundary point against its own boundary could reject it
    arbitrarily. The independent joint1/joint2 ranges (the ones also
    backed by the servo's own hardware registers) are still enforced."""
    controller = jc.build_controller(servos, calib)
    if controller.joint_limits is not None:
        controller.joint_limits = {**controller.joint_limits, "coupled_boundary": []}
    return controller


def _replay_waypoints(boundary):
    """The saved boundary, walked in order and back to the first vertex --
    one full lap, matching what was traced."""
    return list(boundary) + [boundary[0]]


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Live coupled dead-zone boundary trace + quick servo2_offset_deg fix")
    parser.add_argument(
        "--elbow-ref-deg", type=float, default=90.0,
        help="the L1-L2 angle you'll fold the arm to for the 'k' quick-fix key, "
             "as read by a protractor/set-square at the elbow -- 180=fully "
             "straight, 0=fully folded (default: 90.0)")
    args = parser.parse_args()

    calib = core.load_calib()
    limits = core.calib_joint_limits(calib)
    if limits is None:
        print("calib.json has no joint_limits_deg yet -- run "
              "`python3 main.py set-joint-limits` first to establish each "
              "joint's own independent safe range. This tool only adds the "
              "coupled boundary on top of that.")
        return

    params = core.calib_arm_params(calib)
    hw_cfg = core.calib_hardware_config(calib)
    servos = hw.Servos(hw_cfg.joint_ids)
    servos.connect(hw_cfg.servo_port)
    controller = _build_replay_controller(servos, calib)

    for joint in ("joint1", "joint2"):
        servos.set_torque_enabled(joint, False)

    max_r = params.L1 + params.L2
    layout = Layout(base_x=params.base_x, base_y=params.base_y, max_r_mm=max_r)

    pygame.init()
    pygame.display.set_caption("2R Arm -- coupled dead-zone boundary trace")
    screen = pygame.display.set_mode((layout.win_w, layout.win_h))
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("menlo,consolas,monospace", 18)
    sfont = pygame.font.SysFont("menlo,consolas,monospace", 13)

    boundary_trace: list = []  # (joint1_deg, joint2_deg), in traced order -- exactly as recorded
    recording = False
    replaying = False
    replay_queue: list = []
    save_msg = ""
    save_msg_until = 0.0
    screenshot_pending = False  # deferred to after this frame finishes drawing -- see below
    last_poll = 0.0
    s1 = servos.get_present_deg("joint1")
    s2 = servos.get_present_deg("joint2")

    running = True
    try:
        while running:
            now = time.monotonic()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_q, pygame.K_ESCAPE):
                        running = False
                    elif event.key == pygame.K_b:
                        if replaying:
                            save_msg = "stop the replay ('r') before recording"
                            save_msg_until = now + 3.0
                        else:
                            if not recording:
                                boundary_trace = []  # starting a new loop
                                servos.set_torque_enabled("joint1", False)
                                servos.set_torque_enabled("joint2", False)
                            recording = not recording
                    elif event.key == pygame.K_r:
                        if recording:
                            save_msg = "stop recording ('b') before replaying"
                            save_msg_until = now + 3.0
                        elif replaying:
                            replaying = False
                            replay_queue = []
                            save_msg = "replay stopped"
                            save_msg_until = now + 2.0
                        elif not limits["coupled_boundary"]:
                            save_msg = "no saved coupled_boundary to replay yet -- trace and save one first"
                            save_msg_until = now + 3.0
                        else:
                            servos.set_torque_enabled("joint1", True)
                            servos.set_torque_enabled("joint2", True)
                            replay_queue = _replay_waypoints(limits["coupled_boundary"])
                            controller.set_joint_goal(*replay_queue.pop(0))
                            replaying = True
                    elif event.key == pygame.K_k:
                        flip = bool(event.mod & pygame.KMOD_SHIFT)
                        old = calib["kinematics"]["servo2_offset_deg"]
                        new = core.servo2_offset_from_known_elbow_angle(
                            s2, params.servo2_dir, args.elbow_ref_deg, flip=flip)
                        calib["kinematics"]["servo2_offset_deg"] = new
                        core.save_calib(calib)
                        params = core.calib_arm_params(calib)
                        other_key = "k" if flip else "shift+k"
                        save_msg = (f"servo2_offset_deg: {old:.2f} -> {new:.2f} (saved) -- "
                                    f"if the simulated angle still looks wrong/mirrored, "
                                    f"try {other_key} instead")
                        print(f"[{'shift+k' if flip else 'k'}] servo2_offset_deg: "
                              f"{old:.2f} -> {new:.2f}, saved to calib.json "
                              f"(assumed elbow folded to {args.elbow_ref_deg:.1f}deg "
                              f"{'the flipped way' if flip else ''})")
                        save_msg_until = now + 6.0
                    elif event.key == pygame.K_c:
                        boundary_trace = []
                        recording = False
                    elif event.key in (pygame.K_s, pygame.K_RETURN):
                        if len(boundary_trace) < 3:
                            save_msg = "need >=3 vertices to form a closed loop -- keep tracing"
                        else:
                            j1_indep = limits["joint1"]
                            j2_indep = limits["joint2"]
                            j1_lo = min([j1_indep[0]] + [v[0] for v in boundary_trace])
                            j1_hi = max([j1_indep[1]] + [v[0] for v in boundary_trace])
                            j2_lo = min([j2_indep[0]] + [v[1] for v in boundary_trace])
                            j2_hi = max([j2_indep[1]] + [v[1] for v in boundary_trace])
                            calib["joint_limits_deg"] = {
                                "joint1": [round(j1_lo, 2), round(j1_hi, 2)],
                                "joint2": [round(j2_lo, 2), round(j2_hi, 2)],
                                "coupled_boundary": [{"joint1": round(j1, 2), "joint2": round(j2, 2)}
                                                     for j1, j2 in boundary_trace],
                            }
                            core.save_calib(calib)
                            # Refresh so 'r' replays the boundary just saved,
                            # not whatever was on disk when this session started.
                            limits = core.calib_joint_limits(calib)
                            controller.joint_limits = {**limits, "coupled_boundary": []}
                            # Deferred to right after this frame's drawing
                            # finishes below, so the saved image includes
                            # everything just drawn (not last frame's stale
                            # content) -- see the `pygame.display.flip()` call.
                            screenshot_pending = True
                            save_msg = (f"saved {len(boundary_trace)}-vertex boundary to "
                                        f"calib.json + {SCREENSHOT_PATH}")
                            print(f"saved coupled_boundary ({len(boundary_trace)} vertices) "
                                  f"to calib.json; joint1 range [{j1_lo:.1f},{j1_hi:.1f}], "
                                  f"joint2 range [{j2_lo:.1f},{j2_hi:.1f}] "
                                  f"(hardware registers NOT touched by this tool -- "
                                  f"the coupled boundary is software-only; if the "
                                  f"independent ranges widened, re-run "
                                  f"`main.py set-joint-limits` to also update the "
                                  f"hardware registers)")
                        save_msg_until = now + 4.0

            if now - last_poll >= POLL_INTERVAL_S:
                s1 = servos.get_present_deg("joint1")
                s2 = servos.get_present_deg("joint2")
                if recording:
                    boundary_trace.append((s1, s2))
                last_poll = now

            if replaying:
                controller.tick()
                if not controller.is_moving:
                    if replay_queue:
                        controller.set_joint_goal(*replay_queue.pop(0))
                    else:
                        replaying = False
                        save_msg = "replay finished"
                        save_msg_until = now + 3.0

            screen.fill(BG)

            # reference circle + base
            import math
            steps = [i / 200.0 * 2 * math.pi for i in range(201)]
            pygame.draw.lines(
                screen, GRID,
                False,
                [layout.ws2s(params.base_x + max_r * math.cos(t),
                             params.base_y + max_r * math.sin(t)) for t in steps],
                1)
            pygame.draw.circle(screen, BASE_C, layout.ws2s(params.base_x, params.base_y), 9, 2)

            # traced boundary, drawn as a connected (closed, if >=2 points) loop
            if boundary_trace:
                loop = [layout.ws2s(*core.fk_from_servo_angles(params, j1, j2))
                        for j1, j2 in boundary_trace]
                if len(loop) >= 2:
                    pygame.draw.lines(screen, BOUNDARY_C, True, loop, 2)
                else:
                    pygame.draw.circle(screen, BOUNDARY_C, loop[0], 3)

            # while replaying: the commanded/planned pose, a thin ghost
            # behind the real arm below -- shows the planner is actually
            # driving it, not just coincidentally matching
            if replaying:
                cmd_s1, cmd_s2 = controller.commanded_deg
                g_elbow, g_ee = core.fk_joint_positions(params, cmd_s1, cmd_s2)
                p_base_g, p_elbow_g, p_ee_g = (layout.ws2s(params.base_x, params.base_y),
                                                layout.ws2s(*g_elbow), layout.ws2s(*g_ee))
                pygame.draw.line(screen, GHOST_C, p_base_g, p_elbow_g, 3)
                pygame.draw.line(screen, GHOST_C, p_elbow_g, p_ee_g, 2)
                pygame.draw.circle(screen, GHOST_C, p_ee_g, 5, 1)

            # current (real, encoder-read) arm pose
            elbow, ee = core.fk_joint_positions(params, s1, s2)
            p_base, p_elbow, p_ee = (layout.ws2s(params.base_x, params.base_y),
                                      layout.ws2s(*elbow), layout.ws2s(*ee))
            pygame.draw.line(screen, LINK1_C, p_base, p_elbow, 6)
            pygame.draw.line(screen, LINK2_C, p_elbow, p_ee, 5)
            pygame.draw.circle(screen, JOINT_C, p_elbow, 7)
            pygame.draw.circle(screen, EE_C, p_ee, 8)

            # panel
            px, py = int(2 * max_r * layout.scale) + layout.margin_px * 2, 28

            def row(text, dy, color=TEXT_C, f=None):
                nonlocal py
                screen.blit((f or font).render(text, True, color), (px, py))
                py += dy

            theta2_now = params.servo2_dir * (s2 - params.servo2_offset_deg)
            included_angle_now = 180.0 - theta2_now

            row("COUPLED DEAD-ZONE BOUNDARY", 22, HIGHLIGHT)
            row(f"joint1={s1:6.1f}  joint2={s2:6.1f}", 18, TEXT_C, sfont)
            row(f"L1-L2 angle (sim): {included_angle_now:6.1f} deg -- "
                f"compare to protractor", 22, TEXT_C, sfont)
            row("RECORDING" if recording else "stopped", 18,
                OK_C if recording else LABEL_C, sfont)
            row(f"{len(boundary_trace)} vertices", 22, LABEL_C, sfont)
            if replaying:
                row(f"REPLAYING -- {len(replay_queue)} waypoints left", 22, WARN_C, sfont)
            else:
                saved_n = len(limits["coupled_boundary"])
                row(f"saved boundary: {saved_n} vertices" if saved_n else
                    "saved boundary: none yet", 22, LABEL_C, sfont)
            if save_msg and now < save_msg_until:
                row(save_msg, 22, OK_C, sfont)
            else:
                row(" ", 22)
            row("-- Keys --", 13, LABEL_C, sfont)
            for line in ["b         start/stop recording the boundary",
                         "          (starting clears the current trace)",
                         "c         clear trace, stop recording",
                         f"s/Enter   save to calib.json + {SCREENSHOT_PATH}",
                         "r         replay saved boundary (drives the",
                         "          real arm!); r again to stop early",
                         f"k         fold elbow to {args.elbow_ref_deg:.0f}deg, fix",
                         "          servo2_offset_deg (no camera needed)",
                         "shift+k   same, other fold direction",
                         "q/ESC     quit (no save)"]:
                row(line, 16, LABEL_C, sfont)

            pygame.display.flip()
            if screenshot_pending:
                try:
                    pygame.image.save(screen, SCREENSHOT_PATH)
                    print(f"saved a screenshot of the traced boundary to {SCREENSHOT_PATH}")
                except Exception as e:  # noqa: BLE001 -- a screenshot is optional, never fatal
                    print(f"WARNING: could not save {SCREENSHOT_PATH} ({e}) -- "
                          f"calib.json's data is unaffected")
                screenshot_pending = False
            clock.tick(FPS)
    finally:
        pygame.quit()
        _resync_and_relock(servos)
        servos.close()


if __name__ == "__main__":
    main()
