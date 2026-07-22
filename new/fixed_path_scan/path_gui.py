"""Live pygame tool for a fixed, repeatable inspection path -- no AprilTag/
camera calibration involved at all.

Deliberately independent of the vision pipeline: this only reads
calib.json's kinematics/hardware/motion/joint_limits_deg sections (via
arm_core.calib_arm_params/calib_hardware_config/calib_motion_config/
calib_joint_limits) -- the same four sections manual_test/gui.py and
manual_test/run.py already treat as decoupled from the calibration
*sheet*/homography concept (see arm_core.py's module docstring for that
split). workspace/homography are never touched here.

Two corners define a rectangle, taught by JOGGING THE REAL ARM (arrow
keys, same nudge_workspace()-based jog as manual_test/gui.py) and pressing
'1'/'2' to record wherever the arm currently is -- no mouse, no typed
coordinates. Rows/cols set the grid density (see path_core.PathConfig);
node spacing is derived and shown, not set directly.

TEACH mode (default): jog + record corners + adjust rows/cols/dwell_s +
live preview of the generated serpentine node path, with each node colored
by IK reachability.

RUN mode ('g'/Enter, only once both corners are taught and every node is
reachable): drives the real arm node by node via path_core.PathRunner,
which rides jog_controller.ArmController.set_workspace_goal()/tick() to a
full stop at each node, dwells dwell_s seconds, and calls an on_arrive
hook -- currently path_core.default_on_arrive, a placeholder reserved for
real camera-capture code later. 'q'/ESC aborts a run early and returns to
TEACH; the arm simply holds its last commanded position (no in-flight
segment is force-cancelled).
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
import path_core as pc            # noqa: E402

STEP_MM = 5.0             # fixed jog step per arrow-key press -- no adjustment keys, unlike
                           # manual_test/gui.py's '['/']' (those select rows/cols here instead)
POLL_INTERVAL_S = 0.05
FPS = 60
SCREENSHOT_PATH = "path_preview.png"

BG = (28, 30, 40)
GRID = (48, 54, 72)
LINK1_C = (80, 160, 255)
LINK2_C = (255, 120, 55)
JOINT_C = (220, 220, 220)
EE_C = (55, 215, 95)
BASE_C = (200, 100, 55)
GHOST_C = (170, 90, 220)
CORNER_C = (255, 210, 60)
NODE_OK_C = (80, 220, 130)
NODE_BAD_C = (220, 80, 80)
NODE_CUR_C = (255, 230, 80)
TEXT_C = (200, 210, 225)
LABEL_C = (110, 130, 155)
HIGHLIGHT = (255, 220, 80)
WARN_C = (220, 80, 80)
OK_C = (80, 220, 130)


@dataclass
class Layout:
    """Screen-space layout centered on the arm's base, sized to its full
    reach circle -- same approach as manual_test/trace_boundary_gui.py's
    Layout, deliberately NOT manual_test/gui.py's workspace-rect layout,
    since this tool has no calibration-sheet concept to size a window
    around. Both taught corners and every generated node are guaranteed to
    be within reach (corners are recorded from the real, already-reachable
    arm position; nodes failing reachability are flagged, not hidden), so
    this circle always contains everything worth drawing."""
    base_x: float
    base_y: float
    max_r_mm: float
    scale: float = 1.0
    margin_px: int = 40
    panel_w: int = 280

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


def _reachable_nodes(nodes, params, joint_limits):
    """[(x, y, label, reachable_bool), ...] for preview coloring."""
    out = []
    for x, y, label in nodes:
        r = core.ik_solve(params, x, y, joint_limits=joint_limits)
        out.append((x, y, label, r.reachable))
    return out


def main():
    calib = core.load_calib()
    hw_cfg = core.calib_hardware_config(calib)
    servos = hw.Servos(hw_cfg.joint_ids)
    servos.connect(hw_cfg.servo_port)

    controller = jc.build_controller(servos, calib)
    params = controller.params
    cfg = pc.load_path_config()

    max_r = params.L1 + params.L2
    layout = Layout(base_x=params.base_x, base_y=params.base_y, max_r_mm=max_r)

    pygame.init()
    pygame.display.set_caption("2R Arm -- fixed path inspection")
    screen = pygame.display.set_mode((layout.win_w, layout.win_h))
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("menlo,consolas,monospace", 18)
    sfont = pygame.font.SysFont("menlo,consolas,monospace", 13)

    real_s1 = servos.get_present_deg("joint1")
    real_s2 = servos.get_present_deg("joint2")
    workspace_target = core.fk_from_servo_angles(params, real_s1, real_s2)

    mode = "teach"
    runner = None
    msg = ""
    msg_until = 0.0
    screenshot_pending = False
    last_poll = 0.0

    running = True
    try:
        while running:
            now = time.monotonic()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_q, pygame.K_ESCAPE):
                        if mode == "run":
                            mode = "teach"
                            runner = None
                            msg = "run aborted"
                            msg_until = now + 2.0
                        else:
                            running = False
                    elif mode == "run":
                        pass  # ignore all other keys while a run is in progress
                    elif event.key == pygame.K_1:
                        cfg.corner_a_mm = workspace_target
                        msg = f"corner A recorded: ({workspace_target[0]:.1f}, {workspace_target[1]:.1f}) mm"
                        msg_until = now + 3.0
                    elif event.key == pygame.K_2:
                        cfg.corner_b_mm = workspace_target
                        msg = f"corner B recorded: ({workspace_target[0]:.1f}, {workspace_target[1]:.1f}) mm"
                        msg_until = now + 3.0
                    elif event.key == pygame.K_LEFTBRACKET:
                        cfg.cols = max(cfg.cols - 1, 2)
                    elif event.key == pygame.K_RIGHTBRACKET:
                        cfg.cols += 1
                    elif event.key == pygame.K_SEMICOLON:
                        cfg.rows = max(cfg.rows - 1, 2)
                    elif event.key == pygame.K_QUOTE:
                        cfg.rows += 1
                    elif event.key == pygame.K_MINUS:
                        cfg.dwell_s = max(cfg.dwell_s - 0.2, 0.0)
                    elif event.key == pygame.K_EQUALS:
                        cfg.dwell_s += 0.2
                    elif event.key == pygame.K_s:
                        pc.save_path_config(cfg)
                        screenshot_pending = True
                        msg = f"saved path_config.json + {SCREENSHOT_PATH}"
                        msg_until = now + 3.0
                    elif event.key == pygame.K_p:
                        screenshot_pending = True
                        msg = f"saved {SCREENSHOT_PATH}"
                        msg_until = now + 3.0
                    elif event.key in (pygame.K_g, pygame.K_RETURN):
                        if cfg.corner_a_mm is None or cfg.corner_b_mm is None:
                            msg = "teach both corners ('1' and '2') before running"
                        else:
                            nodes = pc.generate_node_path(cfg)
                            unreachable = sum(
                                1 for x, y, _l in nodes
                                if not core.ik_solve(params, x, y, controller.joint_limits).reachable)
                            if unreachable:
                                msg = f"{unreachable}/{len(nodes)} node(s) unreachable -- adjust the rectangle/grid first"
                            else:
                                runner = pc.PathRunner(controller, nodes, cfg.dwell_s,
                                                        on_arrive=pc.default_on_arrive)
                                mode = "run"
                                msg = f"running {len(nodes)} nodes"
                        msg_until = now + 4.0
                    elif event.key == pygame.K_UP:
                        new_t = controller.nudge_workspace(0.0, STEP_MM, workspace_target)
                        workspace_target = new_t if new_t else workspace_target
                    elif event.key == pygame.K_DOWN:
                        new_t = controller.nudge_workspace(0.0, -STEP_MM, workspace_target)
                        workspace_target = new_t if new_t else workspace_target
                    elif event.key == pygame.K_LEFT:
                        new_t = controller.nudge_workspace(-STEP_MM, 0.0, workspace_target)
                        workspace_target = new_t if new_t else workspace_target
                    elif event.key == pygame.K_RIGHT:
                        new_t = controller.nudge_workspace(STEP_MM, 0.0, workspace_target)
                        workspace_target = new_t if new_t else workspace_target

            if now - last_poll >= POLL_INTERVAL_S:
                real_s1 = servos.get_present_deg("joint1")
                real_s2 = servos.get_present_deg("joint2")
                last_poll = now

            if mode == "run":
                runner.tick(now)
                if runner.done:
                    mode = "teach"
                    runner = None
                    msg = "path finished"
                    msg_until = now + 3.0
            else:
                controller.tick()

            screen.fill(BG)

            import math
            steps = [i / 200.0 * 2 * math.pi for i in range(201)]
            pygame.draw.lines(
                screen, GRID, False,
                [layout.ws2s(params.base_x + max_r * math.cos(t),
                             params.base_y + max_r * math.sin(t)) for t in steps], 1)
            pygame.draw.circle(screen, BASE_C, layout.ws2s(params.base_x, params.base_y), 9, 2)

            if cfg.corner_a_mm is not None:
                pygame.draw.circle(screen, CORNER_C, layout.ws2s(*cfg.corner_a_mm), 6, 2)
            if cfg.corner_b_mm is not None:
                pygame.draw.circle(screen, CORNER_C, layout.ws2s(*cfg.corner_b_mm), 6, 2)

            nodes = []
            if cfg.corner_a_mm is not None and cfg.corner_b_mm is not None:
                nodes = pc.generate_node_path(cfg)
                colored = _reachable_nodes(nodes, params, controller.joint_limits)
                points = [layout.ws2s(x, y) for x, y, _l, _r in colored]
                if len(points) >= 2:
                    pygame.draw.lines(screen, LABEL_C, False, points, 1)
                for i, (x, y, _l, reachable) in enumerate(colored):
                    p = layout.ws2s(x, y)
                    if mode == "run" and runner is not None and i == runner.index:
                        pygame.draw.circle(screen, NODE_CUR_C, p, 6)
                    else:
                        pygame.draw.circle(screen, NODE_OK_C if reachable else NODE_BAD_C, p, 3)

            if mode == "run":
                cmd_s1, cmd_s2 = controller.commanded_deg
                g_elbow, g_ee = core.fk_joint_positions(params, cmd_s1, cmd_s2)
                p_base_g = layout.ws2s(params.base_x, params.base_y)
                p_elbow_g, p_ee_g = layout.ws2s(*g_elbow), layout.ws2s(*g_ee)
                pygame.draw.line(screen, GHOST_C, p_base_g, p_elbow_g, 3)
                pygame.draw.line(screen, GHOST_C, p_elbow_g, p_ee_g, 2)
                pygame.draw.circle(screen, GHOST_C, p_ee_g, 5, 1)

            elbow, ee = core.fk_joint_positions(params, real_s1, real_s2)
            p_base, p_elbow, p_ee = (layout.ws2s(params.base_x, params.base_y),
                                      layout.ws2s(*elbow), layout.ws2s(*ee))
            pygame.draw.line(screen, LINK1_C, p_base, p_elbow, 6)
            pygame.draw.line(screen, LINK2_C, p_elbow, p_ee, 5)
            pygame.draw.circle(screen, JOINT_C, p_elbow, 7)
            pygame.draw.circle(screen, EE_C, p_ee, 8)

            px, py = int(2 * max_r * layout.scale) + layout.margin_px * 2, 28

            def row(text, dy, color=TEXT_C, f=None):
                nonlocal py
                screen.blit((f or font).render(text, True, color), (px, py))
                py += dy

            row("FIXED PATH INSPECTION", 22, HIGHLIGHT)
            row("RUNNING" if mode == "run" else "teach", 18,
                WARN_C if mode == "run" else OK_C, sfont)

            a_txt = f"({cfg.corner_a_mm[0]:.1f}, {cfg.corner_a_mm[1]:.1f})" if cfg.corner_a_mm else "not set"
            b_txt = f"({cfg.corner_b_mm[0]:.1f}, {cfg.corner_b_mm[1]:.1f})" if cfg.corner_b_mm else "not set"
            row(f"corner A: {a_txt}", 16, LABEL_C, sfont)
            row(f"corner B: {b_txt}", 22, LABEL_C, sfont)

            col_sp, row_sp = pc.spacing_mm(cfg)
            row(f"rows={cfg.rows}  cols={cfg.cols}", 16, LABEL_C, sfont)
            row(f"spacing: col={col_sp:.1f}mm  row={row_sp:.1f}mm", 16, LABEL_C, sfont)
            row(f"dwell: {cfg.dwell_s:.1f}s   nodes: {len(nodes)}", 22, LABEL_C, sfont)

            if mode == "run" and runner is not None:
                row(f"node {runner.index + 1}/{len(runner.nodes)}", 22, HIGHLIGHT)
            else:
                row(" ", 22)

            if msg and now < msg_until:
                row(msg, 22, OK_C, sfont)
            else:
                row(" ", 22)

            row("-- Keys --", 13, LABEL_C, sfont)
            for line in ["arrows    jog (5mm/press)",
                         "1 / 2     record corner A / B",
                         "[ ]       cols -/+   ; '   rows -/+",
                         "- =       dwell -/+ 0.2s",
                         "s         save config + screenshot",
                         "p         screenshot only",
                         "g/Enter   run the path",
                         "q/ESC     quit (abort run if running)"]:
                row(line, 16, LABEL_C, sfont)

            pygame.display.flip()
            if screenshot_pending:
                try:
                    pygame.image.save(screen, SCREENSHOT_PATH)
                    print(f"saved a screenshot of the path preview to {SCREENSHOT_PATH}")
                except Exception as e:  # noqa: BLE001 -- a screenshot is optional, never fatal
                    print(f"WARNING: could not save {SCREENSHOT_PATH} ({e})")
                screenshot_pending = False
            clock.tick(FPS)
    finally:
        pygame.quit()
        servos.close()


if __name__ == "__main__":
    main()
