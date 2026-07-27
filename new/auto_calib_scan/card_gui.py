"""Live pygame tool: auto-calibrated card scanner.

Self-contained fork of ../fixed_path_scan/path_gui.py (copied, not
imported -- see this folder's module docstrings for why) with the central
change that motivated this whole folder: fixed_path_scan's scan rectangle
was taught by jogging the real arm to two corners, which only lines up
with the physical card if the arm's own kinematics (L1/L2/base/servo
offsets) are precise -- hand-estimated/hardcoded values weren't, which is
why that tool's scans didn't hug the card tightly. This tool instead:

  1. Expects those kinematics to have been fit from real vision data --
     run `python3 main.py homography` then `python3 main.py calibrate`
     in THIS folder first (a self-contained copy of the project's existing
     AprilTag calibration CLI, unchanged). This tool warns loudly (but
     doesn't refuse to run) if calib.json's kinematics still look like the
     untouched nominal/CAD defaults.
  2. Detects the card's own position+rotation live, from an AprilTag stuck
     to it (see arm_core.detect_card_rect) -- no manual teaching, no
     mouse. The card can be anywhere the camera can see it that's also
     arm-reachable; there's nothing to jog into place.

If the camera can't reliably auto-detect the card's tag (e.g. focus
issues), camera_view_gui.py's 'j'/'1'/'2'/'m' keys let you jog the arm to
two corners of the card by hand instead and save them to calib.json's
card.manual_corner_a_mm/manual_corner_b_mm. This tool reads that manual
rectangle (via card_core.sub_rect_from_corners) as a fallback: used
automatically when the tag isn't detected, or forced with 'm' even when
the tag IS detected but not trusted. Drawn in orange (MANUAL_C) instead of
the auto-detected rectangle's yellow (CARD_C), so it's visually obvious
which source is currently driving the scan.

PREVIEW mode (default): detect the card (auto at startup, 'd' to redetect)
+ adjust rows/cols/dwell_s + live preview of the generated serpentine node
path, with each node colored by IK reachability.

RUN mode ('g'/Enter, only once a card is detected and every node is
reachable): drives the real arm node by node via card_core.PathRunner,
which rides jog_controller.ArmController.set_workspace_goal()/tick() to a
full stop at each node, dwells dwell_s seconds, and calls an on_arrive
hook -- currently card_core.default_on_arrive, a placeholder reserved for
real camera-capture code later. 'q'/ESC aborts a run early and returns to
PREVIEW; the arm simply holds its last commanded position (no in-flight
segment is force-cancelled).
"""

import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pygame

import arm_core as core
import arm_hardware as hw
import jog_controller as jc

import card_core as cc

POLL_INTERVAL_S = 0.05
FPS = 60
SCREENSHOT_PATH = "card_scan_preview.png"

BG = (28, 30, 40)
GRID = (48, 54, 72)
LINK1_C = (80, 160, 255)
LINK2_C = (255, 120, 55)
JOINT_C = (220, 220, 220)
EE_C = (55, 215, 95)
BASE_C = (200, 100, 55)
GHOST_C = (170, 90, 220)
CARD_C = (255, 210, 60)
MANUAL_C = (255, 140, 60)
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
    reach circle -- same approach as fixed_path_scan/path_gui.py's Layout.
    The detected card and every generated node are either within reach
    (nodes failing reachability are flagged, not hidden) or the card
    itself may briefly be drawn outside the circle if it's somewhere the
    arm can't reach at all -- that's exactly the "unreachable" case the
    node coloring and the run-time reachability check are there to catch."""
    base_x: float
    base_y: float
    max_r_mm: float
    scale: float = 1.0
    margin_px: int = 40
    panel_w: int = 300

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


def _looks_uncalibrated(params: core.ArmParams) -> bool:
    """True if calib.json's kinematics are still the untouched nominal/CAD
    defaults -- i.e. `main.py calibrate` has never been run (or was run
    and happened to change nothing, astronomically unlikely). This is the
    direct root cause this whole tool exists to address: a hand-estimated
    kinematic model doesn't map jog/scan coordinates onto the physical
    card precisely enough."""
    nominal = core.ArmParams.nominal()
    return (params.L1 == nominal.L1 and params.L2 == nominal.L2 and
            params.base_x == nominal.base_x and params.base_y == nominal.base_y and
            params.servo1_offset_deg == nominal.servo1_offset_deg and
            params.servo2_offset_deg == nominal.servo2_offset_deg)


def _detect_card(arm_hw, calib: dict, card_cfg: "core.CardConfig"):
    """One snapshot + detection attempt -> a card_rect tuple, or None if
    the homography isn't configured yet or the card's tag wasn't seen this
    frame (out of view, bad lighting, ...) -- both are non-fatal, the
    caller keeps whatever card_rect it already had."""
    H = calib["homography"].get("H")
    if H is None:
        return None
    frame = arm_hw.camera.capture_gray()
    detections = arm_hw.detector.detect(frame)
    return core.detect_card_rect(detections, card_cfg.tag_id, card_cfg.width_mm,
                                  card_cfg.height_mm, np.array(H))


def _card_rect_from_manual(scan_area: tuple, card_cfg: "core.CardConfig"):
    """The manually-jogged-and-saved fallback rectangle (see
    camera_view_gui.py's 'j'/'1'/'2'/'m' keys), or None if it hasn't been
    taught yet."""
    if card_cfg.manual_corner_a_mm is None or card_cfg.manual_corner_b_mm is None:
        return None
    return cc.sub_rect_from_corners(scan_area, card_cfg.manual_corner_a_mm, card_cfg.manual_corner_b_mm)


def main():
    calib = core.load_calib()
    hw_cfg = core.calib_hardware_config(calib)
    card_cfg = core.calib_card_config(calib)

    arm_hw = hw.ArmHardware(hw_cfg.servo_port, hw_cfg.joint_ids,
                            camera_backend=hw_cfg.camera_backend,
                            usb_camera_index=hw_cfg.usb_camera_index)
    arm_hw.connect()

    controller = jc.build_controller(arm_hw.servos, calib)
    params = controller.params
    cfg = cc.load_card_scan_config()

    uncalibrated = _looks_uncalibrated(params)
    homography_missing = calib["homography"].get("H") is None

    scan_area = core.calib_scan_area(calib)
    auto_rect = _detect_card(arm_hw, calib, card_cfg)
    manual_rect = _card_rect_from_manual(scan_area, card_cfg)
    if auto_rect is not None:
        card_source = "auto"
    elif manual_rect is not None:
        card_source = "manual"
    else:
        card_source = None
    card_rect = auto_rect if card_source == "auto" else manual_rect
    card_msg = "" if card_rect is not None else "no card detected yet -- press 'd'"

    max_r = params.L1 + params.L2
    layout = Layout(base_x=params.base_x, base_y=params.base_y, max_r_mm=max_r)

    pygame.init()
    pygame.display.set_caption("2R Arm -- auto-calibrated card scan")
    screen = pygame.display.set_mode((layout.win_w, layout.win_h))
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("menlo,consolas,monospace", 18)
    sfont = pygame.font.SysFont("menlo,consolas,monospace", 13)

    real_s1 = arm_hw.servos.get_present_deg("joint1")
    real_s2 = arm_hw.servos.get_present_deg("joint2")

    mode = "preview"
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
                            mode = "preview"
                            runner = None
                            msg = "run aborted"
                            msg_until = now + 2.0
                        else:
                            running = False
                    elif mode == "run":
                        pass  # ignore all other keys while a run is in progress
                    elif event.key == pygame.K_d:
                        new_rect = _detect_card(arm_hw, calib, card_cfg)
                        if new_rect is not None:
                            auto_rect = new_rect
                            if card_source != "manual":
                                card_source = "auto"
                            card_msg = ""
                            msg = f"card detected: center=({auto_rect[0]:.1f}, {auto_rect[1]:.1f})mm rot={auto_rect[4]:.1f}deg"
                        else:
                            # A transient miss must not discard an
                            # already-known-good auto_rect (same contract
                            # as arm_core.detect_card_rect itself) --
                            # leave card_source/auto_rect untouched unless
                            # there was never a good detection to begin
                            # with, in which case fall back to manual if
                            # one's available.
                            card_msg = ("homography not configured -- run main.py homography first"
                                        if homography_missing else
                                        f"card tag {card_cfg.tag_id} not seen -- check position/lighting")
                            if card_source is None and manual_rect is not None:
                                card_source = "manual"
                                card_msg += " -- fell back to manual rect"
                            msg = card_msg
                        card_rect = auto_rect if card_source == "auto" else manual_rect
                        msg_until = now + 4.0
                    elif event.key == pygame.K_m:
                        if card_source == "manual" and auto_rect is not None:
                            card_source = "auto"
                            msg = "switched to auto (tag-detected) card rect"
                        elif manual_rect is not None:
                            card_source = "manual"
                            msg = "switched to manual card rect"
                        else:
                            msg = ("no manual card rect configured yet -- teach one with "
                                    "camera_view_gui.py's 'j'/'1'/'2'/'m' keys first")
                        card_rect = auto_rect if card_source == "auto" else manual_rect
                        msg_until = now + 3.0
                    elif event.key == pygame.K_LEFTBRACKET:
                        cfg.cols = max(cfg.cols - 1, 2)
                    elif event.key == pygame.K_RIGHTBRACKET:
                        cfg.cols += 1
                    elif event.key == pygame.K_SEMICOLON:
                        cfg.rows = max(cfg.rows - 1, 2)
                    elif event.key == pygame.K_QUOTE:
                        cfg.rows += 1
                    elif event.key == pygame.K_COMMA:
                        cfg.dwell_s = max(cfg.dwell_s - 0.2, 0.0)
                    elif event.key == pygame.K_PERIOD:
                        cfg.dwell_s += 0.2
                    elif event.key == pygame.K_s:
                        cc.save_card_scan_config(cfg)
                        screenshot_pending = True
                        msg = f"saved card_scan_config.json + {SCREENSHOT_PATH}"
                        msg_until = now + 3.0
                    elif event.key == pygame.K_p:
                        screenshot_pending = True
                        msg = f"saved {SCREENSHOT_PATH}"
                        msg_until = now + 3.0
                    elif event.key in (pygame.K_g, pygame.K_RETURN):
                        if card_rect is None:
                            msg = "no card detected -- press 'd' first"
                        else:
                            nodes = cc.generate_node_path(cfg, card_rect)
                            unreachable = sum(
                                1 for x, y, _l in nodes
                                if not core.ik_solve(params, x, y, controller.joint_limits).reachable)
                            if unreachable:
                                msg = f"{unreachable}/{len(nodes)} node(s) unreachable -- move the card or arm setup"
                            else:
                                runner = cc.PathRunner(controller, nodes, cfg.dwell_s,
                                                        on_arrive=cc.default_on_arrive)
                                mode = "run"
                                msg = f"running {len(nodes)} nodes"
                        msg_until = now + 4.0

            if now - last_poll >= POLL_INTERVAL_S:
                real_s1 = arm_hw.servos.get_present_deg("joint1")
                real_s2 = arm_hw.servos.get_present_deg("joint2")
                last_poll = now

            if mode == "run":
                runner.tick(now)
                if runner.done:
                    mode = "preview"
                    runner = None
                    msg = "scan finished"
                    msg_until = now + 3.0
            else:
                controller.tick()

            screen.fill(BG)

            steps = [i / 200.0 * 2 * math.pi for i in range(201)]
            pygame.draw.lines(
                screen, GRID, False,
                [layout.ws2s(params.base_x + max_r * math.cos(t),
                             params.base_y + max_r * math.sin(t)) for t in steps], 1)
            pygame.draw.circle(screen, BASE_C, layout.ws2s(params.base_x, params.base_y), 9, 2)

            nodes = []
            if card_rect is not None:
                card_points = [layout.ws2s(*c) for c in core.scan_area_corners(*card_rect)]
                rect_color = MANUAL_C if card_source == "manual" else CARD_C
                pygame.draw.polygon(screen, rect_color, card_points, 2)

                nodes = cc.generate_node_path(cfg, card_rect)
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

            row("AUTO-CALIB CARD SCAN", 22, HIGHLIGHT)
            row("RUNNING" if mode == "run" else "preview", 18,
                WARN_C if mode == "run" else OK_C, sfont)

            if uncalibrated:
                row("WARNING: kinematics look uncalibrated", 13, WARN_C, sfont)
                row("run main.py homography + calibrate first", 22, WARN_C, sfont)
            else:
                row(" ", 13)
                row(" ", 22)

            if card_rect is not None:
                row(f"card [{card_source}]: ({card_rect[0]:.1f}, {card_rect[1]:.1f})mm "
                    f"rot={card_rect[4]:.1f}deg", 16, LABEL_C, sfont)
            else:
                row(f"card: {card_msg or 'not detected'}", 16, WARN_C, sfont)
            row(f"tag_id={card_cfg.tag_id}  size={card_cfg.width_mm:.1f}x{card_cfg.height_mm:.1f}mm", 16,
                LABEL_C, sfont)
            row(f"manual rect: {'available' if manual_rect is not None else 'not taught yet'}", 22,
                LABEL_C, sfont)

            row(f"rows={cfg.rows}  cols={cfg.cols}  dwell={cfg.dwell_s:.1f}s", 22, LABEL_C, sfont)
            row(f"nodes: {len(nodes)}", 22, LABEL_C, sfont)

            if mode == "run" and runner is not None:
                row(f"node {runner.index + 1}/{len(runner.nodes)}", 22, HIGHLIGHT)
            else:
                row(" ", 22)

            if msg and now < msg_until:
                row(msg, 22, OK_C, sfont)
            else:
                row(" ", 22)

            row("-- Keys --", 13, LABEL_C, sfont)
            for line in ["d         (re)detect the card",
                         "m         switch auto (tag) / manual card rect",
                         "[ ]       cols -/+   ; '   rows -/+",
                         ", .       dwell -/+ 0.2s",
                         "s         save config + screenshot",
                         "p         screenshot only",
                         "g/Enter   run the scan",
                         "q/ESC     quit (abort run if running)"]:
                row(line, 16, LABEL_C, sfont)

            pygame.display.flip()
            if screenshot_pending:
                try:
                    pygame.image.save(screen, SCREENSHOT_PATH)
                    print(f"saved a screenshot of the scan preview to {SCREENSHOT_PATH}")
                except Exception as e:  # noqa: BLE001 -- a screenshot is optional, never fatal
                    print(f"WARNING: could not save {SCREENSHOT_PATH} ({e})")
                screenshot_pending = False
            clock.tick(FPS)
    finally:
        pygame.quit()
        arm_hw.close()


if __name__ == "__main__":
    main()
