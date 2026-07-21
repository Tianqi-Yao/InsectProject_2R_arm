"""Live pygame visualizer + jog control.

Draws two overlaid arms every frame:
  - solid  = computed from the *real* servo encoder feedback right now --
             this is what the code currently believes the physical arm
             looks like.
  - ghost  = wherever you're currently jogging to (the IK target).

Put this window next to the real arm and compare by eye: if the solid
arm's pose doesn't match what you're looking at, that's L1/L2/base/offset/
direction being wrong, not a bug in this viewer -- this tool exists to
make that mismatch visible instead of having to picture it from raw angle
numbers.

Visual style borrowed from sim/sim.py (the old pure-simulation tool, no
hardware ties); this one is wired to the real arm instead. Same jog
behaviour and safety rule as run.py: nothing is sent to the servos until
you press a key.

Keep SERVO_PORT / JOINT_IDS / L1 / L2 / BASE_X / BASE_Y / offsets / dirs
in sync with run.py if you change one -- they're duplicated on purpose
(this is a standalone tool, not a shared module) but they describe the
same physical arm.
"""

import sys
import time
from pathlib import Path

import pygame

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "software"))
import arm_core as core   # noqa: E402
import arm_hardware as hw  # noqa: E402

# --- Edit these to match your arm (same values as run.py) ---
SERVO_PORT = "/dev/cu.usbserial-0001"
JOINT_IDS = {"joint1": 1, "joint2": 2}

L1 = 125.0
L2 = 95.0
BASE_X = 100.0
BASE_Y = -45.0
TICKS_AT_STRAIGHT_LINE = 2040
SERVO1_OFFSET_DEG = TICKS_AT_STRAIGHT_LINE * hw.DEG_PER_TICK
SERVO2_OFFSET_DEG = TICKS_AT_STRAIGHT_LINE * hw.DEG_PER_TICK
SERVO1_DIR = 1
SERVO2_DIR = -1
# ---------------------------------------------------------------

PARAMS = core.ArmParams(L1=L1, L2=L2, base_x=BASE_X, base_y=BASE_Y,
                         servo1_offset_deg=SERVO1_OFFSET_DEG,
                         servo2_offset_deg=SERVO2_OFFSET_DEG,
                         servo1_dir=SERVO1_DIR, servo2_dir=SERVO2_DIR)

WS_W, WS_H = 200.0, 150.0

# Serpentine (boustrophedon) scan for the 't' positioning test: starts at
# the top-left (x=0 side, y=max/"away from base" side, matching the
# corner_world_mm "tl" convention used elsewhere) and snakes row by row.
SCAN_MARGIN_MM = 20.0
SCAN_NX, SCAN_NY = 50, 40
# Set to e.g. 3 to only run the first 3 rows (same row spacing as the full
# SCAN_NY grid, just fewer of them) -- handy for a quick check before
# committing to the full sweep. None runs every row.
SCAN_ROWS_LIMIT = 3


def _generate_serpentine_points(width_mm, height_mm, nx, ny, margin_mm, rows_limit=None):
    xs = [margin_mm + i * (width_mm - 2 * margin_mm) / (nx - 1) for i in range(nx)]
    ys = [height_mm - margin_mm - j * (height_mm - 2 * margin_mm) / (ny - 1) for j in range(ny)]
    if rows_limit is not None:
        ys = ys[:rows_limit]
    points = []
    for row, y in enumerate(ys):
        row_xs = xs if row % 2 == 0 else list(reversed(xs))
        for x in row_xs:
            points.append((x, y, f"row{row + 1}"))
    return points


DEMO_POINTS = _generate_serpentine_points(WS_W, WS_H, SCAN_NX, SCAN_NY, SCAN_MARGIN_MM, SCAN_ROWS_LIMIT)

HOME_X, HOME_Y = 100.0, 75.0
STEP_MIN, STEP_MAX = 0.5, 40.0
POLL_INTERVAL_S = 0.08       # how often to re-read real encoder positions

# Arrow-key jog moves: slower than the servo's raw default, plus a bit of
# acceleration ramp (0 = snap straight to speed and stop dead on arrival,
# which is what was feeling too abrupt/shaky). Tune JOG_ACC up for a more
# gradual start/stop, down (toward 0) for snappier response.
JOG_SPEED = 400
JOG_ACC = 30

# How long to hold each scan point before advancing to the next one --
# deliberately *not* waited on for the servo to actually arrive. With a
# dense grid the points are only a few mm apart, so continuously feeding a
# slightly-ahead target makes the servo track smoothly instead of
# decelerating to a stop at every point (a settle-and-wait per point looks
# like visible jitter/stutter on a fine grid). Tune together with
# SCAN_NX/SCAN_NY: denser wants a shorter interval to stay smooth; sparser
# needs a longer one or it'll dash through without getting close to each point.
SCAN_STEP_INTERVAL_S = 0.03

# Slower than the default jog speed (800) on purpose: at full speed the
# servo tries to sprint toward each point and is still accelerating (or
# starting to brake for arrival) when the next one replaces it, which
# looks like jitter even with continuous streaming. A gentler speed suits
# the short hops between closely-spaced scan points better.
SCAN_SPEED = 400

# --- Display (layout/palette borrowed from sim/sim.py) ---
SCALE = 2.0
WS_OX, WS_OY = 55, 30
WIN_W = int(WS_OX + WS_W * SCALE) + 260
WIN_H = int(WS_OY + WS_H * SCALE) + 40
FPS = 60

BG = (28, 30, 40)
WS_FILL = (38, 46, 62)
WS_BORDER = (70, 110, 190)
GRID = (48, 54, 72)
LINK1_C = (80, 160, 255)
LINK2_C = (255, 120, 55)
JOINT_C = (220, 220, 220)
EE_OK = (55, 215, 95)
EE_ERR = (220, 55, 55)
BASE_C = (200, 100, 55)
GHOST_LINK1 = (80, 160, 255, 100)
GHOST_LINK2 = (255, 120, 55, 100)
GHOST_JOINT = (220, 220, 220, 100)
TARGET_C = (255, 230, 80)
TEXT_C = (200, 210, 225)
LABEL_C = (110, 130, 155)
HIGHLIGHT = (255, 220, 80)


def ws2s(wx, wy):
    """Workspace mm -> screen px (Y flipped: workspace +Y = screen up)."""
    return (int(WS_OX + wx * SCALE), int(WS_OY + (WS_H - wy) * SCALE))


def draw_workspace(surf):
    w, h = int(WS_W * SCALE), int(WS_H * SCALE)
    pygame.draw.rect(surf, WS_FILL, (WS_OX, WS_OY, w, h))
    for gx in range(0, int(WS_W) + 1, 25):
        pygame.draw.line(surf, GRID, ws2s(gx, 0), ws2s(gx, WS_H))
    for gy in range(0, int(WS_H) + 1, 25):
        pygame.draw.line(surf, GRID, ws2s(0, gy), ws2s(WS_W, gy))
    pygame.draw.rect(surf, WS_BORDER, (WS_OX, WS_OY, w, h), 2)


def draw_solid_arm(surf, elbow, ee):
    p_base, p_elbow, p_ee = ws2s(BASE_X, BASE_Y), ws2s(*elbow), ws2s(*ee)
    pygame.draw.line(surf, LINK1_C, p_base, p_elbow, 6)
    pygame.draw.line(surf, LINK2_C, p_elbow, p_ee, 5)
    pygame.draw.circle(surf, JOINT_C, p_base, 9)
    pygame.draw.circle(surf, BASE_C, p_base, 9, 2)
    pygame.draw.circle(surf, JOINT_C, p_elbow, 7)
    pygame.draw.circle(surf, EE_OK, p_ee, 8)
    pygame.draw.circle(surf, (255, 255, 255), p_ee, 8, 2)


def draw_ghost_arm(surf, elbow, ee):
    overlay = pygame.Surface((WIN_W, WIN_H), pygame.SRCALPHA)
    p_base, p_elbow, p_ee = ws2s(BASE_X, BASE_Y), ws2s(*elbow), ws2s(*ee)
    pygame.draw.line(overlay, GHOST_LINK1, p_base, p_elbow, 4)
    pygame.draw.line(overlay, GHOST_LINK2, p_elbow, p_ee, 3)
    pygame.draw.circle(overlay, GHOST_JOINT, p_elbow, 5)
    pygame.draw.circle(overlay, (*TARGET_C, 160), p_ee, 6, 2)
    surf.blit(overlay, (0, 0))


def draw_panel(surf, font, sfont, info):
    px = int(WS_OX + WS_W * SCALE) + 24
    py = 28

    def row(text, dy, color=TEXT_C, f=None):
        nonlocal py
        surf.blit((f or font).render(text, True, color), (px, py))
        py += dy

    row("REAL vs TARGET", 22, HIGHLIGHT)
    row("solid = real (encoders)", 15, LABEL_C, sfont)
    row("ghost = jog target", 22, LABEL_C, sfont)

    row("Real", 13, LABEL_C, sfont)
    row(f"s1={info['real_s1']:6.1f}  s2={info['real_s2']:6.1f}", 16)
    row(f"x={info['real_x']:6.1f}  y={info['real_y']:6.1f} mm", 22)

    row("Target", 13, LABEL_C, sfont)
    if info["target_reachable"]:
        row(f"s1={info['target_s1']:6.1f}  s2={info['target_s2']:6.1f}", 16)
        row(f"x={info['target_x']:6.1f}  y={info['target_y']:6.1f} mm", 22)
    else:
        row("NOT REACHABLE", 16, EE_ERR)
        row(" ", 22)

    row("Step", 13, LABEL_C, sfont)
    row(f"{info['step']:.1f} mm", 22)

    if info["demo_active"]:
        row("Positioning test", 13, LABEL_C, sfont)
        row(f"point {info['demo_index'] + 1}/{len(DEMO_POINTS)}", 22, HIGHLIGHT)
    else:
        row(" ", 13)
        row(" ", 22)

    row("-- Keys --", 13, LABEL_C, sfont)
    for line in ["arrows   move target", "[ ]      step size",
                 "h        home", "t        positioning test",
                 "q / ESC  quit"]:
        row(line, 16, LABEL_C, sfont)


def main():
    servos = hw.Servos(JOINT_IDS)
    servos.connect(SERVO_PORT)

    pygame.init()
    pygame.display.set_caption("2R Arm -- real vs target")
    screen = pygame.display.set_mode((WIN_W, WIN_H))
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("menlo,consolas,monospace", 18)
    sfont = pygame.font.SysFont("menlo,consolas,monospace", 13)

    real_s1 = servos.get_present_deg("joint1")
    real_s2 = servos.get_present_deg("joint2")
    real_elbow, real_ee = core.fk_joint_positions(PARAMS, real_s1, real_s2)

    # Target starts wherever the arm actually is right now -- nothing gets
    # sent to the servos until an arrow key (or 't') actually changes it.
    target_x, target_y = real_ee
    last_sent = (target_x, target_y)
    step = 5.0
    last_poll = time.monotonic()
    demo = {"active": False, "index": 0, "next_advance": 0.0}

    def send_if_changed(tx, ty, speed=JOG_SPEED, acc=JOG_ACC):
        nonlocal last_sent
        if last_sent == (tx, ty):
            return
        r = core.ik_solve(PARAMS, tx, ty)
        if r.reachable:
            servos.set_target_deg("joint1", r.servo1_deg, speed=speed, acc=acc)
            servos.set_target_deg("joint2", r.servo2_deg, speed=speed, acc=acc)
        last_sent = (tx, ty)

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
                    elif event.key == pygame.K_LEFTBRACKET:
                        step = max(step / 1.5, STEP_MIN)
                    elif event.key == pygame.K_RIGHTBRACKET:
                        step = min(step * 1.5, STEP_MAX)
                    elif event.key == pygame.K_t and not demo["active"]:
                        demo = {"active": True, "index": 0,
                                "next_advance": now + SCAN_STEP_INTERVAL_S}
                        target_x, target_y, _ = DEMO_POINTS[0]
                        send_if_changed(target_x, target_y, speed=SCAN_SPEED, acc=0)
                    elif event.key == pygame.K_h:
                        # 'h' always wins, even mid-scan: interrupts the
                        # positioning test and heads straight home.
                        demo["active"] = False
                        target_x, target_y = HOME_X, HOME_Y
                        send_if_changed(target_x, target_y)
                    elif not demo["active"]:
                        if event.key == pygame.K_UP:
                            target_y += step
                        elif event.key == pygame.K_DOWN:
                            target_y -= step
                        elif event.key == pygame.K_LEFT:
                            target_x -= step
                        elif event.key == pygame.K_RIGHT:
                            target_x += step
                        else:
                            continue
                        send_if_changed(target_x, target_y)

            if now - last_poll >= POLL_INTERVAL_S:
                real_s1 = servos.get_present_deg("joint1")
                real_s2 = servos.get_present_deg("joint2")
                real_elbow, real_ee = core.fk_joint_positions(PARAMS, real_s1, real_s2)
                last_poll = now

            if demo["active"] and now >= demo["next_advance"]:
                demo["index"] += 1
                if demo["index"] >= len(DEMO_POINTS):
                    demo["active"] = False
                else:
                    target_x, target_y, _ = DEMO_POINTS[demo["index"]]
                    send_if_changed(target_x, target_y, speed=SCAN_SPEED, acc=0)
                    demo["next_advance"] = now + SCAN_STEP_INTERVAL_S

            target_result = core.ik_solve(PARAMS, target_x, target_y)

            screen.fill(BG)
            draw_workspace(screen)
            if target_result.reachable:
                t_elbow, t_ee = core.fk_joint_positions(
                    PARAMS, target_result.servo1_deg, target_result.servo2_deg)
                draw_ghost_arm(screen, t_elbow, t_ee)
            draw_solid_arm(screen, real_elbow, real_ee)
            draw_panel(screen, font, sfont, {
                "real_s1": real_s1, "real_s2": real_s2,
                "real_x": real_ee[0], "real_y": real_ee[1],
                "target_reachable": target_result.reachable,
                "target_s1": target_result.servo1_deg,
                "target_s2": target_result.servo2_deg,
                "target_x": target_x, "target_y": target_y,
                "step": step,
                "demo_active": demo["active"], "demo_index": demo["index"],
            })
            pygame.display.flip()
            clock.tick(FPS)
    finally:
        pygame.quit()
        servos.close()


if __name__ == "__main__":
    main()
