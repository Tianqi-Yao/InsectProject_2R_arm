#!/usr/bin/env python3
"""
2R Arm WASD Simulator — top-down view of horizontal workspace

Controls:
  W / S     +Y / -Y  (forward / backward)
  A / D     -X / +X  (left / right)
  + / -     speed up / slow down
  C         clear trail
  ESC / Q   quit
"""

import sys
import math
import pygame

# ── Arm / workspace parameters ─────────────────────────────────
L1 = 125.0              # shoulder→elbow link length (mm)
L2 =  95.0              # elbow→end-effector link length (mm)
BASE_X = 100.0          # arm base X in workspace coords (mm) — centre of 200mm edge
BASE_Y = -45.0          # arm base Y in workspace coords (mm, below workspace)
WS_W  = 200.0           # workspace width  (mm) — horizontal axis
WS_H  = 150.0           # workspace height (mm) — depth axis

SERVO1_OFFSET = 23.08   # used for servo angle display only
SERVO2_OFFSET =  0.0

# ── Display ─────────────────────────────────────────────────────
SCALE     = 2.0         # pixels per mm
WIN_W     = 860         # 200mm×2 + margins + panel
WIN_H     = 560         # 150mm×2 + 80mm base zone + margins
WS_OX     = 55          # workspace top-left on screen (px)
WS_OY     = 30
FPS       = 60
MAX_TRAIL = 500

SPEED_DEFAULT = 1.5     # mm / frame
SPEED_MIN     = 0.2
SPEED_MAX     = 12.0

L_STEP = 5.0            # link-length adjustment per keypress (mm)
L_MIN  = 50.0           # minimum link length (mm)
L_MAX  = 300.0          # maximum link length (mm)

S_MARGIN      = 0.0     # servo protection margin (deg); effective range = [M, 180-M]
S_MARGIN_STEP = 5.0
S_MARGIN_MAX  = 60.0

BASE_STEP  = 5.0        # base position adjustment per keypress (mm)
BASE_X_MIN = -100.0
BASE_X_MAX =  300.0
BASE_Y_MIN = -300.0
BASE_Y_MAX =  float(150)   # WS_H, filled at runtime

# ── Colour palette ───────────────────────────────────────────────
BG         = ( 28,  30,  40)
WS_FILL    = ( 38,  46,  62)
WS_BORDER  = ( 70, 110, 190)
GRID       = ( 48,  54,  72)
LINK1_C    = ( 80, 160, 255)
LINK2_C    = (255, 120,  55)
JOINT_C    = (220, 220, 220)
EE_OK      = ( 55, 215,  95)
EE_ERR     = (220,  55,  55)
BASE_C     = (200, 100,  55)
TARGET_C   = (255, 230,  80)
TEXT_C     = (200, 210, 225)
LABEL_C    = (110, 130, 155)
HIGHLIGHT  = (255, 220,  80)
DIVIDER_C  = ( 55,  60,  78)

# ── IK / FK ─────────────────────────────────────────────────────

def ik(x, y):
    """
    Solve IK for arm-relative coords (x, y) mm.
    Returns (theta1, theta2, servo1, servo2) degrees, or None if unreachable.
    """
    c2 = (x*x + y*y - L1*L1 - L2*L2) / (2*L1*L2)
    if not (-1.0 <= c2 <= 1.0):
        return None
    s2  = math.sqrt(1.0 - c2*c2)
    t2  = math.degrees(math.atan2(s2, c2))
    t1  = math.degrees(math.atan2(y, x)) - math.degrees(math.atan2(L2*s2, L1+L2*c2))
    s1  = t1 + SERVO1_OFFSET
    s2v = t2 + SERVO2_OFFSET
    lo, hi = S_MARGIN, 180.0 - S_MARGIN
    if not (lo <= s1 <= hi and lo <= s2v <= hi):
        return None
    return t1, t2, s1, s2v

def recompute_offset():
    """Recompute SERVO1_OFFSET from current L1, L2, BASE and S_MARGIN.

    Maps the near-right workspace corner to servo angle S_MARGIN+0.1 so that
    it sits just inside the allowed range regardless of the current margin.
    """
    global SERVO1_OFFSET
    ax = WS_W - BASE_X   # arm-relative X of workspace near-right corner
    ay = -BASE_Y          # arm-relative Y (positive, corner is above base)
    c2 = (ax*ax + ay*ay - L1*L1 - L2*L2) / (2*L1*L2)
    if abs(c2) > 1.0:
        SERVO1_OFFSET = 90.0
        return
    s2 = math.sqrt(1.0 - c2*c2)
    t1 = math.degrees(math.atan2(ay, ax)) - math.degrees(math.atan2(L2*s2, L1+L2*c2))
    SERVO1_OFFSET = -t1 + S_MARGIN + 0.1   # near-right corner → servo S_MARGIN+0.1°


def draw_reach_circles(surf):
    """Draw max-reach and dead-zone circles centred at arm base."""
    bp = ws2s(BASE_X, BASE_Y)
    outer_r = int((L1 + L2) * SCALE)
    inner_r = int(abs(L1 - L2) * SCALE)
    # Outer (max reach) — blue dashed approximated as thin circle
    pygame.draw.circle(surf, (60, 100, 180), bp, outer_r, 1)
    # Inner (dead zone) — red, only when non-trivial
    if inner_r > 4:
        pygame.draw.circle(surf, (160, 60, 60), bp, inner_r, 1)


# ── Coordinate helpers ───────────────────────────────────────────

def ws2s(wx, wy):
    """Workspace mm → screen px (Y flipped: workspace +Y = screen up).
    Works for negative wy too (base zone below workspace rectangle)."""
    return (int(WS_OX + wx * SCALE),
            int(WS_OY + (WS_H - wy) * SCALE))

def arm2s(ax, ay):
    """Arm-relative mm → screen px. Base at workspace (BASE_X, BASE_Y)."""
    return ws2s(ax + BASE_X, ay + BASE_Y)

# ── Drawing ──────────────────────────────────────────────────────

def draw_workspace(surf):
    w, h = int(WS_W * SCALE), int(WS_H * SCALE)
    pygame.draw.rect(surf, WS_FILL, (WS_OX, WS_OY, w, h))
    # 25 mm grid
    for gx in range(0, int(WS_W)+1, 25):
        pygame.draw.line(surf, GRID, ws2s(gx, 0), ws2s(gx, WS_H))
    for gy in range(0, int(WS_H)+1, 25):
        pygame.draw.line(surf, GRID, ws2s(0, gy), ws2s(WS_W, gy))
    pygame.draw.rect(surf, WS_BORDER, (WS_OX, WS_OY, w, h), 2)


def draw_trail(surf, trail):
    if len(trail) < 2:
        return
    overlay = pygame.Surface((WIN_W, WIN_H), pygame.SRCALPHA)
    n = len(trail)
    for i in range(1, n):
        alpha = max(20, int(200 * i / n))
        pygame.draw.line(overlay, (*EE_OK, alpha),
                         ws2s(*trail[i-1]), ws2s(*trail[i]), 2)
    surf.blit(overlay, (0, 0))


def draw_arm(surf, t1, t2, reachable):
    t1r, t2r = math.radians(t1), math.radians(t2)
    ex1 = L1 * math.cos(t1r)
    ey1 = L1 * math.sin(t1r)
    ex2 = ex1 + L2 * math.cos(t1r + t2r)
    ey2 = ey1 + L2 * math.sin(t1r + t2r)

    p_base  = arm2s(0, 0)
    p_elbow = arm2s(ex1, ey1)
    p_ee    = arm2s(ex2, ey2)

    pygame.draw.line(surf, LINK1_C, p_base,  p_elbow, 6)
    pygame.draw.line(surf, LINK2_C, p_elbow, p_ee,    5)

    pygame.draw.circle(surf, JOINT_C, p_base,  9)
    pygame.draw.circle(surf, BASE_C,  p_base,  9, 2)
    pygame.draw.circle(surf, JOINT_C, p_elbow, 7)

    ee_c = EE_OK if reachable else EE_ERR
    pygame.draw.circle(surf, ee_c,            p_ee, 8)
    pygame.draw.circle(surf, (255, 255, 255), p_ee, 8, 2)


def draw_target_cursor(surf, wx, wy, reachable):
    px, py = ws2s(wx, wy)
    col = TARGET_C if reachable else EE_ERR
    r = 6
    pygame.draw.line(surf, col, (px-r, py), (px+r, py), 2)
    pygame.draw.line(surf, col, (px, py-r), (px, py+r), 2)
    pygame.draw.circle(surf, col, (px, py), r+2, 1)


def draw_panel(surf, fonts, state):
    font, sfont = fonts
    px = int(WS_OX + WS_W * SCALE) + 24
    py = 32

    def row(text, dy, color=TEXT_C, f=None):
        nonlocal py
        surf.blit((f or font).render(text, True, color), (px, py))
        py += dy

    row("2R ARM SIM", 24, HIGHLIGHT)
    row("REACHABLE" if state['reachable'] else "OUT OF RANGE",
        22, EE_OK if state['reachable'] else EE_ERR)

    row("End-effector", 13, LABEL_C, sfont)
    row(f"X={state['ex']:6.1f}  Y={state['ey']:6.1f} mm", 20)

    row("Joint   Servo", 13, LABEL_C, sfont)
    if state['reachable']:
        row(f"T1={state['t1']:6.1f}  S1={state['s1']:5.1f}", 18)
        row(f"T2={state['t2']:6.1f}  S2={state['s2']:5.1f}", 20)
    else:
        row("T1 = --    S1 = --", 18, LABEL_C)
        row("T2 = --    S2 = --", 20, LABEL_C)

    row("Links", 13, LABEL_C, sfont)
    row(f"L1={state['l1']:.0f}  L2={state['l2']:.0f} mm", 18)
    mg = state['margin']
    row(f"max={state['l1']+state['l2']:.0f}  dead={abs(state['l1']-state['l2']):.0f} mm",
        20, LABEL_C, sfont)

    row("Base pos", 13, LABEL_C, sfont)
    row(f"Bx={state['bx']:.0f}  By={state['by']:.0f} mm", 20)

    row("Servo margin", 13, LABEL_C, sfont)
    row(f"{mg:.0f} deg  [{mg:.0f}, {180-mg:.0f}]", 20)

    row("Speed", 13, LABEL_C, sfont)
    row(f"{state['speed']:.1f} mm/frame", 22)

    row("-- Keys --", 13, LABEL_C, sfont)
    for line in ["WASD      -> move",
                 "[ ]  ;  ' -> L1 L2",
                 "Arrows    -> base",
                 "Z / X     -> margin",
                 "+/-  C  Q"]:
        row(line, 16, LABEL_C, sfont)

# ── Try to move, falling back to axis-only if combined is blocked ─

def try_move(ex, ey, dx, dy):
    """
    Try full move first; if blocked, try single-axis moves.
    Returns (new_ex, new_ey, ik_result_or_None).
    """
    for ddx, ddy in [(dx, dy), (dx, 0.0), (0.0, dy)]:
        nx = max(0.0, min(WS_W, ex + ddx))
        ny = max(0.0, min(WS_H, ey + ddy))
        r  = ik(nx - BASE_X, ny - BASE_Y)
        if r is not None:
            return nx, ny, r
    # Nothing worked — stay and report current IK
    return ex, ey, ik(ex - BASE_X, ey - BASE_Y)

# ── Main ─────────────────────────────────────────────────────────

def main():
    global L1, L2, BASE_X, BASE_Y, S_MARGIN
    pygame.init()
    pygame.display.set_caption("2R Arm Simulator - WASD")
    screen = pygame.display.set_mode((WIN_W, WIN_H))
    clock  = pygame.time.Clock()

    font  = pygame.font.SysFont("monospace", 15, bold=True)
    sfont = pygame.font.SysFont("monospace", 12)

    ex, ey = BASE_X, WS_H / 2     # start at workspace centre
    speed  = SPEED_DEFAULT
    trail  = []

    # Compute initial IK state
    r0 = ik(ex - BASE_X, ey - BASE_Y)
    t1, t2, s1, s2 = r0 if r0 else (0.0, 0.0, 0.0, 0.0)
    reachable = r0 is not None

    def _revalidate():
        """Re-run IK at current position and update arm state."""
        nonlocal t1, t2, s1, s2, reachable
        r = ik(ex - BASE_X, ey - BASE_Y)
        if r:
            t1, t2, s1, s2 = r
            reachable = True
        else:
            reachable = False

    def _apply_link_change():
        """After L1/L2 change: recompute offset, clear trail, revalidate IK."""
        recompute_offset()
        trail.clear()
        _revalidate()

    def _apply_base_change():
        """After BASE_X/BASE_Y change: recompute offset, clear trail, revalidate IK."""
        recompute_offset()
        trail.clear()
        _revalidate()

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                k = event.key
                if k in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                elif k in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
                    speed = min(SPEED_MAX, round(speed + 0.5, 1))
                elif k in (pygame.K_MINUS, pygame.K_KP_MINUS):
                    speed = max(SPEED_MIN, round(speed - 0.5, 1))
                elif k == pygame.K_c:
                    trail.clear()
                elif k == pygame.K_RIGHTBRACKET:
                    L1 = min(L_MAX, L1 + L_STEP); _apply_link_change()
                elif k == pygame.K_LEFTBRACKET:
                    L1 = max(L_MIN, L1 - L_STEP); _apply_link_change()
                elif k == pygame.K_QUOTE:
                    L2 = min(L_MAX, L2 + L_STEP); _apply_link_change()
                elif k == pygame.K_SEMICOLON:
                    L2 = max(L_MIN, L2 - L_STEP); _apply_link_change()
                elif k == pygame.K_RIGHT:
                    BASE_X = min(BASE_X_MAX, BASE_X + BASE_STEP); _apply_base_change()
                elif k == pygame.K_LEFT:
                    BASE_X = max(BASE_X_MIN, BASE_X - BASE_STEP); _apply_base_change()
                elif k == pygame.K_UP:
                    BASE_Y = min(WS_H, BASE_Y + BASE_STEP); _apply_base_change()
                elif k == pygame.K_DOWN:
                    BASE_Y = max(BASE_Y_MIN, BASE_Y - BASE_STEP); _apply_base_change()
                elif k == pygame.K_x:
                    S_MARGIN = min(S_MARGIN_MAX, S_MARGIN + S_MARGIN_STEP)
                    recompute_offset(); _revalidate()
                elif k == pygame.K_z:
                    S_MARGIN = max(0.0, S_MARGIN - S_MARGIN_STEP)
                    recompute_offset(); _revalidate()

        keys = pygame.key.get_pressed()
        dx = (keys[pygame.K_d] - keys[pygame.K_a]) * speed
        dy = (keys[pygame.K_w] - keys[pygame.K_s]) * speed

        if dx or dy:
            ex, ey, result = try_move(ex, ey, dx, dy)
            if result is not None:
                t1, t2, s1, s2 = result
                reachable = True
                # Append to trail when moving
                if not trail or abs(ex-trail[-1][0]) > 0.3 or abs(ey-trail[-1][1]) > 0.3:
                    trail.append((ex, ey))
                    if len(trail) > MAX_TRAIL:
                        trail.pop(0)
            else:
                reachable = False

        # ── Render ────────────────────────────────────────────
        screen.fill(BG)
        draw_workspace(screen)
        draw_reach_circles(screen)
        draw_trail(screen, trail)

        if reachable:
            draw_arm(screen, t1, t2, True)
        else:
            draw_arm(screen, t1, t2, False)   # arm stays at last valid pose

        draw_target_cursor(screen, ex, ey, reachable)

        # Base marker — drawn below workspace rectangle (BASE_Y = -80mm)
        bp = ws2s(BASE_X, BASE_Y)
        pygame.draw.circle(screen, BASE_C, bp, 9)
        pygame.draw.circle(screen, (255, 255, 255), bp, 9, 2)
        bl = sfont.render("BASE", True, (190, 160, 130))
        screen.blit(bl, (bp[0] - 14, bp[1] + 12))
        # Dashed line from base to workspace near edge (visual guide)
        near_edge = ws2s(BASE_X, 0)
        for i in range(0, int(abs(bp[1] - near_edge[1])), 8):
            y0 = near_edge[1] + i
            y1 = min(bp[1], y0 + 4)
            pygame.draw.line(screen, (80, 80, 100), (bp[0], y0), (bp[0], y1), 1)

        # Workspace size label
        wl = sfont.render(f"{int(WS_W)}x{int(WS_H)} mm", True, WS_BORDER)
        screen.blit(wl, (WS_OX, WS_OY - 20))

        # Divider
        div_x = int(WS_OX + WS_W * SCALE) + 14
        pygame.draw.line(screen, DIVIDER_C, (div_x, 20), (div_x, WIN_H - 20))

        state = dict(ex=ex, ey=ey, t1=t1, t2=t2, s1=s1, s2=s2,
                     speed=speed, reachable=reachable,
                     l1=L1, l2=L2, bx=BASE_X, by=BASE_Y, margin=S_MARGIN)
        draw_panel(screen, (font, sfont), state)

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()
    sys.exit(0)


if __name__ == "__main__":
    main()
