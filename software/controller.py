"""
2R Arm Controller
Usage:
    python controller.py --port /dev/cu.usbserial-XXXX [--baud 115200]
    python controller.py --demo          # run workspace verification pattern
    python controller.py --interactive   # REPL mode
"""

import argparse
import math
import time
import sys

try:
    import serial
except ImportError:
    print("Install pyserial first:  pip install pyserial")
    sys.exit(1)

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    HAS_PLOT = True
except ImportError:
    HAS_PLOT = False

# ── Arm parameters ────────────────────────────────────────────
L1 = 125.0   # mm, shoulder → elbow
L2 =  95.0   # mm, elbow → end-effector

SERVO1_OFFSET = 23.08  # degrees added to theta1 for servo command
SERVO2_OFFSET =  0.0   # degrees added to theta2 for servo command

# Workspace rectangle (mm, origin at bottom-left corner)
WS_X_MIN, WS_X_MAX = 0.0, 200.0   # 200mm is the wide (horizontal) axis
WS_Y_MIN, WS_Y_MAX = 0.0, 150.0   # 150mm is the depth axis
# Base is placed at workspace (100, -80): centre of 200mm near edge, 80mm behind it
BASE_X = 100.0   # workspace X of arm base
BASE_Y = -45.0   # workspace Y of arm base (below workspace)

# ── IK (mirrors firmware logic for preview) ───────────────────

def ik_solve(x, y):
    """
    Compute joint angles for end-effector at (x, y) relative to arm base.
    Returns (theta1_deg, theta2_deg) or raises ValueError if unreachable.
    """
    c2 = (x**2 + y**2 - L1**2 - L2**2) / (2 * L1 * L2)
    if abs(c2) > 1.0:
        raise ValueError(f"Target ({x:.1f}, {y:.1f}) is out of reach")
    s2 = math.sqrt(1 - c2**2)            # elbow-up: s2 > 0
    theta2 = math.degrees(math.atan2(s2, c2))
    alpha   = math.degrees(math.atan2(y, x))
    beta    = math.degrees(math.atan2(L2 * s2, L1 + L2 * c2))
    theta1  = alpha - beta
    return theta1, theta2


def fk_solve(theta1_deg, theta2_deg):
    """Forward kinematics: returns (ex, ey) from joint angles (degrees)."""
    t1 = math.radians(theta1_deg)
    t2 = math.radians(theta2_deg)
    ex = L1 * math.cos(t1) + L2 * math.cos(t1 + t2)
    ey = L1 * math.sin(t1) + L2 * math.sin(t1 + t2)
    return ex, ey


# ── Serial communication ──────────────────────────────────────

class ArmController:
    def __init__(self, port, baud=115200, timeout=5.0):
        self.ser = serial.Serial(port, baud, timeout=timeout)
        time.sleep(2.0)   # wait for Arduino reset
        self._flush_startup()

    def _flush_startup(self):
        while self.ser.in_waiting:
            line = self.ser.readline().decode(errors='replace').strip()
            print(f"[ARM] {line}")

    def _send(self, cmd: str) -> str:
        self.ser.write((cmd.strip() + '\n').encode())
        response = self.ser.readline().decode(errors='replace').strip()
        return response

    def move_to(self, x: float, y: float) -> bool:
        """
        Move end-effector to workspace coordinate (x, y) in mm.
        Coordinate origin is at the bottom-left of the 150×200mm rectangle.
        Returns True on success.
        """
        # Convert workspace coords → arm-base-relative coords
        arm_x = x - BASE_X
        arm_y = y - BASE_Y   # BASE_Y=-80 so arm_y = y + 80
        resp = self._send(f"G X{arm_x:.2f} Y{arm_y:.2f}")
        print(f"[ARM] {resp}")
        return resp.startswith("OK")

    def home(self) -> bool:
        resp = self._send("H")
        print(f"[ARM] {resp}")
        return resp.startswith("OK")

    def where(self) -> str:
        return self._send("W")

    def close(self):
        self.ser.close()


# ── Workspace visualiser ──────────────────────────────────────

def plot_arm(theta1, theta2, target_x=None, target_y=None, history=None):
    if not HAS_PLOT:
        return
    fig, ax = plt.subplots(figsize=(6, 7))

    # Draw workspace rectangle in arm-relative coords (base is at origin)
    # workspace bottom-left in arm-rel: (-BASE_X, -BASE_Y) = (-75, 80)
    rect = patches.Rectangle(
        (-BASE_X, -BASE_Y), WS_X_MAX, WS_Y_MAX,
        linewidth=1.5, edgecolor='steelblue', facecolor='#e8f4fd', label='Workspace'
    )
    ax.add_patch(rect)

    # Draw arm links
    t1 = math.radians(theta1)
    t2 = math.radians(theta2)
    j1 = (0, 0)
    j2 = (L1 * math.cos(t1), L1 * math.sin(t1))
    ee = (j2[0] + L2 * math.cos(t1 + t2), j2[1] + L2 * math.sin(t1 + t2))

    ax.plot([j1[0], j2[0]], [j1[1], j2[1]], 'b-o', lw=3, ms=8, label='Link 1')
    ax.plot([j2[0], ee[0]], [j2[1], ee[1]], 'r-o', lw=3, ms=8, label='Link 2')
    ax.plot(*j1, 'ko', ms=12)
    ax.plot(*ee, 'g*', ms=14, label='End-effector')

    if history:
        hx = [p[0] - BASE_X for p in history]
        hy = [p[1] - BASE_Y for p in history]
        ax.plot(hx, hy, 'g--', lw=1, alpha=0.5)

    if target_x is not None:
        ax.plot(target_x - BASE_X, target_y - BASE_Y, 'rx', ms=12, mew=2, label='Target')

    ax.set_xlim(-L1 - L2 - 10, L1 + L2 + 10)
    ax.set_ylim(-20, L1 + L2 + 10)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=8)
    ax.set_title(f'θ1={theta1:.1f}°  θ2={theta2:.1f}°')
    ax.set_xlabel('X (mm, arm-relative)')
    ax.set_ylabel('Y (mm)')
    plt.tight_layout()
    plt.show()


# ── Demo: verify four corners + centre ───────────────────────

DEMO_POINTS = [
    (100, 75,  "Centre"),
    (0,   0,   "Near-left"),
    (200, 0,   "Near-right"),
    (200, 150, "Far-right"),
    (0,   150, "Far-left"),
    (100, 0,   "Near-centre"),
    (100, 150, "Far-centre"),
]

def run_demo(arm: ArmController):
    print("\n=== Workspace verification demo ===")
    for x, y, label in DEMO_POINTS:
        print(f"\n→ {label}  ({x}, {y}) mm")
        ok = arm.move_to(x, y)
        if not ok:
            print(f"  FAILED at {label}")
        time.sleep(1.5)
    arm.home()
    print("\nDemo complete.")


def run_interactive(arm: ArmController):
    print("\nInteractive mode. Commands:")
    print("  <x> <y>    move to workspace coordinate (mm)")
    print("  h          home")
    print("  w          current position")
    print("  q          quit")
    while True:
        try:
            raw = input("arm> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not raw:
            continue
        if raw in ('q', 'quit', 'exit'):
            break
        if raw in ('h', 'home'):
            arm.home()
            continue
        if raw in ('w', 'where'):
            print(arm.where())
            continue
        parts = raw.split()
        if len(parts) == 2:
            try:
                x, y = float(parts[0]), float(parts[1])
            except ValueError:
                print("Usage: <x_mm> <y_mm>")
                continue
            # Preview IK locally before sending
            try:
                t1, t2 = ik_solve(x - BASE_X, y - BASE_Y)
                print(f"  IK preview: θ1={t1:.1f}°  θ2={t2:.1f}°")
            except ValueError as e:
                print(f"  {e}")
                continue
            arm.move_to(x, y)
        else:
            print("Unknown command")
    arm.home()


# ── Entry point ───────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='2R Arm Controller')
    parser.add_argument('--port',        help='Serial port (e.g. /dev/cu.usbserial-XXXX)')
    parser.add_argument('--baud',        type=int, default=115200)
    parser.add_argument('--demo',        action='store_true', help='Run workspace demo')
    parser.add_argument('--interactive', action='store_true', help='Interactive REPL')
    parser.add_argument('--preview',     nargs=2, type=float, metavar=('X', 'Y'),
                        help='Preview IK for given workspace coordinate (no serial needed)')
    args = parser.parse_args()

    # Local IK preview (no hardware needed)
    if args.preview:
        wx, wy = args.preview
        try:
            t1, t2 = ik_solve(wx - BASE_X, wy - BASE_Y)
            print(f"IK for ({wx}, {wy}) mm:")
            print(f"  θ1 = {t1:.2f}°   θ2 = {t2:.2f}°")
            print(f"  servo1 = {t1 + SERVO1_OFFSET:.2f}°   servo2 = {t2 + SERVO2_OFFSET:.2f}°")
            ex, ey = fk_solve(t1, t2)
            print(f"  FK check: ({ex + BASE_X:.2f}, {ey + BASE_Y:.2f}) mm")
            plot_arm(t1, t2, wx, wy)
        except ValueError as e:
            print(e)
        return

    if not args.port:
        parser.print_help()
        print("\nTip: use --preview X Y to test IK without hardware.")
        return

    arm = ArmController(args.port, args.baud)
    try:
        if args.demo:
            run_demo(arm)
        elif args.interactive:
            run_interactive(arm)
        else:
            run_interactive(arm)
    finally:
        arm.close()


if __name__ == '__main__':
    main()
