"""CLI entry point for v3.

    python3 main.py control   # pygame + hardware: the one control panel --
                               # jog, teach, record/replay paths, run scans,
                               # and live-edit every calib.json parameter
                               # (see control_gui.py)
    python3 main.py preview   # no hardware: offline sanity-check of the
                               # current calib.json rectangle/photo config
"""

from __future__ import annotations

import argparse
import math

import arm_core as core
import path_core as pc


def cmd_control(_args) -> None:
    import control_gui
    control_gui.main()


def cmd_preview(_args) -> None:
    calib = core.load_calib()
    rect = core.calib_rectangle(calib)
    if rect is None:
        print("calib.json has no 'rectangle' yet -- run `python3 main.py teach` first "
              "(or hand-edit calib.json's 'rectangle' section for an offline dry run).")
        return

    photo_cfg = core.calib_photo_config(calib)
    cx, cy, w, h, rot = rect
    photo_points = pc.generate_photo_grid(
        rect, photo_cfg.spacing_x_mm, photo_cfg.spacing_y_mm, photo_cfg.margin_mm)

    print(f"rectangle: center=({cx:.1f},{cy:.1f}) mm  {w:.1f} x {h:.1f} mm  "
          f"rotation={rot:.1f}deg")
    print(f"photo grid: {len(photo_points)} points, spacing="
          f"{photo_cfg.spacing_x_mm:.1f}x{photo_cfg.spacing_y_mm:.1f}mm, "
          f"margin={photo_cfg.margin_mm:.1f}mm, dwell={photo_cfg.dwell_s:.2f}s, "
          f"max_step={photo_cfg.max_step_mm:.1f}mm")

    rows: dict[str, list[tuple[float, float]]] = {}
    for x, y, label in photo_points:
        rows.setdefault(label, []).append((x, y))
    print(f"rows: {len(rows)}, points/row: {len(next(iter(rows.values())))}")

    total_travel_mm = 0.0
    coords = [(x, y) for x, y, _ in photo_points]
    for i in range(len(coords) - 1):
        total_travel_mm += math.dist(coords[i], coords[i + 1])
    print(f"total scan travel (photo-point to photo-point): {total_travel_mm:.1f} mm")

    unreachable = 0
    params = core.calib_arm_params(calib)
    joint_limits = core.calib_joint_limits(calib)
    for x, y, label in photo_points:
        r = core.ik_solve(params, x, y, joint_limits=joint_limits)
        if not r.reachable:
            unreachable += 1
    if unreachable:
        print(f"WARNING: {unreachable}/{len(photo_points)} photo points are NOT reachable "
              f"with the current kinematics/joint_limits_deg -- check L1/L2/base_x/base_y "
              f"and the taught rectangle before running for real.")
    else:
        print(f"all {len(photo_points)} photo points are reachable.")

    print()
    print("row-by-row preview:")
    for label, pts in rows.items():
        xs = ", ".join(f"({x:.1f},{y:.1f})" for x, y in pts)
        print(f"  {label}: {xs}")


def main():
    parser = argparse.ArgumentParser(
        description="2R arm v3: teach a scan rectangle by hand, then run a serpentine "
                    "photo-grid scan over it.")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("control", help="pygame + hardware: the one control panel (see control_gui.py)")
    sub.add_parser("preview", help="offline sanity-check of the current rectangle/photo config")

    args = parser.parse_args()
    {"control": cmd_control, "preview": cmd_preview}[args.command](args)


if __name__ == "__main__":
    main()
