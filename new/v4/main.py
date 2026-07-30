"""CLI entry point for v4: hand-mark a sequence of points, then replay
through them with a pause at each one.

    python3 main.py record [--out PATH.json] [--port PORT]
    python3 main.py replay [--in PATH.json] [--port PORT] [--dwell SECONDS]
    python3 main.py replay --photos photos/ [--photo-delay SECONDS]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import record_replay as rr


def main():
    parser = argparse.ArgumentParser(
        description="v4: hand-record a path (torque released) and replay it later.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_record = sub.add_parser("record", help="torque off, drag the arm, record the path")
    p_record.add_argument("--out", type=Path, default=rr.DEFAULT_PATH)
    p_record.add_argument("--port", default=rr.DEFAULT_SERVO_PORT)

    p_replay = sub.add_parser("replay", help="replay a previously recorded path")
    p_replay.add_argument("--in", dest="in_path", type=Path, default=rr.DEFAULT_PATH)
    p_replay.add_argument("--port", default=rr.DEFAULT_SERVO_PORT)
    p_replay.add_argument("--dwell", type=float, default=rr.DEFAULT_DWELL_S,
                           help=f"seconds to pause at each point (default: {rr.DEFAULT_DWELL_S})")
    p_replay.add_argument("--photos", type=Path, default=None,
                           help="directory to save one imx477 photo per point (default: "
                                "no camera, pure teach-and-replay); each run gets its own "
                                "timestamped subfolder")
    p_replay.add_argument("--photo-delay", type=float, dest="photo_delay",
                           default=rr.DEFAULT_PHOTO_DELAY_S,
                           help=f"seconds into the dwell before the shutter fires, only used "
                                f"with --photos (default: {rr.DEFAULT_PHOTO_DELAY_S})")

    args = parser.parse_args()
    if args.command == "record":
        rr.record(args.out, servo_port=args.port)
    else:
        rr.replay(args.in_path, servo_port=args.port, dwell_s=args.dwell,
                  photo_dir=args.photos, photo_delay_s=args.photo_delay)


if __name__ == "__main__":
    main()
