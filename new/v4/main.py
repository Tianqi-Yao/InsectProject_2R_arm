"""CLI entry point for v4: record a hand-driven path, then replay it.

    python3 main.py record [--out PATH.json] [--port PORT]
    python3 main.py replay [--in PATH.json] [--port PORT]
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

    args = parser.parse_args()
    if args.command == "record":
        rr.record(args.out, servo_port=args.port)
    else:
        rr.replay(args.in_path, servo_port=args.port)


if __name__ == "__main__":
    main()
