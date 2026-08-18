"""Idempotent crontab install/list/uninstall for unattended scheduled
replay. `runner` is injectable (defaults to subprocess.run against the
real system crontab) so tests never touch the actual crontab -- see
tests/test_cron.py's FakeCrontabRunner."""

from __future__ import annotations

import subprocess
from typing import Callable, Optional

MARKER = "# arm2r-replay-timer"

Runner = Callable[..., subprocess.CompletedProcess]


def _read(runner: Runner) -> str:
    result = runner(["crontab", "-l"], capture_output=True, text=True)
    if result.returncode != 0:
        return ""  # no crontab installed yet for this user
    return result.stdout


def _write(content: str, runner: Runner) -> None:
    runner(["crontab", "-"], input=content, capture_output=True, text=True, check=True)


def build_line(schedule: str, command: str) -> str:
    return f"{schedule} {command} {MARKER}"


def install(schedule: str, command: str, runner: Runner = subprocess.run) -> str:
    """Removes any existing arm2r replay timer line (however it was
    scheduled before) and installs this one -- running this twice with
    different parameters UPDATES the schedule rather than accumulating
    duplicate cron entries."""
    existing = _read(runner)
    kept = [line for line in existing.splitlines() if MARKER not in line]
    new_line = build_line(schedule, command)
    kept.append(new_line)
    _write("\n".join(kept) + "\n", runner)
    return new_line


def list_installed(runner: Runner = subprocess.run) -> Optional[str]:
    for line in _read(runner).splitlines():
        if MARKER in line:
            return line
    return None


def uninstall(runner: Runner = subprocess.run) -> bool:
    """Returns True if a timer was actually removed, False if there was
    nothing to remove."""
    existing = _read(runner)
    lines = existing.splitlines()
    kept = [line for line in lines if MARKER not in line]
    if len(kept) == len(lines):
        return False
    _write(("\n".join(kept) + "\n") if kept else "", runner)
    return True
