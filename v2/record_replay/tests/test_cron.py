"""cron.py tests run entirely against an in-memory FakeCrontabRunner --
NEVER against the real system crontab. `runner` is always passed
explicitly here for that reason."""

from __future__ import annotations

import subprocess

from record_replay import cron


class FakeCrontabRunner:
    """Stands in for subprocess.run(["crontab", ...]) against an in-memory
    string, so tests can never touch the real system crontab."""

    def __init__(self, initial: str = ""):
        self.content = initial

    def __call__(self, args, capture_output=None, text=None, input=None, check=None):
        if args == ["crontab", "-l"]:
            if not self.content:
                return subprocess.CompletedProcess(args, returncode=1, stdout="", stderr="no crontab")
            return subprocess.CompletedProcess(args, returncode=0, stdout=self.content, stderr="")
        elif args == ["crontab", "-"]:
            self.content = input
            return subprocess.CompletedProcess(args, returncode=0, stdout="", stderr="")
        raise AssertionError(f"unexpected crontab invocation: {args!r}")


def test_list_installed_returns_none_when_nothing_installed():
    runner = FakeCrontabRunner()
    assert cron.list_installed(runner=runner) is None


def test_install_writes_a_marked_line():
    runner = FakeCrontabRunner()
    line = cron.install("*/30 * * * *", "arm-replay run", runner=runner)
    assert cron.MARKER in line
    assert cron.list_installed(runner=runner) == line


def test_install_preserves_unrelated_existing_cron_lines():
    runner = FakeCrontabRunner(initial="0 3 * * * /usr/bin/some-other-job\n")
    cron.install("*/30 * * * *", "arm-replay run", runner=runner)
    assert "/usr/bin/some-other-job" in runner.content
    assert cron.MARKER in runner.content


def test_install_twice_updates_rather_than_duplicates():
    runner = FakeCrontabRunner()
    cron.install("*/30 * * * *", "arm-replay run", runner=runner)
    cron.install("*/15 * * * *", "arm-replay run --photos /x", runner=runner)

    marker_lines = [l for l in runner.content.splitlines() if cron.MARKER in l]
    assert len(marker_lines) == 1
    assert "*/15" in marker_lines[0]


def test_uninstall_removes_the_line_and_returns_true():
    runner = FakeCrontabRunner()
    cron.install("*/30 * * * *", "arm-replay run", runner=runner)
    removed = cron.uninstall(runner=runner)
    assert removed is True
    assert cron.list_installed(runner=runner) is None


def test_uninstall_returns_false_when_nothing_to_remove():
    runner = FakeCrontabRunner()
    assert cron.uninstall(runner=runner) is False


def test_uninstall_preserves_unrelated_cron_lines():
    runner = FakeCrontabRunner()
    cron.install("*/30 * * * *", "arm-replay run", runner=runner)
    runner.content += "0 3 * * * /usr/bin/some-other-job\n"
    cron.uninstall(runner=runner)
    assert "/usr/bin/some-other-job" in runner.content
    assert cron.MARKER not in runner.content
