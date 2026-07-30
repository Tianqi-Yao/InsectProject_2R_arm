#!/usr/bin/env bash
# 暂停v4-replay的定时任务：把crontab里那一行注释掉（配置还在，不会删掉），
# 之后想恢复重新运行./install_replay_cron.sh即可。反复运行这个脚本是安全
# 的——已经停掉的话再跑一次不会重复处理。
set -euo pipefail

MARKER="# v4-replay-timer"

existing="$(crontab -l 2>/dev/null || true)"
if [[ -z "$existing" ]] || ! printf '%s\n' "$existing" | grep -qF "${MARKER}"; then
    echo "没有找到v4-replay的crontab条目，不用停。"
    exit 0
fi

printf '%s\n' "$existing" | awk -v marker="${MARKER}" '
    index($0, marker) > 0 && substr($0, 1, 1) != "#" { print "# " $0; next }
    { print }
' | crontab -

echo "已停止 -- crontab里那一行被注释掉了，不会再自动触发。"
echo "恢复: 重新运行 ./install_replay_cron.sh"
