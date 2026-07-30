#!/usr/bin/env bash
# 彻底删除v4-replay的定时任务：把crontab里那一行（不管是启用状态还是被
# stop脚本注释掉的）整条删掉。反复运行是安全的——本来就没有的话直接跳过。
set -euo pipefail

MARKER="# v4-replay-timer"

existing="$(crontab -l 2>/dev/null || true)"
if [[ -z "$existing" ]] || ! printf '%s\n' "$existing" | grep -qF "${MARKER}"; then
    echo "没有找到v4-replay的crontab条目，不用卸载。"
    exit 0
fi

filtered="$(printf '%s\n' "$existing" | grep -vF "${MARKER}" || true)"
if [[ -n "$filtered" ]]; then
    printf '%s\n' "$filtered" | crontab -
else
    crontab -r
fi

echo "已彻底删除crontab里v4-replay的那一行。想重新启用，运行 ./install_replay_cron.sh"
