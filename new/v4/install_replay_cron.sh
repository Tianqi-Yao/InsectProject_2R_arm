#!/usr/bin/env bash
# 用crontab设一个最简单的定时任务，每隔 INTERVAL_MIN 分钟跑一次
# `main.py replay --photos ...`。反复运行是幂等的——每次都靠下面这个
# MARKER先删掉旧的那一行（不管是启用状态还是被stop脚本注释掉的），再插入
# 新的一行，不会累积出重复的定时任务，改了参数重新跑一次就是更新配置。
set -euo pipefail

V4_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MARKER="# v4-replay-timer"

PORT="${PORT:-/dev/ttyUSB0}"
PHOTOS_DIR="${PHOTOS_DIR:-$V4_DIR/photos}"
INTERVAL_MIN="${INTERVAL_MIN:-30}"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python3 || true)}"
LOG_FILE="${LOG_FILE:-$V4_DIR/replay.log}"

usage() {
    cat <<EOF
用法: $0 [--port /dev/ttyUSB0] [--photos DIR] [--interval-min 30] [--python PATH]

  --port PORT        舵机串口 (默认: ${PORT})
  --photos DIR       拍照保存目录 (默认: ${PHOTOS_DIR})
  --interval-min N   每隔多少分钟跑一次 (默认: ${INTERVAL_MIN})
  --python PATH      要用哪个python3 (默认: 当前PATH里的python3)
                     如果依赖装在venv里，要么先激活venv再跑这个脚本，要么用
                     这个参数指定venv里的python3路径 (比如.venv/bin/python3)
                     ——cron不会帮你自动激活venv。
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --port) PORT="$2"; shift 2 ;;
        --photos) PHOTOS_DIR="$2"; shift 2 ;;
        --interval-min) INTERVAL_MIN="$2"; shift 2 ;;
        --python) PYTHON_BIN="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "未知参数: $1" >&2; usage; exit 1 ;;
    esac
done

if [[ -z "$PYTHON_BIN" ]]; then
    echo "找不到python3，用--python指定路径" >&2
    exit 1
fi

CRON_LINE="*/${INTERVAL_MIN} * * * * cd ${V4_DIR} && ${PYTHON_BIN} main.py replay --photos ${PHOTOS_DIR} --port ${PORT} >> ${LOG_FILE} 2>&1 ${MARKER}"

existing="$(crontab -l 2>/dev/null || true)"
filtered="$(printf '%s\n' "$existing" | grep -vF "${MARKER}" || true)"
{ [[ -n "$filtered" ]] && printf '%s\n' "$filtered"; printf '%s\n' "$CRON_LINE"; } | crontab -

echo "已写入crontab，每${INTERVAL_MIN}分钟跑一次:"
echo "  ${PYTHON_BIN} main.py replay --photos ${PHOTOS_DIR} --port ${PORT}"
echo
echo "查看:   crontab -l"
echo "日志:   tail -f ${LOG_FILE}"
echo "停止:   ./stop_replay_cron.sh"
echo "卸载:   ./uninstall_replay_cron.sh"
