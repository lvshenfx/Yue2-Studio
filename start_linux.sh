#!/bin/bash
# YuE2 Studio Linux 启动脚本 (监听 0.0.0.0 供 Tailscale 访问)
# 路径自适应：以脚本所在目录为项目根目录
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

export LD_LIBRARY_PATH=/usr/lib64-nvidia:${LD_LIBRARY_PATH:-}
export HF_ENDPOINT=https://hf-mirror.com
export PYTHONUTF8=1
export PYTHONUNBUFFERED=1
export YUE2_KIT="$ROOT/YuE"
export YUE2_PORT=${1:-9090}

PY="$ROOT/YuE/.venv/bin/python"

# 尝试获取本机 Tailscale IP 用于提示
TSIP=$(tailscale ip -4 2>/dev/null | head -1 || true)
echo "YuE2 Web UI 启动于 http://0.0.0.0:${YUE2_PORT}"
[ -n "$TSIP" ] && echo "Tailscale: http://${TSIP}:${YUE2_PORT}"

exec "$PY" -m uvicorn app.server:app --host 0.0.0.0 --port "${YUE2_PORT}"
