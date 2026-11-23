#!/usr/bin/env bash
# ============================================================
# Yue2-Studio 一键部署脚本（Ubuntu / Colab 容器 + NVIDIA GPU）
# 用法: bash deploy_from_github.sh
# 幂等：可重复执行，已完成的步骤会跳过
# 适用：系统重启后恢复部署
#
# 资源获取优先级：
#   源码：本地已有 -> 云盘副本 -> GitHub clone
#   模型：本地已有 -> 云盘副本 -> 网络下载（ModelScope；paraformer 无 HF 源）
#   .venv：每次重建（保证干净可靠，pip 源够快）
#
#   内置补丁（重启自动重新应用，无需手动改代码）：
#   · T4 GPU 阈值 8.0 -> 7.5（让 T4 默认走 cuda）
#   · 解码 budget 24 -> 12（长音频 VAE 解码避免主机内存 OOM）
# ============================================================
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

export LD_LIBRARY_PATH=/usr/lib64-nvidia:${LD_LIBRARY_PATH:-}
export HF_ENDPOINT=https://hf-mirror.com
export PYTHONUTF8=1
export PYTHONUNBUFFERED=1

VENV="$ROOT/YuE/.venv"
PY="$VENV/bin/python"
MODELS="$ROOT/YuE/models"
LOG="/tmp/yue2_deploy.log"
DRIVE_ROOT="/content/drive/MyDrive/yue2-studio-backup/full"
DRIVE_MODELS="$DRIVE_ROOT/YuE/models"

say() { echo "[$(date '+%F %H:%M:%S')] $*" | tee -a "$LOG"; }

say "===== Yue2-Studio 一键部署开始 (ROOT=$ROOT) ====="

# --- 0. 前置检查 --------------------------------------------------------
command -v python3.12 >/dev/null || { say "错误：未找到 python3.12"; exit 1; }
if [ ! -e /dev/nvidia0 ]; then
  say "警告：未检测到 /dev/nvidia0，GPU 可能不可用（将退回 CPU）"
fi

# --- 0.5 源码自举（全新重启 /content 被清空时）--------------------------
if [ ! -f "$ROOT/requirements-deploy.txt" ]; then
  say "检测到源码缺失，尝试恢复 ..."
  if [ -d "$DRIVE_ROOT/app" ]; then
    say "从云盘副本恢复源码到 $ROOT ..."
    mkdir -p "$ROOT"
    rsync -a "$DRIVE_ROOT/" "$ROOT/"
  else
    say "云盘副本不存在，从 GitHub clone ..."
    git clone --depth 1 https://github.com/lvshenfx/Yue2-Studio.git "$ROOT" || {
      say "源码恢复失败：云盘和 GitHub 均不可用"; exit 1;
    }
    cd "$ROOT"
  fi
fi

# --- 1. 系统依赖 --------------------------------------------------------
if ! python3.12 -m venv --help >/dev/null 2>&1; then
  say "安装 python3.12-venv ..."
  apt-get update -qq >/dev/null 2>&1
  DEBIAN_FRONTEND=noninteractive apt-get install -y -qq python3.12-venv >/dev/null 2>&1
fi

# --- 2. 虚拟环境（每次重建，保证干净可靠）------------------------------
if [ -x "$PY" ]; then
  say "删除旧虚拟环境，准备重建 ..."
  rm -rf "$VENV"
fi
say "创建虚拟环境 $VENV ..."
python3.12 -m venv "$VENV" || { say "venv 创建失败"; exit 1; }

# Windows 解释器路径兼容：代码里查找 Scripts/python.exe 与 .venv/python.exe
ln -sfn bin/python "$VENV/python.exe"
mkdir -p "$VENV/Scripts"
ln -sfn ../bin/python "$VENV/Scripts/python.exe"

# --- 3. Python 依赖 -----------------------------------------------------
say "安装 Python 依赖（torch CUDA 版约 2.3GB，请耐心）..."
"$PY" -m pip install -q --upgrade pip 2>>"$LOG"
# torch 必须从官方 cu128 源装，用requirements里的固定版本
"$PY" -m pip install -q torch==2.10.0+cu128 \
    --index-url https://download.pytorch.org/whl/cu128 2>>"$LOG" \
  || { say "torch 安装失败"; exit 1; }
# 其余依赖（去掉 torch 行，避免覆盖 CUDA 版）
grep -v '^torch==' requirements-deploy.txt > /tmp/yue2_reqs.txt
"$PY" -m pip install -q -r /tmp/yue2_reqs.txt \
    -i https://pypi.tuna.tsinghua.edu.cn/simple 2>>"$LOG" \
  || { say "依赖安装失败"; exit 1; }
say "依赖安装完成"

# 验证 CUDA
CUDA_OK=$("$PY" -c "import torch; print('yes' if torch.cuda.is_available() else 'no')" 2>/dev/null)
say "CUDA 可用: $CUDA_OK"

# --- 4. 模型权重（本地 -> 云盘 -> 网络）--------------------------------
mkdir -p "$MODELS"
get_model() {
  local dir="$1" id="$2" name="$3" marker="$4"
  if [ -f "$MODELS/$dir/$marker" ]; then
    say "$name 已存在，跳过"
  elif [ -f "$DRIVE_MODELS/$dir/$marker" ]; then
    say "从云盘恢复 $name ..."
    mkdir -p "$MODELS/$dir"
    cp -a "$DRIVE_MODELS/$dir/." "$MODELS/$dir/"
    say "$name 云盘恢复完成"
  else
    say "从网络下载 $name ($id) ..."
    if "$PY" -c "from modelscope import snapshot_download; snapshot_download('$id', local_dir=r'$MODELS/$dir')" 2>>"$LOG"; then
      say "$name 下载完成"
    else
      say "警告：$name 下载失败（非致命，可稍后重试）"
    fi
  fi
}
get_model "YuE2-3B" "m-a-p/YuE2-3B" "YuE2-3B (6.8G)" "model.safetensors"
get_model "YuE2-Vae" "m-a-p/YuE2-Vae" "YuE2-Vae (507M)" "model.safetensors"
get_model "paraformer-zh-en-ts" "manyeyes/paraformer-large-zh-en-timestamp-int8-onnx-offline" "paraformer (233M)" "model.int8.onnx"

# --- 5. GPU 适配补丁（T4 算力 7.5，默认走 cuda）------------------------
say "检查 GPU 适配补丁 ..."
"$PY" - <<'PYEOF'
from pathlib import Path
p = Path('app/server.py')
s = p.read_text(encoding='utf-8')
old = '    if capability < 8.0:\n        suitable, reason = False, f"{name} 不支持 BF16（算力 {capability:.1f}），使用 CPU"'
new = '    if capability < 7.5:\n        suitable, reason = False, f"{name} 算力 {capability:.1f} 过低，使用 CPU"'
if old in s:
    p.write_text(s.replace(old, new, 1), encoding='utf-8')
    print('[patch] 已应用 T4 GPU 阈值补丁 (8.0 -> 7.5)')
else:
    print('[patch] 补丁已存在或代码结构已变化，跳过')
PYEOF

# --- 5.5 解码内存补丁（budget 24 -> 12，避免长音频解码 OOM）------------
say "检查解码内存补丁 ..."
"$PY" - <<'PYEOF'
from pathlib import Path
p = Path('app/server.py')
s = p.read_text(encoding='utf-8')
old = '    common = {"model": str(MODEL_DIR), "vae": str(VAE_DIR), "budget": 24}'
new = '    common = {"model": str(MODEL_DIR), "vae": str(VAE_DIR), "budget": 12}'
if old in s:
    p.write_text(s.replace(old, new, 1), encoding='utf-8')
    print('[patch] 已应用解码内存补丁 (budget 24 -> 12)，避免长音频解码 OOM')
else:
    print('[patch] 解码内存补丁已存在或代码结构已变化，跳过')
PYEOF

# --- 6. 启动服务 --------------------------------------------------------
mkdir -p runs
if curl -s --max-time 3 http://127.0.0.1:9090/api/health >/dev/null 2>&1; then
  say "服务已在 9090 端口运行，跳过启动"
else
  say "启动 Web 服务 ..."
  nohup bash "$ROOT/start_linux.sh" 9090 > runs/linux_start.log 2>&1 &
  sleep 10
fi

echo ""
say "===== 部署完成 ====="
say "健康检查: $(curl -s --max-time 5 http://127.0.0.1:9090/api/health)"
say "Web UI: http://<本机TailscaleIP>:9090/"
say "下一步：打开页面 → 加载模型 → 创作"
