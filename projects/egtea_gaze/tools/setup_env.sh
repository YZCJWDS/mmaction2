#!/bin/bash
# ============================================================
# EGTEA Gaze+ - 自适应环境配置
#
# 策略:
#   1. 优先复用当前活跃环境中已有的 torch
#   2. 只安装缺失的依赖 (mmengine/mmcv/mmaction/decord)
#   3. 无卡 + 无 torch 时拒绝自动下载，给出提示并退出
#   4. 只有 FORCE_NEW_ENV=1 或 (有GPU+无torch) 时才创建 egtea 环境
#
# Usage:
#   cd /root/code/mmaction2
#   bash projects/egtea_gaze/tools/setup_env.sh
#
# 强制创建独立环境:
#   FORCE_NEW_ENV=1 bash projects/egtea_gaze/tools/setup_env.sh
# ============================================================

set -e

REPO_DIR="${REPO_DIR:-/root/code/mmaction2}"

echo "============================================================"
echo "  EGTEA Gaze+ 自适应环境配置"
echo "============================================================"
echo ""

# ---- 检测 GPU 状态 ----
echo "[1/5] 检测环境..."

HAS_GPU=0
if command -v nvidia-smi &> /dev/null; then
    if nvidia-smi --query-gpu=name --format=csv,noheader &> /dev/null; then
        HAS_GPU=1
        echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
    else
        echo "  GPU: 无卡模式 (nvidia-smi 存在但无设备)"
    fi
else
    echo "  GPU: 无卡模式 (无 nvidia-smi)"
fi

# ---- 检测当前 Python 环境 ----
CURRENT_ENV=$(echo $CONDA_DEFAULT_ENV 2>/dev/null || basename "$CONDA_PREFIX" 2>/dev/null || echo "unknown")
echo "  Conda 环境: $CURRENT_ENV"
echo "  Python: $(python --version 2>/dev/null) @ $(which python 2>/dev/null)"

# 检测 torch
HAS_TORCH=0
TORCH_VER=""
if python -c "import torch" 2>/dev/null; then
    HAS_TORCH=1
    TORCH_VER=$(python -c "import torch; print(torch.__version__)")
    TORCH_CUDA=$(python -c "import torch; print(torch.version.cuda)" 2>/dev/null || echo "none")
    echo "  torch: $TORCH_VER (CUDA built: $TORCH_CUDA)"
    echo "  torch.cuda.is_available: $(python -c 'import torch; print(torch.cuda.is_available())')"
else
    echo "  torch: [未安装]"
fi
echo ""

# ---- 决策: 是否需要创建新环境 ----
echo "[2/5] 环境决策..."

USE_CURRENT=1  # 默认复用当前环境

if [ "${FORCE_NEW_ENV:-0}" = "1" ]; then
    echo "  FORCE_NEW_ENV=1，将创建独立 egtea 环境"
    USE_CURRENT=0
elif [ $HAS_TORCH -eq 0 ]; then
    if [ $HAS_GPU -eq 1 ]; then
        echo "  当前无 torch + 有 GPU，将创建 egtea 环境并安装 torch"
        USE_CURRENT=0
    else
        echo ""
        echo "  [STOP] 当前环境无 torch，且无 GPU 设备。"
        echo "  无卡模式下不自动下载 torch (约 2.2GB)。"
        echo ""
        echo "  解决方案:"
        echo "    A. 切换到已有 torch 的环境 (如 py312)，再运行本脚本"
        echo "    B. 挂载 GPU 后再运行本脚本"
        echo "    C. 手动安装: pip install torch torchvision --index-url ..."
        echo ""
        exit 1
    fi
else
    echo "  复用当前环境 ($CURRENT_ENV)，已有 torch $TORCH_VER"
fi

# 如果需要创建新环境
if [ $USE_CURRENT -eq 0 ]; then
    eval "$(conda shell.bash hook)" 2>/dev/null || true
    if conda env list | grep -q "^egtea "; then
        echo "  激活已有 egtea 环境"
        conda activate egtea
    else
        echo "  创建 egtea 环境 (python=3.9)..."
        conda create -n egtea python=3.9 -y
        conda activate egtea
    fi

    # 安装 torch (仅在有 GPU 时)
    if [ $HAS_GPU -eq 1 ]; then
        CUDA_VER=$(nvidia-smi | grep -oP "CUDA Version: \K[0-9]+\.[0-9]+" || echo "11.8")
        CUDA_MAJOR=$(echo $CUDA_VER | cut -d. -f1)
        if [ "$CUDA_MAJOR" -ge 12 ]; then
            pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121
        else
            pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118
        fi
    fi
    echo "  Python: $(python --version) @ $(which python)"
fi
echo ""

# ---- 安装缺失依赖 ----
echo "[3/5] 安装缺失依赖..."

# openmim
python -c "import mim" 2>/dev/null || { echo "  安装 openmim..."; pip install -U openmim -q; }

# mmengine
python -c "import mmengine" 2>/dev/null || { echo "  安装 mmengine..."; mim install mmengine; }

# mmcv (带快速失败保护)
if ! python -c "import mmcv" 2>/dev/null; then
    echo "  安装 mmcv..."
    echo "  (优先使用预编译 wheel，如果卡住超过 5 分钟请 Ctrl+C)"

    # 尝试 mim install，设置 pip 超时避免无限编译
    if ! pip install mmcv>=2.0.0 --timeout 300 -f https://download.openmmlab.com/mmcv/dist/index.html 2>/dev/null; then
        echo "  [WARN] pip install mmcv 失败，尝试 mim install..."
        if ! timeout 600 mim install "mmcv>=2.0.0"; then
            echo ""
            echo "  [ERROR] mmcv 安装失败。"
            echo "  可能原因: 当前 Python/torch 版本组合没有预编译 wheel，触发了源码编译。"
            echo "  建议:"
            echo "    1. 检查 https://mmcv.readthedocs.io/en/latest/get_started/installation.html"
            echo "    2. 手动安装匹配版本: pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.3/index.html"
            echo ""
            exit 1
        fi
    fi
fi

# mmaction2 (源码安装)
if ! python -c "import mmaction" 2>/dev/null; then
    echo "  安装 mmaction2 (源码开发模式)..."
    cd "$REPO_DIR"
    pip install -v -e . 2>&1 | tail -5
fi

# decord
python -c "import decord" 2>/dev/null || { echo "  安装 decord..."; pip install decord; }

# scipy / tensorboard
python -c "import scipy" 2>/dev/null || pip install scipy -q
python -c "import tensorboard" 2>/dev/null || pip install tensorboard -q

echo ""

# ---- 创建目录 ----
echo "[4/5] 创建目录..."
mkdir -p /root/data/egtea/raw /root/data/egtea/videos /root/data/egtea/action_annotation /root/data/egtea/gaze_data
mkdir -p /root/outputs/egtea_gaze /root/checkpoints /root/logs
echo "  [OK]"
echo ""

# ---- 设置权限 ----
echo "[5/5] 脚本权限..."
cd "$REPO_DIR"
chmod +x projects/egtea_gaze/tools/*.sh 2>/dev/null || true
echo "  [OK]"
echo ""

# ---- 最终验证 ----
echo "============================================================"
echo "  验证结果:"
echo "============================================================"
python -c "
import torch, mmaction, mmengine, mmcv, decord
print(f'  torch:      {torch.__version__} (CUDA built: {torch.version.cuda})')
print(f'  CUDA avail: {torch.cuda.is_available()}')
print(f'  mmengine:   {mmengine.__version__}')
print(f'  mmcv:       {mmcv.__version__}')
print(f'  mmaction:   {mmaction.__version__}')
print(f'  decord:     {decord.__version__}')
print()
print('  [OK] 环境就绪')
"
echo ""
if [ $HAS_GPU -eq 0 ]; then
    echo "  [提醒] 无卡模式。挂卡后可直接训练，无需重新配置。"
fi
echo "============================================================"
