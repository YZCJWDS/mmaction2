#!/bin/bash
# ============================================================
# EGTEA Gaze+ - 环境状态检查
# Usage: bash projects/egtea_gaze/tools/check_env.sh
# ============================================================

echo "============================================================"
echo "  EGTEA Gaze+ 环境状态报告"
echo "============================================================"
echo ""

# ---- GPU 状态 ----
echo "--- GPU 状态 ---"
if ! command -v nvidia-smi &> /dev/null; then
    echo "  状态: nvidia-smi 不存在"
    echo "  结论: 无 CUDA 驱动，无法训练"
    GPU_STATUS="no_driver"
elif ! nvidia-smi --query-gpu=name --format=csv,noheader &> /dev/null; then
    echo "  状态: nvidia-smi 存在，但无 GPU 设备"
    echo "  结论: 无卡模式（可配置环境和数据，不可训练）"
    GPU_STATUS="no_device"
else
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
    GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader | head -1)
    CUDA_DRIVER=$(nvidia-smi | grep -oP "CUDA Version: \K[0-9]+\.[0-9]+" || echo "unknown")
    echo "  状态: GPU 可用"
    echo "  GPU: $GPU_NAME ($GPU_MEM)"
    echo "  CUDA Driver: $CUDA_DRIVER"
    GPU_STATUS="available"
fi
echo ""

# ---- Conda 环境 ----
echo "--- Conda 环境 ---"
if command -v conda &> /dev/null; then
    CONDA_ENV=$(echo $CONDA_DEFAULT_ENV 2>/dev/null || basename "$CONDA_PREFIX" 2>/dev/null || echo "unknown")
    echo "  当前环境: $CONDA_ENV"
    echo "  conda: $(conda --version 2>/dev/null)"
else
    echo "  [WARN] conda 不可用"
fi
echo ""

# ---- Python ----
echo "--- Python ---"
echo "  路径: $(which python 2>/dev/null || echo 'not found')"
echo "  版本: $(python --version 2>/dev/null || echo 'not found')"
echo ""

# ---- 核心依赖 ----
echo "--- 核心依赖 ---"
python -c "import torch; print(f'  torch:       {torch.__version__}')" 2>/dev/null || echo "  torch:       [未安装]"
python -c "import torchvision; print(f'  torchvision: {torchvision.__version__}')" 2>/dev/null || echo "  torchvision: [未安装]"
python -c "import mmengine; print(f'  mmengine:    {mmengine.__version__}')" 2>/dev/null || echo "  mmengine:    [未安装]"
python -c "import mmcv; print(f'  mmcv:        {mmcv.__version__}')" 2>/dev/null || echo "  mmcv:        [未安装]"
python -c "import mmaction; print(f'  mmaction:    {mmaction.__version__}')" 2>/dev/null || echo "  mmaction:    [未安装]"
python -c "import decord; print(f'  decord:      {decord.__version__}')" 2>/dev/null || echo "  decord:      [未安装]"
echo ""

# ---- CUDA 运行时 ----
echo "--- torch CUDA ---"
python -c "
import torch
print(f'  torch.cuda.is_available: {torch.cuda.is_available()}')
print(f'  torch.version.cuda:      {torch.version.cuda}')
print(f'  device_count:            {torch.cuda.device_count()}')
" 2>/dev/null || echo "  [跳过] torch 未安装"
echo ""

# ---- 磁盘 ----
echo "--- 磁盘空间 ---"
df -h / 2>/dev/null | tail -1
echo ""

# ---- 目录 ----
echo "--- 项目目录 ---"
for d in /root/code/mmaction2 /root/data/egtea /root/data/egtea/videos/cropped_clips /root/data/egtea/action_annotation /root/outputs/egtea_gaze /root/checkpoints; do
    echo "  $d: $([ -d $d ] && echo 'EXISTS' || echo '-')"
done
echo ""

# ---- 建议 ----
echo "--- 建议下一步 ---"
if [ "$GPU_STATUS" = "no_device" ] || [ "$GPU_STATUS" = "no_driver" ]; then
    echo "  当前无卡，可执行:"
    echo "    bash projects/egtea_gaze/tools/setup_env.sh      # 安装缺失依赖"
    echo "    bash projects/egtea_gaze/tools/quickstart.sh --mode data-only  # 整理数据"
else
    echo "  GPU 可用，可执行:"
    echo "    bash projects/egtea_gaze/tools/quickstart.sh --mode full"
fi
echo "============================================================"
