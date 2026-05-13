#!/bin/bash
# ============================================================
# EGTEA Gaze+ 项目 - 云端环境检查脚本
# 在 JupyterLab 终端中运行: bash projects/egtea_gaze/tools/check_env.sh
# ============================================================

echo "============================================================"
echo "EGTEA Gaze+ 云端环境检查"
echo "============================================================"

echo ""
echo "--- 系统信息 ---"
echo "OS: $(uname -a)"
echo "GPU:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null || echo "  [INFO] nvidia-smi not found (无卡模式)"
echo ""

echo "--- CUDA ---"
nvcc --version 2>/dev/null || echo "  [INFO] nvcc not found"
if [ -f /usr/local/cuda/version.txt ]; then
    cat /usr/local/cuda/version.txt
fi
echo ""

echo "--- Python ---"
which python
python --version
echo ""

echo "--- Conda ---"
which conda 2>/dev/null && conda --version || echo "  [WARN] conda not found"
conda env list 2>/dev/null || true
echo ""

echo "--- PyTorch ---"
python -c "
import torch
print(f'  PyTorch version: {torch.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  CUDA version: {torch.version.cuda}')
    print(f'  GPU count: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')
        mem = torch.cuda.get_device_properties(i).total_mem / 1024**3
        print(f'    Memory: {mem:.1f} GB')
else:
    print(f'  CUDA built version: {torch.version.cuda}')
    print(f'  [INFO] 无卡模式，挂卡后 CUDA 即可用')
" 2>/dev/null || echo "  [WARN] PyTorch not installed"
echo ""

echo "--- MMAction2 ---"
python -c "
import mmaction
print(f'  mmaction version: {mmaction.__version__}')
" 2>/dev/null || echo "  [WARN] mmaction2 not installed"

python -c "
import mmengine
print(f'  mmengine version: {mmengine.__version__}')
" 2>/dev/null || echo "  [WARN] mmengine not installed"

python -c "
import mmcv
print(f'  mmcv version: {mmcv.__version__}')
" 2>/dev/null || echo "  [WARN] mmcv not installed"
echo ""

echo "--- decord (视频解码) ---"
python -c "import decord; print(f'  decord version: {decord.__version__}')" 2>/dev/null || echo "  [WARN] decord not installed"
echo ""

echo "--- 磁盘空间 ---"
df -h / | tail -1
df -h /root 2>/dev/null | tail -1
echo ""

echo "--- 关键目录检查 ---"
echo "  /root/mmaction2: $([ -d /root/mmaction2 ] && echo 'EXISTS' || echo 'NOT FOUND')"
echo "  /root/data/egtea: $([ -d /root/data/egtea ] && echo 'EXISTS' || echo 'NOT FOUND')"
echo "  /root/data/egtea/videos/cropped_clips: $([ -d /root/data/egtea/videos/cropped_clips ] && echo 'EXISTS' || echo 'NOT FOUND')"
echo "  /root/data/egtea/action_annotation: $([ -d /root/data/egtea/action_annotation ] && echo 'EXISTS' || echo 'NOT FOUND')"
echo "  /root/outputs: $([ -d /root/outputs ] && echo 'EXISTS' || echo 'NOT FOUND')"
echo "  /root/checkpoints: $([ -d /root/checkpoints ] && echo 'EXISTS' || echo 'NOT FOUND')"
echo ""

echo "============================================================"
echo "检查完成。请根据上述输出判断是否需要安装/配置环境。"
echo "============================================================"
