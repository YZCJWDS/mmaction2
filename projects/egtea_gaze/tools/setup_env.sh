#!/bin/bash
# ============================================================
# EGTEA Gaze+ 项目 - 云端一键环境配置脚本
#
# 功能: 自动检测 CUDA 版本，安装 PyTorch + MMAction2 全套依赖
# 前提: 云端镜像已有 CUDA 驱动 + conda
#
# 使用方法 (JupyterLab 终端):
#   cd /root/mmaction2
#   bash projects/egtea_gaze/tools/setup_env.sh
#
# 如果镜像已有 PyTorch，脚本会检测并跳过重复安装
# ============================================================

set -e

echo "============================================================"
echo "  EGTEA Gaze+ 云端一键环境配置"
echo "============================================================"
echo ""

# ---- Step 0: 检测基础环境 ----
echo "[0/6] 检测基础环境..."

# 检测 GPU
if ! command -v nvidia-smi &> /dev/null; then
    echo "  [ERROR] nvidia-smi 不可用，请确认 GPU 实例已启动"
    exit 1
fi
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader | head -1)
echo "  GPU: $GPU_NAME ($GPU_MEM)"

# 检测 CUDA 版本 (从 nvidia-smi 获取驱动支持的最高 CUDA 版本)
CUDA_VER=$(nvidia-smi | grep -oP "CUDA Version: \K[0-9]+\.[0-9]+")
echo "  CUDA Driver Version: $CUDA_VER"

# 检测 conda
if ! command -v conda &> /dev/null; then
    echo "  [ERROR] conda 不可用"
    echo "  请先安装 miniconda:"
    echo "    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
    echo "    bash Miniconda3-latest-Linux-x86_64.sh -b -p /root/miniconda3"
    echo "    eval \"\$(/root/miniconda3/bin/conda shell.bash hook)\""
    exit 1
fi
echo "  conda: $(conda --version)"
echo ""

# ---- Step 1: 创建/激活 conda 环境 ----
echo "[1/6] 配置 conda 环境 (egtea)..."

# conda init for current shell
eval "$(conda shell.bash hook)" 2>/dev/null || true

if conda env list | grep -q "^egtea "; then
    echo "  环境 egtea 已存在，直接激活"
    conda activate egtea
else
    echo "  创建新环境 egtea (python=3.9)..."
    conda create -n egtea python=3.9 -y
    conda activate egtea
fi
echo "  Python: $(python --version)"
echo "  路径: $(which python)"
echo ""

# ---- Step 2: 安装 PyTorch ----
echo "[2/6] 安装 PyTorch..."

# 检查是否已有可用的 PyTorch
PYTORCH_OK=0
python -c "import torch; assert torch.cuda.is_available(); print(f'  已有 PyTorch {torch.__version__}, CUDA={torch.version.cuda}')" 2>/dev/null && PYTORCH_OK=1

if [ $PYTORCH_OK -eq 1 ]; then
    echo "  PyTorch 已可用，跳过安装"
else
    echo "  检测到 CUDA $CUDA_VER，选择对应 PyTorch..."

    # 根据 CUDA 版本选择安装命令
    CUDA_MAJOR=$(echo $CUDA_VER | cut -d. -f1)
    CUDA_MINOR=$(echo $CUDA_VER | cut -d. -f2)

    if [ "$CUDA_MAJOR" -ge 12 ]; then
        echo "  使用 CUDA 12.1 版本 PyTorch"
        pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121
    elif [ "$CUDA_MAJOR" -eq 11 ] && [ "$CUDA_MINOR" -ge 8 ]; then
        echo "  使用 CUDA 11.8 版本 PyTorch"
        pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118
    else
        echo "  [WARN] CUDA $CUDA_VER 较旧，尝试 CUDA 11.8 版本"
        pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118
    fi

    # 验证
    python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'; print(f'  [OK] PyTorch {torch.__version__}, CUDA={torch.version.cuda}')"
fi
echo ""

# ---- Step 3: 安装 MMAction2 生态 ----
echo "[3/6] 安装 MMEngine + MMCV + MMAction2..."

# 检查是否已安装
MMACTION_OK=0
python -c "import mmaction; print(f'  已有 mmaction {mmaction.__version__}')" 2>/dev/null && MMACTION_OK=1

if [ $MMACTION_OK -eq 1 ]; then
    echo "  MMAction2 已安装，跳过"
else
    pip install -U openmim
    mim install mmengine
    mim install "mmcv>=2.0.0"

    # 安装 mmaction2 (开发模式)
    cd /root/mmaction2
    pip install -v -e .

    # 验证
    python -c "
import mmaction, mmengine, mmcv
print(f'  [OK] mmaction={mmaction.__version__}')
print(f'  [OK] mmengine={mmengine.__version__}')
print(f'  [OK] mmcv={mmcv.__version__}')
"
fi
echo ""

# ---- Step 4: 安装额外依赖 ----
echo "[4/6] 安装额外依赖 (decord, scipy, tensorboard)..."

pip install decord scipy tensorboard -q

python -c "import decord; print(f'  [OK] decord={decord.__version__}')"
echo ""

# ---- Step 5: 创建目录结构 ----
echo "[5/6] 创建项目目录结构..."

mkdir -p /root/data/egtea/raw
mkdir -p /root/data/egtea/videos
mkdir -p /root/data/egtea/annotations
mkdir -p /root/data/egtea/gaze_data
mkdir -p /root/outputs/egtea_gaze
mkdir -p /root/checkpoints
mkdir -p /root/logs

echo "  /root/data/egtea/raw          <- 上传压缩包到这里"
echo "  /root/data/egtea/videos       <- 视频解压目标"
echo "  /root/data/egtea/annotations  <- 标注文件"
echo "  /root/outputs/egtea_gaze      <- 训练输出"
echo "  /root/checkpoints             <- 预训练权重"
echo ""

# ---- Step 6: 赋予脚本执行权限 ----
echo "[6/6] 设置脚本权限..."

cd /root/mmaction2
chmod +x projects/egtea_gaze/tools/*.sh
echo "  [OK] 所有 .sh 脚本已设置执行权限"
echo ""

# ---- 最终验证 ----
echo "============================================================"
echo "  环境配置完成！最终验证:"
echo "============================================================"
python -c "
import torch, mmaction, mmengine, mmcv, decord
print(f'  PyTorch:  {torch.__version__} (CUDA={torch.version.cuda})')
print(f'  GPU:      {torch.cuda.get_device_name(0)}')
print(f'  显存:     {torch.cuda.get_device_properties(0).total_mem/1024**3:.1f} GB')
print(f'  mmaction: {mmaction.__version__}')
print(f'  mmengine: {mmengine.__version__}')
print(f'  mmcv:     {mmcv.__version__}')
print(f'  decord:   {decord.__version__}')
print()
print('  ✓ 环境就绪')
"

echo ""
echo "============================================================"
echo "  下一步操作:"
echo "  1. 上传数据到 /root/data/egtea/raw/"
echo "  2. bash projects/egtea_gaze/tools/setup_data.sh"
echo "  3. python projects/egtea_gaze/tools/check_data.py"
echo "  4. bash projects/egtea_gaze/tools/download_pretrained.sh"
echo "  5. python projects/egtea_gaze/tools/smoke_test.py --config projects/egtea_gaze/configs/tsm_r50_egtea.py"
echo "============================================================"
