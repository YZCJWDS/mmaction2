#!/bin/bash
# ============================================================
# EGTEA Gaze+ 项目 - 云端一键环境配置脚本
#
# 功能: 安装 PyTorch + MMAction2 全套依赖
# 支持无卡环境配置（先装环境，后挂卡训练）
#
# 使用方法 (JupyterLab 终端):
#   cd /root/mmaction2
#   bash projects/egtea_gaze/tools/setup_env.sh
#
# 指定 CUDA 版本 (无卡时无法自动检测):
#   CUDA_TARGET=11.8 bash projects/egtea_gaze/tools/setup_env.sh
#   CUDA_TARGET=12.1 bash projects/egtea_gaze/tools/setup_env.sh
# ============================================================

set -e

echo "============================================================"
echo "  EGTEA Gaze+ 云端一键环境配置"
echo "============================================================"
echo ""

# ---- Step 0: 检测基础环境 ----
echo "[0/6] 检测基础环境..."

# 检测 GPU (非强制)
HAS_GPU=0
CUDA_VER=""
if command -v nvidia-smi &> /dev/null; then
    GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true)
    if [ -n "$GPU_INFO" ]; then
        HAS_GPU=1
        echo "  GPU: $GPU_INFO"
        CUDA_VER=$(nvidia-smi 2>/dev/null | grep -oP "CUDA Version: \K[0-9]+\.[0-9]+" || true)
        echo "  CUDA Driver: $CUDA_VER"
    else
        echo "  [INFO] nvidia-smi 存在但无 GPU 设备（无卡模式）"
    fi
else
    echo "  [INFO] 无 GPU / nvidia-smi 不可用（无卡模式）"
fi

# 如果用户通过环境变量指定了 CUDA 版本，优先使用
if [ -n "$CUDA_TARGET" ]; then
    CUDA_VER="$CUDA_TARGET"
    echo "  使用用户指定 CUDA 版本: $CUDA_VER"
fi

# 如果仍然没有 CUDA 版本，尝试从 nvcc 获取
if [ -z "$CUDA_VER" ]; then
    if command -v nvcc &> /dev/null; then
        CUDA_VER=$(nvcc --version | grep -oP "release \K[0-9]+\.[0-9]+" || true)
        echo "  从 nvcc 检测 CUDA: $CUDA_VER"
    fi
fi

# 最终 fallback: 默认 11.8
if [ -z "$CUDA_VER" ]; then
    CUDA_VER="11.8"
    echo "  [WARN] 无法检测 CUDA 版本，默认使用 $CUDA_VER"
    echo "  如需指定: CUDA_TARGET=12.1 bash setup_env.sh"
fi

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

# 检查是否已有 PyTorch (不要求 CUDA 可用，只要能 import)
PYTORCH_OK=0
python -c "import torch; print(f'  已有 PyTorch {torch.__version__}')" 2>/dev/null && PYTORCH_OK=1

if [ $PYTORCH_OK -eq 1 ]; then
    echo "  PyTorch 已安装，跳过"
    # 额外检查 CUDA 支持
    python -c "import torch; print(f'  CUDA available: {torch.cuda.is_available()}')" 2>/dev/null || true
else
    CUDA_MAJOR=$(echo $CUDA_VER | cut -d. -f1)

    if [ "$CUDA_MAJOR" -ge 12 ]; then
        echo "  安装 PyTorch (CUDA 12.1)..."
        pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121
    else
        echo "  安装 PyTorch (CUDA 11.8)..."
        pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118
    fi

    # 验证安装 (不要求 CUDA 可用)
    python -c "import torch; print(f'  [OK] PyTorch {torch.__version__}, CUDA built: {torch.version.cuda}')"
fi
echo ""

# ---- Step 3: 安装 MMAction2 生态 ----
echo "[3/6] 安装 MMEngine + MMCV + MMAction2..."

MMACTION_OK=0
python -c "import mmaction" 2>/dev/null && MMACTION_OK=1

if [ $MMACTION_OK -eq 1 ]; then
    python -c "import mmaction; print(f'  已有 mmaction {mmaction.__version__}')"
    echo "  跳过安装"
else
    pip install -U openmim
    mim install mmengine
    mim install "mmcv>=2.0.0"

    # 安装 mmaction2 (开发模式)
    cd /root/mmaction2
    pip install -v -e .

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

python -c "import decord" 2>/dev/null || pip install decord -q
python -c "import scipy" 2>/dev/null || pip install scipy -q
python -c "import tensorboard" 2>/dev/null || pip install tensorboard -q

python -c "import decord; print(f'  [OK] decord={decord.__version__}')"
echo ""

# ---- Step 5: 创建目录结构 ----
echo "[5/6] 创建项目目录结构..."

mkdir -p /root/data/egtea/raw
mkdir -p /root/data/egtea/videos
mkdir -p /root/data/egtea/action_annotation
mkdir -p /root/data/egtea/gaze_data
mkdir -p /root/outputs/egtea_gaze
mkdir -p /root/checkpoints
mkdir -p /root/logs

echo "  /root/data/egtea/raw               <- 上传压缩包到这里"
echo "  /root/data/egtea/videos            <- 视频解压目标"
echo "  /root/data/egtea/action_annotation <- 标注文件"
echo "  /root/outputs/egtea_gaze           <- 训练输出"
echo "  /root/checkpoints                  <- 预训练权重"
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
print(f'  PyTorch:  {torch.__version__} (CUDA built: {torch.version.cuda})')
print(f'  CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU: {torch.cuda.get_device_name(0)}')
    print(f'  显存: {torch.cuda.get_device_properties(0).total_mem/1024**3:.1f} GB')
else:
    print(f'  [INFO] 当前无 GPU，挂卡后即可训练')
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
if [ $HAS_GPU -eq 0 ]; then
    echo ""
    echo "  [提醒] 当前为无卡环境，环境已配置完成。"
    echo "  挂载 GPU 后可直接开始训练，无需重新配置。"
fi
echo "============================================================"
