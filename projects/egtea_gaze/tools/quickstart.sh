#!/bin/bash
# ============================================================
# EGTEA Gaze+ 云端一键启动脚本
#
# 整合环境配置 + 数据整理 + 权重下载 + 烟雾测试。
# 支持无卡环境（先配置，后挂卡训练）。
#
# 前提:
#   1. 优云智算实例 (有 conda)
#   2. MMAction2 仓库在 /root/mmaction2
#   3. 数据已上传到 /root/data/egtea/raw/
#
# 使用方法:
#   cd /root/mmaction2
#   bash projects/egtea_gaze/tools/quickstart.sh
#
# 数据盘模式:
#   DATA_ROOT=/cloud/egtea bash projects/egtea_gaze/tools/quickstart.sh
#
# 指定 CUDA 版本 (无卡时):
#   CUDA_TARGET=11.8 bash projects/egtea_gaze/tools/quickstart.sh
# ============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
REPO_DIR="$(dirname "$(dirname "$PROJECT_DIR")")"
DATA_DIR="${DATA_ROOT:-/root/data/egtea}"
RAW_DIR="$DATA_DIR/raw"
OUTPUT_DIR="${OUTPUT_ROOT:-/root/outputs/egtea_gaze}"
CKPT_DIR="${CKPT_ROOT:-/root/checkpoints}"

cd "$REPO_DIR"

echo ""
echo "################################################################"
echo "#   EGTEA Gaze+ 云端一键启动                                   #"
echo "################################################################"
echo ""
echo "仓库: $REPO_DIR"
echo "数据: $DATA_DIR"
echo "输出: $OUTPUT_DIR"
echo ""

# ============================================================
# Phase 1: 环境配置
# ============================================================
echo "================================================================"
echo "  Phase 1: 环境配置"
echo "================================================================"

echo ""
echo "[1.1] 检测 GPU..."
HAS_GPU=0
CUDA_VER=""
if command -v nvidia-smi &> /dev/null; then
    GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true)
    if [ -n "$GPU_INFO" ]; then
        HAS_GPU=1
        echo "  GPU: $GPU_INFO"
        CUDA_VER=$(nvidia-smi 2>/dev/null | grep -oP "CUDA Version: \K[0-9]+\.[0-9]+" || true)
        echo "  CUDA: $CUDA_VER"
    else
        echo "  [INFO] 无卡模式"
    fi
else
    echo "  [INFO] 无卡模式"
fi

if [ -n "$CUDA_TARGET" ]; then
    CUDA_VER="$CUDA_TARGET"
    echo "  使用指定 CUDA: $CUDA_VER"
elif [ -z "$CUDA_VER" ] && command -v nvcc &> /dev/null; then
    CUDA_VER=$(nvcc --version | grep -oP "release \K[0-9]+\.[0-9]+" || true)
fi
if [ -z "$CUDA_VER" ]; then
    CUDA_VER="11.8"
    echo "  [INFO] 默认 CUDA $CUDA_VER"
fi

echo ""
echo "[1.2] 配置 conda 环境..."
eval "$(conda shell.bash hook)" 2>/dev/null || true

if conda env list | grep -q "^egtea "; then
    echo "  环境 egtea 已存在，激活"
    conda activate egtea
else
    echo "  创建 egtea (python=3.9)..."
    conda create -n egtea python=3.9 -y
    conda activate egtea
fi
echo "  Python: $(python --version) @ $(which python)"

echo ""
echo "[1.3] 检查/安装 PyTorch..."
PYTORCH_OK=0
python -c "import torch" 2>/dev/null && PYTORCH_OK=1

if [ $PYTORCH_OK -eq 1 ]; then
    TORCH_VER=$(python -c "import torch; print(torch.__version__)")
    echo "  [OK] PyTorch $TORCH_VER 已安装"
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
    python -c "import torch; print(f'  [OK] PyTorch {torch.__version__}, CUDA built: {torch.version.cuda}')"
fi

echo ""
echo "[1.4] 检查/安装 MMAction2..."
MMACTION_OK=0
python -c "import mmaction" 2>/dev/null && MMACTION_OK=1

if [ $MMACTION_OK -eq 1 ]; then
    echo "  [OK] MMAction2 已安装"
else
    pip install -U openmim -q
    mim install mmengine -q
    mim install "mmcv>=2.0.0" -q
    cd "$REPO_DIR"
    pip install -v -e . 2>&1 | tail -3
fi

echo ""
echo "[1.5] 额外依赖..."
python -c "import decord" 2>/dev/null || pip install decord -q
python -c "import scipy" 2>/dev/null || pip install scipy -q
python -c "import tensorboard" 2>/dev/null || pip install tensorboard -q
echo "  [OK]"

chmod +x "$SCRIPT_DIR"/*.sh 2>/dev/null || true

echo ""
echo "[1.6] 验证..."
python -c "
import torch, mmaction, mmengine, mmcv, decord
print(f'  PyTorch: {torch.__version__} (CUDA built: {torch.version.cuda})')
print(f'  CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU: {torch.cuda.get_device_name(0)}')
else:
    print(f'  [INFO] 无卡模式，挂卡后即可训练')
print(f'  mmaction={mmaction.__version__} mmengine={mmengine.__version__} mmcv={mmcv.__version__} decord={decord.__version__}')
"
echo "  [OK] Phase 1 完成"

# ============================================================
# Phase 2: 数据整理
# ============================================================
echo ""
echo "================================================================"
echo "  Phase 2: 数据整理"
echo "================================================================"

mkdir -p "$RAW_DIR" "$DATA_DIR/videos" "$DATA_DIR/action_annotation" "$DATA_DIR/gaze_data"
mkdir -p "$OUTPUT_DIR" "$CKPT_DIR" /root/logs

echo ""
echo "[2.1] 检查数据文件..."
MISSING_DATA=0
for f in video_clips.tar action_annotation.zip; do
    if [ -f "$RAW_DIR/$f" ]; then
        echo "  [OK] $f ($(du -h "$RAW_DIR/$f" | cut -f1))"
    else
        echo "  [MISSING] $f"
        MISSING_DATA=1
    fi
done

if [ $MISSING_DATA -eq 1 ]; then
    echo ""
    echo "  [STOP] 数据文件缺失，请上传到: $RAW_DIR/"
    echo "  上传后重新运行本脚本。"
    exit 1
fi

echo ""
echo "[2.2] 解压视频..."
if [ -d "$DATA_DIR/videos/cropped_clips" ] && [ "$(find "$DATA_DIR/videos/cropped_clips" -name '*.mp4' | head -1)" ]; then
    echo "  [OK] 已解压 ($(find "$DATA_DIR/videos/cropped_clips" -name '*.mp4' | wc -l) clips)"
else
    echo "  解压中..."
    tar -xf "$RAW_DIR/video_clips.tar" -C "$DATA_DIR/videos/"
    if [ ! -d "$DATA_DIR/videos/cropped_clips" ]; then
        FOUND=$(find "$DATA_DIR/videos" -type d -name "cropped_clips" | head -1)
        if [ -n "$FOUND" ]; then
            mv "$FOUND" "$DATA_DIR/videos/cropped_clips"
        fi
    fi
    echo "  [OK] $(find "$DATA_DIR/videos/cropped_clips" -name '*.mp4' | wc -l) clips"
fi

echo ""
echo "[2.3] 解压标注..."
if [ -f "$DATA_DIR/action_annotation/action_idx.txt" ]; then
    echo "  [OK] 已解压"
else
    unzip -qo "$RAW_DIR/action_annotation.zip" -d "$DATA_DIR/action_annotation/"
    echo "  [OK]"
fi

if [ -f "$RAW_DIR/gaze_data.zip" ]; then
    echo ""
    echo "[2.4] 解压 Gaze..."
    if [ "$(ls -A $DATA_DIR/gaze_data 2>/dev/null)" ]; then
        echo "  [OK] 已解压"
    else
        unzip -qo "$RAW_DIR/gaze_data.zip" -d "$DATA_DIR/gaze_data/"
        echo "  [OK]"
    fi
fi

echo ""
echo "[2.5] 标注格式..."
if [ -f "$DATA_DIR/action_annotation/train.txt" ] && [ -f "$DATA_DIR/action_annotation/val.txt" ] && [ -f "$DATA_DIR/action_annotation/test.txt" ]; then
    echo "  [OK] train/val/test.txt 已存在"
else
    echo "  生成中..."
    cd "$REPO_DIR"
    python projects/egtea_gaze/tools/convert_annotations.py \
        --ann-dir "$DATA_DIR/action_annotation" \
        --split 1 \
        --video-dir "$DATA_DIR/videos" \
        --output-dir "$DATA_DIR/action_annotation" \
        --val-ratio 0.1
fi

echo ""
echo "[2.6] 数据验证..."
cd "$REPO_DIR"
python projects/egtea_gaze/tools/check_data.py \
    --data-root "$DATA_DIR/videos" \
    --ann-dir "$DATA_DIR/action_annotation"

echo "  [OK] Phase 2 完成"

# ============================================================
# Phase 3: 预训练权重
# ============================================================
echo ""
echo "================================================================"
echo "  Phase 3: 预训练权重"
echo "================================================================"

SLOWFAST_FILE="$CKPT_DIR/slowfast_r50_8xb8-4x16x1-256e_kinetics400-rgb_20220901-701b0f6f.pth"
RESNET50_CACHE="$HOME/.cache/torch/hub/checkpoints/resnet50-0676ba61.pth"

echo ""
echo "[3.1] SlowFast-R50..."
if [ -f "$SLOWFAST_FILE" ]; then
    echo "  [OK] 已存在"
else
    wget -q --show-progress -O "$SLOWFAST_FILE" \
        "https://download.openmmlab.com/mmaction/v1.0/recognition/slowfast/slowfast_r50_8xb8-4x16x1-256e_kinetics400-rgb/slowfast_r50_8xb8-4x16x1-256e_kinetics400-rgb_20220901-701b0f6f.pth"
    echo "  [OK]"
fi

echo ""
echo "[3.2] ResNet50 ImageNet..."
mkdir -p "$(dirname "$RESNET50_CACHE")"
if [ -f "$RESNET50_CACHE" ]; then
    echo "  [OK] 已存在"
else
    wget -q --show-progress -O "$RESNET50_CACHE" \
        "https://download.pytorch.org/models/resnet50-0676ba61.pth"
    echo "  [OK]"
fi

echo "  [OK] Phase 3 完成"

# ============================================================
# Phase 4: 烟雾测试
# ============================================================
echo ""
echo "================================================================"
echo "  Phase 4: 烟雾测试"
echo "================================================================"

cd "$REPO_DIR"

echo ""
echo "[4.1] TSM-R50..."
python projects/egtea_gaze/tools/smoke_test.py \
    --config projects/egtea_gaze/configs/tsm_r50_egtea.py

echo ""
echo "[4.2] SlowFast-R50..."
python projects/egtea_gaze/tools/smoke_test.py \
    --config projects/egtea_gaze/configs/slowfast_r50_egtea.py

# ============================================================
echo ""
echo "################################################################"
echo "#   ALL DONE                                                   #"
echo "################################################################"
echo ""
if [ $HAS_GPU -eq 1 ]; then
    echo "训练命令:"
    echo "  python tools/train.py projects/egtea_gaze/configs/tsm_r50_egtea.py --work-dir /root/outputs/egtea_gaze/tsm_r50"
    echo "  python tools/train.py projects/egtea_gaze/configs/slowfast_r50_egtea.py --work-dir /root/outputs/egtea_gaze/slowfast_r50"
else
    echo "[INFO] 无卡环境，全部准备工作已完成。"
    echo "挂载 GPU 后直接执行:"
    echo "  conda activate egtea"
    echo "  cd /root/mmaction2"
    echo "  python tools/train.py projects/egtea_gaze/configs/tsm_r50_egtea.py --work-dir /root/outputs/egtea_gaze/tsm_r50"
fi
echo ""
