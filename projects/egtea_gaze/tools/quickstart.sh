#!/bin/bash
# ============================================================
# EGTEA Gaze+ 云端一键启动脚本
#
# 本脚本整合了从环境配置到烟雾测试的全部流程。
# 前提条件:
#   1. 优云智算 GPU 实例已启动 (有 CUDA + conda)
#   2. MMAction2 仓库已在 /root/mmaction2
#   3. 三份数据文件已上传到 /root/data/egtea/raw/
#      - video_clips.tar
#      - gaze_data.zip (可选，baseline 阶段不需要)
#      - action_annotation.zip
#
# 使用方法 (JupyterLab 终端):
#   cd /root/mmaction2
#   bash projects/egtea_gaze/tools/quickstart.sh
#
# 如果某一步已完成，脚本会自动跳过。
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
echo "#                                                              #"
echo "#   EGTEA Gaze+ 云端一键启动                                   #"
echo "#   基于深度学习的第一人称视频行为分类                            #"
echo "#                                                              #"
echo "################################################################"
echo ""
echo "仓库路径: $REPO_DIR"
echo "数据路径: $DATA_DIR"
echo "输出路径: $OUTPUT_DIR"
echo ""

# ============================================================
# Phase 1: 环境配置
# ============================================================
echo "================================================================"
echo "  Phase 1: 环境配置"
echo "================================================================"

# ---- 检测 GPU ----
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
        echo "  [INFO] 无卡模式（nvidia-smi 存在但无设备）"
    fi
else
    echo "  [INFO] 无卡模式（nvidia-smi 不可用）"
fi

# CUDA 版本 fallback
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

# ---- Conda 环境 ----
echo ""
echo "[1.2] 配置 conda 环境..."
eval "$(conda shell.bash hook)" 2>/dev/null || true

if conda env list | grep -q "^egtea "; then
    echo "  环境 egtea 已存在，激活"
    conda activate egtea
else
    echo "  创建新环境 egtea (python=3.9)..."
    conda create -n egtea python=3.9 -y
    conda activate egtea
fi
echo "  Python: $(python --version) @ $(which python)"

# ---- PyTorch ----
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

# ---- MMAction2 生态 ----
echo ""
echo "[1.4] 检查/安装 MMAction2..."
MMACTION_OK=0
python -c "import mmaction" 2>/dev/null && MMACTION_OK=1

if [ $MMACTION_OK -eq 1 ]; then
    echo "  [OK] MMAction2 已安装"
else
    echo "  安装 openmim + mmengine + mmcv..."
    pip install -U openmim -q
    mim install mmengine -q
    mim install "mmcv>=2.0.0" -q
    echo "  安装 mmaction2 (开发模式)..."
    cd "$REPO_DIR"
    pip install -v -e . 2>&1 | tail -3
fi

# ---- 额外依赖 ----
echo ""
echo "[1.5] 检查/安装 decord + scipy + tensorboard..."
python -c "import decord" 2>/dev/null || pip install decord -q
python -c "import scipy" 2>/dev/null || pip install scipy -q
python -c "import tensorboard" 2>/dev/null || pip install tensorboard -q
echo "  [OK] 额外依赖就绪"

# ---- 脚本权限 ----
chmod +x "$SCRIPT_DIR"/*.sh 2>/dev/null || true

# ---- 环境验证 ----
echo ""
echo "[1.6] 环境最终验证..."
python -c "
import torch, mmaction, mmengine, mmcv, decord
print(f'  PyTorch:  {torch.__version__} (CUDA built: {torch.version.cuda})')
print(f'  CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU: {torch.cuda.get_device_name(0)}')
else:
    print(f'  [INFO] 无卡模式，挂卡后即可训练')
print(f'  mmaction: {mmaction.__version__} | mmengine: {mmengine.__version__} | mmcv: {mmcv.__version__}')
print(f'  decord:   {decord.__version__}')
"
echo "  [OK] Phase 1 完成"

# ============================================================
# Phase 2: 数据整理
# ============================================================
echo ""
echo "================================================================"
echo "  Phase 2: 数据整理"
echo "================================================================"

# ---- 创建目录 ----
mkdir -p "$RAW_DIR" "$DATA_DIR/videos" "$DATA_DIR/action_annotation" "$DATA_DIR/gaze_data"
mkdir -p "$OUTPUT_DIR" "$CKPT_DIR" /root/logs

# ---- 检查原始文件 ----
echo ""
echo "[2.1] 检查原始数据文件..."
MISSING_DATA=0
for f in video_clips.tar action_annotation.zip; do
    if [ -f "$RAW_DIR/$f" ]; then
        SIZE=$(du -h "$RAW_DIR/$f" | cut -f1)
        echo "  [OK] $f ($SIZE)"
    else
        echo "  [MISSING] $f"
        MISSING_DATA=1
    fi
done

if [ -f "$RAW_DIR/gaze_data.zip" ]; then
    SIZE=$(du -h "$RAW_DIR/gaze_data.zip" | cut -f1)
    echo "  [OK] gaze_data.zip ($SIZE)"
else
    echo "  [SKIP] gaze_data.zip (baseline 阶段可选)"
fi

if [ $MISSING_DATA -eq 1 ]; then
    echo ""
    echo "  [STOP] 必需的数据文件缺失！"
    echo "  请上传到: $RAW_DIR/"
    echo "  上传后重新运行本脚本即可继续。"
    exit 1
fi

# ---- 解压视频 ----
echo ""
echo "[2.2] 解压视频..."
if [ -d "$DATA_DIR/videos/cropped_clips" ] && [ "$(find "$DATA_DIR/videos/cropped_clips" -name '*.mp4' | head -1)" ]; then
    CLIP_COUNT=$(find "$DATA_DIR/videos/cropped_clips" -name "*.mp4" | wc -l)
    echo "  [OK] 视频已解压 ($CLIP_COUNT 个片段)"
else
    echo "  解压 video_clips.tar (可能需要几分钟)..."
    tar -xf "$RAW_DIR/video_clips.tar" -C "$DATA_DIR/videos/"

    # 处理可能的嵌套目录
    if [ ! -d "$DATA_DIR/videos/cropped_clips" ]; then
        FOUND=$(find "$DATA_DIR/videos" -type d -name "cropped_clips" | head -1)
        if [ -n "$FOUND" ] && [ "$FOUND" != "$DATA_DIR/videos/cropped_clips" ]; then
            echo "  调整目录结构..."
            mv "$FOUND" "$DATA_DIR/videos/cropped_clips"
        fi
    fi

    if [ -d "$DATA_DIR/videos/cropped_clips" ]; then
        CLIP_COUNT=$(find "$DATA_DIR/videos/cropped_clips" -name "*.mp4" | wc -l)
        echo "  [OK] 解压完成: $CLIP_COUNT 个视频片段"
    else
        echo "  [ERROR] 解压后未找到 cropped_clips 目录"
        echo "  请手动检查: ls $DATA_DIR/videos/"
        exit 1
    fi
fi

# ---- 解压标注 ----
echo ""
echo "[2.3] 解压标注..."
if [ -f "$DATA_DIR/action_annotation/action_idx.txt" ]; then
    echo "  [OK] 标注已解压"
else
    unzip -qo "$RAW_DIR/action_annotation.zip" -d "$DATA_DIR/action_annotation/"
    echo "  [OK] 标注解压完成"
fi

# ---- 解压 Gaze 数据 (可选) ----
if [ -f "$RAW_DIR/gaze_data.zip" ]; then
    echo ""
    echo "[2.4] 解压 Gaze 数据..."
    if [ "$(ls -A $DATA_DIR/gaze_data 2>/dev/null)" ]; then
        echo "  [OK] Gaze 数据已解压"
    else
        unzip -qo "$RAW_DIR/gaze_data.zip" -d "$DATA_DIR/gaze_data/"
        echo "  [OK] Gaze 数据解压完成"
    fi
fi

# ---- 生成/检查 MMAction2 格式标注 ----
echo ""
echo "[2.5] 检查 MMAction2 格式标注..."
if [ -f "$DATA_DIR/action_annotation/train.txt" ] && [ -f "$DATA_DIR/action_annotation/val.txt" ] && [ -f "$DATA_DIR/action_annotation/test.txt" ]; then
    TRAIN_N=$(wc -l < "$DATA_DIR/action_annotation/train.txt")
    VAL_N=$(wc -l < "$DATA_DIR/action_annotation/val.txt")
    TEST_N=$(wc -l < "$DATA_DIR/action_annotation/test.txt")
    echo "  [OK] train.txt ($TRAIN_N), val.txt ($VAL_N), test.txt ($TEST_N)"
else
    echo "  生成 MMAction2 格式标注 (split 1)..."
    cd "$REPO_DIR"
    python projects/egtea_gaze/tools/convert_annotations.py \
        --ann-dir "$DATA_DIR/action_annotation" \
        --split 1 \
        --video-dir "$DATA_DIR/videos" \
        --output-dir "$DATA_DIR/action_annotation" \
        --val-ratio 0.1
fi

# ---- 数据完整性验证 ----
echo ""
echo "[2.6] 数据完整性验证..."
cd "$REPO_DIR"
python projects/egtea_gaze/tools/check_data.py \
    --data-root "$DATA_DIR/videos" \
    --ann-dir "$DATA_DIR/action_annotation"

echo ""
echo "  [OK] Phase 2 完成"

# ============================================================
# Phase 3: 预训练权重
# ============================================================
echo ""
echo "================================================================"
echo "  Phase 3: 下载预训练权重"
echo "================================================================"

SLOWFAST_FILE="$CKPT_DIR/slowfast_r50_8xb8-4x16x1-256e_kinetics400-rgb_20220901-701b0f6f.pth"
RESNET50_CACHE="$HOME/.cache/torch/hub/checkpoints/resnet50-0676ba61.pth"

echo ""
echo "[3.1] SlowFast-R50 Kinetics400 权重..."
if [ -f "$SLOWFAST_FILE" ]; then
    echo "  [OK] 已存在 ($(du -h "$SLOWFAST_FILE" | cut -f1))"
else
    echo "  下载中..."
    wget -q --show-progress -O "$SLOWFAST_FILE" \
        "https://download.openmmlab.com/mmaction/v1.0/recognition/slowfast/slowfast_r50_8xb8-4x16x1-256e_kinetics400-rgb/slowfast_r50_8xb8-4x16x1-256e_kinetics400-rgb_20220901-701b0f6f.pth"
    echo "  [OK] 下载完成"
fi

echo ""
echo "[3.2] ResNet50 ImageNet 权重 (TSM backbone)..."
mkdir -p "$(dirname "$RESNET50_CACHE")"
if [ -f "$RESNET50_CACHE" ]; then
    echo "  [OK] 已存在"
else
    echo "  下载中..."
    wget -q --show-progress -O "$RESNET50_CACHE" \
        "https://download.pytorch.org/models/resnet50-0676ba61.pth"
    echo "  [OK] 下载完成"
fi

echo ""
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
echo "[4.1] TSM-R50 烟雾测试..."
python projects/egtea_gaze/tools/smoke_test.py \
    --config projects/egtea_gaze/configs/tsm_r50_egtea.py

echo ""
echo "[4.2] SlowFast-R50 烟雾测试..."
python projects/egtea_gaze/tools/smoke_test.py \
    --config projects/egtea_gaze/configs/slowfast_r50_egtea.py

# ============================================================
# 完成
# ============================================================
echo ""
echo ""
echo "################################################################"
echo "#                                                              #"
echo "#   ALL DONE - 可以开始训练                                     #"
echo "#                                                              #"
echo "################################################################"
echo ""
echo "训练命令:"
echo ""
echo "  # TSM-R50 (约 4-6 小时)"
echo "  python tools/train.py projects/egtea_gaze/configs/tsm_r50_egtea.py \\"
echo "      --work-dir /root/outputs/egtea_gaze/tsm_r50"
echo ""
echo "  # SlowFast-R50 (约 8-12 小时)"
echo "  python tools/train.py projects/egtea_gaze/configs/slowfast_r50_egtea.py \\"
echo "      --work-dir /root/outputs/egtea_gaze/slowfast_r50"
echo ""
echo "  # 断点续训"
echo "  python tools/train.py <config> --work-dir <dir> --resume"
echo ""
echo "================================================================"
