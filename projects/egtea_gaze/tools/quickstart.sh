#!/bin/bash
# ============================================================
# EGTEA Gaze+ 云端快速启动
#
# Usage:
#   cd /root/code/mmaction2
#   bash projects/egtea_gaze/tools/quickstart.sh --mode data-only
#   bash projects/egtea_gaze/tools/quickstart.sh --mode env-only
#   bash projects/egtea_gaze/tools/quickstart.sh --mode full
#
# 默认行为:
#   无卡模式 -> data-only
#   有卡模式 -> full
#
# 环境变量:
#   DATA_ROOT=/cloud/egtea   数据盘模式
#   REPO_DIR=/root/code/mmaction2  仓库路径
# ============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${REPO_DIR:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
DATA_DIR="${DATA_ROOT:-/root/data/egtea}"
RAW_DIR="$DATA_DIR/raw"
OUTPUT_DIR="${OUTPUT_ROOT:-/root/outputs/egtea_gaze}"
CKPT_DIR="${CKPT_ROOT:-/root/checkpoints}"

# ---- 解析参数 ----
MODE=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --mode) MODE="$2"; shift 2 ;;
        *) echo "未知参数: $1"; exit 1 ;;
    esac
done

# ---- 检测 GPU ----
HAS_GPU=0
if command -v nvidia-smi &> /dev/null && nvidia-smi --query-gpu=name --format=csv,noheader &> /dev/null; then
    HAS_GPU=1
fi

# ---- 自动选择模式 ----
if [ -z "$MODE" ]; then
    if [ $HAS_GPU -eq 1 ]; then
        MODE="full"
    else
        MODE="data-only"
        echo "[INFO] 无卡模式，自动选择 --mode data-only"
        echo "       如需安装环境: bash quickstart.sh --mode env-only"
        echo ""
    fi
fi

echo "################################################################"
echo "  EGTEA Gaze+ quickstart (mode: $MODE)"
echo "################################################################"
echo ""
echo "  仓库: $REPO_DIR"
echo "  数据: $DATA_DIR"
echo "  GPU:  $([ $HAS_GPU -eq 1 ] && echo '可用' || echo '无卡')"
echo ""

cd "$REPO_DIR"

# ============================================================
# data-only: 解压数据 + 标注转换 + 验证
# ============================================================
run_data() {
    echo "================================================================"
    echo "  数据整理"
    echo "================================================================"

    mkdir -p "$RAW_DIR" "$DATA_DIR/videos" "$DATA_DIR/action_annotation" "$DATA_DIR/gaze_data"
    mkdir -p "$OUTPUT_DIR" "$CKPT_DIR"

    echo ""
    echo "[D.1] 检查数据文件..."
    MISSING=0
    for f in video_clips.tar action_annotation.zip; do
        if [ -f "$RAW_DIR/$f" ]; then
            echo "  [OK] $f ($(du -h "$RAW_DIR/$f" | cut -f1))"
        else
            echo "  [MISSING] $f"
            MISSING=1
        fi
    done
    if [ $MISSING -eq 1 ]; then
        echo "  [STOP] 请上传到: $RAW_DIR/"
        exit 1
    fi

    echo ""
    echo "[D.2] 解压视频..."
    if [ -d "$DATA_DIR/videos/cropped_clips" ] && [ "$(find "$DATA_DIR/videos/cropped_clips" -name '*.mp4' | head -1)" ]; then
        echo "  [OK] 已解压 ($(find "$DATA_DIR/videos/cropped_clips" -name '*.mp4' | wc -l) clips)"
    else
        echo "  解压中 (可能需要几分钟)..."
        tar -xf "$RAW_DIR/video_clips.tar" -C "$DATA_DIR/videos/"
        if [ ! -d "$DATA_DIR/videos/cropped_clips" ]; then
            FOUND=$(find "$DATA_DIR/videos" -type d -name "cropped_clips" | head -1)
            [ -n "$FOUND" ] && mv "$FOUND" "$DATA_DIR/videos/cropped_clips"
        fi
        if [ -d "$DATA_DIR/videos/cropped_clips" ]; then
            echo "  [OK] $(find "$DATA_DIR/videos/cropped_clips" -name '*.mp4' | wc -l) clips"
        else
            echo "  [ERROR] 未找到 cropped_clips/"; exit 1
        fi
    fi

    echo ""
    echo "[D.3] 解压标注..."
    if [ -f "$DATA_DIR/action_annotation/action_idx.txt" ]; then
        echo "  [OK] 已解压"
    else
        unzip -qo "$RAW_DIR/action_annotation.zip" -d "$DATA_DIR/action_annotation/"
        echo "  [OK]"
    fi

    if [ -f "$RAW_DIR/gaze_data.zip" ]; then
        echo ""
        echo "[D.4] 解压 Gaze..."
        if [ "$(ls -A "$DATA_DIR/gaze_data" 2>/dev/null)" ]; then
            echo "  [OK] 已解压"
        else
            unzip -qo "$RAW_DIR/gaze_data.zip" -d "$DATA_DIR/gaze_data/"
            echo "  [OK]"
        fi
    fi

    echo ""
    echo "[D.5] 标注转换..."
    if [ -f "$DATA_DIR/action_annotation/train.txt" ] && [ -f "$DATA_DIR/action_annotation/val.txt" ] && [ -f "$DATA_DIR/action_annotation/test.txt" ]; then
        echo "  [OK] train/val/test.txt 已存在"
    else
        python projects/egtea_gaze/tools/convert_annotations.py \
            --ann-dir "$DATA_DIR/action_annotation" \
            --split 1 \
            --video-dir "$DATA_DIR/videos" \
            --output-dir "$DATA_DIR/action_annotation" \
            --val-ratio 0.1
    fi

    echo ""
    echo "[D.6] 数据验证..."
    python projects/egtea_gaze/tools/check_data.py \
        --data-root "$DATA_DIR/videos" \
        --ann-dir "$DATA_DIR/action_annotation"

    echo ""
    echo "  [OK] 数据就绪"
}

# ============================================================
# env-only: 安装依赖
# ============================================================
run_env() {
    echo "================================================================"
    echo "  环境配置"
    echo "================================================================"
    bash "$SCRIPT_DIR/setup_env.sh"
}

# ============================================================
# full 额外步骤: 权重 + smoke test
# ============================================================
run_weights() {
    echo ""
    echo "================================================================"
    echo "  预训练权重"
    echo "================================================================"

    mkdir -p "$CKPT_DIR"
    SLOWFAST_FILE="$CKPT_DIR/slowfast_r50_8xb8-4x16x1-256e_kinetics400-rgb_20220901-701b0f6f.pth"
    RESNET50_CACHE="$HOME/.cache/torch/hub/checkpoints/resnet50-0676ba61.pth"

    echo ""
    echo "[W.1] SlowFast-R50 Kinetics400..."
    if [ -f "$SLOWFAST_FILE" ]; then
        echo "  [OK] 已存在"
    else
        wget -q --show-progress -O "$SLOWFAST_FILE" \
            "https://download.openmmlab.com/mmaction/v1.0/recognition/slowfast/slowfast_r50_8xb8-4x16x1-256e_kinetics400-rgb/slowfast_r50_8xb8-4x16x1-256e_kinetics400-rgb_20220901-701b0f6f.pth"
        echo "  [OK]"
    fi

    echo ""
    echo "[W.2] ResNet50 ImageNet..."
    mkdir -p "$(dirname "$RESNET50_CACHE")"
    if [ -f "$RESNET50_CACHE" ]; then
        echo "  [OK] 已存在"
    else
        wget -q --show-progress -O "$RESNET50_CACHE" \
            "https://download.pytorch.org/models/resnet50-0676ba61.pth"
        echo "  [OK]"
    fi
}

run_smoke() {
    echo ""
    echo "================================================================"
    echo "  烟雾测试"
    echo "================================================================"

    cd "$REPO_DIR"
    echo ""
    echo "[S.1] TSM-R50..."
    python projects/egtea_gaze/tools/smoke_test.py \
        --config projects/egtea_gaze/configs/tsm_r50_egtea.py

    echo ""
    echo "[S.2] SlowFast-R50..."
    python projects/egtea_gaze/tools/smoke_test.py \
        --config projects/egtea_gaze/configs/slowfast_r50_egtea.py
}

# ============================================================
# 执行
# ============================================================
case $MODE in
    data-only)
        run_data
        ;;
    env-only)
        run_env
        ;;
    full)
        run_env
        run_data
        run_weights
        run_smoke
        echo ""
        echo "################################################################"
        echo "  ALL DONE - 可以开始训练"
        echo "################################################################"
        echo ""
        echo "  python tools/train.py projects/egtea_gaze/configs/tsm_r50_egtea.py --work-dir /root/outputs/egtea_gaze/tsm_r50"
        echo "  python tools/train.py projects/egtea_gaze/configs/slowfast_r50_egtea.py --work-dir /root/outputs/egtea_gaze/slowfast_r50"
        echo ""
        ;;
    *)
        echo "[ERROR] 未知模式: $MODE"
        echo "  支持: --mode full | --mode env-only | --mode data-only"
        exit 1
        ;;
esac
