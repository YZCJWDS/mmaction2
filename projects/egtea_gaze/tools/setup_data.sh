#!/bin/bash
# ============================================================
# EGTEA Gaze+ 项目 - 数据解压与整理脚本
#
# 前提: 三份数据文件已上传到 $DATA_ROOT/raw/
#   - video_clips.tar
#   - gaze_data.zip
#   - action_annotation.zip
#
# 使用方法:
#   cd /root/mmaction2
#   bash projects/egtea_gaze/tools/setup_data.sh
#
# 支持数据盘模式:
#   DATA_ROOT=/cloud/egtea bash projects/egtea_gaze/tools/setup_data.sh
# ============================================================

set -e

# 路径变量化：默认系统盘，可通过环境变量切换到数据盘
DATA_ROOT="${DATA_ROOT:-/root/data/egtea}"
RAW_DIR="$DATA_ROOT/raw"

echo "============================================================"
echo "EGTEA Gaze+ 数据解压与整理"
echo "============================================================"
echo "DATA_ROOT: $DATA_ROOT"
echo ""

# ---- 创建目录 ----
mkdir -p "$DATA_ROOT/videos"
mkdir -p "$DATA_ROOT/action_annotation"
mkdir -p "$DATA_ROOT/gaze_data"

# ---- 检查原始文件 ----
echo "--- 检查原始文件 ---"
MISSING=0
for f in video_clips.tar action_annotation.zip; do
    if [ -f "$RAW_DIR/$f" ]; then
        SIZE=$(du -h "$RAW_DIR/$f" | cut -f1)
        echo "  [OK] $f ($SIZE)"
    else
        echo "  [MISSING] $f"
        MISSING=1
    fi
done

if [ -f "$RAW_DIR/gaze_data.zip" ]; then
    SIZE=$(du -h "$RAW_DIR/gaze_data.zip" | cut -f1)
    echo "  [OK] gaze_data.zip ($SIZE)"
else
    echo "  [SKIP] gaze_data.zip (baseline 阶段可选)"
fi

if [ $MISSING -eq 1 ]; then
    echo ""
    echo "  [STOP] 必需文件缺失，请上传到 $RAW_DIR/"
    exit 1
fi

# ---- Step 1: 解压视频 ----
echo ""
echo "--- Step 1: 解压视频 ---"
if [ -d "$DATA_ROOT/videos/cropped_clips" ] && [ "$(find "$DATA_ROOT/videos/cropped_clips" -name '*.mp4' | head -1)" ]; then
    CLIP_COUNT=$(find "$DATA_ROOT/videos/cropped_clips" -name "*.mp4" | wc -l)
    echo "  [OK] 已解压 ($CLIP_COUNT 个片段)，跳过"
else
    echo "  解压 video_clips.tar (可能需要几分钟)..."
    tar -xf "$RAW_DIR/video_clips.tar" -C "$DATA_ROOT/videos/"

    # 处理可能的嵌套目录
    if [ ! -d "$DATA_ROOT/videos/cropped_clips" ]; then
        FOUND=$(find "$DATA_ROOT/videos" -type d -name "cropped_clips" | head -1)
        if [ -n "$FOUND" ] && [ "$FOUND" != "$DATA_ROOT/videos/cropped_clips" ]; then
            echo "  调整嵌套目录..."
            mv "$FOUND" "$DATA_ROOT/videos/cropped_clips"
        fi
    fi

    if [ -d "$DATA_ROOT/videos/cropped_clips" ]; then
        CLIP_COUNT=$(find "$DATA_ROOT/videos/cropped_clips" -name "*.mp4" | wc -l)
        echo "  [OK] 解压完成: $CLIP_COUNT 个视频片段"
    else
        echo "  [ERROR] 未找到 cropped_clips/，请手动检查:"
        ls "$DATA_ROOT/videos/" | head -10
        exit 1
    fi
fi

# ---- Step 2: 解压标注 ----
echo ""
echo "--- Step 2: 解压标注 ---"
if [ -f "$DATA_ROOT/action_annotation/action_idx.txt" ] || [ -f "$DATA_ROOT/action_annotation/train_split1.txt" ]; then
    echo "  [OK] 标注已解压，跳过"
else
    unzip -qo "$RAW_DIR/action_annotation.zip" -d "$DATA_ROOT/action_annotation/"
    echo "  [OK] 解压完成"
fi
echo "  内容:"
ls "$DATA_ROOT/action_annotation/" 2>/dev/null | head -10

# ---- Step 3: 解压 Gaze 数据 (可选) ----
echo ""
echo "--- Step 3: 解压 Gaze 数据 ---"
if [ -f "$RAW_DIR/gaze_data.zip" ]; then
    if [ "$(ls -A "$DATA_ROOT/gaze_data" 2>/dev/null)" ]; then
        echo "  [OK] 已解压，跳过"
    else
        unzip -qo "$RAW_DIR/gaze_data.zip" -d "$DATA_ROOT/gaze_data/"
        echo "  [OK] 解压完成"
    fi
else
    echo "  [SKIP] gaze_data.zip 不存在"
fi

# ---- Step 4: 生成 MMAction2 格式标注 ----
echo ""
echo "--- Step 4: 检查/生成 MMAction2 格式标注 ---"

ANN_DIR="$DATA_ROOT/action_annotation"

if [ -f "$ANN_DIR/train.txt" ] && [ -f "$ANN_DIR/val.txt" ] && [ -f "$ANN_DIR/test.txt" ]; then
    echo "  [OK] train.txt ($(wc -l < "$ANN_DIR/train.txt") 行)"
    echo "  [OK] val.txt ($(wc -l < "$ANN_DIR/val.txt") 行)"
    echo "  [OK] test.txt ($(wc -l < "$ANN_DIR/test.txt") 行)"
else
    echo "  生成 MMAction2 格式标注..."
    REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../" && pwd)"
    cd "$REPO_DIR"
    python projects/egtea_gaze/tools/convert_annotations.py \
        --ann-dir "$ANN_DIR" \
        --split 1 \
        --video-dir "$DATA_ROOT/videos" \
        --output-dir "$ANN_DIR" \
        --val-ratio 0.1
fi

# ---- Step 5: 验证 ----
echo ""
echo "--- Step 5: 数据完整性验证 ---"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../" && pwd)"
cd "$REPO_DIR"
python projects/egtea_gaze/tools/check_data.py \
    --data-root "$DATA_ROOT/videos" \
    --ann-dir "$DATA_ROOT/action_annotation"

echo ""
echo "============================================================"
echo "数据整理完成！"
echo ""
echo "目录结构:"
echo "  $DATA_ROOT/"
echo "  ├── raw/                  (原始压缩包)"
echo "  ├── videos/cropped_clips/ (视频片段)"
echo "  ├── action_annotation/    (标注 + train/val/test.txt)"
echo "  └── gaze_data/            (Gaze 数据)"
echo ""
echo "下一步:"
echo "  bash projects/egtea_gaze/tools/download_pretrained.sh"
echo "  python projects/egtea_gaze/tools/smoke_test.py --config projects/egtea_gaze/configs/tsm_r50_egtea.py"
echo "============================================================"
