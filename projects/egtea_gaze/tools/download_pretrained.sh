#!/bin/bash
# ============================================================
# 下载 baseline 所需的预训练权重
# 
# TSM-R50: 使用 torchvision 内置的 ImageNet 预训练 (自动下载)
# SlowFast-R50: 需要下载 Kinetics400 预训练权重
#
# 使用方法:
#   bash projects/egtea_gaze/tools/download_pretrained.sh
# ============================================================

set -e

CKPT_DIR="/root/checkpoints"
mkdir -p "$CKPT_DIR"

echo "============================================================"
echo "下载预训练权重"
echo "============================================================"

# SlowFast-R50 Kinetics400 预训练
SLOWFAST_URL="https://download.openmmlab.com/mmaction/v1.0/recognition/slowfast/slowfast_r50_8xb8-4x16x1-256e_kinetics400-rgb/slowfast_r50_8xb8-4x16x1-256e_kinetics400-rgb_20220901-701b0f6f.pth"
SLOWFAST_FILE="$CKPT_DIR/slowfast_r50_8xb8-4x16x1-256e_kinetics400-rgb_20220901-701b0f6f.pth"

if [ -f "$SLOWFAST_FILE" ]; then
    echo "[OK] SlowFast-R50 权重已存在"
else
    echo "下载 SlowFast-R50 Kinetics400 预训练权重..."
    wget -q --show-progress -O "$SLOWFAST_FILE" "$SLOWFAST_URL"
    echo "[OK] SlowFast-R50 权重下载完成"
fi

# TSM-R50: ImageNet pretrained ResNet50 由 torchvision 自动下载
# 但如果网络不好，可以手动预下载
TSM_RESNET_URL="https://download.pytorch.org/models/resnet50-0676ba61.pth"
TSM_RESNET_FILE="$CKPT_DIR/resnet50-0676ba61.pth"

if [ -f "$TSM_RESNET_FILE" ]; then
    echo "[OK] ResNet50 ImageNet 权重已存在"
else
    echo "下载 ResNet50 ImageNet 预训练权重 (TSM backbone)..."
    wget -q --show-progress -O "$TSM_RESNET_FILE" "$TSM_RESNET_URL"
    echo "[OK] ResNet50 权重下载完成"
    
    # 创建 torchvision 缓存目录的软链接，避免训练时重复下载
    TORCH_CACHE="$HOME/.cache/torch/hub/checkpoints"
    mkdir -p "$TORCH_CACHE"
    if [ ! -f "$TORCH_CACHE/resnet50-0676ba61.pth" ]; then
        ln -sf "$TSM_RESNET_FILE" "$TORCH_CACHE/resnet50-0676ba61.pth"
        echo "  已创建 torchvision 缓存软链接"
    fi
fi

echo ""
echo "============================================================"
echo "预训练权重准备完成！"
echo ""
ls -lh "$CKPT_DIR/"
echo "============================================================"
