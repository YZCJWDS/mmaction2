# EGTEA Gaze+ 第一人称视频行为分类

## 项目概述

基于深度学习的第一人称视频行为分类技术研究。

- 数据集：EGTEA Gaze+ (106 类动作, 10321 视频片段)
- Baseline 模型：TSM-R50, SlowFast-R50
- 改进模型：Gaze-supervised SlowFast（第二阶段实现）

## 数据集统计

| Split | Samples |
|-------|---------|
| Train | 7468    |
| Val   | 831     |
| Test  | 2022    |

## 云端目录结构

```
/root/data/egtea/                         # DATA_ROOT (系统盘)
├── raw/                                  # 原始压缩包
├── videos/cropped_clips/                 # 视频片段
├── action_annotation/                    # 标注 + train/val/test.txt
└── gaze_data/                            # Gaze 数据（第二阶段用）

/root/outputs/egtea_gaze/                 # 训练输出
/root/checkpoints/                        # 预训练权重
```

数据盘模式：`DATA_ROOT=/cloud/egtea`

## 快速开始

```bash
cd /root/mmaction2
bash projects/egtea_gaze/tools/quickstart.sh
```

详见 `docs/cloud_manual.md`

## 训练命令

```bash
cd /root/mmaction2
conda activate egtea

# TSM-R50 baseline
python tools/train.py projects/egtea_gaze/configs/tsm_r50_egtea.py \
    --work-dir /root/outputs/egtea_gaze/tsm_r50

# SlowFast-R50 baseline
python tools/train.py projects/egtea_gaze/configs/slowfast_r50_egtea.py \
    --work-dir /root/outputs/egtea_gaze/slowfast_r50
```

## 测试命令

```bash
python tools/test.py projects/egtea_gaze/configs/tsm_r50_egtea.py \
    /root/outputs/egtea_gaze/tsm_r50/best_acc_top1_epoch_*.pth
```
