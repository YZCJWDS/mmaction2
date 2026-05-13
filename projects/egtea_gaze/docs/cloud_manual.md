# EGTEA Gaze+ 云端训练操作手册

## 一句话说明

上传数据后，一条命令完成全部准备：

```bash
cd /root/mmaction2
bash projects/egtea_gaze/tools/quickstart.sh
```

---

## 前提条件

| 条件 | 说明 |
|------|------|
| 云端实例 | 优云智算 GPU 实例，已启动 |
| GPU | 至少 16GB 显存（推荐 24GB） |
| 系统 | Linux + CUDA 驱动 + conda |
| 仓库 | MMAction2 已 clone 到 `/root/mmaction2` |
| 数据 | 三份文件已上传到 `/root/data/egtea/raw/` |

---

## 数据上传

在运行 quickstart 之前，先把数据传上去：

```bash
mkdir -p /root/data/egtea/raw

# 方式 A: scp 上传
scp -P <port> video_clips.tar root@<host>:/root/data/egtea/raw/
scp -P <port> action_annotation.zip root@<host>:/root/data/egtea/raw/
scp -P <port> gaze_data.zip root@<host>:/root/data/egtea/raw/

# 方式 B: JupyterLab 文件浏览器拖拽上传
# 方式 C: 网盘直链 wget
```

---

## 一键启动 (quickstart.sh)

```bash
cd /root/mmaction2
bash projects/egtea_gaze/tools/quickstart.sh
```

数据盘模式（挂载数据盘后）：
```bash
DATA_ROOT=/cloud/egtea bash projects/egtea_gaze/tools/quickstart.sh
```

脚本自动执行以下全部步骤（已完成的会跳过）：

| Phase | 步骤 | 做什么 |
|-------|------|--------|
| 1.1 | 检测 GPU | nvidia-smi 确认 GPU 和 CUDA 版本 |
| 1.2 | conda 环境 | 创建/激活 `egtea` 环境 (python=3.9) |
| 1.3 | PyTorch | 根据 CUDA 版本自动选 cu118/cu121 安装 |
| 1.4 | MMAction2 | openmim + mmengine + mmcv + pip install -e . |
| 1.5 | 额外依赖 | decord / scipy / tensorboard |
| 1.6 | 验证 | 打印所有版本号确认 |
| 2.1 | 检查数据 | 确认压缩包存在 |
| 2.2 | 解压视频 | tar -xf video_clips.tar，自动处理嵌套目录 |
| 2.3 | 解压标注 | unzip action_annotation.zip |
| 2.4 | 解压 Gaze | unzip gaze_data.zip（可选） |
| 2.5 | 标注转换 | 生成 train.txt / val.txt / test.txt |
| 2.6 | 数据验证 | check_data.py 确认视频与标注匹配 |
| 3.1 | SlowFast 权重 | 下载 Kinetics400 预训练 |
| 3.2 | ResNet50 权重 | 下载 ImageNet 预训练到 torchvision 缓存 |
| 4.1 | TSM 烟雾测试 | config + dataset + model + forward |
| 4.2 | SlowFast 烟雾测试 | 同上 |

全部通过后打印训练命令。

---

## 开始训练

```bash
cd /root/mmaction2
conda activate egtea

# TSM-R50 全量训练 (50 epochs, 约 4-6 小时)
python tools/train.py projects/egtea_gaze/configs/tsm_r50_egtea.py \
    --work-dir /root/outputs/egtea_gaze/tsm_r50

# SlowFast-R50 全量训练 (50 epochs, 约 8-12 小时)
python tools/train.py projects/egtea_gaze/configs/slowfast_r50_egtea.py \
    --work-dir /root/outputs/egtea_gaze/slowfast_r50
```

---

## 训练参数速查

| 参数 | TSM-R50 | SlowFast-R50 |
|------|---------|--------------|
| batch_size | 16 | 8 |
| num_workers | 4 | 4 |
| lr | 0.005 | 0.0075 |
| scheduler | MultiStep [20,40] | Warmup 5ep + Cosine |
| epochs | 50 | 50 |
| 帧采样 | 8×1 (8帧) | 32×2 (32帧,间隔2) |
| 预训练 | ImageNet (自动) | Kinetics400 (load_from) |
| 预计显存 | ~8-10 GB | ~14-18 GB |

---

## 显存不足时

```bash
# 降低 batch_size（不改配置文件）
python tools/train.py projects/egtea_gaze/configs/slowfast_r50_egtea.py \
    --work-dir /root/outputs/egtea_gaze/slowfast_r50 \
    --cfg-options train_dataloader.batch_size=4 val_dataloader.batch_size=4
```

---

## 断点续训

```bash
python tools/train.py projects/egtea_gaze/configs/tsm_r50_egtea.py \
    --work-dir /root/outputs/egtea_gaze/tsm_r50 \
    --resume
```

---

## 测试

```bash
python tools/test.py projects/egtea_gaze/configs/tsm_r50_egtea.py \
    /root/outputs/egtea_gaze/tsm_r50/best_acc_top1_epoch_*.pth

python tools/test.py projects/egtea_gaze/configs/slowfast_r50_egtea.py \
    /root/outputs/egtea_gaze/slowfast_r50/best_acc_top1_epoch_*.pth
```

---

## 查看训练进度

```bash
# 实时日志
tail -f /root/outputs/egtea_gaze/tsm_r50/*/vis_data/scalars.json

# TensorBoard
tensorboard --logdir /root/outputs/egtea_gaze --port 6006 --bind_all
```

---

## 关机前

```bash
# 打包结果
cd /root/outputs
tar -czf egtea_gaze_results.tar.gz egtea_gaze/

# 检查清单:
# □ best checkpoint 存在
# □ 日志已保存
# □ 如需续训，work_dir 中有 latest.pth
```

---

## 分步执行（不用 quickstart）

```bash
cd /root/mmaction2

# 1. 环境
bash projects/egtea_gaze/tools/setup_env.sh

# 2. 数据
bash projects/egtea_gaze/tools/setup_data.sh

# 3. 权重
bash projects/egtea_gaze/tools/download_pretrained.sh

# 4. 验证
python projects/egtea_gaze/tools/check_data.py
python projects/egtea_gaze/tools/smoke_test.py --config projects/egtea_gaze/configs/tsm_r50_egtea.py
python projects/egtea_gaze/tools/smoke_test.py --config projects/egtea_gaze/configs/slowfast_r50_egtea.py
```

---

## 目录结构

```
/root/
├── mmaction2/
│   └── projects/egtea_gaze/
│       ├── configs/
│       │   ├── tsm_r50_egtea.py
│       │   └── slowfast_r50_egtea.py
│       ├── tools/
│       │   ├── quickstart.sh          ← 一键启动（推荐）
│       │   ├── setup_env.sh           ← 单独: 环境配置
│       │   ├── setup_data.sh          ← 单独: 数据解压
│       │   ├── download_pretrained.sh ← 单独: 权重下载
│       │   ├── check_env.sh           ← 单独: 环境检查
│       │   ├── check_data.py          ← 单独: 数据验证
│       │   ├── convert_annotations.py ← 单独: 标注转换
│       │   └── smoke_test.py          ← 单独: 烟雾测试
│       ├── docs/
│       │   └── cloud_manual.md        ← 本手册
│       └── egtea_gaze/
│           └── __init__.py            ← Python 包（后续 gaze 模型扩展）
├── data/egtea/
│   ├── raw/                           ← 上传压缩包到这里
│   ├── videos/cropped_clips/          ← 视频片段
│   ├── action_annotation/             ← 标注 (train/val/test.txt)
│   └── gaze_data/                     ← Gaze 数据（第二阶段用）
├── outputs/egtea_gaze/                ← 训练输出
│   ├── tsm_r50/
│   └── slowfast_r50/
└── checkpoints/                       ← 预训练权重
```

---

## 常见问题

**Q: quickstart.sh 中途失败了？**
修复问题后重新运行，已完成的步骤自动跳过。

**Q: 新开终端后 import mmaction 报错？**
```bash
conda activate egtea
```

**Q: decord 报错？**
```bash
pip uninstall decord -y && pip install decord
```

**Q: CUDA out of memory？**
```bash
--cfg-options train_dataloader.batch_size=4 optim_wrapper.accumulative_counts=2
```

**Q: 标注路径不匹配？**
```bash
head -1 /root/data/egtea/action_annotation/train.txt
# 期望: cropped_clips/xxx/xxx.mp4 0
ls /root/data/egtea/videos/cropped_clips/ | head -3
```
