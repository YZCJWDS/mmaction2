# EGTEA Gaze+ 云端训练操作手册

## 平台与镜像

- 平台：优云智算（容器实例）
- 镜像：PyTorch 基础镜像
- 数据传输：文件管理 → 夸克网盘
- 仓库路径：`/root/code/mmaction2`
- 数据路径：`/root/data/egtea`（系统盘）或 `DATA_ROOT=/cloud/egtea`（数据盘）

---

## 两条部署流程

### 流程 A：无卡模式（配置环境 + 整理数据）

适用于：容器实例未挂 GPU，先把环境和数据准备好。

```bash
cd /root/code/mmaction2

# 1. 检查当前环境状态
bash projects/egtea_gaze/tools/check_env.sh

# 2. 安装缺失依赖（复用当前 torch，不重新下载）
bash projects/egtea_gaze/tools/setup_env.sh

# 3. 上传数据到 /root/data/egtea/raw/ (通过文件管理/夸克网盘)

# 4. 解压数据 + 标注转换 + 验证
bash projects/egtea_gaze/tools/quickstart.sh --mode data-only
```

完成后关机等待挂卡。

### 流程 B：带 GPU 模式（完整流程）

适用于：已挂 GPU，从头到尾一次跑通。

```bash
cd /root/code/mmaction2

# 一键全流程（环境 + 数据 + 权重 + smoke test）
bash projects/egtea_gaze/tools/quickstart.sh --mode full
```

或者分步：

```bash
# 环境（如果 A 阶段已做，会自动跳过已安装的）
bash projects/egtea_gaze/tools/setup_env.sh

# 数据（如果 A 阶段已做，会自动跳过已解压的）
bash projects/egtea_gaze/tools/quickstart.sh --mode data-only

# 权重
bash projects/egtea_gaze/tools/download_pretrained.sh

# 烟雾测试
python projects/egtea_gaze/tools/smoke_test.py --config projects/egtea_gaze/configs/tsm_r50_egtea.py
python projects/egtea_gaze/tools/smoke_test.py --config projects/egtea_gaze/configs/slowfast_r50_egtea.py

# 训练
python tools/train.py projects/egtea_gaze/configs/tsm_r50_egtea.py --work-dir /root/outputs/egtea_gaze/tsm_r50
```

---

## setup_env.sh 行为说明

| 场景 | 行为 |
|------|------|
| 当前环境有 torch（如 py312 + torch 2.3） | 复用，只装缺失的 mmengine/mmcv/mmaction/decord |
| 当前环境无 torch + 有 GPU | 创建 egtea 环境，安装 torch + 全套 |
| 当前环境无 torch + 无 GPU | 拒绝执行，提示切换到有 torch 的环境 |
| FORCE_NEW_ENV=1 | 强制创建 egtea 环境 |

---

## quickstart.sh 模式

| 模式 | 做什么 | 适用场景 |
|------|--------|---------|
| `--mode data-only` | 解压视频/标注/gaze + 标注转换 + 数据验证 | 无卡模式 |
| `--mode env-only` | 调用 setup_env.sh | 只装环境 |
| `--mode full` | env + data + 权重下载 + smoke test | 带 GPU 完整部署 |
| 无参数 | 无卡自动 data-only，有卡自动 full | 默认 |

---

## 训练命令

```bash
cd /root/code/mmaction2

# TSM-R50 (约 4-6 小时)
python tools/train.py projects/egtea_gaze/configs/tsm_r50_egtea.py \
    --work-dir /root/outputs/egtea_gaze/tsm_r50

# SlowFast-R50 (约 8-12 小时)
python tools/train.py projects/egtea_gaze/configs/slowfast_r50_egtea.py \
    --work-dir /root/outputs/egtea_gaze/slowfast_r50

# 断点续训
python tools/train.py <config> --work-dir <dir> --resume

# 显存不足
--cfg-options train_dataloader.batch_size=4
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
| 帧采样 | 8×1 | 32×2 |
| 预训练 | ImageNet (torchvision) | Kinetics400 (load_from) |
| 预计显存 | ~8-10 GB | ~14-18 GB |

---

## 测试

```bash
python tools/test.py projects/egtea_gaze/configs/tsm_r50_egtea.py \
    /root/outputs/egtea_gaze/tsm_r50/best_acc_top1_epoch_*.pth
```

---

## 关机前

```bash
cd /root/outputs
tar -czf egtea_gaze_results.tar.gz egtea_gaze/
# □ best checkpoint 存在
# □ 如需续训，work_dir 中有 latest.pth
```

---

## 目录结构

```
/root/code/mmaction2/                    # 仓库
  └── projects/egtea_gaze/
      ├── configs/                       # 训练配置
      ├── tools/                         # 脚本
      ├── docs/cloud_manual.md           # 本手册
      └── egtea_gaze/                    # Python 包

/root/data/egtea/                        # DATA_ROOT
  ├── raw/                               # 原始压缩包
  ├── videos/cropped_clips/              # 视频
  ├── action_annotation/                 # 标注
  └── gaze_data/                         # Gaze

/root/outputs/egtea_gaze/                # 训练输出
/root/checkpoints/                       # 预训练权重
```

---

## 常见问题

**Q: setup_env.sh 报 "当前环境无 torch" 怎么办？**
切换到镜像自带的 torch 环境：`conda activate py312`，再运行。

**Q: mmcv 安装卡住？**
可能在源码编译。Ctrl+C 中断后手动指定预编译版本：
```bash
pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.3/index.html
```

**Q: 无卡模式下能做什么？**
环境安装 + 数据解压 + 标注转换。不能做 smoke test 的 forward pass 和训练。

**Q: 数据盘模式？**
```bash
DATA_ROOT=/cloud/egtea bash projects/egtea_gaze/tools/quickstart.sh --mode data-only
```
配置文件中的路径需要手动改为 `/cloud/egtea/...`，或训练时用 `--cfg-options` 覆盖。
