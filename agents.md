# AGENTS.md

## 1. 项目身份与背景

你正在协助完成一个本科毕业设计项目。

课题名称：
基于深度学习的第一人称视频行为分类技术研究

英文名称：
Research on First-Person Video Action Classification Technology Based on Deep Learning

当前代码仓库基于 MMAction2，但本项目**不是单纯复现官方模型**，而是要满足任务书要求，完成一个可交付、可运行、可用于论文撰写与答辩的毕业设计项目。

任务书核心要求包括：

1. 研究并综述现有的第一人称视频行为分类方法；
2. 实现若干典型深度学习行为分类模型并进行对比分析；
3. **自主设计并实现一种适用于第一人称视频行为分类的改进模型**；
4. 在典型数据集上完成实验验证。

当前主要数据集：
- EGTEA Gaze+

可选扩展数据集：
- EPIC-Kitchens（仅在时间和资源允许时考虑）

主要开发语言：
- Python

主要技术栈：
- PyTorch
- MMAction2 / MMEngine / MMCV

---

## 2. 项目总体目标

本项目要完成一条完整的毕业设计技术路线，而不仅是运行现有代码。

当前目标分为三个层次：

### 2.1 基线复现
先在 EGTEA Gaze+ 上复现并跑通代表性基线模型：
- TSM-R50
- SlowFast-R50

### 2.2 自主设计模型
在基线模型基础上，设计并实现一个具有毕业设计创新性的改进模型，当前预定方向为：
- Gaze-supervised SlowFast
- 或 Gaze-guided SlowFast

核心思想：
- 训练阶段使用 gaze 信息作为辅助监督
- 测试阶段默认仅使用 RGB 视频进行分类

### 2.3 完整交付
项目最终应具备：
- 可运行的数据集处理脚本
- 可训练的 baseline 配置
- 可训练的改进模型配置
- 评估与测试命令
- 可视化工具（gaze / attention / Grad-CAM 或等价方案）
- 可支持论文图表与实验分析的输出

---

## 3. 什么才算“自主设计”

使用 MMAction2 作为训练框架是允许的，也是推荐的。

但是，仅仅完成以下操作并**不能**视为满足“自主设计”要求：
- clone MMAction2
- 改 `num_classes`
- 直接运行官方 config
- 只汇报 baseline 结果

本项目中的“自主设计”应主要体现在以下方面：

1. 针对 EGTEA Gaze+ 的数据适配与处理流程；
2. gaze 数据读取、对齐与预处理；
3. gaze heatmap 的生成；
4. 自定义模块、损失函数或训练监督方式；
5. 自定义配置与实验设计；
6. 消融实验与可视化分析。

当前推荐的创新方向为：

- 以 SlowFast 为主基线；
- 在训练阶段引入 gaze 引导的注意力监督；
- 在测试阶段只用 RGB 完成行为分类。

一个典型但非强制的目标损失形式可以是：

`L = L_cls + lambda * L_gaze`

其中：
- `L_cls`：动作分类损失
- `L_gaze`：模型注意力图与 gaze heatmap 的一致性约束损失
- `lambda`：权重系数

如果有更合适且更容易落地的实现方式，可以在不偏离项目目标的前提下调整。

---

## 4. 仓库修改原则

### 4.1 总原则
优先进行**最小改动、局部改动、可回退改动**。

### 4.2 强约束
除非确有必要，不要大幅修改 MMAction2 上游核心代码。

优先采用以下方式扩展项目：
- 在独立项目目录中新增代码
- 使用 MMAction2 注册机制
- 使用 `custom_imports`
- 使用 config 继承

### 4.3 推荐做法
优先将本项目的新增代码放在类似下面的目录中：

- `projects/egtea_gaze/`

而不是直接把大量自定义逻辑散落在：
- `mmaction/models/...`
- `mmaction/datasets/...`
- `tools/...`

### 4.4 尽量避免
不要进行以下操作，除非有充分理由：
- 大规模重构原仓库
- 与本项目无关的命名整理
- 一次改动很多无关文件
- 修改官方 baseline 的行为定义
- 在一个回合里同时完成多个大任务

### 4.5 允许做的事
以下内容是推荐且允许的：
- 新增项目专用配置文件
- 新增数据处理脚本
- 新增自定义 transform
- 新增自定义 loss / head / recognizer
- 新增项目 README
- 新增可视化和分析工具

---

## 5. 推荐目录结构

如无明显冲突，优先采用以下结构：

```text
projects/
  egtea_gaze/
    README.md
    configs/
    tools/
    egtea_gaze/
      __init__.py
      datasets/
        __init__.py
        transforms/
      models/
        __init__.py
        heads/
        losses/
        recognizers/
      visualization/


      ## 16. 云端训练补充规则（优云智算 / Linux）

### 16.1 云端环境默认假设
除非用户明确说明，否则默认本项目的训练环境为：
- 优云智算 GPU 实例
- Linux 环境
- 通过 JupyterLab 终端或 SSH 进行操作
- 代码仓库为 MMAction2
- 训练任务以 PyTorch / MMAction2 为主

Agent 在输出命令和路径时，默认优先按 Linux 环境组织，不再使用 Windows 的盘符路径写法。

---

### 16.2 镜像与实例选择约束
对于本项目，优先建议：
- 选择 PyTorch 基础镜像
- 仅在确有必要时选择更底层的 CUDA 基础镜像并手动补环境
- 不要为本项目选择与任务无关的 ComfyUI / SD-WebUI 类镜像

Agent 在给出部署建议时，应优先围绕：
- PyTorch
- CUDA
- Conda
- git
- Jupyter 终端
这些基础能力展开，而不是偏离到图像生成类工作流。

---

### 16.3 目录与路径约定
云端默认推荐使用如下目录约定：

- 仓库目录：`/root/mmaction2`
- 项目自定义目录：`/root/mmaction2/projects/egtea_gaze`
- 原始数据目录：`/root/data/egtea/raw`
- 解压后数据目录：`/root/data/egtea`
- 训练输出目录：`/root/outputs/egtea_gaze`
- 预训练权重目录：`/root/checkpoints`
- 日志目录：`/root/logs`

如果后续用户挂载了独立数据盘，则优先把：
- 数据集
- 训练输出
- checkpoint
迁移到数据盘，再由配置文件统一指向新的路径。

Agent 在未确认数据盘路径前，不要擅自假设特殊挂载目录。

---

### 16.4 虚拟环境规则
云端训练默认采用 Conda 虚拟环境，不建议直接污染基础环境。

推荐约定：
- 虚拟环境名称：`egtea`
- Python 版本：根据 MMAction2 兼容性决定，先检查仓库依赖后再定
- 依赖安装顺序：
  1. 激活 conda 环境
  2. 安装 PyTorch / CUDA 对应依赖
  3. 安装 MMAction2 与相关 requirements
  4. 再安装项目特有依赖

Agent 在输出环境配置方案时必须：
- 先检查当前镜像已有环境
- 再决定是复用还是新建环境
- 不允许不经检查就覆盖系统环境

---

### 16.5 云端操作方式约束
默认使用以下方式工作：
- 在实例页面进入 JupyterLab
- 在 JupyterLab 中新建终端执行命令
- 必要时使用 SSH 远程连接
- 不依赖 Windows 本地整合包直接上传运行

Agent 在提供操作步骤时，应优先给出：
- Jupyter 终端可执行命令
- Linux 路径
- 清晰的工作目录切换命令

---

### 16.6 数据与存储策略
本项目当前已知核心数据文件包括：
- `video_clips.tar`
- `gaze_data.zip`
- `action_annotation.zip`

默认策略：
- 原始压缩包先保存在 `/root/data/egtea/raw`
- 视频解压到 `/root/data/egtea/videos`
- gaze 数据解压到 `/root/data/egtea/gaze_data`
- 标注解压到 `/root/data/egtea/action_annotation`

若空间紧张：
- 优先保留解压后的必要训练数据
- 明确记录是否删除原始压缩包
- 不要在未确认可重复下载前随意删除关键数据

---

### 16.7 公共模型盘与软链接策略
如果云端平台已经挂载公共模型盘或共享模型目录：
- 优先检查是否存在可直接复用的预训练权重
- 若存在，优先使用软链接或直接引用路径
- 不要重复拷贝大型模型文件到项目目录

本项目中，公共模型盘只作为“可选加速项”，不是默认前提。
Agent 不得在未验证文件存在的情况下声称可以直接使用某个权重。

---

### 16.8 训练与计费意识
Agent 必须具备云端训练的成本意识：

- 不要默认长时间空转实例
- 训练命令应优先支持断点续训
- 日志和 checkpoint 保存路径要清楚
- 大训练任务前要先做小规模验证
- 用户完成阶段性任务后，应提醒：
  - 保存关键结果
  - 必要时制作环境说明
  - 空闲时关机实例

---

### 16.9 本项目的云端推进顺序
在云端环境中，默认按以下顺序推进：

1. 检查实例镜像、CUDA、Python、Conda、PyTorch 是否可用
2. 确认 MMAction2 仓库路径与项目目录结构
3. 整理 EGTEA Gaze+ 三份数据文件的上传、解压和目录规范
4. 建立 baseline 所需的数据转换脚本
5. 跑通 SlowFast / TSM baseline
6. 再进入 gaze 数据处理和自主模型阶段

Agent 不要跳过环境检查和数据落盘检查，直接开始改模型。

---

### 16.10 云端回答格式要求
当用户要求云端相关操作时，优先按以下格式回复：

1. 当前目标
2. 云端前置检查
3. 需要执行的命令
4. 需要确认的路径
5. 预期输出
6. 成功验证方式
7. 下一步建议

每一轮都应尽量让用户可以直接复制执行，而不是只讲概念。