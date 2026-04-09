# EGTEA RGB-Only TSN Baseline

## 1. What is already verified

This repository already completed the following EGTEA RGB-only baseline steps locally:

- official split1 converted to MMAction2 `VideoDataset` format
- TSN RGB baseline config created
- `print_config.py` passed
- dataset browsing and video decoding passed
- 1 epoch training finished
- validation finished
- checkpoints were generated

Verified local validation result after 1 epoch:

- `acc/top1 = 0.3586`
- `acc/top5 = 0.6318`
- `acc/mean1 = 0.1696`

Generated checkpoints:

- `work_dirs/egtea_tsn_split1/epoch_1.pth`
- `work_dirs/egtea_tsn_split1/best_acc_top1_epoch_1.pth`

This means the main baseline path is already healthy:

- data paths are correct
- labels are correct
- annotation format is correct
- model config is correct
- training works
- validation works

The unfinished item is only the final full `test.txt` evaluation result.

## 2. Files that matter for cloud training

Core files to keep:

- `configs/recognition/tsn/tsn_r50_egtea_rgb_split1.py`
- `tools/data/egtea/make_egtea_splits.py`
- `tools/cloud/setup_egtea_cloud.sh`
- `tools/cloud/train_egtea_tsn.sh`
- `tools/cloud/test_egtea_tsn.sh`
- `requirements/egtea_cloud.txt`

Data to upload separately:

- `data/egtea/videos/`
- `data/egtea/annotations/`

You do not need the following local-only helper files on cloud:

- `sitecustomize.py`
- `tools/local/run_mmaction_tool.py`
- `.tmp/`
- `.tmp_python/`

## 3. Top-level requirements.txt is not enough

Top-level `requirements.txt` only contains:

```txt
-r requirements/build.txt
-r requirements/optional.txt
-r requirements/tests.txt
```

It is not a full GPU runtime lock file.

For cloud training, the important compatibility set is:

- Python
- PyTorch
- CUDA wheel
- mmengine
- mmcv-lite

So cloud deployment should use the dedicated setup script in this repo instead of only `pip install -r requirements.txt`.

## 4. Recommended cloud machine

Recommended:

- Ubuntu 20.04 or 22.04
- 1 GPU with 16 GB VRAM or more
- 8 vCPU or more
- 16 GB RAM or more
- 80 GB to 120 GB disk

## 5. One-command environment setup

From repo root on cloud:

```bash
bash tools/cloud/setup_egtea_cloud.sh
```

This script will:

- create a Python virtual environment at `.venv_egtea`
- install pinned runtime packages
- install this repo in editable mode
- check that `mmaction`, `mmengine`, `mmcv`, and `torch` import correctly

## 6. One-command full training

After setup:

```bash
bash tools/cloud/train_egtea_tsn.sh
```

This runs the full 10-epoch baseline with cloud-friendly dataloader settings:

- `train num_workers=8`
- `val/test num_workers=8`
- `val/test batch_size=8`
- `persistent_workers=True`

Work dir:

- `work_dirs/egtea_tsn_split1_full`

## 7. One-command final test

After training:

```bash
bash tools/cloud/test_egtea_tsn.sh
```

This will automatically use:

- `work_dirs/egtea_tsn_split1_full/best_acc_top1_epoch_*.pth`

and run final test with faster dataloader settings.

## 8. Expected cloud runtime

Conservative estimate for a stable Linux single-GPU run:

- full 10-epoch training: `4 to 6 hours`
- final test: `10 to 20 minutes`

To avoid local-like timeout problems:

- do not wrap test with a 1-hour hard timeout
- run inside `tmux` or `screen`
- keep logs on disk

## 9. Current split statistics

- classes: `106`
- train: `7468`
- val: `831`
- test: `2022`

## 10. Current annotation files

Generated MMAction2 annotation files:

- `data/egtea/annotations/train.txt`
- `data/egtea/annotations/val.txt`
- `data/egtea/annotations/test.txt`
- `data/egtea/annotations/label_mapping_generated.txt`

## 11. Upload checklist

Push these code files to Gitee:

- `EGTEA_TSN_BASELINE_README.md`
- `configs/recognition/tsn/tsn_r50_egtea_rgb_split1.py`
- `tools/data/egtea/make_egtea_splits.py`
- `tools/cloud/setup_egtea_cloud.sh`
- `tools/cloud/train_egtea_tsn.sh`
- `tools/cloud/test_egtea_tsn.sh`
- `requirements/egtea_cloud.txt`

Do not push local temporary artifacts:

- `.tmp/`
- `.tmp_python/`
- `.torch_cache/`
- `work_dirs/`

Do not push `data/` if you plan to upload dataset separately on cloud.
