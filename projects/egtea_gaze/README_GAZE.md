# EGTEA Gaze Training Pipeline

## Method

This project implements a gaze-supervised SlowFast model for EGTEA Gaze+.
During training, the model uses offline gaze supervision as an auxiliary loss.
During validation and testing, the model only consumes RGB clips and does not
require real gaze input.

## Data Flow

1. Inspect raw gaze files:
   `python projects/egtea_gaze/tools/inspect_gaze_data.py ...`
2. Manually verify alignment with overlays:
   `python projects/egtea_gaze/tools/visualize_gaze_overlay.py ...`
3. Build offline gaze caches:
   `python projects/egtea_gaze/tools/make_gaze_maps.py ...`
4. Validate gaze caches:
   `python projects/egtea_gaze/tools/check_gaze_maps.py ...`

The offline cache stores clip-aligned `gaze_xy`, `gaze_valid`, and debug
`gaze_maps`. Training-time `LoadGazeMap` applies current crop/flip metadata and
rebuilds a low-resolution target heatmap on the fly.

## Model

- Backbone: SlowFast-R50
- Classification head: standard SlowFast global pooling + FC
- Gaze branch: lightweight 1x1x1 conv projections and 1-channel attention logits
- Loss: `loss_cls + lambda * loss_gaze`

## Configs

- Formal training:
  `projects/egtea_gaze/configs/gaze_slowfast_r50_egtea.py`
- Debug training:
  `projects/egtea_gaze/configs/gaze_slowfast_r50_egtea_debug.py`
- Ablations:
  - random gaze
  - center prior
  - shuffled gaze

## 4090 Optimization

Video decoding stays on the standard MMAction2 `VideoDataset + DecordDecode`
pipeline. Performance tuning is handled through config and shell defaults:

- AMP enabled with `AmpOptimWrapper`
- default `BATCH_SIZE=16`
- default `WORKERS=16`
- default `PREFETCH=8`
- `pin_memory=True`
- `persistent_workers=True`
- `cudnn_benchmark=True`
- `OMP_NUM_THREADS=1`
- `MKL_NUM_THREADS=1`

The gaze branch avoids runtime parsing of raw gaze text files. Runtime work is
limited to cache loading, crop/flip coordinate adjustment, and 14x14 heatmap
generation.

## Commands

### Debug training

```bash
bash projects/egtea_gaze/tools/train_gaze_slowfast_debug.sh
```

### Formal training

```bash
bash projects/egtea_gaze/tools/train_gaze_slowfast.sh
```

Override example:

```bash
GPU=0 BATCH_SIZE=24 WORKERS=16 PREFETCH=8 MAX_EPOCHS=30 \
bash projects/egtea_gaze/tools/train_gaze_slowfast.sh
```

### Resume training

```bash
RESUME=1 bash projects/egtea_gaze/tools/train_gaze_slowfast.sh
```

### Test

```bash
bash projects/egtea_gaze/tools/test_gaze_slowfast.sh
```

### Ablations

```bash
bash projects/egtea_gaze/tools/run_gaze_ablation.sh
```

## Output Paths

- Gaze cache root:
  `/root/data/egtea/gaze_maps`
- Training outputs:
  `/root/outputs/egtea_gaze/gaze_slowfast_r50`
- Debug outputs:
  `/root/outputs/egtea_gaze/gaze_slowfast_debug`
- Overlay and attention visualizations:
  `/root/outputs/egtea_gaze/...`

## Troubleshooting

- Missing `metadata.json`:
  run `make_gaze_maps.py` first and check `--out-root`.
- Missing SlowFast checkpoint:
  set `SLOWFAST_CKPT=/path/to/best.pth` before training.
- `custom_imports` failure:
  export `PYTHONPATH=/root/code/mmaction2:$PYTHONPATH`.
- `loss_gaze` becomes NaN:
  inspect cache with `check_gaze_maps.py` and verify valid heatmap sums.
- `data_time` too high:
  inspect cache path layout, worker count, and `LoadGazeMap` cache hits.
- CUDA OOM:
  lower `BATCH_SIZE`, reduce `PREFETCH`, or use debug config first.
