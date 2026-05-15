# EGTEA Gaze Pipeline

## Method

This project implements a gaze-supervised SlowFast model for EGTEA Gaze+.

- Training: RGB clip + gaze supervision
- Validation / testing: RGB only
- Real gaze is not required at inference time

The current implementation keeps MMAction2 video decoding unchanged and adds the
gaze logic under `projects/egtea_gaze/`.

## Current Status

The gaze codebase now contains:

- raw BeGaze parsing
- gaze inspection and overlay debugging
- offline gaze cache generation
- cache loading in the training pipeline
- gaze loss / gaze head / recognizer glue
- training / testing / ablation configs and shell scripts

The current recommended gate before training is:

1. `inspect_gaze_data.py`
2. `visualize_gaze_overlay.py`
3. confirm most valid gaze points are visible and roughly aligned
4. `make_gaze_maps.py`
5. debug training

Do not continue to cache generation or training if overlay is still mostly
invalid.

## BeGaze Format

EGTEA gaze files are BeGaze exports with metadata lines starting with `##`.

Typical header:

```text
Time    Type    Trial    L POR X [px]    L POR Y [px]    Frame    Aux1    L Event Info
```

Current parser behavior:

- skips all `##` metadata lines
- uses the first non-empty, non-`##` line as header
- prefers header-name matching over numeric guessing

Primary column mapping:

- `Time` -> `timestamp`
- `L POR X [px]` -> `x`
- `L POR Y [px]` -> `y`
- `Frame` -> `frame_id`
- `L Event Info` -> `gaze_type`

Important details:

- `Type` is usually `SMP`; it is not the fixation / saccade label
- `Trial` is not a coordinate column
- `L Event Info` is the event source used for:
  - `Fixation -> fixation`
  - `Saccade -> saccade`
  - other / empty -> `unknown`

## Source Resolution

BeGaze metadata may contain:

- `Calibration Area`
- `Stimulus Dimension`

The parser extracts source resolution from those fields with priority:

1. `Calibration Area`
2. `Stimulus Dimension`
3. user override
4. fallback inference

For EGTEA, the common source coordinate system is:

```text
source_width = 1280
source_height = 960
```

Pixel gaze coordinates are normalized as:

```text
x_norm = x_px / source_width
y_norm = y_px / source_height
```

This normalized gaze is then used for:

- overlay visualization
- cache generation
- crop / flip adjusted training targets

## Parse Modes

The parser now has two explicit modes:

- sample parse:
  - `parse_gaze_file_sample(...)`
  - used by `inspect_gaze_data.py`
  - reads a limited number of rows for fast format inspection
- full parse:
  - `parse_gaze_file_full(...)`
  - used by `visualize_gaze_overlay.py`
  - used by `make_gaze_maps.py`
  - reads the full BeGaze table

This split is important. Overlay and cache generation must not reuse sampled
rows.

## Data Flow

1. Inspect raw gaze files
2. Visualize overlays on real RGB clips
3. Build offline gaze caches
4. Validate caches
5. Run debug training
6. Run formal training

Training-time `LoadGazeMap` does not parse raw BeGaze files. It reads offline
cache, applies crop / flip metadata, and builds the final low-resolution target
heatmap.

## Inspect Gaze Data

Script:

```bash
python projects/egtea_gaze/tools/inspect_gaze_data.py ...
```

What it reports per file:

- header
- preview rows
- inferred columns
- `source_resolution`
- `coordinate_mode`
- `x_range / y_range`
- `out_of_source_bounds_ratio`
- gaze type counts

Recommended command:

```bash
python projects/egtea_gaze/tools/inspect_gaze_data.py \
  --gaze-root /root/data/egtea/gaze_data \
  --video-root /root/data/egtea/videos/cropped_clips \
  --ann-root /root/data/egtea/action_annotation \
  --focus-file /root/data/egtea/gaze_data/gaze_data/gaze_data/OP01-R01-PastaSalad.txt \
  --max-files 20 \
  --output /root/outputs/egtea_gaze/gaze_inspect_report.json
```

## Overlay Debugging

Script:

```bash
python projects/egtea_gaze/tools/visualize_gaze_overlay.py ...
```

Overlay behavior:

- uses full parse, not sampled parse
- parses clip frame range from names such as:
  - `F011147-F011334`
  - `F11147-F11334`
- scales gaze from source coordinates into the displayed image
- draws a large red filled point with white outline and crosshair

Important parameters:

- `--gaze-source-width`
- `--gaze-source-height`
- `--auto-scale-gaze`

Default behavior:

- `auto_scale_gaze=True`
- uses metadata-derived source resolution when present

Overlay `summary.json` now includes:

- `frames_with_valid_gaze`
- `frames_with_visible_gaze`
- `frames_out_of_bounds`
- `frames_invalid`
- `frames_fixation`
- `frames_saccade`
- `gaze_source_resolution`
- `frames_scaled_into_view`
- `frames_out_of_source_bounds`
- `coordinate_scale_mode`
- `invalid_reason_counts`
- `clip_debug`

`clip_debug` is the main lookup diagnostic. For each sampled clip it records:

- `video_name`
- `parsed_start_frame`
- `parsed_end_frame`
- `parse_source`
- `num_gaze_rows_total`
- `num_gaze_frames_total`
- `gaze_frame_min`
- `gaze_frame_max`
- sampled per-frame debug:
  - `clip_idx`
  - `computed_global_frame`
  - `matched_frame`
  - `rows_found`
  - `valid`
  - `visible`
  - `invalid_reason`

Recommended command:

```bash
python projects/egtea_gaze/tools/visualize_gaze_overlay.py \
  --data-root /root/data/egtea \
  --video-root /root/data/egtea/videos/cropped_clips \
  --gaze-root /root/data/egtea/gaze_data \
  --ann-file /root/data/egtea/action_annotation/train.txt \
  --out-dir /root/outputs/egtea_gaze/gaze_overlay_debug \
  --num-clips 30 \
  --frames-per-clip 5 \
  --seed 42
```

Only continue when:

- `frames_with_valid_gaze > 0`
- `frames_with_visible_gaze > 0`
- most visible points fall in plausible hand / object regions

## Gaze Cache Generation

Script:

```bash
python projects/egtea_gaze/tools/make_gaze_maps.py ...
```

### Split Input Modes

There are two supported input modes.

### A. Baseline mode

Recommended default. This keeps gaze training aligned with the SlowFast
baseline split.

Input:

- `--ann-root /root/data/egtea/action_annotation`

Reads:

- `train.txt`
- `val.txt`
- `test.txt`

Writes:

- `/root/data/egtea/gaze_maps/train/`
- `/root/data/egtea/gaze_maps/val/`
- `/root/data/egtea/gaze_maps/test/`

### B. Processed mode

Extension mode, not the default.

Input:

- `--processed-root /root/data/egtea/processed --splits 1 2 3`

Reads examples:

- `train_s1.txt`
- `val_s1.txt`
- `test_s1.txt`
- `train_s2.txt`
- `...`

If both `--ann-root` and `--processed-root` are passed, the script now raises
an error to avoid ambiguity.

### Cache Behavior

Current cache generation behavior:

- full parse of gaze rows
- source resolution aware normalization
- stores normalized `gaze_xy`
- stores `gaze_valid`
- stores debug `gaze_maps`
- supports `--only-fixation`
- supports `--out-of-source-policy {clip, invalid}`

Key parameters:

- `--ann-root`
- `--processed-root`
- `--gaze-source-width`
- `--gaze-source-height`
- `--out-of-source-policy`

Recommended baseline-mode command:

```bash
python projects/egtea_gaze/tools/make_gaze_maps.py \
  --data-root /root/data/egtea \
  --video-root /root/data/egtea/videos/cropped_clips \
  --gaze-root /root/data/egtea/gaze_data \
  --ann-root /root/data/egtea/action_annotation \
  --out-root /root/data/egtea/gaze_maps \
  --heatmap-size 14 14 \
  --sigma 1.5 \
  --only-fixation \
  --num-workers 8 \
  --overwrite
```

`metadata.json` now records:

- `split_source`
- `ann_root`
- `processed_root`
- `split_files`
- `splits`
- `num_samples`
- per-sample metadata

## Cache Checking

Script:

```bash
python projects/egtea_gaze/tools/check_gaze_maps.py ...
```

Use this after cache generation to verify:

- file existence
- shapes
- finite values
- valid / invalid consistency
- heatmap normalization

## Model

- Backbone: SlowFast-R50
- Classification branch: standard SlowFast global pooling + FC
- Gaze branch: lightweight 1x1x1 conv projections + 1-channel attention logits
- Loss: `loss_cls + lambda * loss_gaze`

Main components:

- `GazeKLLoss`
- `GazeSlowFastHead`
- `LoadGazeMap`
- thin gaze recognizer glue

## Configs

- formal:
  - `projects/egtea_gaze/configs/gaze_slowfast_r50_egtea.py`
- debug:
  - `projects/egtea_gaze/configs/gaze_slowfast_r50_egtea_debug.py`
- ablations:
  - `gaze_slowfast_r50_egtea_ablation_random.py`
  - `gaze_slowfast_r50_egtea_ablation_center.py`
  - `gaze_slowfast_r50_egtea_ablation_shuffle.py`

## 4090 Optimization

Video decoding remains standard MMAction2:

- `VideoDataset + DecordDecode`

Performance tuning is handled via config and shell:

- `AmpOptimWrapper`
- default `BATCH_SIZE=16`
- default `WORKERS=16`
- default `PREFETCH=8`
- `pin_memory=True`
- `persistent_workers=True`
- `cudnn_benchmark=True`
- `OMP_NUM_THREADS=1`
- `MKL_NUM_THREADS=1`

The gaze path avoids runtime parsing of raw text files. Runtime work is limited
to:

- cache loading
- crop / flip coordinate adjustment
- 14x14 target heatmap generation

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

- gaze cache root:
  - `/root/data/egtea/gaze_maps`
- formal training outputs:
  - `/root/outputs/egtea_gaze/gaze_slowfast_r50`
- debug outputs:
  - `/root/outputs/egtea_gaze/gaze_slowfast_debug`
- overlay / attention visualization:
  - `/root/outputs/egtea_gaze/attention_visualization`
- training curves:
  - `/root/outputs/egtea_gaze/training_curves`
- gaze cache stats:
  - `/root/outputs/egtea_gaze/gaze_cache_stats`
- paper assets:
  - `/root/outputs/egtea_gaze/paper_assets`

## Post-Training Workflow (论文准备)

After training completes, run the following scripts in order to prepare all
materials needed for thesis writing. All scripts are read-only with respect to
training work directories and checkpoint files.

### Step 1: Test best checkpoint

```bash
bash projects/egtea_gaze/tools/test_best_gaze_checkpoint.sh
```

Auto-discovers `best_acc_top1_*.pth` in the gaze training work_dir and runs
10-clip × 3-crop evaluation. Outputs to
`/root/outputs/egtea_gaze/gaze_slowfast_r50_test_best`.

### Step 2: Collect experiment results

```bash
python projects/egtea_gaze/tools/collect_experiment_results.py --pretty-print --overwrite
```

Scans all 6 work directories (baseline + gaze + test_best + 3 ablations),
extracts best metrics from JSONL/log files, and produces:
- `experiment_summary.csv`
- `experiment_summary.md`
- `experiment_summary.json`

### Step 3: Generate comparison table

```bash
python projects/egtea_gaze/tools/compare_baseline_gaze.py --pretty-print --overwrite
```

Reads `experiment_summary.csv` and produces the main paper table with delta
columns vs the SlowFast baseline:
- `baseline_vs_gaze.csv`
- `baseline_vs_gaze.md`
- `baseline_vs_gaze_latex.txt` (paste directly into LaTeX)

### Step 4: Export training curves

```bash
python projects/egtea_gaze/tools/export_training_curves.py --overwrite
```

Parses JSONL logs and produces per-experiment CSVs and per-metric PNG plots
(loss, loss_cls, loss_gaze, acc/top1, acc/top5, lr, etc.).

### Step 5: Analyze gaze cache quality

```bash
python projects/egtea_gaze/tools/analyze_gaze_cache_stats.py --pretty-print --overwrite
```

Read-only scan of the gaze cache. Reports per-split and per-class valid gaze
ratios for the "data preprocessing" thesis chapter.

### Step 6: Visualize attention vs GT gaze

```bash
python projects/egtea_gaze/tools/visualize_gaze_attention.py --overwrite
```

Generates side-by-side images (RGB | GT gaze | model attention) with
correct/wrong grouping. GT gaze is for comparison only — the model uses
RGB-only input at test time.

### Step 7: Package paper assets

```bash
bash projects/egtea_gaze/tools/package_paper_assets.sh --overwrite
```

Copies all tables, figures, logs, and checkpoint paths into a single
`paper_assets/` directory with a README explaining which figure goes in which
thesis section.

### Safety guarantees

All post-training scripts satisfy:

1. Checkpoint missing → `[WARN] training may not be finished`, exit 0
2. Log missing → warning, no crash
3. Metric not found → field shows `NA`
4. Never overwrites without explicit `--overwrite`
5. Never modifies training work directories or checkpoint files
6. All paths have cloud defaults but accept CLI overrides

## Troubleshooting

- `metadata.json` missing:
  - run `make_gaze_maps.py` first and check `--out-root`
- `custom_imports` failure:
  - `export PYTHONPATH=/root/code/mmaction2:$PYTHONPATH`
- `loss_gaze` becomes NaN:
  - inspect cache with `check_gaze_maps.py`
- `data_time` too high:
  - inspect cache layout, worker count, and `LoadGazeMap` cache behavior
- CUDA OOM:
  - lower `BATCH_SIZE`, reduce `PREFETCH`, or start from debug config
- overlay all invalid:
  - inspect `clip_debug`
  - confirm `num_gaze_rows_total` is full-file scale, not sampled scale
  - confirm `gaze_frame_max` reaches the real session range
  - confirm `computed_global_frame` and `matched_frame` are reasonable
