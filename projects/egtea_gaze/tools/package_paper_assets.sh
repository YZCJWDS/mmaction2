#!/bin/bash
# =============================================================================
# Package all paper-ready assets into a single directory for thesis writing.
#
# This script COPIES lightweight files (tables, figures, stats) and RECORDS
# checkpoint paths without copying the large .pth files themselves.
#
# Usage:
#   bash projects/egtea_gaze/tools/package_paper_assets.sh
#   bash projects/egtea_gaze/tools/package_paper_assets.sh --overwrite
#   PAPER_ASSETS_DIR=/cloud/paper bash projects/egtea_gaze/tools/package_paper_assets.sh
#
# Defaults (overridable via environment variables):
#   OUTPUT_ROOT       /root/outputs/egtea_gaze
#   PAPER_ASSETS_DIR  /root/outputs/egtea_gaze/paper_assets
#   OVERWRITE         0
# =============================================================================

set -u
set -o pipefail

# ---------------------------------------------------------------------------
# Parse --overwrite flag from positional args
# ---------------------------------------------------------------------------
OVERWRITE="${OVERWRITE:-0}"
for arg in "$@"; do
  case "$arg" in
    --overwrite) OVERWRITE=1 ;;
    *) echo "[WARN] unknown argument: $arg" ;;
  esac
done

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
OUTPUT_ROOT="${OUTPUT_ROOT:-/root/outputs/egtea_gaze}"
PAPER_ASSETS_DIR="${PAPER_ASSETS_DIR:-$OUTPUT_ROOT/paper_assets}"

# Source directories (read-only access)
TABLES_SRC="$OUTPUT_ROOT"
CURVES_SRC="$OUTPUT_ROOT/training_curves"
ATTENTION_SRC="$OUTPUT_ROOT/attention_visualization"
CACHE_STATS_SRC="$OUTPUT_ROOT/gaze_cache_stats"

# Work directories for log collection
BASELINE_WORK_DIR="$OUTPUT_ROOT/slowfast_r50_bs24_amp_25ep"
GAZE_WORK_DIR="$OUTPUT_ROOT/gaze_slowfast_r50"
GAZE_TEST_DIR="$OUTPUT_ROOT/gaze_slowfast_r50_test_best"
ABL_RANDOM_DIR="$OUTPUT_ROOT/gaze_slowfast_ablation_random"
ABL_CENTER_DIR="$OUTPUT_ROOT/gaze_slowfast_ablation_center"
ABL_SHUFFLE_DIR="$OUTPUT_ROOT/gaze_slowfast_ablation_shuffle"

# ---------------------------------------------------------------------------
# Output directory handling
# ---------------------------------------------------------------------------
if [ -d "$PAPER_ASSETS_DIR" ] && [ -n "$(ls -A "$PAPER_ASSETS_DIR" 2>/dev/null)" ]; then
  if [ "$OVERWRITE" != "1" ]; then
    TIMESTAMP=$(date +%Y%m%d_%H%M%S 2>/dev/null || date +%s)
    PAPER_ASSETS_DIR="${PAPER_ASSETS_DIR}_${TIMESTAMP}"
    echo "[INFO] existing paper_assets found, creating timestamped dir: $PAPER_ASSETS_DIR"
  else
    echo "[INFO] --overwrite: reusing existing $PAPER_ASSETS_DIR"
  fi
fi

echo "[INFO] paper assets output: $PAPER_ASSETS_DIR"

# ---------------------------------------------------------------------------
# Create directory structure
# ---------------------------------------------------------------------------
mkdir -p "$PAPER_ASSETS_DIR/tables"
mkdir -p "$PAPER_ASSETS_DIR/figures/training_curves"
mkdir -p "$PAPER_ASSETS_DIR/figures/attention_visualization"
mkdir -p "$PAPER_ASSETS_DIR/figures/gaze_cache_stats"
mkdir -p "$PAPER_ASSETS_DIR/logs/slowfast_baseline"
mkdir -p "$PAPER_ASSETS_DIR/logs/gaze_slowfast"
mkdir -p "$PAPER_ASSETS_DIR/logs/ablations"
mkdir -p "$PAPER_ASSETS_DIR/checkpoints_index"

# ---------------------------------------------------------------------------
# Helper: safe copy (skip if source missing)
# ---------------------------------------------------------------------------
safe_cp() {
  local src="$1"
  local dst="$2"
  if [ -f "$src" ]; then
    cp "$src" "$dst"
    echo "  [OK] $src -> $dst"
  else
    echo "  [SKIP] not found: $src"
  fi
}

safe_cp_dir() {
  local src="$1"
  local dst="$2"
  if [ -d "$src" ] && [ -n "$(ls -A "$src" 2>/dev/null)" ]; then
    cp -r "$src"/* "$dst/" 2>/dev/null || true
    echo "  [OK] $src/* -> $dst/"
  else
    echo "  [SKIP] dir empty or missing: $src"
  fi
}

# ---------------------------------------------------------------------------
# 1. Tables
# ---------------------------------------------------------------------------
echo ""
echo "=== Copying tables ==="
safe_cp "$TABLES_SRC/experiment_summary.csv" "$PAPER_ASSETS_DIR/tables/experiment_summary.csv"
safe_cp "$TABLES_SRC/experiment_summary.md" "$PAPER_ASSETS_DIR/tables/experiment_summary.md"
safe_cp "$TABLES_SRC/experiment_summary.json" "$PAPER_ASSETS_DIR/tables/experiment_summary.json"
safe_cp "$TABLES_SRC/baseline_vs_gaze.csv" "$PAPER_ASSETS_DIR/tables/baseline_vs_gaze.csv"
safe_cp "$TABLES_SRC/baseline_vs_gaze.md" "$PAPER_ASSETS_DIR/tables/baseline_vs_gaze.md"
safe_cp "$TABLES_SRC/baseline_vs_gaze_latex.txt" "$PAPER_ASSETS_DIR/tables/baseline_vs_gaze_latex.txt"

# ---------------------------------------------------------------------------
# 2. Figures
# ---------------------------------------------------------------------------
echo ""
echo "=== Copying figures ==="
echo "  -- training curves --"
safe_cp_dir "$CURVES_SRC" "$PAPER_ASSETS_DIR/figures/training_curves"

echo "  -- attention visualization --"
# Copy only PNGs and summary (not the full HTML with embedded base64)
if [ -d "$ATTENTION_SRC" ]; then
  # Copy comparison images and summary
  find "$ATTENTION_SRC" -maxdepth 3 -name '*_comparison.png' -exec cp {} "$PAPER_ASSETS_DIR/figures/attention_visualization/" \; 2>/dev/null
  safe_cp "$ATTENTION_SRC/summary.json" "$PAPER_ASSETS_DIR/figures/attention_visualization/summary.json"
  # Count copied
  N_ATTN=$(find "$PAPER_ASSETS_DIR/figures/attention_visualization" -name '*.png' 2>/dev/null | wc -l)
  echo "  [OK] copied $N_ATTN attention comparison images"
else
  echo "  [SKIP] attention_visualization dir not found"
fi

echo "  -- gaze cache stats --"
safe_cp "$CACHE_STATS_SRC/gaze_cache_stats.csv" "$PAPER_ASSETS_DIR/figures/gaze_cache_stats/gaze_cache_stats.csv"
safe_cp "$CACHE_STATS_SRC/gaze_cache_stats.md" "$PAPER_ASSETS_DIR/figures/gaze_cache_stats/gaze_cache_stats.md"
safe_cp "$CACHE_STATS_SRC/gaze_cache_stats.json" "$PAPER_ASSETS_DIR/figures/gaze_cache_stats/gaze_cache_stats.json"

# ---------------------------------------------------------------------------
# 3. Logs (copy .log files, not full work_dir)
# ---------------------------------------------------------------------------
echo ""
echo "=== Collecting logs ==="

copy_logs_from() {
  local src_dir="$1"
  local dst_dir="$2"
  local label="$3"
  if [ ! -d "$src_dir" ]; then
    echo "  [SKIP] $label: dir not found ($src_dir)"
    return
  fi
  # Find .log files in timestamped subdirs
  local count=0
  while IFS= read -r -d '' logfile; do
    cp "$logfile" "$dst_dir/" 2>/dev/null && count=$((count + 1))
  done < <(find "$src_dir" -maxdepth 3 -name '*.log' -print0 2>/dev/null)
  # Also copy test_best.log if present
  if [ -f "$src_dir/test_best.log" ]; then
    cp "$src_dir/test_best.log" "$dst_dir/" && count=$((count + 1))
  fi
  echo "  [OK] $label: $count log file(s)"
}

copy_logs_from "$BASELINE_WORK_DIR" "$PAPER_ASSETS_DIR/logs/slowfast_baseline" "SlowFast baseline"
copy_logs_from "$GAZE_WORK_DIR" "$PAPER_ASSETS_DIR/logs/gaze_slowfast" "Gaze-SlowFast"
copy_logs_from "$GAZE_TEST_DIR" "$PAPER_ASSETS_DIR/logs/gaze_slowfast" "Gaze-SlowFast test"
copy_logs_from "$ABL_RANDOM_DIR" "$PAPER_ASSETS_DIR/logs/ablations" "Ablation random"
copy_logs_from "$ABL_CENTER_DIR" "$PAPER_ASSETS_DIR/logs/ablations" "Ablation center"
copy_logs_from "$ABL_SHUFFLE_DIR" "$PAPER_ASSETS_DIR/logs/ablations" "Ablation shuffle"

# ---------------------------------------------------------------------------
# 4. Checkpoint index (paths only, NOT the .pth files)
# ---------------------------------------------------------------------------
echo ""
echo "=== Building checkpoint index ==="
CKPT_INDEX="$PAPER_ASSETS_DIR/checkpoints_index/checkpoint_paths.txt"
{
  echo "# Checkpoint Path Index"
  echo "# Generated by package_paper_assets.sh"
  echo "# These are PATH REFERENCES only. The .pth files are NOT copied."
  echo "# ================================================================"
  echo ""

  echo "## SlowFast-R50 Baseline"
  BEST_BL=$(find "$BASELINE_WORK_DIR" -maxdepth 1 -name 'best_*.pth' -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -1 | awk '{print $2}')
  [ -n "$BEST_BL" ] && echo "  best: $BEST_BL" || echo "  best: (not found)"
  [ -f "$BASELINE_WORK_DIR/latest.pth" ] && echo "  latest: $BASELINE_WORK_DIR/latest.pth" || echo "  latest: (not found)"
  echo ""

  echo "## Gaze-SlowFast-R50 (Ours)"
  BEST_GZ=$(find "$GAZE_WORK_DIR" -maxdepth 1 -name 'best_acc_top1_*.pth' -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -1 | awk '{print $2}')
  [ -n "$BEST_GZ" ] && echo "  best: $BEST_GZ" || echo "  best: (not found)"
  [ -f "$GAZE_WORK_DIR/latest.pth" ] && echo "  latest: $GAZE_WORK_DIR/latest.pth" || echo "  latest: (not found)"
  echo ""

  echo "## Ablation: Random Gaze"
  BEST_AR=$(find "$ABL_RANDOM_DIR" -maxdepth 1 -name 'best_*.pth' -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -1 | awk '{print $2}')
  [ -n "$BEST_AR" ] && echo "  best: $BEST_AR" || echo "  best: (not found)"
  echo ""

  echo "## Ablation: Center Gaze"
  BEST_AC=$(find "$ABL_CENTER_DIR" -maxdepth 1 -name 'best_*.pth' -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -1 | awk '{print $2}')
  [ -n "$BEST_AC" ] && echo "  best: $BEST_AC" || echo "  best: (not found)"
  echo ""

  echo "## Ablation: Shuffle Gaze"
  BEST_AS=$(find "$ABL_SHUFFLE_DIR" -maxdepth 1 -name 'best_*.pth' -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -1 | awk '{print $2}')
  [ -n "$BEST_AS" ] && echo "  best: $BEST_AS" || echo "  best: (not found)"
  echo ""
} > "$CKPT_INDEX"
echo "  [OK] $CKPT_INDEX"

# ---------------------------------------------------------------------------
# 5. README_PAPER_ASSETS.md
# ---------------------------------------------------------------------------
echo ""
echo "=== Generating README ==="
README="$PAPER_ASSETS_DIR/README_PAPER_ASSETS.md"
cat > "$README" << 'READMEEOF'
# Paper Assets

This directory contains all materials needed for writing the thesis chapter on
**Gaze-Supervised SlowFast for First-Person Action Recognition**.

## Important Note

> **Testing uses RGB-only input. Real gaze is NOT used at inference time.**
>
> Gaze supervision is applied exclusively during training via an auxiliary KL
> divergence loss. At test time, the model receives only RGB video clips and
> produces action predictions without any gaze input.

---

## Directory Structure

```
paper_assets/
├── tables/                         # Quantitative results
├── figures/                        # Qualitative results & curves
│   ├── training_curves/            # Loss & accuracy over epochs
│   ├── attention_visualization/    # Model attention vs GT gaze
│   └── gaze_cache_stats/           # Gaze data quality analysis
├── logs/                           # Training logs for reference
│   ├── slowfast_baseline/
│   ├── gaze_slowfast/
│   └── ablations/
├── checkpoints_index/              # Checkpoint paths (NOT the .pth files)
└── README_PAPER_ASSETS.md          # This file
```

---

## Tables

| File | Usage | Thesis Section |
|------|-------|----------------|
| `experiment_summary.csv/md` | Full results across all experiments | 第四章 实验结果与分析 |
| `baseline_vs_gaze.csv/md` | Main comparison table (baseline vs ours) | 第四章 4.x 主实验结果 |
| `baseline_vs_gaze_latex.txt` | LaTeX tabular, paste directly into paper | 第四章 Table X |

### baseline_vs_gaze table columns:
- **Method**: model name
- **Test Input**: always RGB (gaze is NOT used at test time)
- **Training Supervision**: what loss was used during training
- **Top-1 / Top-5 / Mean Class Acc**: test set performance
- **Delta vs SlowFast**: improvement over the RGB-only baseline

---

## Figures

### training_curves/

Training loss and validation accuracy curves over epochs.

| Figure | Thesis Section |
|--------|----------------|
| `curve_loss.png` | 第四章 训练过程分析 |
| `curve_loss_cls.png` | 第四章 分类损失收敛 |
| `curve_loss_gaze.png` | 第四章 注视损失收敛 |
| `curve_loss_cls_vs_gaze.png` | 第四章 双损失对比 |
| `curve_acc_top1.png` | 第四章 验证集精度曲线 |
| `curve_acc_top5.png` | 第四章 Top-5 精度 |
| `curve_acc_mean1.png` | 第四章 平均类别精度 |
| `curve_lr.png` | 附录 学习率调度 |

### attention_visualization/

Side-by-side comparison: RGB | GT Gaze (red) | Model Attention (blue).

| Content | Thesis Section |
|---------|----------------|
| `*_correct_*_comparison.png` | 第四章 注意力可视化（正确样本） |
| `*_wrong_*_comparison.png` | 第四章 失败案例分析 |
| `summary.json` | 统计信息 |

**Key point for thesis**: GT gaze overlays are shown for qualitative comparison
only. The model never receives gaze as input during testing.

### gaze_cache_stats/

Gaze data quality statistics supporting the "data preprocessing" chapter.

| File | Thesis Section |
|------|----------------|
| `gaze_cache_stats.md` | 第三章 数据预处理 / 注视数据有效性分析 |
| `gaze_cache_stats.csv` | 第三章 Table: 注视数据统计 |

---

## Logs

| Directory | Experiment |
|-----------|-----------|
| `slowfast_baseline/` | SlowFast-R50 baseline (RGB-only, no gaze) |
| `gaze_slowfast/` | Gaze-SlowFast-R50 (ours, gaze supervision during training) |
| `ablations/` | Random / Center / Shuffle gaze ablation experiments |

---

## Checkpoints

`checkpoints_index/checkpoint_paths.txt` lists the paths to all best checkpoints.

**These are path references only.** The actual .pth files remain in their
original work directories and are NOT copied here to save disk space.

The formal Gaze-SlowFast best checkpoint is typically at:
```
/root/outputs/egtea_gaze/gaze_slowfast_r50/best_acc_top1_epoch_*.pth
```

---

## Reproducing Results

```bash
cd /root/code/mmaction2

# 1. Collect experiment results
python projects/egtea_gaze/tools/collect_experiment_results.py --pretty-print --overwrite

# 2. Generate comparison table
python projects/egtea_gaze/tools/compare_baseline_gaze.py --pretty-print --overwrite

# 3. Export training curves
python projects/egtea_gaze/tools/export_training_curves.py --overwrite

# 4. Analyze gaze cache quality
python projects/egtea_gaze/tools/analyze_gaze_cache_stats.py --pretty-print --overwrite

# 5. Visualize attention
python projects/egtea_gaze/tools/visualize_gaze_attention.py --overwrite

# 6. Package everything
bash projects/egtea_gaze/tools/package_paper_assets.sh --overwrite
```
READMEEOF

echo "  [OK] $README"

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
echo ""
echo "============================================"
echo "[OK] Paper assets packaged to: $PAPER_ASSETS_DIR"
echo "============================================"
echo ""
echo "Next steps:"
echo "  1. Review tables/ for quantitative results"
echo "  2. Select best figures/ for thesis chapters"
echo "  3. Check checkpoints_index/ for model paths"
echo "  4. Copy this directory to your local machine for writing"
