#!/bin/bash
set -e

REPO_DIR="${REPO_DIR:-/root/code/mmaction2}"
CONFIG="${CONFIG:-projects/egtea_gaze/configs/gaze_slowfast_r50_egtea.py}"
WORK_DIR="${WORK_DIR:-/root/outputs/egtea_gaze/gaze_slowfast_r50}"
OUT_DIR="${OUT_DIR:-/root/outputs/egtea_gaze/gaze_slowfast_r50_test}"
CHECKPOINT="${CHECKPOINT:-}"
GPU_ID="${GPU:-0}"
WORKERS="${WORKERS:-16}"

cd "$REPO_DIR"
export PYTHONPATH="$REPO_DIR:$PYTHONPATH"

if [ -z "$CHECKPOINT" ]; then
  CANDIDATE="$(find "$WORK_DIR" -maxdepth 1 -type f -name 'best_*.pth' -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -n 1 | awk '{print $2}')"
  CHECKPOINT="${CANDIDATE}"
fi

[ -n "$CHECKPOINT" ] || { echo "[ERROR] no best_*.pth found in WORK_DIR and CHECKPOINT not provided"; exit 1; }
[ -f "$CHECKPOINT" ] || { echo "[ERROR] missing checkpoint: $CHECKPOINT"; exit 1; }

CUDA_VISIBLE_DEVICES="${GPU_ID}" python tools/test.py "$CONFIG" "$CHECKPOINT" \
  --work-dir "$OUT_DIR" \
  --cfg-options \
  val_dataloader.num_workers="${WORKERS}" \
  test_dataloader.num_workers="${WORKERS}" \
  env_cfg.cudnn_benchmark=True
