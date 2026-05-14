#!/bin/bash
set -e

REPO_DIR="${REPO_DIR:-/root/code/mmaction2}"
CONFIG="${CONFIG:-projects/egtea_gaze/configs/gaze_slowfast_r50_egtea.py}"
WORK_DIR="${WORK_DIR:-/root/outputs/egtea_gaze/gaze_slowfast_r50}"
GPU_ID="${GPU:-0}"
BATCH_SIZE="${BATCH_SIZE:-16}"
WORKERS="${WORKERS:-16}"
PREFETCH="${PREFETCH:-8}"
MAX_EPOCHS="${MAX_EPOCHS:-30}"
RESUME="${RESUME:-0}"
GAZE_MAP_ROOT="${GAZE_MAP_ROOT:-/root/data/egtea/gaze_maps}"
GAZE_METADATA="${GAZE_METADATA:-$GAZE_MAP_ROOT/metadata.json}"
SLOWFAST_CKPT="${SLOWFAST_CKPT:-/root/outputs/egtea_gaze/slowfast_r50_bs24_amp_25ep/best_acc_top1_epoch_24.pth}"

cd "$REPO_DIR"
export PYTHONPATH="$REPO_DIR:$PYTHONPATH"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export EGTEA_GAZE_MAP_ROOT="$GAZE_MAP_ROOT"
export EGTEA_SLOWFAST_CKPT="$SLOWFAST_CKPT"

[ -f "$GAZE_METADATA" ] || { echo "[ERROR] missing gaze metadata: $GAZE_METADATA"; exit 1; }
[ -f "$SLOWFAST_CKPT" ] || { echo "[ERROR] missing slowfast checkpoint: $SLOWFAST_CKPT"; exit 1; }

EXTRA_ARGS=()
if [ "$RESUME" = "1" ]; then
  EXTRA_ARGS+=(--resume)
fi

CUDA_VISIBLE_DEVICES="${GPU_ID}" python tools/train.py "$CONFIG" \
  --work-dir "$WORK_DIR" \
  "${EXTRA_ARGS[@]}" \
  --cfg-options \
  train_cfg.max_epochs="${MAX_EPOCHS}" \
  train_dataloader.batch_size="${BATCH_SIZE}" \
  train_dataloader.num_workers="${WORKERS}" \
  val_dataloader.num_workers="${WORKERS}" \
  test_dataloader.num_workers="${WORKERS}" \
  train_dataloader.pin_memory=True \
  val_dataloader.pin_memory=True \
  test_dataloader.pin_memory=True \
  train_dataloader.persistent_workers=True \
  val_dataloader.persistent_workers=True \
  test_dataloader.persistent_workers=True \
  train_dataloader.prefetch_factor="${PREFETCH}" \
  val_dataloader.prefetch_factor="${PREFETCH}" \
  test_dataloader.prefetch_factor="${PREFETCH}" \
  env_cfg.cudnn_benchmark=True

