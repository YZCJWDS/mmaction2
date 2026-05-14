#!/bin/bash
set -e

REPO_DIR="${REPO_DIR:-/root/code/mmaction2}"
CONFIG="${CONFIG:-projects/egtea_gaze/configs/gaze_slowfast_r50_egtea_debug.py}"
WORK_DIR="${WORK_DIR:-/root/outputs/egtea_gaze/gaze_slowfast_debug}"
GPUT="${GPU:-0}"
BATCH_SIZE="${BATCH_SIZE:-4}"
WORKERS="${WORKERS:-8}"
PREFETCH="${PREFETCH:-4}"
MAX_EPOCHS="${MAX_EPOCHS:-1}"
GAZE_MAP_ROOT="${GAZE_MAP_ROOT:-/root/data/egtea/gaze_maps}"
GAZE_METADATA="${GAZE_METADATA:-$GAZE_MAP_ROOT/metadata.json}"
SLOWFAST_CKPT="${SLOWFAST_CKPT:-/root/outputs/egtea_gaze/slowfast_r50_bs24_amp_25ep/best_acc_top1_epoch_24.pth}"

cd "$REPO_DIR"
export PYTHONPATH="$REPO_DIR:$PYTHONPATH"

[ -d "$REPO_DIR/projects/egtea_gaze" ] || { echo "[ERROR] missing project dir"; exit 1; }
[ -f "$GAZE_METADATA" ] || { echo "[ERROR] missing gaze metadata: $GAZE_METADATA"; exit 1; }
[ -f "$SLOWFAST_CKPT" ] || { echo "[ERROR] missing slowfast checkpoint: $SLOWFAST_CKPT"; exit 1; }

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

CUDA_VISIBLE_DEVICES="${GPUT}" python tools/train.py "$CONFIG" \
  --work-dir "$WORK_DIR" \
  --cfg-options \
  train_cfg.max_epochs="${MAX_EPOCHS}" \
  train_dataloader.batch_size="${BATCH_SIZE}" \
  train_dataloader.num_workers="${WORKERS}" \
  val_dataloader.num_workers="${WORKERS}" \
  test_dataloader.num_workers="${WORKERS}" \
  train_dataloader.prefetch_factor="${PREFETCH}" \
  val_dataloader.prefetch_factor="${PREFETCH}" \
  test_dataloader.prefetch_factor="${PREFETCH}" \
  train_dataloader.dataset.pipeline.6.gaze_map_root="${GAZE_MAP_ROOT}" \
  train_dataloader.dataset.pipeline.6.metadata_file="${GAZE_METADATA}" \
  load_from="${SLOWFAST_CKPT}" \
  env_cfg.cudnn_benchmark=True
