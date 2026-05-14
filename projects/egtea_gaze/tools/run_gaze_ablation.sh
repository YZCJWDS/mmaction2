#!/bin/bash
set -e

REPO_DIR="${REPO_DIR:-/root/code/mmaction2}"
MODES="${MODES:-real random center shuffle}"
BASE_WORK_DIR="${BASE_WORK_DIR:-/root/outputs/egtea_gaze/ablation}"

declare -A CONFIG_MAP
CONFIG_MAP[real]="projects/egtea_gaze/configs/gaze_slowfast_r50_egtea.py"
CONFIG_MAP[random]="projects/egtea_gaze/configs/gaze_slowfast_r50_egtea_ablation_random.py"
CONFIG_MAP[center]="projects/egtea_gaze/configs/gaze_slowfast_r50_egtea_ablation_center.py"
CONFIG_MAP[shuffle]="projects/egtea_gaze/configs/gaze_slowfast_r50_egtea_ablation_shuffle.py"

for mode in $MODES; do
  CONFIG="${CONFIG_MAP[$mode]}"
  [ -n "$CONFIG" ] || { echo "[ERROR] unsupported mode: $mode"; exit 1; }
  WORK_DIR="$BASE_WORK_DIR/$mode"
  echo "[INFO] Running ablation: $mode"
  CONFIG="$CONFIG" WORK_DIR="$WORK_DIR" bash "$REPO_DIR/projects/egtea_gaze/tools/train_gaze_slowfast.sh"
done

