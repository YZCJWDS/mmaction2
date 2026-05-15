#!/bin/bash
# =============================================================================
# Test the best checkpoint of the gaze-supervised SlowFast model.
#
# This script is read-only with respect to the training work directory. It
# only LOOKS UP a checkpoint inside WORK_DIR; it never writes, deletes, or
# resumes training there.
#
# Usage examples:
#   bash projects/egtea_gaze/tools/test_best_gaze_checkpoint.sh
#   GPU=1 bash projects/egtea_gaze/tools/test_best_gaze_checkpoint.sh
#   CHECKPOINT=/path/to/ckpt.pth bash projects/egtea_gaze/tools/test_best_gaze_checkpoint.sh
#   OVERWRITE=1 bash projects/egtea_gaze/tools/test_best_gaze_checkpoint.sh
#
# Defaults (overridable via environment variables):
#   CODE_ROOT     /root/code/mmaction2
#   CONFIG        projects/egtea_gaze/configs/gaze_slowfast_r50_egtea.py
#   WORK_DIR      /root/outputs/egtea_gaze/gaze_slowfast_r50
#   TEST_OUT_DIR  /root/outputs/egtea_gaze/gaze_slowfast_r50_test_best
#   GPU           0
#   CHECKPOINT    (auto-discovered inside WORK_DIR)
#   OVERWRITE     0   set to 1 to allow writing into a non-empty TEST_OUT_DIR
#
# Checkpoint discovery priority (newest mtime first within each tier):
#   1. best_acc_top1_*.pth
#   2. best_*.pth
#   3. latest.pth
# If none is found, the script prints a hint and exits with code 0 so that it
# can be safely scheduled before training has finished.
# =============================================================================

set -u
set -o pipefail

CODE_ROOT="${CODE_ROOT:-/root/code/mmaction2}"
CONFIG="${CONFIG:-projects/egtea_gaze/configs/gaze_slowfast_r50_egtea.py}"
WORK_DIR="${WORK_DIR:-/root/outputs/egtea_gaze/gaze_slowfast_r50}"
TEST_OUT_DIR="${TEST_OUT_DIR:-/root/outputs/egtea_gaze/gaze_slowfast_r50_test_best}"
GPU_ID="${GPU:-0}"
CHECKPOINT="${CHECKPOINT:-}"
OVERWRITE="${OVERWRITE:-0}"

# ---------------------------------------------------------------------------
# Sanity: code root and config
# ---------------------------------------------------------------------------
if [ ! -d "$CODE_ROOT" ]; then
  echo "[ERROR] CODE_ROOT does not exist: $CODE_ROOT"
  exit 1
fi

cd "$CODE_ROOT"
export PYTHONPATH="$CODE_ROOT:${PYTHONPATH:-}"

if [ ! -f "$CONFIG" ]; then
  echo "[ERROR] CONFIG not found: $CONFIG"
  echo "[HINT] expected path is relative to CODE_ROOT ($CODE_ROOT)."
  exit 1
fi

# ---------------------------------------------------------------------------
# Sanity: work dir (read-only access)
# ---------------------------------------------------------------------------
if [ ! -d "$WORK_DIR" ]; then
  echo "[WARN] WORK_DIR not found: $WORK_DIR"
  echo "[WARN] training may not be finished, or WORK_DIR is wrong."
  echo "[WARN] nothing to test, exiting cleanly."
  exit 0
fi

# ---------------------------------------------------------------------------
# Discover checkpoint by priority unless user passed CHECKPOINT explicitly
# ---------------------------------------------------------------------------
find_latest_in_workdir() {
  # $1: glob pattern, e.g. 'best_acc_top1_*.pth'
  # Prints the newest matching file path on stdout, or empty string.
  find "$WORK_DIR" -maxdepth 1 -type f -name "$1" -printf '%T@ %p\n' 2>/dev/null \
    | sort -nr | head -n 1 | awk '{print $2}'
}

if [ -z "$CHECKPOINT" ]; then
  CHECKPOINT="$(find_latest_in_workdir 'best_acc_top1_*.pth')"
  PICKED_TIER="best_acc_top1_*.pth"
  if [ -z "$CHECKPOINT" ]; then
    CHECKPOINT="$(find_latest_in_workdir 'best_*.pth')"
    PICKED_TIER="best_*.pth"
  fi
  if [ -z "$CHECKPOINT" ] && [ -f "$WORK_DIR/latest.pth" ]; then
    CHECKPOINT="$WORK_DIR/latest.pth"
    PICKED_TIER="latest.pth"
  fi
else
  PICKED_TIER="user-specified"
fi

if [ -z "$CHECKPOINT" ] || [ ! -f "$CHECKPOINT" ]; then
  echo "[WARN] no usable checkpoint found in: $WORK_DIR"
  echo "[WARN] searched (in order): best_acc_top1_*.pth -> best_*.pth -> latest.pth"
  echo "[WARN] training may not be finished, or checkpoint hook has not saved yet."
  exit 0
fi

echo "[INFO] code root      : $CODE_ROOT"
echo "[INFO] config         : $CONFIG"
echo "[INFO] work dir       : $WORK_DIR"
echo "[INFO] picked tier    : $PICKED_TIER"
echo "[INFO] checkpoint     : $CHECKPOINT"
echo "[INFO] test out dir   : $TEST_OUT_DIR"
echo "[INFO] GPU            : $GPU_ID"

# ---------------------------------------------------------------------------
# Safe write: refuse to clobber a non-empty existing TEST_OUT_DIR
# ---------------------------------------------------------------------------
if [ -d "$TEST_OUT_DIR" ] && [ -n "$(ls -A "$TEST_OUT_DIR" 2>/dev/null)" ]; then
  if [ "$OVERWRITE" != "1" ]; then
    echo "[ERROR] TEST_OUT_DIR already exists and is not empty: $TEST_OUT_DIR"
    echo "[HINT]  pass OVERWRITE=1 to allow writing into it,"
    echo "[HINT]  or set TEST_OUT_DIR to a fresh path."
    exit 1
  fi
  echo "[WARN] OVERWRITE=1, writing into existing TEST_OUT_DIR: $TEST_OUT_DIR"
fi
mkdir -p "$TEST_OUT_DIR"

LOG_FILE="$TEST_OUT_DIR/test_best.log"
echo "[INFO] log file       : $LOG_FILE"

# ---------------------------------------------------------------------------
# Run test (RGB-only, gaze not used at inference)
# ---------------------------------------------------------------------------
{
  echo "==== test_best_gaze_checkpoint.sh ===="
  echo "timestamp     : $(date -Is 2>/dev/null || date)"
  echo "code root     : $CODE_ROOT"
  echo "config        : $CONFIG"
  echo "work dir      : $WORK_DIR"
  echo "picked tier   : $PICKED_TIER"
  echo "checkpoint    : $CHECKPOINT"
  echo "test out dir  : $TEST_OUT_DIR"
  echo "GPU           : $GPU_ID"
  echo "PYTHONPATH    : $PYTHONPATH"
  echo "======================================"
} | tee "$LOG_FILE"

set +e
CUDA_VISIBLE_DEVICES="${GPU_ID}" python tools/test.py "$CONFIG" "$CHECKPOINT" \
  --work-dir "$TEST_OUT_DIR" 2>&1 | tee -a "$LOG_FILE"
STATUS=${PIPESTATUS[0]}
set -e

if [ "$STATUS" -ne 0 ]; then
  echo "[ERROR] tools/test.py failed with exit code $STATUS"
  echo "[ERROR] see log: $LOG_FILE"
  exit "$STATUS"
fi

echo "[OK] test finished. log: $LOG_FILE"
