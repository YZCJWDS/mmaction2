#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

VENV_DIR="${VENV_DIR:-.venv_egtea}"
if [ ! -f "${VENV_DIR}/bin/activate" ]; then
  echo "Virtual environment not found: ${VENV_DIR}"
  echo "Run: bash tools/cloud/setup_egtea_cloud.sh"
  exit 1
fi

source "${VENV_DIR}/bin/activate"

BEST_CKPT="$(ls work_dirs/egtea_tsn_split1_full/best_acc_top1_epoch_*.pth 2>/dev/null | head -n 1 || true)"
if [ -z "$BEST_CKPT" ]; then
  echo "Best checkpoint not found under work_dirs/egtea_tsn_split1_full"
  exit 1
fi

python tools/test.py configs/recognition/tsn/tsn_r50_egtea_rgb_split1.py \
  "$BEST_CKPT" \
  --cfg-options \
  test_dataloader.batch_size=8 \
  test_dataloader.num_workers=8 \
  test_dataloader.persistent_workers=True
