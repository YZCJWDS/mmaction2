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

python tools/train.py configs/recognition/tsn/tsn_r50_egtea_rgb_split1.py \
  --work-dir work_dirs/egtea_tsn_split1_full \
  --cfg-options \
  train_dataloader.num_workers=8 \
  train_dataloader.persistent_workers=True \
  val_dataloader.batch_size=8 \
  val_dataloader.num_workers=8 \
  val_dataloader.persistent_workers=True \
  test_dataloader.batch_size=8 \
  test_dataloader.num_workers=8 \
  test_dataloader.persistent_workers=True
