#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-.venv_egtea}"

"$PYTHON_BIN" -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip
python -m pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements/egtea_cloud.txt
python -m pip install -v -e .

python - <<'PY'
import mmaction
import mmcv
import mmengine
import torch

print('mmaction', mmaction.__version__)
print('mmcv', mmcv.__version__)
print('mmengine', mmengine.__version__)
print('torch', torch.__version__)
print('cuda_available', torch.cuda.is_available())
PY

echo "Environment setup completed."
echo "Activate it with: source ${VENV_DIR}/bin/activate"
