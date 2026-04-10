#!/usr/bin/env bash
# ============================================================================
# Computer Vision Projects — Environment Setup (Linux / macOS / WSL)
# ============================================================================
# Usage:
#   bash scripts/setup_env.sh
#   bash scripts/setup_env.sh --skip-torch
# ============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

SKIP_TORCH=false
for arg in "$@"; do
    case "$arg" in --skip-torch) SKIP_TORCH=true ;; esac
done

echo ""
echo "========================================"
echo "  CV Projects — Environment Setup"
echo "========================================"
echo ""

# --- 1. Install requirements.txt ---
echo "[1/4] Installing requirements.txt ..."
pip install -r requirements.txt

# --- 1b. Verify OpenCV ---
if ! python -c "import cv2; print(f'  OpenCV {cv2.__version__} OK')" 2>/dev/null; then
    echo "[WARN] opencv-python failed to install."
    echo "  If you are on Python 3.13 free-threaded (cp313t), binary wheels may"
    echo "  not exist yet.  Options:"
    echo "    1. Use standard CPython 3.13 (non-free-threaded)"
    echo "    2. conda install -c conda-forge opencv"
    echo "    3. Build from source: pip install --no-binary opencv-python opencv-python"
fi

# --- 2. Install PyTorch with CUDA 13.0 ---
if [ "$SKIP_TORCH" = false ]; then
    echo ""
    echo "[2/4] Installing PyTorch + torchvision (CUDA 13.0) ..."
    pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu130
else
    echo ""
    echo "[2/4] Skipping PyTorch install (--skip-torch)"
fi

# --- 3. Verify critical imports ---
echo ""
echo "[3/4] Verifying critical imports ..."
python -c "
import torch, torchvision, cv2, ultralytics
print(f'  torch       {torch.__version__}  CUDA={torch.cuda.is_available()}')
print(f'  torchvision {torchvision.__version__}')
print(f'  OpenCV      {cv2.__version__}')
print(f'  Ultralytics {ultralytics.__version__}')
"

# --- 4. Run smoke tests ---
echo ""
echo "[4/4] Running smoke tests ..."
python scripts/smoke_test.py || echo "[WARN] Some smoke tests failed."

# --- 5. Configure git hooks ---
echo ""
echo "[+] Configuring git hooks path ..."
git config core.hooksPath .githooks 2>/dev/null || true
echo "  Git hooks path set to .githooks"

echo ""
echo "========================================"
echo "  Setup complete!"
echo "========================================"
echo ""
