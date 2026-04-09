#!/usr/bin/env python3
"""
Notebook Normalizer — Adds standard setup cells to all .ipynb files.
Inserts a markdown header and a Python setup cell at the top of each notebook.

Usage:
    python scripts/normalize_notebooks.py
"""

import json
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

SETUP_MARKDOWN = """# Standard Setup
This cell ensures **reproducibility**, **CUDA detection**, and proper **path management** across all notebooks in this monorepo."""

SETUP_CODE = """# ── Standard Setup Cell ──────────────────────────────────────
# Reproducibility | CUDA Detection | Path Management
import os
import random
import logging
from pathlib import Path

import numpy as np

# ── Reproducibility ─────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# ── PyTorch + CUDA ──────────────────────────────────────────
try:
    import torch
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"PyTorch {torch.__version__} | Device: {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
except ImportError:
    DEVICE = "cpu"
    print("PyTorch not installed — using CPU")

# ── Paths (relative, portable) ──────────────────────────────
PROJECT_DIR = Path(".").resolve()
DATA_DIR = PROJECT_DIR / "data"
OUTPUT_DIR = PROJECT_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"Project : {PROJECT_DIR.name}")
print(f"Data    : {DATA_DIR}")

# ── Logging ─────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)"""

MARKER = "Standard Setup Cell"


def normalize_notebook(filepath: Path) -> bool:
    """Add standard setup cells to the top of a notebook."""
    with open(filepath, "r", encoding="utf-8") as f:
        nb = json.load(f)

    # Check if already normalized
    for cell in nb.get("cells", []):
        source = "".join(cell.get("source", []))
        if MARKER in source:
            return False

    # Create new cells
    md_cell = {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in SETUP_MARKDOWN.split("\n")]
    }
    # Remove trailing \n from last line
    if md_cell["source"]:
        md_cell["source"][-1] = md_cell["source"][-1].rstrip("\n")

    code_cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in SETUP_CODE.split("\n")]
    }
    if code_cell["source"]:
        code_cell["source"][-1] = code_cell["source"][-1].rstrip("\n")

    # Insert at top
    nb["cells"] = [md_cell, code_cell] + nb.get("cells", [])

    # Ensure kernel metadata has Python
    if "metadata" not in nb:
        nb["metadata"] = {}
    if "kernelspec" not in nb["metadata"]:
        nb["metadata"]["kernelspec"] = {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        }
    if "language_info" not in nb["metadata"]:
        nb["metadata"]["language_info"] = {
            "name": "python",
            "version": "3.11.0"
        }

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)

    return True


def main():
    notebooks = list(ROOT.rglob("*.ipynb"))
    print(f"\nFound {len(notebooks)} notebooks\n")

    normalized = 0
    for nb_path in sorted(notebooks):
        rel = nb_path.relative_to(ROOT)
        if normalize_notebook(nb_path):
            normalized += 1
            print(f"  ✓ {rel}")
        else:
            print(f"  - {rel} (already normalized)")

    print(f"\nNormalized {normalized}/{len(notebooks)} notebooks\n")


if __name__ == "__main__":
    main()
