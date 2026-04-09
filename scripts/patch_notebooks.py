#!/usr/bin/env python
"""Patch all code.ipynb notebooks in the workspace.

Changes applied (idempotent):
  1. Set kernel metadata to 'nlp-projects' / 'NLP Projects (py313)'.
  2. Ensure the FIRST code cell prepends the workspace root to sys.path.

Run from the workspace root:
    python scripts/patch_notebooks.py
"""

from __future__ import annotations

import json
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parent.parent
KERNEL_NAME = "nlp-projects"
KERNEL_DISPLAY = "NLP Projects (py313)"

BOOTSTRAP_SOURCE = [
    "# --- workspace bootstrap (auto-generated) ---\n",
    "import os, sys\n",
    "\n",
    "# Ensure the workspace root is on the Python path so shared utils are importable\n",
    "_ws_root = os.path.abspath(os.path.join(os.getcwd(), os.pardir, os.pardir, os.pardir))\n",
    "if _ws_root not in sys.path:\n",
    "    sys.path.insert(0, _ws_root)\n",
    "\n",
    "from utils.seed import set_global_seed\n",
    "set_global_seed(42)\n",
]

BOOTSTRAP_MARKER = "workspace bootstrap"


def patch_notebook(nb_path: Path) -> list[str]:
    """Patch a single notebook and return a list of changes made."""
    changes: list[str] = []

    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    # --- 1. Kernel metadata ---------------------------------------------------
    meta = nb.setdefault("metadata", {})
    ks = meta.setdefault("kernelspec", {})
    if ks.get("name") != KERNEL_NAME or ks.get("display_name") != KERNEL_DISPLAY:
        ks["name"] = KERNEL_NAME
        ks["display_name"] = KERNEL_DISPLAY
        ks["language"] = "python"
        changes.append("kernel → nlp-projects")

    # Ensure language_info is set
    li = meta.setdefault("language_info", {})
    if li.get("name") != "python":
        li["name"] = "python"

    # --- 2. Bootstrap cell ----------------------------------------------------
    cells = nb.setdefault("cells", [])

    # Check if the first code cell already contains the bootstrap marker
    has_bootstrap = False
    for cell in cells:
        if cell.get("cell_type") == "code":
            src = "".join(cell.get("source", []))
            if BOOTSTRAP_MARKER in src:
                has_bootstrap = True
            break  # only check the first code cell

    if not has_bootstrap:
        bootstrap_cell = {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": BOOTSTRAP_SOURCE,
        }
        # Insert before the first code cell (preserve any leading markdown cells)
        insert_idx = 0
        for i, cell in enumerate(cells):
            if cell.get("cell_type") == "code":
                insert_idx = i
                break
        cells.insert(insert_idx, bootstrap_cell)
        changes.append("bootstrap cell inserted")

    # --- Write back -----------------------------------------------------------
    if changes:
        with open(nb_path, "w", encoding="utf-8", newline="\n") as f:
            json.dump(nb, f, indent=1, ensure_ascii=False)
            f.write("\n")

    return changes


def main() -> None:
    notebooks = sorted(WORKSPACE.rglob("code.ipynb"))
    print(f"Found {len(notebooks)} notebooks.\n")

    total_patched = 0
    for nb in notebooks:
        rel = nb.relative_to(WORKSPACE)
        changes = patch_notebook(nb)
        if changes:
            total_patched += 1
            print(f"  PATCHED  {rel}")
            for c in changes:
                print(f"           - {c}")
        else:
            print(f"  OK       {rel}")

    print(f"\nDone. {total_patched}/{len(notebooks)} notebooks patched.")


if __name__ == "__main__":
    main()
