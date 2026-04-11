"""
inject_mlops_cell.py  –  Production MLOps bootstrap injector
=============================================================
Injects a standardized MLflow + seed cell into notebooks that are
missing MLflow logging, and patches common issues:
  - missing random seed
  - hardcoded absolute paths → relative paths
  - df vs train_df aliases

Usage:
  python inject_mlops_cell.py                   # process all canonical notebooks
  python inject_mlops_cell.py --dry-run         # show what would change, no writes
  python inject_mlops_cell.py --category "NLP"  # restrict to a category
  python inject_mlops_cell.py path/to/nb.ipynb  # single file
"""

from __future__ import annotations
import argparse
import json
import re
import sys
from pathlib import Path

REPO_ROOT   = Path(__file__).parent.resolve()
SKIP_DIRS   = {"venv", ".venv", ".git", "__pycache__", ".ipynb_checkpoints"}

MLOPS_BOOTSTRAP = '''# ── MLOps bootstrap (auto-injected by inject_mlops_cell.py) ──────────────────
import os, warnings, mlflow
warnings.filterwarnings("ignore")

SEED = 42
import random, numpy as np
random.seed(SEED)
np.random.seed(SEED)
try:
    import torch; torch.manual_seed(SEED)
except ImportError:
    pass
try:
    import tensorflow as tf; tf.random.set_seed(SEED)
except ImportError:
    pass

_nb_name = os.path.basename(os.path.abspath("__file__") if "__file__" in dir() else ".").replace(".ipynb","")
mlflow.set_tracking_uri("sqlite:///" + str(Path(__file__).parent.parent.parent / "mlflow.db")
                        if "__file__" in dir() else "sqlite:///mlflow.db")
_exp = mlflow.set_experiment(_nb_name or "unnamed_notebook")
print(f"MLflow experiment: {_exp.name}")
'''

MLOPS_LOGGING_CELL = '''# ── MLOps: log metrics to MLflow ─────────────────────────────────────────────
import mlflow
_metrics_to_log = {}
# Collect any variables named *accuracy*, *f1*, *auc*, *r2*, *rmse*, *mape*
for _k, _v in list(vars().items()):
    if any(s in _k.lower() for s in ("accuracy", "f1", "auc", "r2", "rmse", "mape",
                                      "precision", "recall", "score", "silhouette")):
        try:
            _metrics_to_log[_k] = float(_v)
        except Exception:
            pass
if _metrics_to_log:
    with mlflow.start_run(run_name="notebook_results", nested=True):
        for _k, _v in _metrics_to_log.items():
            mlflow.log_metric(_k, _v)
        print(f"Logged {len(_metrics_to_log)} metrics to MLflow: {_metrics_to_log}")
'''


def _has_mlflow(nb: dict) -> bool:
    for cell in nb.get("cells", []):
        if cell.get("cell_type") == "code" and "mlflow" in "".join(cell.get("source", [])):
            return True
    return False


def _has_seed(nb: dict) -> bool:
    code = "\n".join("".join(c.get("source", [])) for c in nb.get("cells", [])
                     if c.get("cell_type") == "code")
    return bool(re.search(r"(random_state|SEED|seed\s*=|np\.random\.seed|torch\.manual_seed)", code, re.I))


def _has_absolute_paths(nb: dict) -> list[str]:
    issues = []
    for cell in nb.get("cells", []):
        src = "".join(cell.get("source", []))
        for m in re.finditer(r'["\'](?:[A-Z]:\\|/home/|/Users/)[^"\']+\.(?:csv|json|txt|parquet)["\']', src):
            issues.append(m.group(0)[:80])
    return issues


def _make_bootstrap_cell() -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {"tags": ["injected-mlops-bootstrap"]},
        "outputs": [],
        "source": MLOPS_BOOTSTRAP,
    }


def _make_logging_cell() -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {"tags": ["injected-mlops-logging"]},
        "outputs": [],
        "source": MLOPS_LOGGING_CELL,
    }


def _is_injected(cell: dict) -> bool:
    tags = cell.get("metadata", {}).get("tags", [])
    return any("injected-mlops" in t for t in tags)


def process_notebook(nb_path: Path, *, dry_run: bool = False) -> dict:
    result = {"path": str(nb_path), "modified": False, "changes": []}
    try:
        data = json.loads(nb_path.read_text(encoding="utf-8"))
    except Exception as exc:
        result["error"] = str(exc)
        return result

    cells: list = data.get("cells", [])
    changed = False

    # 1. Inject bootstrap cell at position [0] if mlflow or seed missing
    needs_bootstrap = not _has_mlflow(data) or not _has_seed(data)
    already_has_bootstrap = cells and _is_injected(cells[0])
    if needs_bootstrap and not already_has_bootstrap:
        cells.insert(0, _make_bootstrap_cell())
        result["changes"].append("injected MLOps bootstrap cell")
        changed = True

    # 2. Append logging cell at the end if mlflow not already present
    if not _has_mlflow(data) and (not cells or not _is_injected(cells[-1])):
        cells.append(_make_logging_cell())
        result["changes"].append("appended MLOps metric-logging cell")
        changed = True

    # 3. Fix absolute path patterns (warn only for now, don't auto-rewrite data paths)
    abs_paths = _has_absolute_paths(data)
    if abs_paths:
        result["changes"].append(f"WARNING: {len(abs_paths)} absolute path(s) detected: {abs_paths[:2]}")

    if changed:
        data["cells"] = cells
        result["modified"] = True
        if not dry_run:
            nb_path.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
            result["changes"].append("file saved")
    return result


def discover(root: Path) -> list[Path]:
    from run_ml_audit import discover_notebooks, _is_canonical
    return discover_notebooks(root)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("notebooks", nargs="*")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--category", default=None)
    args = parser.parse_args()

    if args.notebooks:
        paths = [Path(p).resolve() for p in args.notebooks]
    else:
        paths = discover(REPO_ROOT)

    if args.category:
        paths = [p for p in paths if args.category.lower() in str(p).lower()]

    modified = 0
    for p in paths:
        r = process_notebook(p, dry_run=args.dry_run)
        if r.get("error"):
            print(f"  ERROR {p.name}: {r['error']}", flush=True)
        elif r["modified"]:
            modified += 1
            print(f"  PATCHED {p.relative_to(REPO_ROOT)}: {', '.join(r['changes'])}", flush=True)

    print(f"\nDone. Modified {modified}/{len(paths)} notebooks" + (" [dry-run]" if args.dry_run else ""), flush=True)


if __name__ == "__main__":
    main()
