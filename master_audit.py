#!/usr/bin/env python3
"""
ML Audit — Notebook Quality Scorer
====================================
Scans Data Analysis notebooks and scores each on structural quality,
data handling, analysis depth, and educational value.

Outputs:
    reports/summary.json   — machine-readable scores + ranking
    (stdout)               — human-readable summary table

Usage:
    python master_audit.py                # audit all Data Analysis notebooks
    python master_audit.py --skip-inject  # same (flag accepted for compat)
"""

from __future__ import annotations

import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_ANALYSIS_DIR = ROOT / "Data Analysis"
REPORTS_DIR = ROOT / "reports"

# ── Scoring weights (sum = 100) ──────────────────────────────────────────────

# Structure (30 pts)
PTS_TITLE = 5
PTS_MARKDOWN_CELLS = 5        # ≥5 markdown cells
PTS_CODE_CELLS = 5            # ≥5 code cells
PTS_TOTAL_CELLS = 5           # ≥15 total cells
PTS_CONCLUSION = 5            # has summary/conclusion section
PTS_NO_EMPTY_CODE = 5         # no completely empty code cells

# Data Handling (25 pts)
PTS_IMPORTS = 5               # imports libraries
PTS_DATA_LOAD = 5             # loads data (read_csv, load_dataset, etc.)
PTS_DATASET_SOURCE = 5        # mentions dataset URL/source
PTS_DATA_VALIDATION = 5       # checks nulls/dtypes/shape
PTS_DOWNLOAD_CELL = 5         # in-notebook download (kaggle/wget/urllib/etc.)

# Analysis Quality (25 pts)
PTS_VISUALIZATION = 10        # imports matplotlib/seaborn/plotly
PTS_STATISTICS = 5            # statistical analysis (describe/corr/value_counts)
PTS_MULTIPLE_ANALYSES = 5     # ≥3 substantive code cells
PTS_MD_CODE_RATIO = 5         # markdown-to-code ratio ≥ 0.5

# Educational Value (20 pts)
PTS_METHODOLOGY = 5           # explains methodology in markdown
PTS_LIMITATIONS = 5           # has limitations/improvements section
PTS_EXERCISES = 5             # has exercises or challenges
PTS_FORMATTING = 5            # professional formatting (h2+ headings)

# Verdict thresholds
PRODUCTION_READY = 90
NEEDS_IMPROVEMENT = 80
WEAK = 70
# < 70 = FAILED


def _join_sources(cells: list[dict], cell_type: str) -> str:
    """Concatenate all source text from cells of a given type."""
    parts = []
    for c in cells:
        if c.get("cell_type") == cell_type:
            src = c.get("source", "")
            if isinstance(src, list):
                src = "".join(src)
            parts.append(src)
    return "\n".join(parts)


def _count_cells(cells: list[dict], cell_type: str) -> int:
    return sum(1 for c in cells if c.get("cell_type") == cell_type)


def _has_pattern(text: str, patterns: list[str]) -> bool:
    """Case-insensitive check for any pattern in text."""
    low = text.lower()
    return any(p.lower() in low for p in patterns)


def score_notebook(nb_path: Path) -> dict:
    """Score a single notebook. Returns dict with score, verdict, checks."""
    result = {
        "notebook": str(nb_path.relative_to(ROOT)),
        "score": 0,
        "verdict": "FAILED",
        "checks": {},
    }

    try:
        raw = nb_path.read_text(encoding="utf-8")
        nb = json.loads(raw)
    except Exception as exc:
        result["checks"]["parse_error"] = str(exc)
        return result

    cells = nb.get("cells", [])
    if not cells:
        result["checks"]["empty"] = True
        return result

    md_text = _join_sources(cells, "markdown")
    code_text = _join_sources(cells, "code")
    all_text = md_text + "\n" + code_text

    n_md = _count_cells(cells, "markdown")
    n_code = _count_cells(cells, "code")
    n_total = len(cells)

    score = 0
    checks: dict[str, bool] = {}

    # ── Structure (30) ────────────────────────────────────────────────────
    # Title: first markdown cell starts with #
    first_md = None
    for c in cells:
        if c.get("cell_type") == "markdown":
            src = c.get("source", "")
            if isinstance(src, list):
                src = "".join(src)
            first_md = src
            break
    checks["title"] = bool(first_md and re.search(r"^#\s", first_md, re.M))
    if checks["title"]:
        score += PTS_TITLE

    checks["markdown_cells_5"] = n_md >= 5
    if checks["markdown_cells_5"]:
        score += PTS_MARKDOWN_CELLS

    checks["code_cells_5"] = n_code >= 5
    if checks["code_cells_5"]:
        score += PTS_CODE_CELLS

    checks["total_cells_15"] = n_total >= 15
    if checks["total_cells_15"]:
        score += PTS_TOTAL_CELLS

    checks["conclusion"] = _has_pattern(md_text, [
        "conclusion", "summary", "key takeaway", "final summary",
        "wrap up", "recap",
    ])
    if checks["conclusion"]:
        score += PTS_CONCLUSION

    # Empty code cells
    empty_code = 0
    for c in cells:
        if c.get("cell_type") == "code":
            src = c.get("source", "")
            if isinstance(src, list):
                src = "".join(src)
            stripped = src.strip()
            # Allow cells that are only comments or exercise placeholders
            if not stripped or stripped == "pass":
                empty_code += 1
    checks["no_empty_code"] = empty_code == 0
    if checks["no_empty_code"]:
        score += PTS_NO_EMPTY_CODE

    # ── Data Handling (25) ────────────────────────────────────────────────
    checks["imports"] = _has_pattern(code_text, [
        "import pandas", "import numpy", "import pyspark",
        "from sklearn", "import torch", "import tensorflow",
    ])
    if checks["imports"]:
        score += PTS_IMPORTS

    checks["data_load"] = _has_pattern(code_text, [
        "read_csv", "read_excel", "read_json", "read_parquet",
        "load_dataset", "spark.read", "pd.read_",
        "fetch_", "from_pandas", "load(",
    ])
    if checks["data_load"]:
        score += PTS_DATA_LOAD

    checks["dataset_source"] = _has_pattern(all_text, [
        "kaggle.com", "kagglehub", "huggingface.co", "github.com",
        "dataset source", "data source", "download the dataset",
        "yahoo finance", "yfinance", "uci.edu", "data.gov",
        "https://", "http://",
    ])
    if checks["dataset_source"]:
        score += PTS_DATASET_SOURCE

    checks["data_validation"] = _has_pattern(code_text, [
        "isnull()", "isna()", "info()", "dtypes", ".shape",
        "describe()", "nunique()", "value_counts",
        "printSchema", "count()",
    ])
    if checks["data_validation"]:
        score += PTS_DATA_VALIDATION

    checks["download_cell"] = _has_pattern(code_text, [
        "kaggle", "kagglehub", "wget", "urllib", "requests.get",
        "download", "curl", "gdown", "yfinance",
        "huggingface", "load_dataset",
    ])
    if checks["download_cell"]:
        score += PTS_DOWNLOAD_CELL

    # ── Analysis Quality (25) ────────────────────────────────────────────
    checks["visualization"] = _has_pattern(code_text, [
        "import matplotlib", "import seaborn", "import plotly",
        "from matplotlib", "plt.plot", "plt.show", "plt.figure",
        "sns.", "px.", ".plot(", ".hist(", ".bar(",
    ])
    if checks["visualization"]:
        score += PTS_VISUALIZATION

    checks["statistics"] = _has_pattern(code_text, [
        "describe()", "corr()", "value_counts", "mean()", "std()",
        "median()", "groupby", "agg(", "pivot_table",
        "chi2", "ttest", "shapiro", "pearsonr",
    ])
    if checks["statistics"]:
        score += PTS_STATISTICS

    # Count substantive code cells (>3 non-trivial lines)
    substantive = 0
    for c in cells:
        if c.get("cell_type") != "code":
            continue
        src = c.get("source", "")
        if isinstance(src, list):
            src = "".join(src)
        lines = [ln for ln in src.strip().splitlines()
                 if ln.strip() and not ln.strip().startswith("#")]
        if len(lines) >= 3:
            substantive += 1
    checks["multiple_analyses"] = substantive >= 3
    if checks["multiple_analyses"]:
        score += PTS_MULTIPLE_ANALYSES

    checks["md_code_ratio"] = (n_md / max(n_code, 1)) >= 0.5
    if checks["md_code_ratio"]:
        score += PTS_MD_CODE_RATIO

    # ── Educational Value (20) ────────────────────────────────────────────
    checks["methodology"] = _has_pattern(md_text, [
        "approach", "methodology", "strategy", "method",
        "why", "because", "reason", "explain",
        "we will", "we use", "the goal",
    ])
    if checks["methodology"]:
        score += PTS_METHODOLOGY

    checks["limitations"] = _has_pattern(md_text, [
        "limitation", "improvement", "future work", "next step",
        "could be improved", "extensions", "how to improve",
    ])
    if checks["limitations"]:
        score += PTS_LIMITATIONS

    checks["exercises"] = _has_pattern(md_text, [
        "exercise", "challenge", "try it", "your turn",
        "mini challenge", "practice",
    ])
    if checks["exercises"]:
        score += PTS_EXERCISES

    # Professional formatting: has ≥3 h2+ headings
    h2_count = len(re.findall(r"^#{2,}\s", md_text, re.M))
    checks["formatting"] = h2_count >= 3
    if checks["formatting"]:
        score += PTS_FORMATTING

    # ── Verdict ───────────────────────────────────────────────────────────
    if score >= PRODUCTION_READY:
        verdict = "PRODUCTION_READY"
    elif score >= NEEDS_IMPROVEMENT:
        verdict = "NEEDS_IMPROVEMENT"
    elif score >= WEAK:
        verdict = "WEAK"
    else:
        verdict = "FAILED"

    result["score"] = score
    result["verdict"] = verdict
    result["checks"] = checks
    return result


def discover_notebooks(base_dir: Path) -> list[Path]:
    """Find all .ipynb files in immediate subdirectories of base_dir."""
    notebooks = []
    if not base_dir.is_dir():
        return notebooks
    for project_dir in sorted(base_dir.iterdir()):
        if not project_dir.is_dir():
            continue
        for f in sorted(project_dir.iterdir()):
            if f.suffix == ".ipynb" and not f.name.startswith("."):
                notebooks.append(f)
    return notebooks


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    notebooks = discover_notebooks(DATA_ANALYSIS_DIR)
    if not notebooks:
        print("No notebooks found in Data Analysis/", file=sys.stderr)
        sys.exit(1)

    results = []
    for nb_path in notebooks:
        r = score_notebook(nb_path)
        results.append(r)

    # Sort ascending by score
    results.sort(key=lambda r: r["score"])

    # Compute summary stats
    scores = [r["score"] for r in results]
    total = len(results)
    passed = sum(1 for s in scores if s >= WEAK)
    failed = total - passed
    avg = round(sum(scores) / max(total, 1), 1)
    prod_ready = sum(1 for s in scores if s >= PRODUCTION_READY)
    needs_imp = sum(1 for s in scores if NEEDS_IMPROVEMENT <= s < PRODUCTION_READY)
    weak_cnt = sum(1 for s in scores if WEAK <= s < NEEDS_IMPROVEMENT)
    failed_cnt = sum(1 for s in scores if s < WEAK)

    summary = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S"),
        "total_notebooks": total,
        "passed": passed,
        "failed": failed,
        "pass_rate_pct": round(100 * passed / max(total, 1), 1),
        "avg_score": avg,
        "production_ready": prod_ready,
        "needs_improvement": needs_imp,
        "weak": weak_cnt,
        "failed_cnt": failed_cnt,
        "ranking_asc": [
            {"notebook": r["notebook"], "score": r["score"], "verdict": r["verdict"]}
            for r in results
        ],
    }

    out_path = REPORTS_DIR / "summary.json"
    out_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    # Print human-readable table
    print(f"\n{'='*90}")
    print(f"  ML AUDIT — {total} notebooks scored")
    print(f"{'='*90}")
    print(f"{'#':>3}  {'Score':>5}  {'Verdict':<20}  Notebook")
    print(f"{'-'*90}")
    for i, r in enumerate(results, 1):
        print(f"{i:>3}  {r['score']:>5}  {r['verdict']:<20}  {r['notebook']}")
    print(f"{'-'*90}")
    print(f"  Pass: {passed}/{total}  Fail: {failed}  Avg: {avg}")
    print(f"  Production-ready: {prod_ready}  Needs improvement: {needs_imp}")
    print(f"  Weak: {weak_cnt}  Failed: {failed_cnt}")
    print(f"{'='*90}")
    print(f"\nReport: {out_path}")

    # Exit non-zero if hard failures exist
    hard_fails = [r for r in results if r["score"] < WEAK]
    if hard_fails:
        print(f"\n{len(hard_fails)} HARD FAILURE(S) (score < {WEAK}):")
        for r in hard_fails:
            print(f"  {r['notebook']} → {r['score']}")
        sys.exit(1)


if __name__ == "__main__":
    main()
