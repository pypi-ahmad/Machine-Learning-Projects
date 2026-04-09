#!/usr/bin/env python
"""Phase 1 — Patch notebooks + discover datasets + generate reports.

Operations (all idempotent):
  1. Discover / prepare datasets for all 21 projects.
  2. Patch every code.ipynb with Phase-1 cells (data-load + EDA + baseline).
  3. Generate reports:
       reports/dataset_map.json
       reports/dataset_links.md
       reports/phase1_summary.md
       reports/phase1_failures.md

Run from the workspace root:
    .venv\\Scripts\\python.exe scripts/phase1_patch_notebooks.py
"""

from __future__ import annotations

import json
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
WORKSPACE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKSPACE))
# ---------------------------------------------------------------------------

from utils.dataset_finder import (
    PROJECT_REGISTRY,
    generate_dataset_links_report,
    generate_dataset_map,
    prepare_all_projects,
)
from utils.logger import get_logger

logger = get_logger(__name__)

PHASE1_MARKER = "phase 1: data + baselines"
BOOTSTRAP_MARKER = "workspace bootstrap"
REPORTS = WORKSPACE / "reports"

# ======================================================================
# Cell builders
# ======================================================================

def _code_cell(src: list[str]) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": src,
    }


def _md_cell(src: list[str]) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": src}


# ======================================================================
# Project-specific LOAD code
# ======================================================================

def _load_lines(slug: str, info: dict) -> list[str]:
    """Return source lines for the 'Load data + EDA' cell."""
    L: list[str] = ["# Phase 1 · Load data + quick EDA\n"]

    if slug == "fake-news-detection":
        L += [
            "import pandas as pd\n",
            "from utils.data_io import load_csv\n",
            "\n",
            '_p1_true = load_csv(_p1_project_dir / "True.csv"); _p1_true["label"] = 1\n',
            '_p1_fake = load_csv(_p1_project_dir / "Fake.csv"); _p1_fake["label"] = 0\n',
            "_p1_df = pd.concat([_p1_true, _p1_fake], ignore_index=True)"
            ".sample(frac=1, random_state=42).reset_index(drop=True)\n",
        ]
    elif slug == "review-classification":
        L += [
            "import pandas as pd\n",
            "from utils.data_io import load_labeled_text\n",
            "\n",
            "_p1_df = pd.concat([\n",
            '    load_labeled_text(_p1_project_dir / "amazon.txt"),\n',
            '    load_labeled_text(_p1_project_dir / "imdb.txt"),\n',
            '    load_labeled_text(_p1_project_dir / "yelp.txt"),\n',
            "], ignore_index=True)\n",
        ]
    elif slug == "e-commerce-product-classification":
        L += [
            "from utils.data_io import load_csv\n",
            "\n",
            "_p1_df = load_csv(\n",
            '    _p1_project_dir / "data.csv",\n',
            '    header=None, names=["category", "description"],\n',
            ")\n",
        ]
    elif slug == "news-headline-classification":
        L += [
            "from utils.data_io import load_json\n",
            "\n",
            '_p1_df = load_json(_p1_project_dir / "data.json")\n',
        ]
    elif slug == "paper-subject-prediction":
        L += [
            "from utils.data_io import load_csv\n",
            "\n",
            '_p1_df = load_csv(_p1_project_dir / "arxiv_data.csv")\n',
        ]
    elif slug in ("toxic-comment-classification", "twitter-sentiment-analysis"):
        L += [
            "from utils.data_io import load_csv\n",
            "\n",
            '_p1_df = load_csv(_p1_project_dir / "train.csv")\n',
        ]
    elif slug == "english-to-french-translation":
        L += [
            "import pandas as pd\n",
            "\n",
            "_p1_raw = (_p1_project_dir / 'eng-fra.txt').read_text(encoding='utf-8').strip()\n",
            "_p1_pairs = [l.split('\\t')[:2] for l in _p1_raw.split('\\n')]\n",
            '_p1_df = pd.DataFrame(_p1_pairs, columns=["english", "french"])\n',
        ]
    elif slug == "automated-image-captioning":
        L += [
            "# Captioning — locate captions + images\n",
            '_p1_captions_path = _p1_project_dir / "captions.txt"\n',
            '_p1_images_dir   = _p1_project_dir / "images"\n',
            'print(f"Captions file exists: {_p1_captions_path.exists()}")\n',
            'print(f"Images dir exists:    {_p1_images_dir.exists()}")\n',
        ]
        return L  # no DataFrame, skip generic tail
    elif slug == "name-generate-from-languages":
        L += [
            "# Name generation — list language files\n",
            '_p1_names_dir = _p1_project_dir / "names"\n',
            'print(f"Names dir exists: {_p1_names_dir.exists()}")\n',
            "if _p1_names_dir.exists():\n",
            '    _p1_lang_files = sorted(_p1_names_dir.glob("*.txt"))\n',
            '    print(f"Languages: {len(_p1_lang_files)}")\n',
            "    for _f in _p1_lang_files[:5]:\n",
            '        print(f"  {_f.stem}: {sum(1 for _ in open(_f, encoding=\'utf-8\'))} names")\n',
        ]
        return L
    elif slug == "bbc-articles-summarization":
        L += [
            "# BBC Summarization — data may need Kaggle download\n",
            "import os\n",
            'if _p1_ds["status"] == "READY":\n',
            "    _p1_raw = Path(_p1_ds['raw_dir'])\n",
            "    print('Files in raw dir:', os.listdir(_p1_raw)[:20])\n",
            "else:\n",
            '    print("\\u26a0 Data not available. Configure Kaggle API or download manually.")\n',
        ]
        return L
    elif slug == "newsgroups-posts-clustering":
        L += [
            "import pandas as pd\n",
            'if _p1_ds["status"] == "READY":\n',
            "    from utils.data_io import load_csv\n",
            "    _p1_df = load_csv(Path(_p1_ds['raw_dir']) / 'newsgroups.csv')\n",
            "else:\n",
            "    from sklearn.datasets import fetch_20newsgroups\n",
            '    _ng = fetch_20newsgroups(subset="all", remove=("headers","footers","quotes"))\n',
            "    _p1_df = pd.DataFrame({'text': _ng.data, "
            "'newsgroup': [_ng.target_names[t] for t in _ng.target]})\n",
        ]
    elif slug == "stories-clustering":
        L += [
            "import os, pandas as pd\n",
            'if _p1_ds["status"] == "READY":\n',
            "    from utils.data_io import load_csv\n",
            "    _p1_raw = Path(_p1_ds['raw_dir'])\n",
            "    _p1_csvs = [f for f in os.listdir(_p1_raw) if f.endswith('.csv')]\n",
            "    _p1_df = load_csv(_p1_raw / _p1_csvs[0]) if _p1_csvs else None\n",
            "else:\n",
            '    print("\\u26a0 Data not available. Download from Kaggle.")\n',
            '    _p1_df = None\n',
        ]
    elif slug == "kaggle-survey-questions-clustering":
        L += [
            "from utils.data_io import load_csv\n",
            "\n",
            '_p1_df = load_csv(_p1_project_dir / "questions.csv")\n',
        ]
    elif slug == "medium-articles-clustering":
        L += [
            "from utils.data_io import load_csv\n",
            "\n",
            '_p1_df = load_csv(_p1_project_dir / "articles.csv")\n',
        ]
    else:
        # Default: first data_file listed in registry
        first = (info.get("data_files") or ["data.csv"])[0]
        L += [
            "from utils.data_io import load_csv\n",
            "\n",
            f'_p1_df = load_csv(_p1_project_dir / "{first}")\n',
        ]

    # Generic tail: print shape + head (only for projects with a DataFrame)
    L += [
        "\n",
        'print(f"Shape: {_p1_df.shape}")\n',
        "_p1_df.head()\n",
    ]
    return L


# ======================================================================
# Project-specific BASELINE code
# ======================================================================

def _baseline_lines(slug: str, info: dict) -> list[str]:
    task = info["task"]
    text_col = info.get("text_col")
    target_col = info.get("target_col")

    L: list[str] = ["# Phase 1 · Run baseline\n"]

    # -- Analysis / EDA only ------------------------------------------------
    if task == "analysis":
        L += [
            "from utils.baselines import run_eda\n",
            "\n",
            f"_p1_eda = run_eda(_p1_df, PROJECT_SLUG, text_col={_q(text_col)})\n",
            'print(f"EDA complete — shape {_p1_eda[\'shape\']}")\n',
        ]

    # -- Single-label classification ----------------------------------------
    elif task == "classification":
        tc, tg = _text_target(slug, info)
        L += [
            "from utils.baselines import run_eda, run_classification_baselines\n",
            "\n",
            f"_p1_eda = run_eda(_p1_df, PROJECT_SLUG, text_col={tc})\n",
            f"_p1_bl  = run_classification_baselines(_p1_df, {tc}, {tg}, PROJECT_SLUG)\n",
            "\n",
            'print("\\n--- Classification Baseline Results ---")\n',
            "if 'logreg_metrics' in _p1_bl:\n",
            "    _m = _p1_bl['logreg_metrics']\n",
            "    print(f\"  LogReg  acc={_m['accuracy']:.4f}  f1={_m['f1_weighted']:.4f}\")\n",
            "if 'svm_metrics' in _p1_bl:\n",
            "    _m = _p1_bl['svm_metrics']\n",
            "    print(f\"  SVM     acc={_m['accuracy']:.4f}  f1={_m['f1_weighted']:.4f}\")\n",
        ]

    # -- Multi-label classification -----------------------------------------
    elif task == "classification_multilabel":
        L += [
            "from utils.baselines import run_eda, run_multilabel_classification\n",
            "\n",
            f"_p1_eda = run_eda(_p1_df, PROJECT_SLUG, text_col={_q(text_col)})\n",
            f"_p1_bl  = run_multilabel_classification(\n",
            f"    _p1_df, {_q(text_col)}, {repr(target_col)}, PROJECT_SLUG,\n",
            ")\n",
            "\n",
            'print("\\n--- Multi-label Baseline ---")\n',
            "if 'metrics' in _p1_bl:\n",
            "    _m = _p1_bl['metrics']\n",
            "    print(f\"  F1  micro={_m.get('f1_micro',0):.4f}  "
            "macro={_m.get('f1_macro',0):.4f}\")\n",
        ]

    # -- Summarization ------------------------------------------------------
    elif task == "summarization":
        L += [
            "from utils.baselines import run_summarization_baseline\n",
            "\n",
            'if _p1_ds["status"] == "READY":\n',
            "    import os\n",
            "    _p1_raw = Path(_p1_ds['raw_dir'])\n",
            "    _p1_articles = []\n",
            "    for root, dirs, files in os.walk(_p1_raw):\n",
            "        for f in files:\n",
            "            if f.endswith('.txt'):\n",
            "                _p1_articles.append("
            "open(os.path.join(root, f), encoding='utf-8', errors='ignore').read())\n",
            '    print(f"Loaded {len(_p1_articles)} articles")\n',
            "    if _p1_articles:\n",
            "        _p1_bl = run_summarization_baseline(_p1_articles[:100], slug=PROJECT_SLUG)\n",
            "else:\n",
            '    print("\\u26a0 Data not available for baseline.")\n',
        ]

    # -- Translation --------------------------------------------------------
    elif task == "translation":
        L += [
            "from utils.baselines import run_translation_baseline\n",
            "\n",
            "_p1_bl = run_translation_baseline(\n",
            '    _p1_df["english"].tolist(), _p1_df["french"].tolist(),\n',
            "    slug=PROJECT_SLUG,\n",
            ")\n",
            'print("\\n--- Translation Baseline ---")\n',
            "if 'metrics' in _p1_bl:\n",
            "    _m = _p1_bl['metrics']\n",
            "    print(f\"  BLEU={_m.get('bleu',0):.4f}  chrF={_m.get('chrf',0):.2f}\")\n",
        ]

    # -- Image captioning ---------------------------------------------------
    elif task == "image_captioning":
        L += [
            "from utils.baselines import run_captioning_baseline\n",
            "\n",
            "_p1_bl = run_captioning_baseline(_p1_captions_path, _p1_images_dir, slug=PROJECT_SLUG)\n",
            'print(f"Captions: {_p1_bl.get(\'n_captions\',0)}  '
            'Images: {_p1_bl.get(\'n_image_files\',0)}")\n',
        ]

    # -- Name generation ----------------------------------------------------
    elif task == "generation":
        L += [
            "from utils.baselines import run_generation_baseline\n",
            "\n",
            "_p1_bl = run_generation_baseline(_p1_names_dir, slug=PROJECT_SLUG)\n",
            'print(f"Languages: {_p1_bl.get(\'n_languages\',0)}  '
            'Total names: {_p1_bl.get(\'total_names\',0)}")\n',
        ]

    # -- Clustering ---------------------------------------------------------
    elif task == "clustering":
        if text_col:
            L += [
                "from utils.baselines import run_clustering_baseline, run_eda\n",
                "\n",
                f"_p1_eda = run_eda(_p1_df, PROJECT_SLUG, text_col={_q(text_col)})\n",
                f"_p1_bl  = run_clustering_baseline(_p1_df[{_q(text_col)}], slug=PROJECT_SLUG)\n",
                "\n",
                'print("\\n--- Clustering Baseline ---")\n',
                "if 'kmeans_metrics' in _p1_bl:\n",
                "    print(f\"  Silhouette: {_p1_bl['kmeans_metrics'].get('silhouette',0):.4f}\")\n",
            ]
        else:
            # Must auto-detect text column at runtime
            L += [
                "from utils.baselines import run_clustering_baseline, run_eda\n",
                "\n",
                "# Auto-detect longest-string column as text\n",
                "_p1_text_col = None\n",
                "for _c in _p1_df.select_dtypes('object').columns:\n",
                "    if _p1_df[_c].dropna().str.len().mean() > 20:\n",
                "        _p1_text_col = _c; break\n",
                "\n",
                "if _p1_text_col:\n",
                "    _p1_eda = run_eda(_p1_df, PROJECT_SLUG, text_col=_p1_text_col)\n",
                "    _p1_bl  = run_clustering_baseline(_p1_df[_p1_text_col], slug=PROJECT_SLUG)\n",
                "else:\n",
                '    print("\\u26a0 No suitable text column found for clustering.")\n',
                "    _p1_eda = run_eda(_p1_df, PROJECT_SLUG)\n",
            ]
    else:
        L += ['print(f"\\u26a0 No baseline runner for task type: {_p1_info[\'task\']}")\n']

    return L


# ---- helpers for quoting column names ------------------------------------

def _q(val):
    """repr() a value — for embedding in generated source."""
    return repr(val)


def _text_target(slug: str, info: dict) -> tuple[str, str]:
    """Return (text_col_repr, target_col_repr) for a classification project."""
    overrides = {
        "e-commerce-product-classification": ('"description"', '"category"'),
        "fake-news-detection": ('"text"', '"label"'),
        "review-classification": ('"text"', '"label"'),
    }
    if slug in overrides:
        return overrides[slug]
    return _q(info.get("text_col")), _q(info.get("target_col"))


# ======================================================================
# Full cell bundle per project
# ======================================================================

def generate_phase1_cells(slug: str) -> list[dict]:
    info = PROJECT_REGISTRY[slug]
    task = info["task"]

    cells: list[dict] = []

    # 0 — Markdown header
    cells.append(_md_cell([
        "## Phase 1 — Automated Data & Baseline Pipeline\n",
        "\n",
        f"> Auto-generated for **{slug}** (task: `{task}`).  \n",
        "> Re-running the patching script will NOT duplicate these cells.\n",
    ]))

    # 1 — Setup + data preparation
    cells.append(_code_cell([
        f"# --- {PHASE1_MARKER} (auto-generated) ---\n",
        "from pathlib import Path\n",
        "from utils.dataset_finder import prepare_project_data, PROJECT_REGISTRY\n",
        "\n",
        f'PROJECT_SLUG = "{slug}"\n',
        "_p1_info       = PROJECT_REGISTRY[PROJECT_SLUG]\n",
        '_p1_project_dir = Path(_ws_root) / _p1_info["dir"]\n',
        "\n",
        "_p1_ds = prepare_project_data(_p1_project_dir, PROJECT_SLUG)\n",
        "print(f\"Dataset: {PROJECT_SLUG}\")\n",
        "print(f\"  status : {_p1_ds['status']}\")\n",
        "print(f\"  source : {_p1_ds.get('chosen_source','N/A')}\")\n",
        "print(f\"  files  : {_p1_ds.get('raw_paths',[])}\")\n",
    ]))

    # 2 — Load data + EDA
    cells.append(_code_cell(_load_lines(slug, info)))

    # 3 — Baseline + metrics
    cells.append(_code_cell(_baseline_lines(slug, info)))

    return cells


# ======================================================================
# Notebook patcher
# ======================================================================

def _slug_from_path(nb_path: Path) -> str | None:
    """Map a notebook path to its project slug via the registry."""
    rel = nb_path.parent.relative_to(WORKSPACE).as_posix()
    for slug, info in PROJECT_REGISTRY.items():
        if info["dir"].replace("\\", "/") == rel:
            return slug
    return None


def patch_notebook(nb_path: Path) -> tuple[str, list[str]]:
    """Patch a single notebook. Returns (slug, list_of_changes)."""
    changes: list[str] = []
    slug = _slug_from_path(nb_path)
    if slug is None:
        return ("?", ["SKIP — not in registry"])

    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    cells = nb.get("cells", [])

    # --- idempotency check ------------------------------------------------
    for cell in cells:
        src = "".join(cell.get("source", []))
        if PHASE1_MARKER in src:
            return (slug, ["already patched"])

    # --- find bootstrap cell index ----------------------------------------
    bootstrap_idx: int | None = None
    for i, cell in enumerate(cells):
        if cell.get("cell_type") == "code":
            src = "".join(cell.get("source", []))
            if BOOTSTRAP_MARKER in src:
                bootstrap_idx = i
                break

    if bootstrap_idx is None:
        return (slug, ["WARN — bootstrap cell not found; inserting at position 1"])

    insert_at = (bootstrap_idx + 1) if bootstrap_idx is not None else 1

    # --- generate & insert cells ------------------------------------------
    new_cells = generate_phase1_cells(slug)
    for j, c in enumerate(reversed(new_cells)):
        cells.insert(insert_at, c)
    changes.append(f"{len(new_cells)} Phase-1 cells inserted after pos {insert_at - 1}")

    # --- write back -------------------------------------------------------
    nb["cells"] = cells
    with open(nb_path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
        f.write("\n")

    return (slug, changes)


# ======================================================================
# Report generators
# ======================================================================

def _generate_phase1_summary(ds_results: dict[str, dict], patch_log: dict) -> Path:
    lines = [
        "# Phase 1 — Summary Report",
        "",
        f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        "",
        f"**Projects**: {len(PROJECT_REGISTRY)}  ",
        f"**Datasets ready**: "
        f"{sum(1 for r in ds_results.values() if r.get('status') == 'READY')} / {len(ds_results)}  ",
        f"**Notebooks patched**: "
        f"{sum(1 for v in patch_log.values() if 'already patched' not in str(v) and 'SKIP' not in str(v))} "
        f"(+ {sum(1 for v in patch_log.values() if 'already patched' in str(v))} already done)",
        "",
        "---",
        "",
        "| # | Project Slug | Task | Dataset Status | Source | Text Col | Target Col | Baseline Ready |",
        "|---|---|---|---|---|---|---|---|",
    ]

    for i, (slug, info) in enumerate(sorted(PROJECT_REGISTRY.items()), 1):
        ds = ds_results.get(slug, {})
        status = ds.get("status", "UNKNOWN")
        source = ds.get("chosen_source", "-")[:40]
        tc = info.get("text_col") or "-"
        if isinstance(tc, list):
            tc = ", ".join(tc[:2]) + "..."
        tg = info.get("target_col") or "-"
        if isinstance(tg, list):
            tg = ", ".join(tg[:2]) + "..."
        bl = "yes" if status == "READY" else "no"
        lines.append(
            f"| {i} | `{slug}` | {info['task']} | {status} | {source} | `{tc}` | `{tg}` | {bl} |"
        )

    lines += [
        "",
        "---",
        "",
        "## Per-Project Details",
        "",
    ]

    for slug in sorted(PROJECT_REGISTRY):
        info = PROJECT_REGISTRY[slug]
        ds = ds_results.get(slug, {})
        lines.append(f"### {slug}")
        lines.append("")
        lines.append(f"- **Task:** {info['task']}")
        lines.append(f"- **Directory:** `{info['dir']}`")
        lines.append(f"- **Dataset status:** {ds.get('status', 'UNKNOWN')}")
        lines.append(f"- **Source:** {ds.get('chosen_source', 'N/A')}")
        lines.append(f"- **Files:** {ds.get('raw_paths', [])}")
        urls = info.get("official_urls", [])
        if urls:
            lines.append("- **Official links:**")
            for u in urls:
                lines.append(f"  - {u}")
        lines.append(f"- **License:** {info.get('license', 'Unknown')}")
        patch = patch_log.get(slug, [])
        lines.append(f"- **Notebook patch:** {'; '.join(patch) if patch else 'N/A'}")
        if info.get("note"):
            lines.append(f"- **Note:** {info['note']}")
        lines.append("")

    out = REPORTS / "phase1_summary.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def _generate_phase1_failures(ds_results: dict[str, dict], patch_log: dict) -> Path:
    lines = [
        "# Phase 1 — Failures & Warnings",
        "",
        f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        "",
    ]
    has_issues = False

    for slug in sorted(PROJECT_REGISTRY):
        ds = ds_results.get(slug, {})
        issues: list[str] = []
        if ds.get("status") not in ("READY",):
            issues.append(f"Dataset status: {ds.get('status', 'UNKNOWN')}")
        if ds.get("error"):
            issues.append(f"Error: {ds['error']}")
        patch = patch_log.get(slug, [])
        for p in patch:
            if "SKIP" in p or "WARN" in p:
                issues.append(f"Patch: {p}")

        if issues:
            has_issues = True
            lines.append(f"## {slug}")
            for iss in issues:
                lines.append(f"- {iss}")
            lines.append("")

    if not has_issues:
        lines.append("**No failures or warnings — all clear!**")

    out = REPORTS / "phase1_failures.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


# ======================================================================
# Main
# ======================================================================

def main() -> None:
    print("=" * 70)
    print("  Phase 1: Dataset Discovery + Notebook Patching + Reports")
    print("=" * 70)
    print()

    # --- 1. Discover / prepare datasets -----------------------------------
    print("[1/4] Discovering and preparing datasets ...")
    ds_results = prepare_all_projects()
    ready = sum(1 for r in ds_results.values() if r.get("status") == "READY")
    print(f"       {ready}/{len(ds_results)} datasets READY\n")

    # --- 2. Generate dataset reports --------------------------------------
    print("[2/4] Generating dataset reports ...")
    map_path = generate_dataset_map(ds_results)
    print(f"       -> {map_path.relative_to(WORKSPACE)}")
    links_path = generate_dataset_links_report()
    print(f"       -> {links_path.relative_to(WORKSPACE)}\n")

    # --- 3. Patch notebooks -----------------------------------------------
    print("[3/4] Patching notebooks ...")
    notebooks = sorted(WORKSPACE.rglob("code.ipynb"))
    print(f"       Found {len(notebooks)} notebooks\n")

    patch_log: dict[str, list[str]] = {}
    for nb in notebooks:
        rel = nb.relative_to(WORKSPACE)
        try:
            slug, changes = patch_notebook(nb)
            patch_log[slug] = changes
            status = changes[0] if changes else "no changes"
            print(f"  {'OK' if 'already' in status else 'PATCHED':>8}  {slug:<45}  {status}")
        except Exception as exc:
            print(f"  {'ERROR':>8}  {rel}  {exc}")
            patch_log[str(rel)] = [f"ERROR: {exc}"]
            traceback.print_exc()

    print()

    # --- 4. Summary + failures reports ------------------------------------
    print("[4/4] Generating summary reports ...")
    summary_path = _generate_phase1_summary(ds_results, patch_log)
    print(f"       -> {summary_path.relative_to(WORKSPACE)}")
    failures_path = _generate_phase1_failures(ds_results, patch_log)
    print(f"       -> {failures_path.relative_to(WORKSPACE)}")

    print()
    print("=" * 70)
    print("  Phase 1 complete.")
    print(f"  Datasets ready : {ready}/{len(ds_results)}")
    print(f"  Notebooks      : {len(notebooks)}")
    print(f"  Reports dir    : reports/")
    print("=" * 70)


if __name__ == "__main__":
    main()
