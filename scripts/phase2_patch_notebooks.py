#!/usr/bin/env python
"""Phase 2 -- Patch notebooks with model-training cells + generate reports.

Operations (idempotent):
  1. Insert Phase 2 cells (training + evaluation) into each code.ipynb.
  2. Generate reports/phase2_summary.md and reports/phase2_failures.md.

Run from workspace root:
    .venv\\Scripts\\python.exe scripts/phase2_patch_notebooks.py
"""

from __future__ import annotations

import json
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKSPACE))

from utils.dataset_finder import PROJECT_REGISTRY
from utils.logger import get_logger

logger = get_logger(__name__)

PHASE2_MARKER = "phase 2: model training"
PHASE1_MARKER = "phase 1: data + baselines"
REPORTS = WORKSPACE / "reports"

# ======================================================================
# Phase 2 configuration per project
# ======================================================================

PHASE2_CONFIG: dict[str, dict] = {
    # -- Analysis projects (with usable target -> classify) -----------------
    "e-commerce-clothing-reviews": {
        "action": "classify",
        "text_col": "Review Text",
        "target_col": "Recommended IND",
        "model": "microsoft/deberta-v3-base",
        "max_length": 256,
        "notes": "Binary: recommended vs not recommended",
    },
    "trip-advisor-hotel-reviews": {
        "action": "classify",
        "text_col": "Review",
        "target_col": "Rating",
        "model": "microsoft/deberta-v3-base",
        "max_length": 256,
        "notes": "5-class rating prediction",
    },
    "world-war-i-letters": {
        "action": "embed_cluster",
        "text_col": "_auto",
        "notes": "No classification target; embeddings + clustering + topics",
    },
    # -- Classification projects --------------------------------------------
    "cyberbullying-classification": {
        "action": "classify",
        "text_col": "tweet_text",
        "target_col": "cyberbullying_type",
        "model": "microsoft/deberta-v3-base",
        "max_length": 256,
    },
    "e-commerce-product-classification": {
        "action": "classify",
        "text_col": "description",
        "target_col": "category",
        "model": "microsoft/deberta-v3-base",
        "max_length": 256,
        "notes": "CSV headerless: col0=category, col1=description (renamed in P1)",
    },
    "economic-news-articles": {
        "action": "classify",
        "text_col": "text",
        "target_col": "positivity",
        "model": "microsoft/deberta-v3-base",
        "max_length": 256,
    },
    "fake-news-detection": {
        "action": "classify",
        "text_col": "text",
        "target_col": "label",
        "model": "microsoft/deberta-v3-base",
        "max_length": 512,
        "notes": "Merged True.csv + Fake.csv in Phase 1",
    },
    "news-headline-classification": {
        "action": "classify",
        "text_col": "headline",
        "target_col": "category",
        "model": "microsoft/deberta-v3-base",
        "max_length": 128,
        "notes": "~42 HuffPost categories",
    },
    "paper-subject-prediction": {
        "action": "classify",
        "text_col": "summaries",
        "target_col": "terms",
        "model": "microsoft/deberta-v3-base",
        "max_length": 384,
        "notes": "arXiv subject terms; treating first term as primary label",
    },
    "review-classification": {
        "action": "classify",
        "text_col": "text",
        "target_col": "label",
        "model": "microsoft/deberta-v3-base",
        "max_length": 256,
        "notes": "Merged Amazon + IMDb + Yelp",
    },
    "spam-message-detection": {
        "action": "classify",
        "text_col": "text",
        "target_col": "label",
        "model": "microsoft/deberta-v3-base",
        "max_length": 128,
    },
    "toxic-comment-classification": {
        "action": "classify_multilabel",
        "text_col": "comment_text",
        "target_cols": ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"],
        "model": "microsoft/deberta-v3-base",
        "max_length": 256,
    },
    "twitter-sentiment-analysis": {
        "action": "classify",
        "text_col": "OriginalTweet",
        "target_col": "Sentiment",
        "model": "microsoft/deberta-v3-base",
        "max_length": 128,
        "notes": "COVID-related tweets, 5 sentiment classes",
    },
    # -- Generation projects -----------------------------------------------
    "automated-image-captioning": {
        "action": "caption",
        "model": "Salesforce/blip-image-captioning-large",
    },
    "bbc-articles-summarization": {
        "action": "summarize",
        "model": "facebook/bart-large-cnn",
        "notes": "LoRA fine-tune; data from Kaggle raw dir",
    },
    "english-to-french-translation": {
        "action": "translate",
        "model": "facebook/nllb-200-distilled-600M",
    },
    "name-generate-from-languages": {
        "action": "char_rnn",
        "notes": "Custom LSTM char-level generator",
    },
    # -- Clustering / Topic Modelling --------------------------------------
    "kaggle-survey-questions-clustering": {
        "action": "embed_cluster",
        "text_col": "_auto",
        "notes": "Questions / survey text; auto-detect text column",
    },
    "medium-articles-clustering": {
        "action": "embed_cluster",
        "text_col": "text",
    },
    "newsgroups-posts-clustering": {
        "action": "embed_cluster",
        "text_col": "text",
    },
    "stories-clustering": {
        "action": "embed_cluster",
        "text_col": "_auto",
        "notes": "Auto-detect text column from downloaded CSV",
    },
}


# ======================================================================
# Cell helpers
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


def _q(s):
    """Quote a string for code generation."""
    if s is None:
        return "None"
    return repr(s)


# ======================================================================
# Cell generators per action type
# ======================================================================

_RESULTS_FOOTER = [
    "\n",
    "# Print summary\n",
    "print(f\"\\n{'='*60}\")\n",
    "print(f\"Phase 2 Results: {PROJECT_SLUG}\")\n",
    "print(f\"{'='*60}\")\n",
    "print(f\"  Status: {_p2_results.get('status', 'UNKNOWN')}\")\n",
]

_CLASSIFY_FOOTER = [
    *_RESULTS_FOOTER,
    "if 'test_metrics' in _p2_results:\n",
    "    _m = _p2_results['test_metrics']\n",
    "    print(f\"  Accuracy:     {_m.get('accuracy', 0):.4f}\")\n",
    "    print(f\"  F1 weighted:  {_m.get('f1_weighted', 0):.4f}\")\n",
    "    print(f\"  F1 macro:     {_m.get('f1_macro', 0):.4f}\")\n",
    "if 'kfold_summary' in _p2_results:\n",
    "    _kf = _p2_results['kfold_summary']\n",
    "    print(f\"  K-Fold F1:    {_kf['mean_f1']:.4f} +/- {_kf['std_f1']:.4f}\")\n",
]

_MULTILABEL_FOOTER = [
    *_RESULTS_FOOTER,
    "if 'test_metrics' in _p2_results:\n",
    "    _m = _p2_results['test_metrics']\n",
    "    print(f\"  F1 micro:    {_m.get('f1_micro', 0):.4f}\")\n",
    "    print(f\"  F1 macro:    {_m.get('f1_macro', 0):.4f}\")\n",
    "    if 'per_label' in _m:\n",
    "        for _lbl, _f1 in _m['per_label'].items():\n",
    "            print(f\"    {_lbl}: {_f1:.4f}\")\n",
]


def _classify_cell(slug: str, cfg: dict) -> list[str]:
    model = cfg.get("model", "microsoft/deberta-v3-base")
    text_col = cfg["text_col"]
    target_col = cfg["target_col"]
    ml = cfg.get("max_length", 256)
    return [
        f"# --- {PHASE2_MARKER} (auto-generated) ---\n",
        f"# DeBERTa text classifier for {slug}\n",
        "from utils.train_text_classifier import train_text_classifier\n",
        "\n",
        "_p2_results = train_text_classifier(\n",
        f"    slug=PROJECT_SLUG,\n",
        f"    df=_p1_df,\n",
        f"    text_col={_q(text_col)},\n",
        f"    target_col={_q(target_col)},\n",
        f"    model_name={_q(model)},\n",
        f"    max_length={ml},\n",
        ")\n",
        *_CLASSIFY_FOOTER,
    ]


def _multilabel_cell(slug: str, cfg: dict) -> list[str]:
    model = cfg.get("model", "microsoft/deberta-v3-base")
    text_col = cfg["text_col"]
    target_cols = cfg["target_cols"]
    ml = cfg.get("max_length", 256)
    return [
        f"# --- {PHASE2_MARKER} (auto-generated) ---\n",
        f"# DeBERTa multi-label classifier for {slug}\n",
        "from utils.train_text_classifier import train_multilabel_classifier\n",
        "\n",
        "_p2_results = train_multilabel_classifier(\n",
        f"    slug=PROJECT_SLUG,\n",
        f"    df=_p1_df,\n",
        f"    text_col={_q(text_col)},\n",
        f"    target_cols={repr(target_cols)},\n",
        f"    model_name={_q(model)},\n",
        f"    max_length={ml},\n",
        ")\n",
        *_MULTILABEL_FOOTER,
    ]


def _summarize_cell(slug: str, cfg: dict) -> list[str]:
    model = cfg.get("model", "facebook/bart-large-cnn")
    return [
        f"# --- {PHASE2_MARKER} (auto-generated) ---\n",
        f"# BART-large-CNN summarizer for {slug}\n",
        "from utils.train_summarizer import train_summarizer\n",
        "\n",
        "_p2_results = train_summarizer(\n",
        f"    slug=PROJECT_SLUG,\n",
        f"    model_name={_q(model)},\n",
        ")\n",
        *_RESULTS_FOOTER,
        "if 'test_metrics' in _p2_results:\n",
        "    _m = _p2_results['test_metrics']\n",
        "    print(f\"  ROUGE-1: {_m.get('rouge1', 0):.4f}\")\n",
        "    print(f\"  ROUGE-2: {_m.get('rouge2', 0):.4f}\")\n",
        "    print(f\"  ROUGE-L: {_m.get('rougeL', 0):.4f}\")\n",
        "if 'sample_predictions' in _p2_results:\n",
        "    for _s in _p2_results['sample_predictions'][:2]:\n",
        "        print(f\"\\n  Ref:  {_s['reference'][:120]}...\")\n",
        "        print(f\"  Gen:  {_s['generated'][:120]}...\")\n",
    ]


def _translate_cell(slug: str, cfg: dict) -> list[str]:
    model = cfg.get("model", "facebook/nllb-200-distilled-600M")
    return [
        f"# --- {PHASE2_MARKER} (auto-generated) ---\n",
        f"# NLLB-200 EN->FR translator for {slug}\n",
        "from utils.train_translator import train_translator\n",
        "\n",
        "_p2_results = train_translator(\n",
        f"    slug=PROJECT_SLUG,\n",
        f"    model_name={_q(model)},\n",
        ")\n",
        *_RESULTS_FOOTER,
        "if 'test_metrics' in _p2_results:\n",
        "    _m = _p2_results['test_metrics']\n",
        "    print(f\"  BLEU:  {_m.get('bleu', 0):.2f}\")\n",
        "    print(f\"  chrF:  {_m.get('chrf', 0):.2f}\")\n",
        "if 'sample_translations' in _p2_results:\n",
        "    for _s in _p2_results['sample_translations'][:3]:\n",
        "        print(f\"\\n  EN:  {_s['source'][:80]}\")\n",
        "        print(f\"  FR:  {_s['reference'][:80]}\")\n",
        "        print(f\"  ->:  {_s['predicted'][:80]}\")\n",
    ]


def _embed_cluster_cell(slug: str, cfg: dict) -> list[str]:
    text_col = cfg.get("text_col", "_auto")
    # For auto-detect, build runtime logic
    if text_col == "_auto":
        text_src = [
            "# Auto-detect the best text column\n",
            "_p2_text_col = None\n",
            "for _c in _p1_df.columns:\n",
            "    if _p1_df[_c].dtype == object and _p1_df[_c].str.len().mean() > 20:\n",
            "        _p2_text_col = _c\n",
            "        break\n",
            "if _p2_text_col is None:\n",
            "    _p2_text_col = _p1_df.select_dtypes(include='object').columns[0]\n",
            "print(f'Using text column: {_p2_text_col}')\n",
            "_p2_texts = _p1_df[_p2_text_col].dropna().astype(str).tolist()\n",
        ]
    else:
        text_src = [
            f"_p2_texts = _p1_df[{_q(text_col)}].dropna().astype(str).tolist()\n",
        ]

    return [
        f"# --- {PHASE2_MARKER} (auto-generated) ---\n",
        f"# Sentence-transformer embeddings + clustering for {slug}\n",
        "from utils.embeddings_and_topics import run_embedding_pipeline\n",
        "\n",
        *text_src,
        "\n",
        "_p2_results = run_embedding_pipeline(\n",
        f"    slug=PROJECT_SLUG,\n",
        "    texts=_p2_texts,\n",
        ")\n",
        *_RESULTS_FOOTER,
        "if 'clustering' in _p2_results:\n",
        "    _c = _p2_results['clustering']\n",
        "    print(f\"  Clusters:   {_c.get('n_clusters', '?')}\")\n",
        "    print(f\"  Silhouette: {_c.get('silhouette', 0):.4f}\")\n",
        "    print(f\"  Method:     {_c.get('method', '?')}\")\n",
        "if 'topics' in _p2_results:\n",
        "    print(f\"  Topics found: {_p2_results['topics'].get('n_topics', 0)}\")\n",
    ]


def _caption_cell(slug: str, cfg: dict) -> list[str]:
    model = cfg.get("model", "Salesforce/blip-image-captioning-large")
    return [
        f"# --- {PHASE2_MARKER} (auto-generated) ---\n",
        f"# BLIP image captioning for {slug}\n",
        "from utils.captioning import run_captioning_pipeline\n",
        "\n",
        "_p2_results = run_captioning_pipeline(\n",
        f"    slug=PROJECT_SLUG,\n",
        "    images_dir=_p1_images_dir,\n",
        "    captions_path=_p1_captions_path,\n",
        f"    model_name={_q(model)},\n",
        ")\n",
        *_RESULTS_FOOTER,
        "if 'eval_metrics' in _p2_results:\n",
        "    _m = _p2_results['eval_metrics']\n",
        "    print(f\"  ROUGE-1: {_m.get('rouge1', 0):.4f}\")\n",
        "    print(f\"  ROUGE-L: {_m.get('rougeL', 0):.4f}\")\n",
        "if 'sample_captions' in _p2_results:\n",
        "    for _s in _p2_results['sample_captions'][:3]:\n",
        "        print(f\"\\n  Image: {_s['image']}\")\n",
        "        print(f\"  Gen:   {_s['generated']}\")\n",
        "        if 'references' in _s:\n",
        "            print(f\"  Ref:   {_s['references'][0][:80]}\")\n",
    ]


def _char_rnn_cell(slug: str, cfg: dict) -> list[str]:
    return [
        f"# --- {PHASE2_MARKER} (auto-generated) ---\n",
        f"# Character-level RNN for {slug}\n",
        "from utils.training_common import train_char_rnn\n",
        "\n",
        "_p2_results = train_char_rnn(\n",
        f"    slug=PROJECT_SLUG,\n",
        "    names_dir=_p1_names_dir,\n",
        ")\n",
        *_RESULTS_FOOTER,
        "print(f\"  Vocab:     {_p2_results.get('vocab_size', '?')}\")\n",
        "print(f\"  Val loss:  {_p2_results.get('best_val_loss', '?'):.4f}\")\n",
        "print('\\n  Generated samples:')\n",
        "for _s in _p2_results.get('generated_samples', [])[:10]:\n",
        "    print(f'    {_s}')\n",
    ]


# ======================================================================
# Notebook patching
# ======================================================================

CELL_GENERATORS = {
    "classify": _classify_cell,
    "classify_multilabel": _multilabel_cell,
    "summarize": _summarize_cell,
    "translate": _translate_cell,
    "embed_cluster": _embed_cluster_cell,
    "caption": _caption_cell,
    "char_rnn": _char_rnn_cell,
}


def _find_notebook(slug: str) -> Path | None:
    info = PROJECT_REGISTRY.get(slug, {})
    rel = info.get("dir", "")
    if rel:
        nb = WORKSPACE / rel / "code.ipynb"
        if nb.exists():
            return nb
    # Fallback: search
    for p in WORKSPACE.rglob("code.ipynb"):
        if slug.replace("-", " ").lower() in str(p).lower():
            return p
    return None


def _has_marker(nb: dict, marker: str) -> bool:
    for cell in nb.get("cells", []):
        src = "".join(cell.get("source", []))
        if marker in src:
            return True
    return False


def _last_phase1_index(nb: dict) -> int:
    """Return the index of the last Phase 1 cell, or last cell index."""
    last = len(nb["cells"]) - 1
    for i, cell in enumerate(nb["cells"]):
        src = "".join(cell.get("source", []))
        if PHASE1_MARKER in src or "Phase 1" in src:
            last = i
    return last


def patch_notebook(slug: str, cfg: dict) -> str:
    """Patch one notebook. Returns status string."""
    nb_path = _find_notebook(slug)
    if nb_path is None:
        return f"SKIP: notebook not found for {slug}"

    nb = json.loads(nb_path.read_text(encoding="utf-8"))

    if _has_marker(nb, PHASE2_MARKER):
        return f"SKIP: already patched ({slug})"

    action = cfg.get("action")
    gen_fn = CELL_GENERATORS.get(action)
    if gen_fn is None:
        return f"SKIP: unknown action '{action}' for {slug}"

    # Generate cells
    md = _md_cell(["## Phase 2 -- Deep Learning Model Training\n",
                    f"**Project:** `{slug}`  \n",
                    f"**Action:** {action}  \n",
                    f"**Model:** {cfg.get('model', 'see below')}  \n"])
    code_lines = gen_fn(slug, cfg)
    code = _code_cell(code_lines)

    # Find insertion point (after last Phase 1 cell)
    insert_idx = _last_phase1_index(nb) + 1
    nb["cells"].insert(insert_idx, md)
    nb["cells"].insert(insert_idx + 1, code)

    nb_path.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    return f"OK: patched {slug} (action={action}, cells inserted at {insert_idx})"


# ======================================================================
# Report generation
# ======================================================================

def generate_reports(results: dict[str, str]) -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    # -- Summary --
    lines = [
        f"# Phase 2 Summary\n\n",
        f"Generated: {ts}\n\n",
        "## Overview\n\n",
        f"Total projects: {len(PHASE2_CONFIG)}\n\n",
        "| # | Project | Action | Model | Status |\n",
        "|---|---------|--------|-------|--------|\n",
    ]
    ok, fail, skip = 0, 0, 0
    for i, (slug, cfg) in enumerate(PHASE2_CONFIG.items(), 1):
        status = results.get(slug, "NOT RUN")
        action = cfg.get("action", "?")
        model = cfg.get("model", "-")
        st = "OK" if status.startswith("OK") else ("SKIP" if status.startswith("SKIP") else "FAIL")
        if st == "OK":
            ok += 1
        elif st == "SKIP":
            skip += 1
        else:
            fail += 1
        lines.append(f"| {i} | {slug} | {action} | `{model}` | {st} |\n")

    lines += [
        f"\n**OK: {ok}  |  Skipped: {skip}  |  Failed: {fail}**\n\n",
    ]

    # Per-project details
    lines.append("## Per-Project Details\n\n")
    for slug, cfg in PHASE2_CONFIG.items():
        action = cfg.get("action", "?")
        model = cfg.get("model", "-")
        notes = cfg.get("notes", "")
        status = results.get(slug, "NOT RUN")

        lines.append(f"### {slug}\n\n")
        lines.append(f"- **Action:** {action}\n")
        lines.append(f"- **Model:** `{model}`\n")
        if notes:
            lines.append(f"- **Notes:** {notes}\n")
        lines.append(f"- **Patch status:** {status}\n")
        lines.append(f"- **Outputs:** `outputs/{slug}/metrics/phase2_metrics.json`\n")

        if action == "classify":
            lines.append(f"- **Text col:** `{cfg.get('text_col')}`\n")
            lines.append(f"- **Target col:** `{cfg.get('target_col')}`\n")
            lines.append(f"- **Max length:** {cfg.get('max_length', 256)}\n")
        elif action == "classify_multilabel":
            lines.append(f"- **Text col:** `{cfg.get('text_col')}`\n")
            lines.append(f"- **Target cols:** {cfg.get('target_cols')}\n")
        elif action == "embed_cluster":
            lines.append(f"- **Text col:** `{cfg.get('text_col', 'auto')}`\n")
            lines.append(f"- **Embedding model:** `all-MiniLM-L6-v2`\n")
        lines.append("\n")

    # GPU / training notes
    lines += [
        "## Training Configuration\n\n",
        "| Setting | Value |\n",
        "|---------|-------|\n",
        "| Framework | PyTorch + HuggingFace Transformers |\n",
        "| Classification | DeBERTa-v3-base (full fine-tune) |\n",
        "| Summarization | BART-large-CNN + LoRA (r=16) |\n",
        "| Translation | NLLB-200-distilled-600M + LoRA (r=16) |\n",
        "| Clustering | sentence-transformers/all-MiniLM-L6-v2 |\n",
        "| Captioning | BLIP (inference only) |\n",
        "| Generation | CharRNN (custom LSTM) |\n",
        "| Precision | fp16/bf16 (auto-detect) |\n",
        "| Batch size | 2-8 (project-dependent) |\n",
        "| K-Fold CV | 5-fold LogReg on TF-IDF (>2000 samples) |\n",
        "| Early stopping | patience=2 on val metric |\n",
        "| GPU target | RTX 4060 8GB |\n",
        "\n",
        "## How to run training\n\n",
        "Open any `code.ipynb` and run all cells. Phase 2 cells will:\n",
        "1. Load data from Phase 1 variables\n",
        "2. Train the configured model\n",
        "3. Evaluate on test set\n",
        "4. Save metrics + model to `outputs/<slug>/`\n",
    ]

    (REPORTS / "phase2_summary.md").write_text("".join(lines), encoding="utf-8")
    logger.info("Wrote reports/phase2_summary.md")

    # -- Failures --
    fail_lines = [f"# Phase 2 Failures\n\nGenerated: {ts}\n\n"]
    failures = {s: r for s, r in results.items() if not r.startswith("OK") and not r.startswith("SKIP")}
    if failures:
        for s, r in failures.items():
            fail_lines.append(f"- **{s}**: {r}\n")
    else:
        fail_lines.append("No failures -- all projects patched or already up to date!\n")

    (REPORTS / "phase2_failures.md").write_text("".join(fail_lines), encoding="utf-8")
    logger.info("Wrote reports/phase2_failures.md")


# ======================================================================
# Main
# ======================================================================

def main() -> None:
    logger.info("=" * 60)
    logger.info("Phase 2: Patch notebooks with model-training cells")
    logger.info("=" * 60)

    results: dict[str, str] = {}

    for slug, cfg in PHASE2_CONFIG.items():
        try:
            status = patch_notebook(slug, cfg)
            results[slug] = status
            logger.info("  %s -> %s", slug, status)
        except Exception as exc:
            tb = traceback.format_exc()
            results[slug] = f"ERROR: {exc}\n{tb}"
            logger.error("  %s -> ERROR: %s", slug, exc)

    generate_reports(results)

    # Summary counts
    ok = sum(1 for v in results.values() if v.startswith("OK"))
    skip = sum(1 for v in results.values() if v.startswith("SKIP"))
    fail = len(results) - ok - skip
    logger.info("-" * 60)
    logger.info("Phase 2 patching complete: %d OK, %d skipped, %d failed", ok, skip, fail)
    logger.info("Reports: reports/phase2_summary.md, reports/phase2_failures.md")


if __name__ == "__main__":
    main()
