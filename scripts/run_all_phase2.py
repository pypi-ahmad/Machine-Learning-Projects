#!/usr/bin/env python
"""Phase 2.1 -- Run ALL project training/eval pipelines headlessly.

Orchestrates training for all 21 NLP projects, dispatches to the
appropriate trainer, saves metrics/models/samples, generates reports.

Usage:
    .venv\\Scripts\\python.exe scripts/run_all_phase2.py [--force]
"""

from __future__ import annotations

import csv
import json
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKSPACE))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import pandas as pd

from utils.dataset_finder import PROJECT_REGISTRY
from utils.logger import get_logger
from utils.training_common import (
    OUTPUTS_DIR, SEED, cleanup_gpu, ensure_output_dirs, get_device, save_json, seed_everything,
)

logger = get_logger(__name__)
REPORTS = WORKSPACE / "reports"
DATA_ROOT = WORKSPACE / "data"

# ======================================================================
# GPU-safe training defaults (RTX 4060 8 GB)
# ======================================================================

TRAIN_DEFAULTS = {
    "batch_size": 2,
    "grad_accum": 8,
    "epochs": 2,
    "patience": 1,
    "lr": 2e-5,
    "max_samples": 10_000,
    "seed": SEED,
}

# ======================================================================
# Per-project configuration
# ======================================================================

PHASE2_PROJECTS: dict[str, dict] = {
    # -- CLASSIFICATION (11 single-label) -----------------------------------
    "e-commerce-clothing-reviews": {
        "action": "classify", "data_file": "data.csv",
        "text_col": "Review Text", "target_col": "Recommended IND",
        "max_length": 256, "model_name": "microsoft/deberta-v3-base",
    },
    "trip-advisor-hotel-reviews": {
        "action": "classify", "data_file": "data.csv",
        "text_col": "Review", "target_col": "Rating",
        "max_length": 256, "model_name": "microsoft/deberta-v3-base",
    },
    "cyberbullying-classification": {
        "action": "classify", "data_file": "data.csv",
        "text_col": "tweet_text", "target_col": "cyberbullying_type",
        "max_length": 256, "model_name": "microsoft/deberta-v3-base",
    },
    "e-commerce-product-classification": {
        "action": "classify", "data_file": "data.csv",
        "text_col": "description", "target_col": "category",
        "max_length": 256, "model_name": "microsoft/deberta-v3-base",
        "csv_kwargs": {"header": None, "names": ["category", "description"]},
    },
    "economic-news-articles": {
        "action": "classify", "data_file": "data.csv",
        "text_col": "text", "target_col": "positivity",
        "max_length": 256, "model_name": "microsoft/deberta-v3-base",
    },
    "fake-news-detection": {
        "action": "classify", "data_loader": "fake_news",
        "text_col": "text", "target_col": "label",
        "max_length": 512, "model_name": "microsoft/deberta-v3-base",
    },
    "news-headline-classification": {
        "action": "classify", "data_loader": "json", "data_file": "data.json",
        "text_col": "headline", "target_col": "category",
        "max_length": 128, "model_name": "microsoft/deberta-v3-base",
    },
    "paper-subject-prediction": {
        "action": "classify", "data_file": "arxiv_data.csv",
        "text_col": "summaries", "target_col": "terms",
        "max_length": 384, "model_name": "microsoft/deberta-v3-base",
        "preprocess_target": "first_term",
    },
    "review-classification": {
        "action": "classify", "data_loader": "review_merge",
        "text_col": "text", "target_col": "label",
        "max_length": 256, "model_name": "microsoft/deberta-v3-base",
    },
    "spam-message-detection": {
        "action": "classify", "data_file": "data.csv",
        "text_col": "text", "target_col": "label",
        "max_length": 128, "model_name": "microsoft/deberta-v3-base",
    },
    "twitter-sentiment-analysis": {
        "action": "classify", "data_file": "train.csv",
        "text_col": "OriginalTweet", "target_col": "Sentiment",
        "max_length": 128, "model_name": "microsoft/deberta-v3-base",
    },
    # -- MULTI-LABEL --------------------------------------------------------
    "toxic-comment-classification": {
        "action": "classify_multilabel", "data_file": "train.csv",
        "text_col": "comment_text",
        "target_cols": ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"],
        "max_length": 256, "model_name": "microsoft/deberta-v3-base",
    },
    # -- EMBEDDING / CLUSTERING (5) ----------------------------------------
    "world-war-i-letters": {
        "action": "embed_cluster", "data_file": "data.csv", "text_col": "_auto",
    },
    "kaggle-survey-questions-clustering": {
        "action": "embed_cluster", "data_file": "questions.csv", "text_col": "_auto",
    },
    "medium-articles-clustering": {
        "action": "embed_cluster", "data_file": "articles.csv", "text_col": "text",
    },
    "newsgroups-posts-clustering": {
        "action": "embed_cluster", "data_loader": "newsgroups", "text_col": "text",
    },
    "stories-clustering": {
        "action": "embed_cluster", "data_loader": "stories", "text_col": "_auto",
    },
    # -- SUMMARIZATION ------------------------------------------------------
    "bbc-articles-summarization": {
        "action": "summarize", "model_name": "facebook/bart-large-cnn",
    },
    # -- TRANSLATION --------------------------------------------------------
    "english-to-french-translation": {
        "action": "translate", "model_name": "facebook/nllb-200-distilled-600M",
    },
    # -- CAPTIONING ---------------------------------------------------------
    "automated-image-captioning": {
        "action": "caption", "model_name": "Salesforce/blip-image-captioning-large",
    },
    # -- CHAR RNN -----------------------------------------------------------
    "name-generate-from-languages": {
        "action": "char_rnn",
    },
}

# ======================================================================
# Data utilities
# ======================================================================

def _find_raw_dir(slug: str) -> Path | None:
    raw = DATA_ROOT / slug / "raw"
    if raw.exists() and any(raw.iterdir()):
        return raw
    info = PROJECT_REGISTRY.get(slug, {})
    pdir = WORKSPACE / info.get("dir", "")
    if pdir.exists():
        return pdir
    return None


def _find_file(d: Path, *names: str) -> Path | None:
    for name in names:
        f = d / name
        if f.exists():
            return f
    for name in names:
        for f in d.rglob(name):
            return f
    return None


def _load_csv(path: Path, **kwargs) -> pd.DataFrame:
    for enc in ["utf-8", "latin-1", "cp1252"]:
        try:
            return pd.read_csv(path, encoding=enc, on_bad_lines="skip", **kwargs)
        except Exception:
            continue
    return pd.read_csv(path, encoding="utf-8", errors="replace", on_bad_lines="skip", **kwargs)


def _find_text_col(df: pd.DataFrame, hint: str) -> str:
    if hint and hint != "_auto" and hint in df.columns:
        return hint
    candidates = [
        "text", "Text", "review", "Review", "comment_text", "content",
        "description", "tweet_text", "headline", "v2", "message",
        "OriginalTweet", "summaries", "article", "question", "story",
        "Story", "Question", "Title", "title", "body", "Body",
    ]
    for c in candidates:
        if c in df.columns:
            return c
    str_cols = df.select_dtypes(include="object").columns.tolist()
    if str_cols:
        avg_len = {c: df[c].astype(str).str.len().mean() for c in str_cols}
        return max(avg_len, key=avg_len.get)
    return df.columns[0]


def _find_target_col(df: pd.DataFrame, hint: str) -> str | None:
    if hint and hint in df.columns:
        return hint
    candidates = [
        "label", "Label", "class", "category", "target", "sentiment",
        "v1", "positivity", "Rating", "Sentiment", "cyberbullying_type",
        "genre", "Genre", "Category", "Class", "Recommended IND",
    ]
    for c in candidates:
        if c in df.columns:
            return c
    return None


# ======================================================================
# Data loading (project-specific)
# ======================================================================

def _load_data(slug: str, cfg: dict, raw_dir: Path) -> dict:
    """Return {'df': DataFrame} or {'texts': list[str]}."""
    loader = cfg.get("data_loader")

    if loader == "fake_news":
        true_f = _find_file(raw_dir, "True.csv")
        fake_f = _find_file(raw_dir, "Fake.csv")
        if not true_f or not fake_f:
            raise FileNotFoundError(f"True.csv / Fake.csv not found in {raw_dir}")
        true_df = _load_csv(true_f); true_df["label"] = 1
        fake_df = _load_csv(fake_f); fake_df["label"] = 0
        df = pd.concat([true_df, fake_df], ignore_index=True)
        return {"df": df.sample(frac=1, random_state=SEED).reset_index(drop=True)}

    if loader == "review_merge":
        dfs = []
        for nm in ["amazon.txt", "imdb.txt", "yelp.txt"]:
            f = _find_file(raw_dir, nm)
            if not f:
                continue
            rows = []
            for line in f.read_text(encoding="utf-8", errors="replace").splitlines():
                parts = line.split("\t")
                if len(parts) >= 2:
                    rows.append({"text": parts[0].strip(), "label": parts[1].strip()})
            if rows:
                dfs.append(pd.DataFrame(rows))
        if not dfs:
            raise FileNotFoundError("No review files found in " + str(raw_dir))
        return {"df": pd.concat(dfs, ignore_index=True)}

    if loader == "json":
        f = _find_file(raw_dir, cfg.get("data_file", "data.json"))
        if not f:
            raise FileNotFoundError("JSON not found in " + str(raw_dir))
        try:
            df = pd.read_json(f, lines=True)
        except ValueError:
            df = pd.read_json(f)
        return {"df": df}

    if loader == "newsgroups":
        # CSV first
        f = _find_file(raw_dir, "newsgroups.csv", "list.csv", "20newsgroups.csv")
        if f:
            df = _load_csv(f)
            if len(df) > 10:
                return {"df": df}
        # Read individual text files
        texts: list[str] = []
        for tf in sorted(raw_dir.rglob("*.txt")):
            content = tf.read_text(encoding="utf-8", errors="replace")
            for doc in content.split("\n\n"):
                doc = doc.strip()
                if len(doc) > 50:
                    texts.append(doc)
        if texts:
            return {"texts": texts}
        # Fall back sklearn
        from sklearn.datasets import fetch_20newsgroups
        ng = fetch_20newsgroups(subset="all", remove=("headers", "footers", "quotes"))
        return {"df": pd.DataFrame({"text": ng.data, "newsgroup": [ng.target_names[t] for t in ng.target]})}

    if loader == "stories":
        csvs = sorted(raw_dir.glob("*.csv")) or sorted(raw_dir.rglob("*.csv"))
        if csvs:
            return {"df": _load_csv(csvs[0])}
        raise FileNotFoundError("No CSV found for stories in " + str(raw_dir))

    # ---- Default CSV loader -----------------------------------------------
    data_file = cfg.get("data_file", "data.csv")
    csv_kwargs = cfg.get("csv_kwargs", {})
    f = _find_file(raw_dir, data_file)
    if not f:
        csvs = sorted(raw_dir.glob("*.csv"))
        if csvs:
            f = csvs[0]
    if not f:
        raise FileNotFoundError(f"No data file for {slug} in {raw_dir}")
    return {"df": _load_csv(f, **csv_kwargs)}


# ======================================================================
# Training dispatch
# ======================================================================

def _process_project(slug: str, cfg: dict) -> dict:
    """Run the full Phase 2 pipeline for one project."""
    t0 = time.time()
    action = cfg["action"]

    raw_dir = _find_raw_dir(slug)
    if raw_dir is None:
        return {"status": "FAILED", "error": "Data directory not found",
                "slug": slug, "action": action, "time_sec": 0}

    dirs = ensure_output_dirs(slug)
    outputs_dir = OUTPUTS_DIR / slug

    # Build config for run_project
    config = {**TRAIN_DEFAULTS}
    config.update({k: v for k, v in cfg.items()
                   if k not in ("action", "data_file", "data_loader",
                                "csv_kwargs", "preprocess_target")})

    force = "--force" in sys.argv

    # ---- CLASSIFY ---------------------------------------------------------
    if action == "classify":
        data = _load_data(slug, cfg, raw_dir)
        df = data["df"]
        text_col = _find_text_col(df, cfg.get("text_col", "text"))
        target_col = _find_target_col(df, cfg.get("target_col", "label"))
        if target_col is None:
            return {"status": "FAILED", "slug": slug, "action": action, "time_sec": 0,
                    "error": f"No target column in {list(df.columns)}"}
        if cfg.get("preprocess_target") == "first_term":
            df[target_col] = (df[target_col].astype(str)
                              .str.strip("[]' ").str.split(",").str[0].str.strip("' "))
        config.update({"df": df, "text_col": text_col, "target_col": target_col,
                       "run_kfold": True, "kfold_n": 5})
        from utils.train_text_classifier import run_project
        result = run_project(slug, raw_dir, None, None, outputs_dir, config, force=force)

    # ---- MULTI-LABEL ------------------------------------------------------
    elif action == "classify_multilabel":
        data = _load_data(slug, cfg, raw_dir)
        df = data["df"]
        text_col = _find_text_col(df, cfg.get("text_col", "comment_text"))
        target_cols = cfg.get("target_cols", [])
        missing = [c for c in target_cols if c not in df.columns]
        if missing:
            return {"status": "FAILED", "slug": slug, "action": action, "time_sec": 0,
                    "error": f"Missing cols {missing}; have {list(df.columns)}"}
        config.update({"df": df, "text_col": text_col,
                       "target_cols": target_cols, "multilabel": True})
        from utils.train_text_classifier import run_project
        result = run_project(slug, raw_dir, None, None, outputs_dir, config, force=force)

    # ---- EMBED / CLUSTER --------------------------------------------------
    elif action == "embed_cluster":
        data = _load_data(slug, cfg, raw_dir)
        if "texts" in data:
            texts = data["texts"]
        else:
            df = data["df"]
            tc = _find_text_col(df, cfg.get("text_col", "_auto"))
            texts = df[tc].dropna().astype(str).tolist()
        # If very few texts (e.g. survey schema file), extract all string values
        if len(texts) < 5 and "df" in data:
            df = data["df"]
            all_texts = []
            for _, row in df.iterrows():
                for val in row:
                    s = str(val).strip()
                    if len(s) > 10 and not s.replace(".", "").replace("-", "").isdigit():
                        all_texts.append(s)
            if len(all_texts) > len(texts):
                logger.info("Expanded %d texts → %d by extracting all string cells", len(texts), len(all_texts))
                texts = all_texts
        config["texts"] = texts
        from utils.embeddings_and_topics import run_project
        result = run_project(slug, raw_dir, None, None, outputs_dir, config, force=force)

    # ---- SUMMARIZE --------------------------------------------------------
    elif action == "summarize":
        config["raw_dir"] = raw_dir
        config["max_samples"] = 5_000
        from utils.train_summarizer import run_project
        result = run_project(slug, raw_dir, None, None, outputs_dir, config, force=force)

    # ---- TRANSLATE --------------------------------------------------------
    elif action == "translate":
        config["raw_dir"] = raw_dir
        config["max_samples"] = 30_000
        from utils.train_translator import run_project
        result = run_project(slug, raw_dir, None, None, outputs_dir, config, force=force)

    # ---- CAPTION ----------------------------------------------------------
    elif action == "caption":
        images_dir = None
        for cand in [raw_dir / "images", raw_dir / "Images", raw_dir]:
            if cand.is_dir() and (list(cand.glob("*.jpg")) or list(cand.glob("*.png"))):
                images_dir = cand
                break
        captions_path = None
        for nm in ["captions.txt", "captions.csv", "token.txt", "results.csv"]:
            c = raw_dir / nm
            if c.exists():
                captions_path = c
                break
        config.update({"images_dir": images_dir, "captions_path": captions_path,
                       "max_samples": 200})
        from utils.captioning import run_project
        result = run_project(slug, raw_dir, None, None, outputs_dir, config, force=force)

    # ---- CHAR RNN ---------------------------------------------------------
    elif action == "char_rnn":
        names_dir = None
        for cand in [raw_dir / "names", raw_dir]:
            if cand.is_dir() and any(cand.glob("*.txt")):
                names_dir = cand
                break
        if names_dir is None:
            info = PROJECT_REGISTRY.get(slug, {})
            pdir = WORKSPACE / info.get("dir", "")
            for cand in [pdir / "names", pdir]:
                if cand.is_dir() and any(cand.glob("*.txt")):
                    names_dir = cand
                    break
        if names_dir is None:
            return {"status": "FAILED", "slug": slug, "action": action,
                    "error": "names dir not found", "time_sec": 0}
        config.update({"names_dir": names_dir, "epochs": 20})
        from utils.training_common import run_char_rnn_project as run_project
        result = run_project(slug, raw_dir, None, None, outputs_dir, config, force=force)

    else:
        return {"status": "FAILED", "slug": slug, "action": action,
                "error": f"Unknown action: {action}", "time_sec": 0}

    result["slug"] = slug
    result["action"] = action
    result["time_sec"] = round(time.time() - t0, 1)

    # ---- Save extra files requested by spec -------------------------------
    _save_extra_outputs(slug, result, dirs)

    return result


def _save_extra_outputs(slug: str, result: dict, dirs: dict) -> None:
    """Save val metrics, CV metrics, and sample files separately."""
    fr = result.get("full_result", {})

    if fr.get("val_metrics"):
        save_json(fr["val_metrics"], dirs["metrics"] / "phase2_val_metrics.json")

    kfold = fr.get("kfold_summary")
    if kfold:
        save_json(kfold, dirs["metrics"] / "phase2_cv_metrics.json")

    samples_dir = OUTPUTS_DIR / slug / "samples"
    for key in ("sample_predictions", "sample_translations",
                "sample_captions", "generated_samples"):
        if key in fr:
            samples_dir.mkdir(parents=True, exist_ok=True)
            save_json(fr[key], samples_dir / f"{key}.json")


# ======================================================================
# Report generation
# ======================================================================

def _metric_str(action: str, mm: dict) -> str:
    if action == "classify":
        return f"F1w={mm.get('f1_weighted', mm.get('f1', 0)):.4f}"
    if action == "classify_multilabel":
        return f"F1mi={mm.get('f1_micro', 0):.4f}"
    if action == "summarize":
        return f"R-L={mm.get('rougeL', 0):.4f}"
    if action == "translate":
        return f"BLEU={mm.get('bleu', 0):.1f}"
    if action == "embed_cluster":
        return f"Sil={mm.get('silhouette', 0):.4f}"
    if action == "caption":
        return f"R-L={mm.get('rougeL', 0):.4f}"
    if action == "char_rnn":
        return f"VL={mm.get('best_val_loss', 0):.4f}"
    return "?"


def _generate_summary(all_results: dict) -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    ok = sum(1 for r in all_results.values() if r.get("status") == "OK")
    fail = len(all_results) - ok
    total_t = sum(r.get("time_sec", 0) for r in all_results.values())

    L = [
        f"# Phase 2.1 -- Training Results Summary\n\n",
        f"Generated: {ts}\n\n",
        "## Overview\n\n",
        f"| Metric | Value |\n|--------|-------|\n",
        f"| Total projects | {len(all_results)} |\n",
        f"| Succeeded | {ok} |\n",
        f"| Failed | {fail} |\n",
        f"| Total time | {total_t:.0f}s ({total_t/60:.1f} min) |\n\n",
        "## Results Table\n\n",
        "| # | Project | Action | Model | Mode | Size | Main Metric | Time |\n",
        "|---|---------|--------|-------|------|------|-------------|------|\n",
    ]
    for i, (slug, r) in enumerate(all_results.items(), 1):
        st = r.get("status", "?")
        act = r.get("action", "?")
        model = (r.get("model_name") or "?").split("/")[-1][:28]
        mode = r.get("training_mode", "?")
        size = r.get("dataset_size", 0)
        mm = r.get("main_metrics", {})
        t = r.get("time_sec", 0)
        ms = _metric_str(act, mm) if st == "OK" else f"**{st}**"
        L.append(f"| {i} | {slug} | {act} | {model} | {mode} | {size} | {ms} | {t:.0f}s |\n")

    L.append(f"\n**{ok} succeeded, {fail} failed, total {total_t:.0f}s**\n\n")

    # Per-project detail
    L.append("## Per-Project Details\n\n")
    for slug, r in all_results.items():
        L.append(f"### {slug}\n\n")
        L.append(f"- **Status:** {r.get('status', '?')}\n")
        L.append(f"- **Action:** {r.get('action', '?')}\n")
        L.append(f"- **Model:** `{r.get('model_name', '?')}`\n")
        L.append(f"- **Training mode:** {r.get('training_mode', '?')}\n")
        L.append(f"- **Dataset size:** {r.get('dataset_size', 0)}\n")
        L.append(f"- **Time:** {r.get('time_sec', 0):.1f}s\n")
        vm = r.get("val_metrics", {})
        if vm:
            L.append(f"- **Val metrics:** `{json.dumps(vm, default=str)[:200]}`\n")
        mm = r.get("main_metrics", {})
        if mm:
            L.append(f"- **Test metrics:** `{json.dumps(mm, default=str)[:200]}`\n")
        if r.get("notes"):
            L.append(f"- **Notes:** {r['notes']}\n")
        if r.get("error"):
            L.append(f"- **Error:** `{r['error']}`\n")
        L.append(f"- **Outputs:** `outputs/{slug}/`\n\n")

    (REPORTS / "phase2_summary.md").write_text("".join(L), encoding="utf-8")
    logger.info("Wrote reports/phase2_summary.md")


def _generate_leaderboard(all_results: dict) -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for slug, r in all_results.items():
        if r.get("status") != "OK":
            continue
        act = r.get("action", "?")
        mm = r.get("main_metrics", {})
        row = {
            "slug": slug, "action": act,
            "model": (r.get("model_name") or "?"),
            "mode": r.get("training_mode", "?"),
            "dataset_size": r.get("dataset_size", 0),
        }
        if act == "classify":
            row["metric_name"] = "f1_macro"
            row["metric_value"] = round(mm.get("f1_macro", mm.get("f1_weighted", 0)), 4)
            row["accuracy"] = round(mm.get("accuracy", 0), 4)
            row["f1_weighted"] = round(mm.get("f1_weighted", 0), 4)
        elif act == "classify_multilabel":
            row["metric_name"] = "f1_micro"
            row["metric_value"] = round(mm.get("f1_micro", 0), 4)
            row["f1_macro"] = round(mm.get("f1_macro", 0), 4)
        elif act == "summarize":
            row["metric_name"] = "rougeL"
            row["metric_value"] = round(mm.get("rougeL", 0), 4)
            row["rouge1"] = round(mm.get("rouge1", 0), 4)
        elif act == "translate":
            row["metric_name"] = "bleu"
            row["metric_value"] = round(mm.get("bleu", 0), 2)
            row["chrf"] = round(mm.get("chrf", 0), 2)
        elif act == "embed_cluster":
            row["metric_name"] = "silhouette"
            row["metric_value"] = round(mm.get("silhouette", 0), 4)
        elif act == "caption":
            row["metric_name"] = "rougeL"
            row["metric_value"] = round(mm.get("rougeL", 0), 4)
        elif act == "char_rnn":
            row["metric_name"] = "val_loss"
            row["metric_value"] = round(mm.get("best_val_loss", 0), 4)
        else:
            row["metric_name"] = "unknown"
            row["metric_value"] = 0
        rows.append(row)

    rows.sort(key=lambda r: r.get("metric_value", 0), reverse=True)

    path = REPORTS / "phase2_leaderboard.csv"
    if rows:
        all_keys: list[str] = []
        for r in rows:
            for k in r:
                if k not in all_keys:
                    all_keys.append(k)
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
    else:
        path.write_text("slug,action,metric_name,metric_value\n", encoding="utf-8")
    logger.info("Wrote reports/phase2_leaderboard.csv")


def _generate_failures(all_results: dict) -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    L = [f"# Phase 2.1 Failures\n\nGenerated: {ts}\n\n"]
    failures = {s: r for s, r in all_results.items() if r.get("status") != "OK"}
    if failures:
        L.append(f"**{len(failures)} project(s) failed:**\n\n")
        for slug, r in failures.items():
            L.append(f"## {slug}\n\n")
            L.append(f"- **Action:** {r.get('action', '?')}\n")
            L.append(f"- **Error:** {r.get('error', 'Unknown')}\n")
            if r.get("traceback"):
                L.append(f"\n```\n{r['traceback']}\n```\n\n")
    else:
        L.append("No failures -- all 21 projects completed successfully!\n")
    (REPORTS / "phase2_failures.md").write_text("".join(L), encoding="utf-8")
    logger.info("Wrote reports/phase2_failures.md")


# ======================================================================
# Main
# ======================================================================

def main() -> None:
    seed_everything(SEED)
    device = get_device()

    logger.info("=" * 70)
    logger.info("Phase 2.1 -- Run ALL project training pipelines")
    logger.info("  Device: %s", device)
    logger.info("  Projects: %d", len(PHASE2_PROJECTS))
    logger.info("  Defaults: bs=%d  accum=%d  epochs=%d  patience=%d",
                TRAIN_DEFAULTS["batch_size"], TRAIN_DEFAULTS["grad_accum"],
                TRAIN_DEFAULTS["epochs"], TRAIN_DEFAULTS["patience"])
    logger.info("  force=%s", "--force" in sys.argv)
    logger.info("=" * 70)

    all_results: dict[str, dict] = {}

    for i, (slug, cfg) in enumerate(PHASE2_PROJECTS.items(), 1):
        logger.info("-" * 70)
        logger.info("[%d/%d] %s  (action=%s)", i, len(PHASE2_PROJECTS), slug, cfg["action"])
        logger.info("-" * 70)

        try:
            result = _process_project(slug, cfg)
            all_results[slug] = result
            st = result.get("status", "UNKNOWN")
            mm = result.get("main_metrics", {})
            logger.info("  => %s  (%.1fs)  %s", st, result.get("time_sec", 0),
                        json.dumps(mm, default=str)[:120])
        except Exception as exc:
            tb = traceback.format_exc()
            logger.error("  => FAILED: %s", exc)
            all_results[slug] = {
                "status": "FAILED", "slug": slug, "action": cfg.get("action", "?"),
                "error": str(exc), "traceback": tb, "time_sec": 0,
            }

        cleanup_gpu()

    # ---- Reports ----------------------------------------------------------
    logger.info("=" * 70)
    logger.info("Generating reports...")
    _generate_summary(all_results)
    _generate_leaderboard(all_results)
    _generate_failures(all_results)

    ok = sum(1 for r in all_results.values() if r.get("status") == "OK")
    fail = len(all_results) - ok
    total_t = sum(r.get("time_sec", 0) for r in all_results.values())

    # Print terminal table
    print("\n" + "=" * 95)
    print(f"{'#':>3} | {'Slug':<42} | {'Status':<7} | {'Main Metric':<20} | {'Time':>7}")
    print("-" * 95)
    for i, (slug, r) in enumerate(all_results.items(), 1):
        st = r.get("status", "?")
        act = r.get("action", "?")
        mm = r.get("main_metrics", {})
        ms = _metric_str(act, mm) if st == "OK" else f"[{st}]"
        t = r.get("time_sec", 0)
        print(f"{i:>3} | {slug:<42} | {st:<7} | {ms:<20} | {t:>6.1f}s")
    print("=" * 95)
    print(f"Total: {ok} OK, {fail} FAILED, {total_t:.0f}s ({total_t/60:.1f} min)")
    print("=" * 95)


if __name__ == "__main__":
    main()
