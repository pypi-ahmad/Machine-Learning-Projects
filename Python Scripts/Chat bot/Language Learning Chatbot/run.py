#!/usr/bin/env python3
"""
Language Learning Chatbot — Sentence Classification
=====================================================
Classifies English sentences by complexity/length using PyCaret.

The dataset contains English sentences (Tatoeba corpus). Since there are no
pre-existing category labels, we engineer a sentence-complexity target based
on sentence length (short / medium / long) and use TF-IDF features with
PyCaret's automated ML pipeline.

Dataset: https://www.kaggle.com/datasets/mayakaripel/eng-sentences
Run:     python run.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import logging

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from shared.utils import (
    download_kaggle_dataset,
    set_seed,
    setup_logging,
    project_paths,
    run_tabular_auto,
    parse_common_args,
    save_metrics,
    dataset_missing_metrics,
    configure_cuda_allocator,
    make_tabular_splits,
    write_split_manifest,
    dataset_fingerprint,
    run_metadata,
)

logger = logging.getLogger(__name__)

KAGGLE_SLUG = "mayakaripel/eng-sentences"
SEED = 42
MAX_ROWS = 30_000
TFIDF_FEATURES = 200

# Length-based complexity bins (character count)
LENGTH_BINS = [0, 30, 70, 150, 10_000]
LENGTH_LABELS = ["short", "medium", "long", "very_long"]


# ═════════════════════════════════════════════════════════════
#  Data loading
# ═════════════════════════════════════════════════════════════
def get_data(data_dir: Path) -> pd.DataFrame:
    ds_path = download_kaggle_dataset(
        KAGGLE_SLUG, data_dir,
        dataset_name="English Sentences (Tatoeba)",
    )

    # Try CSV first, then TSV
    csvs = sorted(ds_path.glob("*.csv")) + sorted(data_dir.glob("**/*.csv"))
    tsvs = sorted(ds_path.glob("*.tsv")) + sorted(data_dir.glob("**/*.tsv"))
    txts = sorted(ds_path.glob("*.txt")) + sorted(data_dir.glob("**/*.txt"))

    df = None
    if csvs:
        chosen = max(csvs, key=lambda f: f.stat().st_size)
        df = pd.read_csv(chosen, on_bad_lines="skip")
        logger.info("Loaded %d rows from %s", len(df), chosen.name)
    elif tsvs:
        chosen = max(tsvs, key=lambda f: f.stat().st_size)
        df = pd.read_csv(chosen, sep="\t", on_bad_lines="skip")
        # Tatoeba format: no header → first col name is a numeric id
        try:
            int(str(df.columns[0]))
            names = (["id", "lang", "text"] if df.shape[1] == 3
                     else [f"col{i}" for i in range(df.shape[1])])
            df = pd.read_csv(chosen, sep="\t", header=None,
                             on_bad_lines="skip", names=names)
        except (ValueError, IndexError):
            pass  # has a proper header
        logger.info("Loaded %d rows from %s", len(df), chosen.name)
    elif txts:
        chosen = max(txts, key=lambda f: f.stat().st_size)
        # Try tab-separated (Tatoeba format: id \t lang \t text)
        df = pd.read_csv(
            chosen, sep="\t", header=None, on_bad_lines="skip",
            names=["id", "lang", "text"],
        )
        logger.info("Loaded %d rows from %s", len(df), chosen.name)
    else:
        raise FileNotFoundError(f"No data files found in {ds_path}")

    return df


def _detect_text_column(df: pd.DataFrame) -> str:
    """Find the column most likely containing sentence text."""
    cols_lower = {c.lower().strip(): c for c in df.columns}
    for cand in ["text", "sentence", "content", "eng", "english", "message"]:
        if cand in cols_lower:
            return cols_lower[cand]
    # Heuristic: pick the object column with the longest average string length
    obj_cols = [c for c in df.columns if df[c].dtype == object]
    if obj_cols:
        avg_lens = {c: df[c].astype(str).str.len().mean() for c in obj_cols}
        return max(avg_lens, key=avg_lens.get)
    raise ValueError(f"Cannot find text column in {list(df.columns)}")


def _detect_label_column(df: pd.DataFrame) -> str | None:
    """Check if the dataset already has a categorical label column."""
    cols_lower = {c.lower().strip(): c for c in df.columns}
    for cand in ["category", "label", "class", "type", "tag", "level"]:
        if cand in cols_lower:
            col = cols_lower[cand]
            if 2 <= df[col].nunique() <= 50:
                return col
    return None


def preprocess(df: pd.DataFrame):
    """Prepare text and labels. Uses existing categories or creates length-based ones."""
    text_col = _detect_text_column(df)
    logger.info("Text column: '%s'", text_col)

    df = df.dropna(subset=[text_col]).reset_index(drop=True)
    df["_text"] = df[text_col].astype(str).str.strip()
    df = df[df["_text"].str.len() > 5].reset_index(drop=True)  # drop trivial rows

    # Check for existing label column
    label_col = _detect_label_column(df)

    if label_col is not None:
        logger.info("Found existing label column: '%s'", label_col)
        target_col = label_col
    else:
        # Create sentence-length complexity labels
        logger.info("No label column found — creating length-based complexity labels")
        df["complexity"] = pd.cut(
            df["_text"].str.len(),
            bins=LENGTH_BINS,
            labels=LENGTH_LABELS,
            include_lowest=True,
        )
        df = df.dropna(subset=["complexity"]).reset_index(drop=True)
        target_col = "complexity"

    df[target_col] = df[target_col].astype(str)

    # Sample down
    if len(df) > MAX_ROWS:
        df = df.sample(n=MAX_ROWS, random_state=SEED).reset_index(drop=True)
        logger.info("Sampled down to %d rows", MAX_ROWS)

    logger.info("Class distribution:\n%s", df[target_col].value_counts().to_string())
    return df, text_col, target_col


def build_tfidf_frame(df: pd.DataFrame, text_col: str, target_col: str) -> pd.DataFrame:
    """Create a DataFrame with TF-IDF features + additional text stats + target."""
    texts = df[text_col].astype(str) if text_col != "_text" else df["_text"]

    vectorizer = TfidfVectorizer(
        max_features=TFIDF_FEATURES,
        stop_words="english" if len(texts) > 500 else None,
        sublinear_tf=True,
    )
    tfidf_matrix = vectorizer.fit_transform(texts)
    feature_names = [f"tfidf_{i}" for i in range(tfidf_matrix.shape[1])]
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=feature_names)

    # Add text statistics as extra features
    tfidf_df["char_count"] = df["_text"].str.len().values
    tfidf_df["word_count"] = df["_text"].str.split().str.len().values
    tfidf_df["avg_word_len"] = (tfidf_df["char_count"] / tfidf_df["word_count"].replace(0, 1))
    tfidf_df["punct_count"] = df["_text"].str.count(r"[^\w\s]").values
    tfidf_df["upper_ratio"] = (
        df["_text"].str.count(r"[A-Z]").values / tfidf_df["char_count"].replace(0, 1)
    )

    tfidf_df[target_col] = df[target_col].values
    logger.info("Built feature matrix: %d rows × %d cols", *tfidf_df.shape)
    return tfidf_df


# ═════════════════════════════════════════════════════════════
#  Main
# ═════════════════════════════════════════════════════════════
def main():
    args = parse_common_args("Language Learning Chatbot — Sentence Classification")
    setup_logging()
    set_seed(args.seed, deterministic=True)
    configure_cuda_allocator()
    paths = project_paths(__file__)

    # -- download-only gate
    if args.download_only:
        try:
            get_data(paths["data"])
            logger.info("Download complete.")
        except Exception as e:
            logger.error("Download failed: %s", e)
        sys.exit(0)

    # 1. Load & preprocess
    try:
        df = get_data(paths["data"])
    except (FileNotFoundError, Exception) as exc:
        logger.error("Dataset error: %s", exc)
        dataset_missing_metrics(
            paths["outputs"],
            "English Sentences (Tatoeba)",
            ["https://www.kaggle.com/datasets/mayakaripel/eng-sentences"],
        )
        return

    if args.mode == "smoke":
        df = df.sample(n=min(200, len(df)), random_state=args.seed)
        logger.info("SMOKE TEST: %d rows", len(df))

    df, text_col, target_col = preprocess(df)

    if len(df) == 0:
        logger.error("Preprocessing left 0 rows — cannot train.")
        save_metrics(paths["outputs"], {"status": "error", "error": "0 rows after preprocessing"},
                     task_type="classification", mode=args.mode)
        return

    # 2. Build TF-IDF feature matrix
    try:
        model_df = build_tfidf_frame(df, text_col, target_col)
    except ValueError as exc:
        logger.warning("TF-IDF failed (%s) — falling back to text stats only", exc)
        # Build minimal feature set from text statistics
        model_df = pd.DataFrame()
        model_df["char_count"] = df["_text"].str.len()
        model_df["word_count"] = df["_text"].str.split().str.len()
        model_df["avg_word_len"] = model_df["char_count"] / model_df["word_count"].replace(0, 1)
        model_df["punct_count"] = df["_text"].str.count(r"[^\w\s]")
        model_df["upper_ratio"] = df["_text"].str.count(r"[A-Z]") / model_df["char_count"].replace(0, 1)
        model_df[target_col] = df[target_col].values

    # 3. Split data
    X_tr, y_tr, X_v, y_v, X_te, y_te = make_tabular_splits(model_df, target_col, "classification", args.seed)
    splits = {"X_train": X_tr, "y_train": y_tr, "X_val": X_v, "y_val": y_v, "X_test": X_te, "y_test": y_te}
    write_split_manifest(
        paths["outputs"],
        dataset_fp=dataset_fingerprint(paths["data"]),
        split_method="stratified_random",
        seed=args.seed,
        counts={"train": len(y_tr), "val": len(y_v), "test": len(y_te)},
    )

    # 4. Run auto-ML classification (PyCaret -> LazyPredict -> sklearn)
    logger.info("Running tabular auto-ML classification pipeline …")
    metrics = run_tabular_auto(model_df, target=target_col, output_dir=paths["outputs"],
                               task="classification", session_id=args.seed, splits=splits)
    metrics["run_metadata"] = run_metadata(args)
    save_metrics(paths["outputs"], metrics, task_type="classification", mode=args.mode)

    logger.info("Done.")


if __name__ == "__main__":
    main()
