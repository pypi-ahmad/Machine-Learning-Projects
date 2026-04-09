"""Standard NLP preprocessing pipeline.

Provides text normalization, feature extraction, target inference,
and reproducible train/val/test splitting.
"""

from __future__ import annotations

import html
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import issparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

from utils.logger import get_logger
from utils.seed import set_global_seed

logger = get_logger(__name__)

# --------------------------------------------------------------------------
# Text normalization
# --------------------------------------------------------------------------

_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_EMAIL_RE = re.compile(r"\S+@\S+\.\S+")
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_MULTI_WS_RE = re.compile(r"\s+")
_USERNAME_RE = re.compile(r"@\w+")


def normalize_text(
    text: str,
    *,
    lowercase: bool = True,
    strip_html: bool = True,
    remove_urls: bool = True,
    remove_emails: bool = True,
    remove_usernames: bool = False,
    unicode_norm: str = "NFKD",
) -> str:
    """Normalize a text string for NLP processing."""
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize(unicode_norm, text)
    if strip_html:
        text = _HTML_TAG_RE.sub(" ", text)
        text = html.unescape(text)
    if remove_urls:
        text = _URL_RE.sub(" ", text)
    if remove_emails:
        text = _EMAIL_RE.sub(" ", text)
    if remove_usernames:
        text = _USERNAME_RE.sub(" ", text)
    if lowercase:
        text = text.lower()
    text = _MULTI_WS_RE.sub(" ", text).strip()
    return text


def normalize_series(series: pd.Series, **kwargs) -> pd.Series:
    """Apply normalize_text to a pandas Series."""
    return series.fillna("").apply(lambda t: normalize_text(t, **kwargs))


# --------------------------------------------------------------------------
# Feature extraction
# --------------------------------------------------------------------------


def build_tfidf_features(
    train_texts: list[str] | pd.Series,
    test_texts: list[str] | pd.Series | None = None,
    max_features: int = 10_000,
    ngram_range: tuple[int, int] = (1, 2),
    min_df: int = 2,
) -> dict:
    """Build TF-IDF feature matrices.

    Returns dict with keys: X_train, X_test (if given), vectorizer.
    """
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,
        sublinear_tf=True,
        strip_accents="unicode",
    )
    X_train = vectorizer.fit_transform(train_texts)
    result = {"X_train": X_train, "vectorizer": vectorizer}
    if test_texts is not None:
        result["X_test"] = vectorizer.transform(test_texts)
    logger.info(
        "TF-IDF: %d features, train shape %s",
        len(vectorizer.vocabulary_),
        X_train.shape,
    )
    return result


def get_tokenizer(model_name: str = "bert-base-uncased"):
    """Return a HuggingFace AutoTokenizer."""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_name)


# --------------------------------------------------------------------------
# Target column inference
# --------------------------------------------------------------------------

_TARGET_CANDIDATES = [
    "label", "target", "y", "class", "sentiment", "category",
    "cyberbullying_type", "positivity", "relevance", "toxic",
    "Sentiment", "Label", "Category", "Class", "Rating",
    "Recommended IND",
]
_TARGET_ANTI = {"id", "text", "comment_text", "review", "title", "headline", "date", "author"}


def infer_target_column(df: pd.DataFrame, task_hint: str | None = None) -> str | None:
    """Heuristically infer the target/label column from a DataFrame."""
    cols = set(df.columns)
    # Exact match priority
    for cand in _TARGET_CANDIDATES:
        if cand in cols:
            logger.info("Inferred target column: '%s'", cand)
            return cand
    # Case-insensitive match
    col_lower = {c.lower(): c for c in df.columns}
    for cand in _TARGET_CANDIDATES:
        if cand.lower() in col_lower:
            match = col_lower[cand.lower()]
            logger.info("Inferred target column (case-insensitive): '%s'", match)
            return match
    # Last resort: look for columns with few unique values relative to rows
    for col in df.columns:
        if col.lower() in _TARGET_ANTI:
            continue
        if df[col].dtype in ("object", "category"):
            nunique = df[col].nunique()
            if 2 <= nunique <= 50:
                logger.info("Inferred target column (heuristic): '%s' (%d classes)", col, nunique)
                return col
    logger.warning("Could not infer target column from columns: %s", list(df.columns))
    return None


# --------------------------------------------------------------------------
# Splitting
# --------------------------------------------------------------------------


def load_seed() -> int:
    """Load random seed from config/base_config.yaml."""
    try:
        import yaml

        cfg_path = Path(__file__).resolve().parent.parent / "config" / "base_config.yaml"
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
        return int(cfg.get("random_seed", 42))
    except Exception:
        return 42


def train_val_test_split(
    df: pd.DataFrame,
    target_col: str | None = None,
    test_size: float = 0.2,
    val_size: float = 0.1,
    seed: int | None = None,
    max_rows: int | None = None,
) -> dict[str, pd.DataFrame]:
    """Split a DataFrame into train / val / test sets.

    Returns dict with keys: train, val, test.
    Stratifies by target_col if classification (few unique values).
    """
    if seed is None:
        seed = load_seed()
    set_global_seed(seed)

    if max_rows and len(df) > max_rows:
        df = df.sample(n=max_rows, random_state=seed).reset_index(drop=True)
        logger.info("Subsampled to %d rows", max_rows)

    stratify = None
    if target_col and target_col in df.columns:
        nunique = df[target_col].nunique()
        if 2 <= nunique <= 200:
            # Check minimum class count
            min_count = df[target_col].value_counts().min()
            if min_count >= 3:
                stratify = df[target_col]

    train_val, test = train_test_split(
        df, test_size=test_size, random_state=seed, stratify=stratify,
    )

    # Val from train
    relative_val = val_size / (1 - test_size)
    stratify_tv = None
    if stratify is not None and target_col in train_val.columns:
        min_count_tv = train_val[target_col].value_counts().min()
        if min_count_tv >= 2:
            stratify_tv = train_val[target_col]

    train, val = train_test_split(
        train_val, test_size=relative_val, random_state=seed, stratify=stratify_tv,
    )

    logger.info(
        "Split: train=%d, val=%d, test=%d (seed=%d)",
        len(train), len(val), len(test), seed,
    )
    return {"train": train.reset_index(drop=True),
            "val": val.reset_index(drop=True),
            "test": test.reset_index(drop=True)}


def save_splits(splits: dict[str, pd.DataFrame], output_dir: str | Path) -> None:
    """Save train/val/test DataFrames as CSV."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, df in splits.items():
        path = output_dir / f"{name}.csv"
        df.to_csv(path, index=False)
    logger.info("Saved splits to %s", output_dir)
