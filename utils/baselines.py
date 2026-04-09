"""Baseline model implementations for all NLP project types.

Provides functions for:
- Classification (LazyPredict + LogReg/SVM on TF-IDF)
- Text analysis / EDA
- Summarization (transformers pipeline)
- Translation (transformers pipeline)
- Clustering (TF-IDF → KMeans + LDA)
- Image captioning (dataset stats + sample pairs)
- Name generation (dataset stats)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from utils.logger import get_logger
from utils.metrics import (
    classification_metrics,
    clustering_metrics,
    multilabel_metrics,
    save_metrics,
    summarization_metrics,
    topic_modeling_metrics,
    translation_metrics,
)
from utils.nlp_preprocess import (
    build_tfidf_features,
    infer_target_column,
    normalize_series,
    train_val_test_split,
)
from utils.seed import set_global_seed

logger = get_logger(__name__)

_WORKSPACE = Path(__file__).resolve().parent.parent
_OUTPUTS = _WORKSPACE / "outputs"


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


# ======================================================================
# A) Text Classification baselines
# ======================================================================

def run_classification_baselines(
    df: pd.DataFrame,
    text_col: str,
    target_col: str,
    slug: str,
    *,
    max_rows: int = 50_000,
    seed: int = 42,
) -> dict:
    """Run LazyPredict + LogReg/SVM baselines on TF-IDF features."""
    set_global_seed(seed)
    out_dir = _ensure_dir(_OUTPUTS / slug)

    # Prep data
    df = df[[text_col, target_col]].dropna().reset_index(drop=True)
    if len(df) > max_rows:
        df = df.sample(n=max_rows, random_state=seed).reset_index(drop=True)

    df["_clean"] = normalize_series(df[text_col])

    splits = train_val_test_split(df, target_col=target_col, seed=seed)
    train, test = splits["train"], splits["test"]

    # TF-IDF
    tfidf = build_tfidf_features(
        train["_clean"], test["_clean"], max_features=10_000,
    )
    X_train, X_test = tfidf["X_train"], tfidf["X_test"]
    y_train, y_test = train[target_col], test[target_col]

    results = {"slug": slug, "task": "classification", "n_train": len(train), "n_test": len(test)}

    # --- Baseline 1: LazyPredict ---
    try:
        from lazypredict.Supervised import LazyClassifier

        clf = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=None)
        models_df, _ = clf.fit(X_train, X_test, y_train, y_test)
        models_df = models_df.reset_index().rename(columns={"index": "Model"})
        models_df.to_csv(out_dir / "leaderboard.csv", index=False)
        results["lazypredict_top5"] = models_df.head(5).to_dict(orient="records")
        logger.info("LazyPredict: %d models evaluated", len(models_df))
    except Exception as exc:
        logger.warning("LazyPredict failed: %s", exc)
        results["lazypredict_error"] = str(exc)

    # --- Baseline 2: LogReg on TF-IDF ---
    try:
        from sklearn.linear_model import LogisticRegression

        lr = LogisticRegression(max_iter=1000, random_state=seed, n_jobs=-1)
        lr.fit(X_train, y_train)
        y_pred = lr.predict(X_test)
        y_proba = None
        try:
            y_proba = lr.predict_proba(X_test)
        except Exception:
            pass
        lr_metrics = classification_metrics(y_test, y_pred, y_proba)
        results["logreg_metrics"] = lr_metrics
        save_metrics(lr_metrics, out_dir / "logreg_metrics.json")
        logger.info("LogReg: acc=%.4f, f1=%.4f", lr_metrics["accuracy"], lr_metrics["f1_weighted"])
    except Exception as exc:
        logger.warning("LogReg failed: %s", exc)
        results["logreg_error"] = str(exc)

    # --- Baseline 3: Linear SVM ---
    try:
        from sklearn.svm import LinearSVC

        svm = LinearSVC(max_iter=2000, random_state=seed)
        svm.fit(X_train, y_train)
        y_pred_svm = svm.predict(X_test)
        svm_metrics = classification_metrics(y_test, y_pred_svm)
        results["svm_metrics"] = svm_metrics
        save_metrics(svm_metrics, out_dir / "svm_metrics.json")
        logger.info("SVM: acc=%.4f, f1=%.4f", svm_metrics["accuracy"], svm_metrics["f1_weighted"])
    except Exception as exc:
        logger.warning("SVM failed: %s", exc)
        results["svm_error"] = str(exc)

    # Save combined results
    save_metrics(results, out_dir / "baseline_metrics.json")
    return results


def run_multilabel_classification(
    df: pd.DataFrame,
    text_col: str,
    target_cols: list[str],
    slug: str,
    *,
    max_rows: int = 50_000,
    seed: int = 42,
) -> dict:
    """Run multi-label classification baselines (e.g. toxic comments)."""
    set_global_seed(seed)
    out_dir = _ensure_dir(_OUTPUTS / slug)

    cols = [text_col] + target_cols
    df = df[cols].dropna(subset=[text_col]).fillna(0).reset_index(drop=True)
    if len(df) > max_rows:
        df = df.sample(n=max_rows, random_state=seed).reset_index(drop=True)

    df["_clean"] = normalize_series(df[text_col])

    from sklearn.model_selection import train_test_split as sk_split

    train, test = sk_split(df, test_size=0.2, random_state=seed)

    tfidf = build_tfidf_features(train["_clean"], test["_clean"], max_features=10_000)
    X_train, X_test = tfidf["X_train"], tfidf["X_test"]
    y_train = train[target_cols].values
    y_test = test[target_cols].values

    results = {"slug": slug, "task": "multilabel_classification"}

    try:
        from sklearn.multiclass import OneVsRestClassifier
        from sklearn.linear_model import LogisticRegression

        ovr = OneVsRestClassifier(LogisticRegression(max_iter=1000, random_state=seed))
        ovr.fit(X_train, y_train)
        y_pred = ovr.predict(X_test)
        m = multilabel_metrics(y_test, y_pred)
        results["metrics"] = m
        save_metrics(m, out_dir / "baseline_metrics.json")
    except Exception as exc:
        results["error"] = str(exc)

    return results


# ======================================================================
# B) Text Analysis / EDA
# ======================================================================

def run_eda(
    df: pd.DataFrame,
    slug: str,
    text_col: str | None = None,
) -> dict:
    """Run EDA: token stats, n-grams, vocab size, label dist, missing values."""
    out_dir = _ensure_dir(_OUTPUTS / slug)
    fig_dir = _ensure_dir(out_dir / "figures")

    results = {
        "slug": slug,
        "task": "eda",
        "shape": list(df.shape),
        "columns": list(df.columns),
        "dtypes": {str(k): str(v) for k, v in df.dtypes.items()},
        "missing_values": df.isnull().sum().to_dict(),
        "missing_pct": (df.isnull().sum() / len(df) * 100).round(2).to_dict(),
    }

    # Text stats
    if text_col and text_col in df.columns:
        texts = df[text_col].dropna().astype(str)
        tokens = texts.str.split()
        results["text_stats"] = {
            "n_texts": int(len(texts)),
            "mean_token_count": float(tokens.str.len().mean()),
            "median_token_count": float(tokens.str.len().median()),
            "max_token_count": int(tokens.str.len().max()),
            "min_token_count": int(tokens.str.len().min()),
            "vocab_size": int(len(set(w for ws in tokens for w in ws))),
        }

        # Top unigrams
        from collections import Counter

        all_words = [w.lower() for ws in tokens for w in ws]
        top_unigrams = Counter(all_words).most_common(20)
        results["top_unigrams"] = top_unigrams

        # Save token length distribution chart
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(8, 4))
            tokens.str.len().hist(bins=50, ax=ax, edgecolor="black")
            ax.set_xlabel("Token count")
            ax.set_ylabel("Frequency")
            ax.set_title(f"{slug}: Token Length Distribution")
            fig.tight_layout()
            fig.savefig(fig_dir / "token_length_dist.png", dpi=100)
            plt.close(fig)
        except Exception:
            pass

    # Label distribution
    for col in df.columns:
        if df[col].dtype in ("object", "category") and df[col].nunique() <= 50:
            results[f"label_dist_{col}"] = df[col].value_counts().head(20).to_dict()

    # Numeric column summary
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if numeric_cols:
        results["numeric_summary"] = df[numeric_cols].describe().to_dict()

    save_metrics(results, out_dir / "eda_report.json")
    logger.info("EDA complete for %s: %d rows, %d cols", slug, df.shape[0], df.shape[1])
    return results


# ======================================================================
# C) Summarization baseline
# ======================================================================

def run_summarization_baseline(
    articles: list[str],
    references: list[str] | None = None,
    slug: str = "",
    max_samples: int = 100,
) -> dict:
    """Run a small transformers summarization baseline."""
    out_dir = _ensure_dir(_OUTPUTS / slug)
    results = {"slug": slug, "task": "summarization", "n_articles": len(articles)}

    articles = articles[:max_samples]
    if references:
        references = references[:max_samples]

    try:
        from transformers import pipeline

        summarizer = pipeline(
            "summarization",
            model="sshleifer/distilbart-cnn-6-6",
            device=-1,  # CPU for quick baseline
        )
        predictions = []
        for art in articles:
            # Truncate to 1024 chars for speed
            truncated = art[:1024]
            try:
                out = summarizer(truncated, max_length=80, min_length=20, do_sample=False)
                predictions.append(out[0]["summary_text"])
            except Exception:
                predictions.append("")

        results["n_predictions"] = len(predictions)

        # Save samples
        samples = []
        for i, pred in enumerate(predictions[:25]):
            sample = {"article_preview": articles[i][:300], "prediction": pred}
            if references:
                sample["reference"] = references[i][:300]
            samples.append(sample)
        save_metrics(samples, out_dir / "samples.json")

        # Metrics
        if references:
            m = summarization_metrics(predictions, references)
            results["metrics"] = m
            save_metrics(m, out_dir / "baseline_metrics.json")
        else:
            results["metrics"] = {"note": "NO_REFERENCE — cannot compute ROUGE"}

    except Exception as exc:
        results["error"] = str(exc)
        logger.error("Summarization baseline failed: %s", exc)

    return results


# ======================================================================
# D) Translation baseline
# ======================================================================

def run_translation_baseline(
    source_texts: list[str],
    reference_texts: list[str] | None = None,
    slug: str = "",
    max_samples: int = 200,
) -> dict:
    """Run a small transformers translation baseline."""
    out_dir = _ensure_dir(_OUTPUTS / slug)
    results = {"slug": slug, "task": "translation", "n_pairs": len(source_texts)}

    source_texts = source_texts[:max_samples]
    if reference_texts:
        reference_texts = reference_texts[:max_samples]

    try:
        from transformers import pipeline

        translator = pipeline(
            "translation_en_to_fr",
            model="Helsinki-NLP/opus-mt-en-fr",
            device=-1,
        )
        predictions = []
        for text in source_texts:
            try:
                out = translator(text[:512], max_length=128)
                predictions.append(out[0]["translation_text"])
            except Exception:
                predictions.append("")

        results["n_predictions"] = len(predictions)

        if reference_texts:
            m = translation_metrics(predictions, reference_texts)
            results["metrics"] = m
            save_metrics(m, out_dir / "baseline_metrics.json")

        # Save samples
        samples = []
        for i, pred in enumerate(predictions[:25]):
            sample = {"source": source_texts[i], "prediction": pred}
            if reference_texts:
                sample["reference"] = reference_texts[i]
            samples.append(sample)
        save_metrics(samples, out_dir / "samples.json")

    except Exception as exc:
        results["error"] = str(exc)
        logger.error("Translation baseline failed: %s", exc)

    return results


# ======================================================================
# E) Clustering / Topic Modelling baseline
# ======================================================================

def run_clustering_baseline(
    texts: list[str] | pd.Series,
    slug: str = "",
    n_clusters: int = 5,
    seed: int = 42,
) -> dict:
    """TF-IDF → KMeans + gensim LDA baseline."""
    set_global_seed(seed)
    out_dir = _ensure_dir(_OUTPUTS / slug)

    if isinstance(texts, pd.Series):
        texts = texts.dropna().tolist()

    cleaned = [t.lower().strip() for t in texts if isinstance(t, str) and len(t.strip()) > 10]
    if len(cleaned) < 20:
        return {"slug": slug, "error": "Too few texts for clustering", "n_texts": len(cleaned)}

    results = {"slug": slug, "task": "clustering", "n_texts": len(cleaned)}

    # TF-IDF + KMeans
    try:
        from sklearn.cluster import KMeans
        from sklearn.feature_extraction.text import TfidfVectorizer

        vec = TfidfVectorizer(max_features=5000, stop_words="english", min_df=2)
        X = vec.fit_transform(cleaned)
        km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10, max_iter=300)
        labels = km.fit_predict(X)
        m = clustering_metrics(X, labels)
        results["kmeans_metrics"] = m
        save_metrics(m, out_dir / "kmeans_metrics.json")

        # Top terms per cluster
        order_centroids = km.cluster_centers_.argsort()[:, ::-1]
        terms = vec.get_feature_names_out()
        top_terms = {}
        for i in range(n_clusters):
            top_terms[f"cluster_{i}"] = [str(terms[ind]) for ind in order_centroids[i, :10]]
        results["kmeans_top_terms"] = top_terms

    except Exception as exc:
        results["kmeans_error"] = str(exc)

    # LDA + coherence
    try:
        from gensim import corpora, models as gensim_models

        tokenized = [t.split() for t in cleaned]
        dictionary = corpora.Dictionary(tokenized)
        dictionary.filter_extremes(no_below=2, no_above=0.8)
        corpus = [dictionary.doc2bow(doc) for doc in tokenized]

        lda = gensim_models.LdaMulticore(
            corpus, num_topics=n_clusters, id2word=dictionary,
            passes=5, random_state=seed, workers=2,
        )
        lda_m = topic_modeling_metrics(lda, corpus, dictionary, tokenized)
        results["lda_metrics"] = lda_m

        # Save topics
        topics = {}
        for t_id in range(n_clusters):
            topics[f"topic_{t_id}"] = lda.print_topic(t_id, topn=10)
        results["lda_topics"] = topics
        save_metrics(results, out_dir / "baseline_metrics.json")

    except Exception as exc:
        results["lda_error"] = str(exc)

    return results


# ======================================================================
# F) Image Captioning baseline
# ======================================================================

def run_captioning_baseline(
    captions_path: str | Path,
    images_dir: str | Path,
    slug: str = "",
) -> dict:
    """Dataset integrity check + sample pairs for image captioning."""
    out_dir = _ensure_dir(_OUTPUTS / slug)
    captions_path = Path(captions_path)
    images_dir = Path(images_dir)

    results = {"slug": slug, "task": "image_captioning"}

    # Parse captions
    try:
        lines = captions_path.read_text(encoding="utf-8").strip().split("\n")
        header = lines[0]
        records = []
        for line in lines[1:]:
            parts = line.split(",", 1)
            if len(parts) == 2:
                records.append({"image": parts[0].strip(), "caption": parts[1].strip()})
        results["n_captions"] = len(records)
        results["n_unique_images"] = len(set(r["image"] for r in records))

        # Check how many images exist
        if images_dir.exists():
            image_files = set(f.name for f in images_dir.iterdir() if f.is_file())
            results["n_image_files"] = len(image_files)
            referenced = set(r["image"] for r in records)
            results["images_present"] = len(referenced & image_files)
            results["images_missing"] = len(referenced - image_files)
        else:
            results["images_dir_missing"] = True

        # Save 25 sample pairs
        samples = records[:25]
        save_metrics(samples, out_dir / "samples.json")

    except Exception as exc:
        results["error"] = str(exc)

    save_metrics(results, out_dir / "baseline_metrics.json")
    return results


# ======================================================================
# G) Name Generation baseline (dataset stats)
# ======================================================================

def run_generation_baseline(
    names_dir: str | Path,
    slug: str = "",
) -> dict:
    """Dataset stats for name generation project."""
    out_dir = _ensure_dir(_OUTPUTS / slug)
    names_dir = Path(names_dir)

    results = {"slug": slug, "task": "generation"}

    try:
        langs = {}
        for fp in sorted(names_dir.glob("*.txt")):
            lang = fp.stem
            names = [n.strip() for n in fp.read_text(encoding="utf-8").splitlines() if n.strip()]
            langs[lang] = {"count": len(names), "samples": names[:5]}

        results["n_languages"] = len(langs)
        results["total_names"] = sum(v["count"] for v in langs.values())
        results["languages"] = langs

    except Exception as exc:
        results["error"] = str(exc)

    save_metrics(results, out_dir / "baseline_metrics.json")
    return results
