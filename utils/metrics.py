"""Universal metrics module for all NLP task types.

Supports: classification, regression, clustering, summarization,
translation, text generation, and topic modelling.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from utils.logger import get_logger

logger = get_logger(__name__)


# ======================================================================
# Classification
# ======================================================================

def classification_metrics(
    y_true,
    y_pred,
    y_proba=None,
    *,
    average: str = "weighted",
    labels=None,
) -> dict:
    """Compute comprehensive classification metrics."""
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        classification_report,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
    )

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    m: dict = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision_weighted": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_micro": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
    }

    # Confusion matrix as nested list
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    m["confusion_matrix"] = cm.tolist()

    # Classification report as dict
    m["classification_report"] = classification_report(
        y_true, y_pred, output_dict=True, zero_division=0, labels=labels,
    )

    # ROC-AUC (binary & multiclass OVR)
    if y_proba is not None:
        try:
            from sklearn.metrics import roc_auc_score

            y_proba = np.asarray(y_proba)
            n_classes = len(np.unique(y_true))
            if n_classes == 2:
                if y_proba.ndim == 2:
                    m["roc_auc"] = float(roc_auc_score(y_true, y_proba[:, 1]))
                else:
                    m["roc_auc"] = float(roc_auc_score(y_true, y_proba))
            elif n_classes > 2:
                m["roc_auc_ovr"] = float(
                    roc_auc_score(y_true, y_proba, multi_class="ovr", average="weighted")
                )
        except Exception as exc:
            m["roc_auc_error"] = str(exc)

    # PR-AUC (binary only)
    if y_proba is not None:
        try:
            from sklearn.metrics import average_precision_score

            n_classes = len(np.unique(y_true))
            if n_classes == 2:
                if y_proba.ndim == 2:
                    m["pr_auc"] = float(average_precision_score(y_true, y_proba[:, 1]))
                else:
                    m["pr_auc"] = float(average_precision_score(y_true, y_proba))
        except Exception as exc:
            m["pr_auc_error"] = str(exc)

    logger.info("Classification metrics: acc=%.4f, f1_w=%.4f", m["accuracy"], m["f1_weighted"])
    return m


def multilabel_metrics(y_true, y_pred) -> dict:
    """Metrics for multi-label classification (e.g. toxic comment)."""
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        hamming_loss,
        precision_score,
        recall_score,
    )

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    m = {
        "subset_accuracy": float(accuracy_score(y_true, y_pred)),
        "hamming_loss": float(hamming_loss(y_true, y_pred)),
        "f1_micro": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1_samples": float(f1_score(y_true, y_pred, average="samples", zero_division=0)),
        "precision_micro": float(precision_score(y_true, y_pred, average="micro", zero_division=0)),
        "recall_micro": float(recall_score(y_true, y_pred, average="micro", zero_division=0)),
    }
    logger.info("Multi-label: subset_acc=%.4f, f1_micro=%.4f", m["subset_accuracy"], m["f1_micro"])
    return m


# ======================================================================
# Regression
# ======================================================================

def regression_metrics(y_true, y_pred) -> dict:
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    m = {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mse": float(mean_squared_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }
    logger.info("Regression: MAE=%.4f, RMSE=%.4f, R2=%.4f", m["mae"], m["rmse"], m["r2"])
    return m


# ======================================================================
# Clustering
# ======================================================================

def clustering_metrics(X, labels) -> dict:
    """Compute clustering quality metrics."""
    from sklearn.metrics import (
        calinski_harabasz_score,
        davies_bouldin_score,
        silhouette_score,
    )

    labels = np.asarray(labels)
    n_clusters = len(set(labels) - {-1})
    if n_clusters < 2:
        logger.warning("Fewer than 2 clusters; skipping cluster metrics.")
        return {"n_clusters": n_clusters, "error": "fewer_than_2_clusters"}

    X_dense = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
    m = {
        "n_clusters": n_clusters,
        "silhouette": float(silhouette_score(X_dense, labels, sample_size=min(5000, len(labels)))),
        "calinski_harabasz": float(calinski_harabasz_score(X_dense, labels)),
        "davies_bouldin": float(davies_bouldin_score(X_dense, labels)),
    }
    logger.info(
        "Clustering: k=%d, silhouette=%.4f, CH=%.1f, DB=%.4f",
        m["n_clusters"], m["silhouette"], m["calinski_harabasz"], m["davies_bouldin"],
    )
    return m


# ======================================================================
# Summarization (ROUGE)
# ======================================================================

def summarization_metrics(predictions: list[str], references: list[str]) -> dict:
    """Compute ROUGE metrics via evaluate library."""
    try:
        import evaluate

        rouge = evaluate.load("rouge")
        result = rouge.compute(predictions=predictions, references=references)
        m = {k: float(v) for k, v in result.items()}
        logger.info("ROUGE: R1=%.4f, R2=%.4f, RL=%.4f", m.get("rouge1", 0), m.get("rouge2", 0), m.get("rougeL", 0))
        return m
    except Exception as exc:
        logger.error("ROUGE computation failed: %s", exc)
        return {"error": str(exc)}


# ======================================================================
# Translation (BLEU / chrF)
# ======================================================================

def translation_metrics(predictions: list[str], references: list[str]) -> dict:
    """Compute BLEU and chrF metrics via evaluate library."""
    m: dict = {}
    try:
        import evaluate

        # BLEU expects list-of-list references
        refs_nested = [[r] for r in references]
        bleu = evaluate.load("bleu")
        bleu_result = bleu.compute(predictions=predictions, references=refs_nested)
        m["bleu"] = float(bleu_result["bleu"])
        m["bleu_precisions"] = [float(p) for p in bleu_result.get("precisions", [])]
    except Exception as exc:
        m["bleu_error"] = str(exc)

    try:
        import evaluate

        chrf = evaluate.load("chrf")
        chrf_result = chrf.compute(predictions=predictions, references=references)
        m["chrf"] = float(chrf_result["score"])
    except Exception as exc:
        m["chrf_error"] = str(exc)

    logger.info("Translation: BLEU=%.4f, chrF=%.2f", m.get("bleu", 0), m.get("chrf", 0))
    return m


# ======================================================================
# Text Generation
# ======================================================================

def generation_metrics(
    predictions: list[str],
    references: list[str] | None = None,
) -> dict:
    """Compute text generation metrics (perplexity approx + BLEU/ROUGE if refs)."""
    m: dict = {}

    # Attempt approximate perplexity via model
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        ppl_model = "gpt2"
        tokenizer = AutoTokenizer.from_pretrained(ppl_model)
        model = AutoModelForCausalLM.from_pretrained(ppl_model)
        model.eval()

        losses = []
        for text in predictions[:50]:  # limit for speed
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            with torch.no_grad():
                out = model(**inputs, labels=inputs["input_ids"])
            losses.append(out.loss.item())
        m["approx_perplexity"] = float(np.exp(np.mean(losses)))
    except Exception as exc:
        m["perplexity_error"] = str(exc)

    if references:
        m.update(summarization_metrics(predictions, references))

    logger.info("Generation: perplexity≈%.2f", m.get("approx_perplexity", 0))
    return m


# ======================================================================
# Topic Modelling
# ======================================================================

def topic_modeling_metrics(
    model,
    corpus,
    dictionary,
    texts: list[list[str]],
) -> dict:
    """Compute topic coherence (c_v) for a gensim LDA model."""
    try:
        from gensim.models import CoherenceModel

        cm = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence="c_v")
        coherence = cm.get_coherence()
        m = {
            "coherence_cv": float(coherence),
            "num_topics": model.num_topics,
        }
        logger.info("Topic model: coherence_cv=%.4f, n_topics=%d", coherence, model.num_topics)
        return m
    except Exception as exc:
        logger.error("Topic coherence failed: %s", exc)
        return {"error": str(exc)}


# ======================================================================
# Helpers
# ======================================================================

def save_metrics(metrics: dict, path: str | Path) -> Path:
    """Save metrics dict to JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, default=str)
    logger.info("Saved metrics to %s", path)
    return path
