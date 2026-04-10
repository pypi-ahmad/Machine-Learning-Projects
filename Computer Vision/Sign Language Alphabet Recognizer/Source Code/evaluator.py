"""Sign Language Alphabet Recognizer — evaluation with confusion matrix."""

from __future__ import annotations

from typing import Any

import numpy as np


def evaluate_model(
    clf,
    X_test: np.ndarray,
    y_test: np.ndarray,
    labels: list[str],
) -> dict[str, Any]:
    """Run evaluation and return a structured report.

    Returns
    -------
    dict with keys:
        accuracy        – overall accuracy (float)
        per_class       – list of {label, precision, recall, f1, support}
        confusion_matrix – 2-D list (true × predicted)
        labels          – ordered label list
    """
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
    )

    preds, confs = clf.predict_batch(X_test)
    acc = accuracy_score(y_test, preds)
    cm = confusion_matrix(y_test, preds, labels=list(range(len(labels))))

    # Per-class metrics via classification_report
    report_dict = classification_report(
        y_test, preds, target_names=labels, output_dict=True, zero_division=0,
    )

    per_class: list[dict[str, Any]] = []
    for label in labels:
        entry = report_dict.get(label, {})
        per_class.append(
            {
                "label": label,
                "precision": round(entry.get("precision", 0.0), 4),
                "recall": round(entry.get("recall", 0.0), 4),
                "f1": round(entry.get("f1-score", 0.0), 4),
                "support": int(entry.get("support", 0)),
            }
        )

    return {
        "accuracy": round(float(acc), 4),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "labels": labels,
    }


def print_confusion_matrix(
    cm: list[list[int]],
    labels: list[str],
) -> None:
    """Pretty-print a confusion matrix to stdout."""
    n = len(labels)
    # Header
    hdr = "     " + " ".join(f"{l:>4}" for l in labels)
    print(hdr)
    print("     " + "-" * (5 * n))
    for i, row in enumerate(cm):
        vals = " ".join(f"{v:4d}" for v in row)
        print(f"  {labels[i]:>2} |{vals}")


def save_confusion_matrix_image(
    cm: list[list[int]],
    labels: list[str],
    path: str,
) -> None:
    """Save confusion matrix as a PNG heatmap (if matplotlib available)."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed — skipping confusion matrix image.")
        return

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.6), max(6, len(labels) * 0.5)))
    cm_arr = np.array(cm)
    im = ax.imshow(cm_arr, interpolation="nearest", cmap=plt.cm.Blues)
    ax.set(
        xticks=range(len(labels)),
        yticks=range(len(labels)),
        xticklabels=labels,
        yticklabels=labels,
        xlabel="Predicted",
        ylabel="True",
        title="Confusion Matrix",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Confusion matrix image saved to {path}")
