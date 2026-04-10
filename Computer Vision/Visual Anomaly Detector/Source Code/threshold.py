"""Visual Anomaly Detector — threshold selection utilities.

Provides automatic and manual threshold-tuning strategies for the
anomaly scorer. Separated from scoring so thresholds can be swept
after training without recomputing features.

Usage::

    from threshold import ThresholdSelector

    sel = ThresholdSelector()
    thresh = sel.percentile(normal_scores, pct=95)
    thresh = sel.f1_optimal(normal_scores, anomaly_scores)
    report = sel.sweep(normal_scores, anomaly_scores, steps=50)
"""

from __future__ import annotations

import logging

import numpy as np

log = logging.getLogger("visual_anomaly.threshold")


class ThresholdSelector:
    """Threshold selection and sweep utilities."""

    @staticmethod
    def percentile(normal_scores: np.ndarray, pct: float = 95.0) -> float:
        """Select threshold as a percentile of normal scores.

        Parameters
        ----------
        normal_scores : np.ndarray
            Anomaly scores computed on normal training/validation images.
        pct : float
            Percentile (0–100). Higher = fewer false positives.

        Returns
        -------
        float
            Threshold value.
        """
        t = float(np.percentile(normal_scores, pct))
        log.info("Percentile threshold (p=%.1f): %.4f", pct, t)
        return t

    @staticmethod
    def mean_std(
        normal_scores: np.ndarray, n_sigma: float = 3.0,
    ) -> float:
        """Select threshold as mean + n_sigma * std of normal scores."""
        t = float(np.mean(normal_scores) + n_sigma * np.std(normal_scores))
        log.info("Mean+%.1fσ threshold: %.4f", n_sigma, t)
        return t

    @staticmethod
    def f1_optimal(
        normal_scores: np.ndarray,
        anomaly_scores: np.ndarray,
        steps: int = 200,
    ) -> float:
        """Find threshold that maximises F1 score.

        Requires both normal and anomaly scores (validation set).
        """
        all_scores = np.concatenate([normal_scores, anomaly_scores])
        lo, hi = float(all_scores.min()), float(all_scores.max())

        best_t, best_f1 = lo, 0.0
        for t in np.linspace(lo, hi, steps):
            tp = int(np.sum(anomaly_scores > t))
            fp = int(np.sum(normal_scores > t))
            fn = int(np.sum(anomaly_scores <= t))

            precision = tp / (tp + fp) if (tp + fp) else 0.0
            recall = tp / (tp + fn) if (tp + fn) else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

            if f1 > best_f1:
                best_f1 = f1
                best_t = float(t)

        log.info("F1-optimal threshold: %.4f (F1=%.4f)", best_t, best_f1)
        return best_t

    @staticmethod
    def sweep(
        normal_scores: np.ndarray,
        anomaly_scores: np.ndarray,
        steps: int = 50,
    ) -> list[dict]:
        """Sweep thresholds and return metrics at each step.

        Returns
        -------
        list[dict]
            Each entry has: threshold, tp, fp, tn, fn, accuracy,
            precision, recall, f1, fpr.
        """
        all_scores = np.concatenate([normal_scores, anomaly_scores])
        lo, hi = float(all_scores.min()), float(all_scores.max())
        results = []

        for t in np.linspace(lo, hi, steps):
            tp = int(np.sum(anomaly_scores > t))
            fp = int(np.sum(normal_scores > t))
            tn = int(np.sum(normal_scores <= t))
            fn = int(np.sum(anomaly_scores <= t))
            total = tp + fp + tn + fn

            precision = tp / (tp + fp) if (tp + fp) else 0.0
            recall = tp / (tp + fn) if (tp + fn) else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
            fpr = fp / (fp + tn) if (fp + tn) else 0.0
            acc = (tp + tn) / total if total else 0.0

            results.append({
                "threshold": round(float(t), 4),
                "tp": tp, "fp": fp, "tn": tn, "fn": fn,
                "accuracy": round(acc, 4),
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f1, 4),
                "fpr": round(fpr, 4),
            })

        return results

    @staticmethod
    def auroc(normal_scores: np.ndarray, anomaly_scores: np.ndarray) -> float:
        """Compute AUROC using the trapezoidal rule.

        Simple implementation without sklearn dependency.
        """
        labels = np.concatenate([
            np.zeros(len(normal_scores)),
            np.ones(len(anomaly_scores)),
        ])
        scores = np.concatenate([normal_scores, anomaly_scores])

        # Sort by descending score
        order = np.argsort(-scores)
        labels_sorted = labels[order]

        # Compute TPR/FPR at each unique threshold
        n_pos = int(np.sum(labels))
        n_neg = len(labels) - n_pos
        if n_pos == 0 or n_neg == 0:
            return 0.0

        tp_cumsum = np.cumsum(labels_sorted)
        fp_cumsum = np.cumsum(1 - labels_sorted)

        tpr = tp_cumsum / n_pos
        fpr = fp_cumsum / n_neg

        # Prepend (0, 0)
        tpr = np.concatenate([[0], tpr])
        fpr = np.concatenate([[0], fpr])

        # Trapezoidal integration
        auc = float(np.trapz(tpr, fpr))
        log.info("AUROC: %.4f", auc)
        return auc
