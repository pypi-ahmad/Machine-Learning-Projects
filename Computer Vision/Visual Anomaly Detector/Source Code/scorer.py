"""Visual Anomaly Detector — anomaly scoring.
"""Visual Anomaly Detector — anomaly scoring.

Implements Mahalanobis distance and k-NN anomaly scoring against a
learned distribution of normal features. Scoring is fully separated
from feature extraction and thresholding.

Usage::

    from scorer import AnomalyScorer

    scorer = AnomalyScorer(method="mahalanobis")
    scorer.fit(normal_features)               # (N, D) array
    score = scorer.score(feature_vector)       # scalar anomaly score
    scores = scorer.score_batch(features)      # (M,) array
"""
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

log = logging.getLogger("visual_anomaly.scorer")


class AnomalyScorer:
    """Compute anomaly scores from feature vectors."""

    def __init__(
        self,
        method: str = "mahalanobis",
        knn_k: int = 5,
        regularization: float = 1e-5,
    ) -> None:
        self.method = method
        self.knn_k = knn_k
        self.regularization = regularization

        self._mean: np.ndarray | None = None
        self._cov_inv: np.ndarray | None = None
        self._normal_features: np.ndarray | None = None
        self._fitted = False

    @property
    def fitted(self) -> bool:
        return self._fitted

    def fit(self, features: np.ndarray) -> dict:
        """Fit the scorer to normal feature vectors.
        """Fit the scorer to normal feature vectors.

        Parameters
        ----------
        features : np.ndarray
            Normal features of shape ``(N, D)`` where N >= 2.

        Returns
        -------
        dict
            Fitting summary.
        """
        """
        if features.ndim != 2 or features.shape[0] < 2:
            raise ValueError(
                f"Need at least 2 samples with 2D shape, got {features.shape}"
            )

        self._normal_features = features.astype(np.float64)
        self._mean = np.mean(self._normal_features, axis=0)

        # Regularized covariance inverse for Mahalanobis
        cov = np.cov(self._normal_features.T)
        cov += np.eye(cov.shape[0]) * self.regularization
        self._cov_inv = np.linalg.inv(cov)

        self._fitted = True
        log.info(
            "Fitted on %d samples (dim=%d, method=%s)",
            features.shape[0], features.shape[1], self.method,
        )

        return {
            "num_samples": features.shape[0],
            "feature_dim": features.shape[1],
            "mean_norm": float(np.linalg.norm(self._mean)),
        }

    def score(self, feature: np.ndarray) -> dict[str, float]:
        """Score a single feature vector.
        """Score a single feature vector.

        Returns
        -------
        dict
            ``mahalanobis``, ``knn``, and ``combined`` scores.
        """
        """
        if not self._fitted:
            raise RuntimeError("Scorer not fitted. Call fit() first.")

        diff = feature.astype(np.float64) - self._mean

        # Mahalanobis distance
        mahal = float(np.sqrt(np.clip(diff @ self._cov_inv @ diff, 0, None)))

        # k-NN distance (mean of k nearest)
        dists = np.linalg.norm(
            self._normal_features - feature.astype(np.float64), axis=1,
        )
        knn = float(np.mean(np.sort(dists)[: self.knn_k]))

        return {
            "mahalanobis": mahal,
            "knn": knn,
            "combined": (mahal + knn) / 2.0,
        }

    def score_batch(self, features: np.ndarray) -> list[dict[str, float]]:
        """Score a batch of feature vectors."""
        return [self.score(f) for f in features]

    def score_primary(self, feature: np.ndarray) -> float:
        """Return the primary anomaly score based on configured method."""
        scores = self.score(feature)
        return scores.get(self.method, scores["mahalanobis"])

    # ── Persistence ────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        """Save fitted scorer to NPZ file."""
        if not self._fitted:
            raise RuntimeError("Scorer not fitted")
        np.savez_compressed(
            str(path),
            mean=self._mean,
            cov_inv=self._cov_inv,
            normal_features=self._normal_features,
            method=np.array(self.method),
            knn_k=np.array(self.knn_k),
            regularization=np.array(self.regularization),
        )
        log.info("Saved scorer -> %s", path)

    def load(self, path: str | Path) -> None:
        """Load fitted scorer from NPZ file."""
        data = np.load(str(path), allow_pickle=False)
        self._mean = data["mean"]
        self._cov_inv = data["cov_inv"]
        self._normal_features = data["normal_features"]
        self.method = str(data["method"])
        self.knn_k = int(data["knn_k"])
        self.regularization = float(data["regularization"])
        self._fitted = True
        log.info(
            "Loaded scorer (%d samples, dim=%d)",
            self._normal_features.shape[0], self._normal_features.shape[1],
        )
