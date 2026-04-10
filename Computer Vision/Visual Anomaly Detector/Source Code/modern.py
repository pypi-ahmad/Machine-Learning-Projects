"""Modern v2 pipeline — Visual Anomaly Detector.

Uses:     ResNet feature extraction + Mahalanobis / k-NN anomaly scoring
Pipeline: Feature extractor → scorer → threshold → heatmap

Delegates feature extraction to ``feature_extractor.py``, scoring to
``scorer.py``, threshold selection to ``threshold.py``, heatmap
generation to ``heatmap.py``, and visualisation to ``visualize.py``.

This file is the thin CVProject adapter that plugs into the repo's
global registry.
"""

import sys
from pathlib import Path

_PROJECT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_PROJECT_DIR))
sys.path.insert(0, str(_PROJECT_DIR.parents[1]))

import json

import cv2
import numpy as np

from core.base import CVProject
from core.registry import register

from config import AnomalyConfig, load_config
from feature_extractor import FeatureExtractor
from heatmap import HeatmapGenerator
from scorer import AnomalyScorer
from visualize import draw_result


@register("visual_anomaly_detector")
class VisualAnomalyDetectorModern(CVProject):
    project_type = "anomaly"
    description = "Unsupervised visual anomaly detection — train on normal, flag abnormal"
    legacy_tech = "N/A (new project)"
    modern_tech = "ResNet feature extraction + Mahalanobis / k-NN anomaly scoring + patch heatmaps"

    def __init__(self, config: AnomalyConfig | None = None) -> None:
        super().__init__()
        self._cfg = config or AnomalyConfig()
        self._extractor: FeatureExtractor | None = None
        self._scorer = AnomalyScorer(
            method=self._cfg.scoring_method,
            knn_k=self._cfg.knn_k,
            regularization=self._cfg.regularization,
        )
        self._threshold = self._cfg.anomaly_threshold
        self._heatmap_gen: HeatmapGenerator | None = None

    # ── CVProject interface ────────────────────────────────

    def load(self) -> None:
        self._extractor = FeatureExtractor(
            backbone=self._cfg.backbone, imgsz=self._cfg.imgsz,
        )
        self._extractor.load()

        # Try to load a previously trained model
        model_path = Path(self._cfg.model_save_path)
        if not model_path.is_absolute():
            model_path = _PROJECT_DIR / model_path

        if model_path.exists():
            self._scorer.load(str(model_path))
            # Load threshold from metadata if available
            meta_path = model_path.parent / "train_meta.json"
            if meta_path.exists():
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                self._threshold = meta.get("threshold", self._threshold)
            print(f"[visual_anomaly_detector] Loaded trained model from {model_path}")
        else:
            print("[visual_anomaly_detector] No trained model found — "
                  "run train.py first or call train_normal()")

        self._heatmap_gen = HeatmapGenerator(
            self._extractor, self._scorer,
            patch_size=self._cfg.patch_size,
            stride=self._cfg.patch_stride,
            alpha=self._cfg.heatmap_alpha,
        )

    def predict(self, input_data) -> dict:
        frame = input_data if isinstance(input_data, np.ndarray) else cv2.imread(str(input_data))

        if not self._scorer.fitted:
            return {"error": "Model not trained. Call train_normal() or run train.py first.",
                    "is_anomaly": None}

        feat = self._extractor.extract(frame)
        scores = self._scorer.score(feat)
        primary = scores.get(self._cfg.scoring_method, scores["mahalanobis"])
        is_anomaly = primary > self._threshold

        return {
            "is_anomaly": is_anomaly,
            "anomaly_score": round(primary, 4),
            "mahalanobis": round(scores["mahalanobis"], 4),
            "knn": round(scores["knn"], 4),
            "threshold": round(self._threshold, 4),
            "label": "ANOMALY" if is_anomaly else "NORMAL",
        }

    def visualize(self, input_data, output) -> np.ndarray:
        frame = input_data if isinstance(input_data, np.ndarray) else cv2.imread(str(input_data))
        return draw_result(frame, output, self._cfg)

    # ── Project-specific API ───────────────────────────────

    def train_normal(self, image_paths: list[str]) -> dict:
        """Train the anomaly scorer on normal images.

        Parameters
        ----------
        image_paths : list[str]
            Paths to normal training images.

        Returns
        -------
        dict
            Training summary.
        """
        features = self._extractor.extract_from_paths(image_paths, batch_size=32)
        summary = self._scorer.fit(features)

        # Auto-threshold on training scores
        from threshold import ThresholdSelector
        train_scores = np.array([
            self._scorer.score_primary(f) for f in features
        ])
        self._threshold = ThresholdSelector.percentile(
            train_scores, self._cfg.auto_threshold_percentile,
        )
        summary["threshold"] = self._threshold
        return summary

    def generate_heatmap(self, input_data) -> np.ndarray:
        """Generate a patch-level anomaly heatmap overlay."""
        frame = input_data if isinstance(input_data, np.ndarray) else cv2.imread(str(input_data))
        if not self._scorer.fitted:
            return frame
        _, overlay = self._heatmap_gen.generate(frame)
        return overlay

    def save_model(self, path: str) -> None:
        """Save the trained scorer to disk."""
        self._scorer.save(path)

    def load_model(self, path: str) -> None:
        """Load a trained scorer from disk."""
        self._scorer.load(path)

    def set_threshold(self, threshold: float) -> None:
        """Manually set the anomaly threshold."""
        self._threshold = threshold
