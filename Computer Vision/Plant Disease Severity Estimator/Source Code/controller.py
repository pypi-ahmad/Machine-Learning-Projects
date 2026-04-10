"""Plant Disease Severity Estimator — high-level controller facade.

Wires together the classifier, visualiser, and exporter into a single
convenient API.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from classifier import PlantDiseaseClassifier, PredictionResult
from config import SeverityConfig, load_config
from export import export_csv, export_json
from validator import collect_images, validate_image
from visualize import save_annotated, save_grid


class PlantDiseaseController:
    """Facade for the full classification + severity pipeline."""

    def __init__(self, cfg: SeverityConfig | None = None) -> None:
        self.cfg = cfg or SeverityConfig()
        self.classifier = PlantDiseaseClassifier(self.cfg)

    # ── Lifecycle ─────────────────────────────────────────

    def load(self, weights: str | Path | None = None) -> None:
        self.classifier.load(weights)

    def close(self) -> None:
        self.classifier.close()

    # ── Prediction ────────────────────────────────────────

    def predict(self, image_bgr: np.ndarray) -> PredictionResult:
        return self.classifier.classify(image_bgr)

    def predict_batch(
        self, images: Sequence[np.ndarray]
    ) -> list[PredictionResult]:
        return self.classifier.classify_batch(images)

    # ── File-based helpers ────────────────────────────────

    def predict_file(self, path: str | Path) -> PredictionResult:
        img = validate_image(path)
        return self.predict(img)

    def predict_directory(
        self, directory: str | Path
    ) -> list[tuple[Path, PredictionResult]]:
        paths = collect_images(directory)
        results: list[tuple[Path, PredictionResult]] = []
        for p in paths:
            img = validate_image(p)
            results.append((p, self.predict(img)))
        return results

    # ── Save helpers ──────────────────────────────────────

    def save_annotated(
        self,
        image_bgr: np.ndarray,
        result: PredictionResult,
        output_dir: str | Path | None = None,
        filename: str = "annotated.jpg",
    ) -> Path:
        out = output_dir or self.cfg.output_dir
        return save_annotated(image_bgr, result, out, filename, self.cfg)

    def save_grid(
        self,
        images: list[np.ndarray],
        results: list[PredictionResult],
        output_dir: str | Path | None = None,
        filename: str = "grid.jpg",
    ) -> Path:
        out = output_dir or self.cfg.output_dir
        return save_grid(images, results, out, filename, self.cfg)

    def export_json(
        self,
        results: Sequence[PredictionResult],
        output_path: str | Path,
        sources: Sequence[str] | None = None,
    ) -> Path:
        return export_json(results, output_path, sources)

    def export_csv(
        self,
        results: Sequence[PredictionResult],
        output_path: str | Path,
        sources: Sequence[str] | None = None,
    ) -> Path:
        return export_csv(results, output_path, sources)
