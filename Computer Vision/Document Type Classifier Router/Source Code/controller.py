"""Document Type Classifier Router — high-level controller facade.

Wires together the classifier, router, visualiser, and exporter
into a single convenient API.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from classifier import ClassificationResult, DocumentClassifier
from config import RouterConfig, load_config
from export import export_csv, export_json
from router import DocumentRouter, RoutingDecision
from validator import collect_images, validate_image
from visualize import save_annotated, save_grid


class DocumentController:
    """Facade for classify → route → export pipeline."""

    def __init__(self, cfg: RouterConfig | None = None) -> None:
        self.cfg = cfg or RouterConfig()
        self.classifier = DocumentClassifier(self.cfg)
        self.router = DocumentRouter(self.cfg)

    # ── Lifecycle ─────────────────────────────────────────

    def load(self, weights: str | Path | None = None) -> None:
        self.classifier.load(weights)

    def close(self) -> None:
        self.classifier.close()

    # ── Single image ──────────────────────────────────────

    def classify_and_route(
        self, image_bgr: np.ndarray
    ) -> tuple[ClassificationResult, RoutingDecision]:
        cr = self.classifier.classify(image_bgr)
        rd = self.router.route(cr)
        return cr, rd

    def process_file(
        self, path: str | Path
    ) -> tuple[ClassificationResult, RoutingDecision]:
        img = validate_image(path)
        return self.classify_and_route(img)

    # ── Batch ─────────────────────────────────────────────

    def process_batch(
        self, images: Sequence[np.ndarray]
    ) -> list[tuple[ClassificationResult, RoutingDecision]]:
        cls_results = self.classifier.classify_batch(images)
        routes = self.router.route_batch(cls_results)
        return list(zip(cls_results, routes))

    def process_directory(
        self, directory: str | Path
    ) -> list[tuple[Path, ClassificationResult, RoutingDecision]]:
        paths = collect_images(directory)
        results = []
        for p in paths:
            img = validate_image(p)
            cr, rd = self.classify_and_route(img)
            results.append((p, cr, rd))
        return results

    # ── Save helpers ──────────────────────────────────────

    def save_annotated(
        self,
        image_bgr: np.ndarray,
        cr: ClassificationResult,
        rd: RoutingDecision,
        output_dir: str | Path | None = None,
        filename: str = "annotated.jpg",
    ) -> Path:
        out = output_dir or self.cfg.output_dir
        return save_annotated(image_bgr, cr, rd, out, filename, self.cfg)

    def save_grid(
        self,
        images: list[np.ndarray],
        cls_results: list[ClassificationResult],
        routes: list[RoutingDecision],
        output_dir: str | Path | None = None,
        filename: str = "grid.jpg",
    ) -> Path:
        out = output_dir or self.cfg.output_dir
        return save_grid(images, cls_results, routes, out, filename, self.cfg)

    def export_json(
        self,
        cls_results: Sequence[ClassificationResult],
        routes: Sequence[RoutingDecision],
        output_path: str | Path,
        sources: Sequence[str] | None = None,
    ) -> Path:
        return export_json(cls_results, routes, output_path, sources)

    def export_csv(
        self,
        cls_results: Sequence[ClassificationResult],
        routes: Sequence[RoutingDecision],
        output_path: str | Path,
        sources: Sequence[str] | None = None,
    ) -> Path:
        return export_csv(cls_results, routes, output_path, sources)
