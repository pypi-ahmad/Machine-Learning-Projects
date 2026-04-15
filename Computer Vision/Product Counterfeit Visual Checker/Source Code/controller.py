"""Product Counterfeit Visual Checker — high-level controller.
"""Product Counterfeit Visual Checker — high-level controller.

Orchestrates embedding, reference management, comparison,
visualisation, and export for counterfeit screening.
"""
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2

from comparator import ProductComparator, ScreeningResult
from config import CounterfeitConfig
from embedder import ProductEmbedder
from export import export_csv, export_json
from reference_store import ReferenceStore
from validator import collect_images, infer_product, validate_image
from visualize import make_region_heatmap, make_screening_grid

logger = logging.getLogger(__name__)


class CounterfeitController:
    """Top-level facade for the counterfeit visual screening pipeline."""

    def __init__(self, cfg: CounterfeitConfig | None = None) -> None:
        self.cfg = cfg or CounterfeitConfig()
        self._embedder: ProductEmbedder | None = None
        self._store: ReferenceStore | None = None
        self._comparator: ProductComparator | None = None

    # ── lifecycle ─────────────────────────────────────────

    def load(self) -> None:
        """Initialise embedder and load existing reference store."""
        self._embedder = ProductEmbedder(
            backbone=self.cfg.backbone,
            embedding_dim=self.cfg.embedding_dim,
            imgsz=self.cfg.imgsz,
            device=self.cfg.device,
        )
        self._embedder.load()

        ref_path = Path(self.cfg.reference_path)
        self._store = ReferenceStore(metric=self.cfg.similarity_metric)
        if ref_path.exists():
            self._store.load(str(ref_path))
            logger.info("Loaded reference store: %s", self._store.summary())

        self._comparator = ProductComparator(self._embedder, self._store, self.cfg)

    def close(self) -> None:
        if self._embedder:
            self._embedder.close()

    # ── reference management ──────────────────────────────

    def build_references(
        self,
        image_dir: str | Path,
        *,
        batch_size: int = 32,
        force: bool = False,
    ) -> ReferenceStore:
        """Build the reference store from a directory of approved product images."""
        assert self._embedder is not None, "Call load() first"

        ref_path = Path(self.cfg.reference_path)
        if ref_path.exists() and not force:
            logger.info("Reference store exists at %s -- use force=True to rebuild",
                        ref_path)
            if self._store and len(self._store) > 0:
                return self._store

        root = Path(image_dir)
        images = collect_images(root, recursive=True)
        if not images:
            raise FileNotFoundError(f"No images found in {root}")

        logger.info("Building references from %d images in %s ...", len(images), root)

        self._store = ReferenceStore(metric=self.cfg.similarity_metric)

        for start in range(0, len(images), batch_size):
            batch_paths = images[start: start + batch_size]
            batch_imgs = []
            valid_paths = []
            for p in batch_paths:
                img = cv2.imread(str(p))
                if img is not None:
                    batch_imgs.append(img)
                    valid_paths.append(p)

            if not batch_imgs:
                continue

            embeddings = self._embedder.embed_batch(batch_imgs)
            products = [infer_product(p, root) for p in valid_paths]
            str_paths = [str(p) for p in valid_paths]
            self._store.add_batch(embeddings, str_paths, products)

            n_done = min(start + batch_size, len(images))
            logger.info("  %d / %d images embedded", n_done, len(images))

        ref_path.parent.mkdir(parents=True, exist_ok=True)
        self._store.save(str(ref_path))
        logger.info("Reference store saved -> %s  (%s)", ref_path, self._store.summary())

        self._comparator = ProductComparator(self._embedder, self._store, self.cfg)
        return self._store

    # ── screening ─────────────────────────────────────────

    def screen(
        self,
        image_path: str | Path,
        *,
        product_filter: str | None = None,
    ) -> ScreeningResult:
        """Screen a single suspect image against the reference store."""
        assert self._comparator is not None, "Call load() or build_references() first"
        validated = validate_image(image_path)
        suspect = cv2.imread(str(validated))
        if suspect is None:
            raise FileNotFoundError(f"Cannot read image: {validated}")
        return self._comparator.screen(suspect, suspect_path=str(validated),
                                       product_filter=product_filter)

    # ── visualisation ─────────────────────────────────────

    def screen_and_visualise(
        self,
        image_path: str | Path,
        *,
        save_path: str | Path | None = None,
        show: bool = False,
        product_filter: str | None = None,
    ) -> ScreeningResult:
        """Screen and build a visual comparison grid."""
        result = self.screen(image_path, product_filter=product_filter)
        suspect_img = cv2.imread(str(image_path))
        if suspect_img is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")

        grid = make_screening_grid(
            suspect_img,
            result,
            thumb_size=self.cfg.grid_thumb_size,
            cols=self.cfg.grid_cols,
        )

        if save_path:
            p = Path(save_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(p), grid)
            logger.info("Screening grid saved -> %s", p)

        if show:
            cv2.imshow("Counterfeit Screening", grid)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return result

    def make_heatmap(
        self,
        image_path: str | Path,
        result: ScreeningResult,
        *,
        save_path: str | Path | None = None,
    ) -> None:
        """Generate a region heatmap for the best-matching comparison."""
        if not result.details:
            return

        suspect_img = cv2.imread(str(image_path))
        if suspect_img is None:
            return

        best = max(result.details, key=lambda d: d.composite_score)
        if not best.region_scores:
            return

        heatmap = make_region_heatmap(
            suspect_img,
            best.region_scores,
            self.cfg.region_grid,
        )

        if save_path:
            p = Path(save_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(p), heatmap)
            logger.info("Region heatmap saved -> %s", p)

    # ── export ────────────────────────────────────────────

    def export_results(
        self,
        result: ScreeningResult,
        *,
        json_path: str | Path | None = None,
        csv_path: str | Path | None = None,
    ) -> None:
        if json_path:
            export_json(result, json_path)
            logger.info("JSON exported -> %s", json_path)
        if csv_path:
            export_csv(result, csv_path)
            logger.info("CSV exported -> %s", csv_path)
