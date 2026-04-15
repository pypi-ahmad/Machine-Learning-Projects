"""Similar Image Finder — high-level controller.
"""Similar Image Finder — high-level controller.

Orchestrates embedding, indexing, retrieval, visualisation and export.
"""
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2

from config import SimilarityConfig
from embedder import ImageEmbedder
from export import export_csv, export_json
from index import ImageIndex
from retriever import ImageRetriever, RetrievalResult
from validator import collect_images, infer_category, validate_image
from visualize import draw_query_overlay, make_result_grid

logger = logging.getLogger(__name__)


class SimilarityController:
    """Top-level facade for the Similar Image Finder pipeline."""

    def __init__(self, cfg: SimilarityConfig | None = None) -> None:
        self.cfg = cfg or SimilarityConfig()
        self._embedder: ImageEmbedder | None = None
        self._index: ImageIndex | None = None
        self._retriever: ImageRetriever | None = None

    # ── lifecycle ─────────────────────────────────────────

    def load(self) -> None:
        """Initialise embedder and load existing index (if any)."""
        self._embedder = ImageEmbedder(
            backbone=self.cfg.backbone,
            embedding_dim=self.cfg.embedding_dim,
            imgsz=self.cfg.imgsz,
            device=self.cfg.device,
        )
        self._embedder.load()

        index_path = Path(self.cfg.index_path)
        self._index = ImageIndex(metric=self.cfg.similarity_metric)
        if index_path.exists():
            self._index.load(str(index_path))
            logger.info("Loaded index with %d images from %s",
                        len(self._index), index_path)

        self._retriever = ImageRetriever(self._embedder, self._index, cfg=self.cfg)

    def close(self) -> None:
        if self._embedder:
            self._embedder.close()

    # ── index management ──────────────────────────────────

    def build_index(
        self,
        image_dir: str | Path,
        *,
        batch_size: int = 32,
        force: bool = False,
    ) -> ImageIndex:
        """Build (or rebuild) the embedding index from a directory of images."""
        assert self._embedder is not None, "Call load() first"

        index_path = Path(self.cfg.index_path)
        if index_path.exists() and not force:
            logger.info("Index already exists at %s -- use force=True to rebuild", index_path)
            if self._index and len(self._index) > 0:
                return self._index

        root = Path(image_dir)
        images = collect_images(root, recursive=True)
        if not images:
            raise FileNotFoundError(f"No images found in {root}")

        logger.info("Building index from %d images in %s ...", len(images), root)

        self._index = ImageIndex(metric=self.cfg.similarity_metric)

        # Process in batches
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
            categories = [infer_category(p, root) for p in valid_paths]
            str_paths = [str(p) for p in valid_paths]
            self._index.add_batch(embeddings, str_paths, categories)

            n_done = min(start + batch_size, len(images))
            logger.info("  %d / %d images embedded", n_done, len(images))

        # Save index
        index_path.parent.mkdir(parents=True, exist_ok=True)
        self._index.save(str(index_path))
        logger.info("Index saved -> %s  (%s)", index_path, self._index.summary())

        # Update retriever
        self._retriever = ImageRetriever(self._embedder, self._index, cfg=self.cfg)
        return self._index

    # ── query ─────────────────────────────────────────────

    def query(self, image_path: str | Path) -> RetrievalResult:
        """Find similar images for a single query."""
        assert self._retriever is not None, "Call load() or build_index() first"
        validated = validate_image(image_path)
        return self._retriever.query_image(str(validated))

    # ── visualisation ─────────────────────────────────────

    def query_and_visualise(
        self,
        image_path: str | Path,
        *,
        save_path: str | Path | None = None,
        show: bool = False,
    ) -> RetrievalResult:
        """Query and build a visual result grid."""
        result = self.query(image_path)
        query_img = cv2.imread(str(image_path))
        if query_img is None:
            raise FileNotFoundError(f"Cannot read query image: {image_path}")

        grid = make_result_grid(
            query_img,
            result.hits,
            thumb_size=self.cfg.grid_thumb_size,
            cols=self.cfg.grid_cols,
            border_color=self.cfg.border_color,
        )

        if save_path:
            p = Path(save_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(p), grid)
            logger.info("Result grid saved -> %s", p)

        if show:
            cv2.imshow("Similar Images", grid)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return result

    # ── export ────────────────────────────────────────────

    def export_results(
        self,
        result: RetrievalResult,
        *,
        json_path: str | Path | None = None,
        csv_path: str | Path | None = None,
    ) -> None:
        """Export retrieval results to JSON and/or CSV."""
        if json_path:
            export_json(
                result.query_path, result.hits, json_path,
                category_votes=result.category_votes,
            )
            logger.info("JSON exported -> %s", json_path)
        if csv_path:
            export_csv(result.query_path, result.hits, csv_path)
            logger.info("CSV exported -> %s", csv_path)
