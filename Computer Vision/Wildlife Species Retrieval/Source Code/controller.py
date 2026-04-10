"""Wildlife Species Retrieval — high-level controller.

Orchestrates embedding, indexing, retrieval, optional classifier
reranking, visualisation and export.
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2

from config import WildlifeConfig
from embedder import WildlifeEmbedder
from export import export_csv, export_json
from index import WildlifeIndex
from retriever import RetrievalResult, WildlifeRetriever
from validator import collect_images, infer_species, validate_image
from visualize import make_result_grid, save_result_grid

logger = logging.getLogger(__name__)


class WildlifeController:
    """Top-level facade for the wildlife species retrieval pipeline."""

    def __init__(self, cfg: WildlifeConfig | None = None) -> None:
        self.cfg = cfg or WildlifeConfig()
        self._embedder: WildlifeEmbedder | None = None
        self._index: WildlifeIndex | None = None
        self._retriever: WildlifeRetriever | None = None
        self._classifier = None   # lazy-loaded when reranking

    # ── lifecycle ─────────────────────────────────────────

    def load(self) -> None:
        """Initialise embedder and load existing index (if any)."""
        self._embedder = WildlifeEmbedder(self.cfg)
        self._embedder.load()

        index_path = Path(self.cfg.index_path)
        if index_path.exists():
            self._index = WildlifeIndex.load(str(index_path))
            logger.info("Loaded index with %d images from %s",
                        self._index.size, index_path)
        else:
            self._index = WildlifeIndex()

        self._retriever = WildlifeRetriever(self.cfg)
        self._retriever._embedder = self._embedder
        self._retriever._index = self._index

        if self.cfg.enable_rerank:
            self._load_classifier()

    def close(self) -> None:
        if self._embedder:
            self._embedder.close()
        if self._classifier:
            self._classifier.close()

    def _load_classifier(self) -> None:
        from classifier import WildlifeClassifier
        clf_path = Path(self.cfg.classifier_weights)
        if not clf_path.exists():
            logger.warning("Classifier weights not found at %s — "
                           "reranking disabled", clf_path)
            self.cfg.enable_rerank = False
            return
        self._classifier = WildlifeClassifier(
            weights_path=str(clf_path),
            model_name=self.cfg.classifier_model,
            num_classes=self.cfg.num_classes,
            imgsz=self.cfg.imgsz,
            device=self.cfg.device,
        )
        self._classifier.load()
        logger.info("Classifier loaded for reranking")

    # ── index management ──────────────────────────────────

    def build_index(
        self,
        image_dir: str | Path,
        *,
        batch_size: int = 32,
        force: bool = False,
    ) -> WildlifeIndex:
        """Build (or rebuild) the embedding index from an image directory."""
        if self._embedder is None or not self._embedder.is_loaded:
            raise RuntimeError("Call load() first")

        index_path = Path(self.cfg.index_path)
        if index_path.exists() and not force:
            logger.info("Index exists at %s — use force=True to rebuild",
                        index_path)
            if self._index and self._index.size > 0:
                return self._index

        root = Path(image_dir)
        images = collect_images(root, recursive=True)
        if not images:
            raise FileNotFoundError(f"No images found in {root}")

        logger.info("Building index from %d images in %s …", len(images), root)

        self._index = WildlifeIndex()

        for start in range(0, len(images), batch_size):
            batch_paths = images[start:start + batch_size]
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
            species = [infer_species(p, root) for p in valid_paths]
            str_paths = [str(p) for p in valid_paths]
            self._index.add_batch(embeddings, str_paths, species)

            n_done = min(start + batch_size, len(images))
            logger.info("  %d / %d images embedded", n_done, len(images))

        index_path.parent.mkdir(parents=True, exist_ok=True)
        self._index.save(str(index_path))
        logger.info("Index saved → %s  (%s)", index_path,
                     self._index.summary())

        # Update retriever reference
        if self._retriever:
            self._retriever._index = self._index
        return self._index

    # ── query ─────────────────────────────────────────────

    def query(
        self,
        image_path: str | Path,
        top_k: int | None = None,
    ) -> RetrievalResult:
        """Find similar wildlife images for a query."""
        if self._retriever is None:
            raise RuntimeError("Call load() or build_index() first")

        img = validate_image(image_path)
        result = self._retriever.query_image(
            img, top_k=top_k, source=str(image_path), exclude_self=True,
        )

        # Optional classifier reranking
        if self.cfg.enable_rerank and self._classifier and self._classifier.is_loaded:
            cls_result = self._classifier.classify(img)
            result.hits = self._classifier.rerank(
                cls_result.species, result.hits,
                weight=self.cfg.rerank_weight,
            )

        return result

    # ── visualisation ─────────────────────────────────────

    def query_and_visualise(
        self,
        image_path: str | Path,
        *,
        save_path: str | Path | None = None,
        show: bool = False,
        top_k: int | None = None,
    ) -> RetrievalResult:
        """Query and build a visual result grid."""
        result = self.query(image_path, top_k=top_k)
        query_img = cv2.imread(str(image_path))
        if query_img is None:
            raise FileNotFoundError(f"Cannot read query image: {image_path}")

        grid = make_result_grid(
            query_img,
            result.hits,
            thumb_size=self.cfg.grid_thumb_size,
            cols=self.cfg.grid_cols,
            border_color=self.cfg.border_color,
            text_color=self.cfg.text_color,
            font_scale=self.cfg.font_scale,
        )

        if save_path:
            p = Path(save_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(p), grid)
            logger.info("Result grid saved → %s", p)

        if show:
            cv2.imshow("Wildlife Retrieval", grid)
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
        if json_path:
            export_json(result.query_path, result.hits, json_path,
                        species_votes=result.species_votes)
            logger.info("JSON exported → %s", json_path)
        if csv_path:
            export_csv(result.query_path, result.hits, csv_path)
            logger.info("CSV exported → %s", csv_path)
