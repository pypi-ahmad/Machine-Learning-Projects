"""Food Freshness Grader — high-level controller.
"""Food Freshness Grader — high-level controller.

Orchestrates grading, visualisation, export, and batch inference.
"""
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

from config import FreshnessConfig
from export import export_csv, export_json, result_to_dict
from grader import FreshnessGrader, GradeResult
from validator import collect_images, validate_image
from visualize import annotate_image, make_batch_grid, make_summary_bar

logger = logging.getLogger(__name__)


class FreshnessController:
    """Top-level facade for the food freshness grading pipeline."""

    def __init__(self, cfg: FreshnessConfig | None = None) -> None:
        self.cfg = cfg or FreshnessConfig()
        self._grader: FreshnessGrader | None = None

    # ── lifecycle ─────────────────────────────────────────

    def load(self) -> None:
        self._grader = FreshnessGrader(self.cfg)
        self._grader.load()

    def close(self) -> None:
        if self._grader:
            self._grader.close()

    # ── single inference ──────────────────────────────────

    def grade(self, image_path: str | Path) -> GradeResult:
        """Grade a single image."""
        assert self._grader is not None, "Call load() first"
        validated = validate_image(image_path)
        img = cv2.imread(str(validated))
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {validated}")
        return self._grader.grade(img)

    def grade_and_annotate(
        self,
        image_path: str | Path,
        *,
        save_path: str | Path | None = None,
        show: bool = False,
    ) -> GradeResult:
        """Grade and annotate a single image."""
        validated = validate_image(image_path)
        img = cv2.imread(str(validated))
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {validated}")

        result = self._grader.grade(img)
        annotated = annotate_image(img, result, self.cfg)

        if save_path:
            p = Path(save_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(p), annotated)
            logger.info("Annotated image saved -> %s", p)

        if show:
            cv2.imshow("Freshness Grade", annotated)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return result

    # ── batch inference ───────────────────────────────────

    def grade_batch(
        self,
        source: str | Path,
        *,
        batch_size: int = 16,
        save_grid: str | Path | None = None,
        save_summary: str | Path | None = None,
    ) -> list[tuple[str, GradeResult]]:
        """Grade all images in a directory.
        """Grade all images in a directory.

        Returns list of (image_path, GradeResult) tuples.
        """
        """
        assert self._grader is not None, "Call load() first"
        images_paths = collect_images(source)
        if not images_paths:
            raise FileNotFoundError(f"No images found in {source}")

        logger.info("Grading %d images from %s ...", len(images_paths), source)

        all_results: list[tuple[str, GradeResult]] = []
        all_images: list[np.ndarray] = []
        all_grades: list[GradeResult] = []

        for start in range(0, len(images_paths), batch_size):
            batch_paths = images_paths[start: start + batch_size]
            batch_imgs = []
            valid_paths = []
            for p in batch_paths:
                img = cv2.imread(str(p))
                if img is not None:
                    batch_imgs.append(img)
                    valid_paths.append(p)

            if not batch_imgs:
                continue

            grades = self._grader.grade_batch(batch_imgs)

            for p, img, g in zip(valid_paths, batch_imgs, grades):
                all_results.append((str(p), g))
                all_images.append(img)
                all_grades.append(g)

            n_done = min(start + batch_size, len(images_paths))
            logger.info("  %d / %d images graded", n_done, len(images_paths))

        # Summary
        n_fresh = sum(1 for _, g in all_results if g.freshness == "fresh")
        n_stale = sum(1 for _, g in all_results if g.freshness == "stale")
        logger.info("Batch complete: %d fresh, %d stale (total %d)",
                     n_fresh, n_stale, len(all_results))

        # Grid visualisation
        if save_grid and all_images:
            grid = make_batch_grid(
                all_images[:32],   # limit grid size
                all_grades[:32],
                thumb_size=self.cfg.grid_thumb_size,
                cols=self.cfg.grid_cols,
                cfg=self.cfg,
            )
            p = Path(save_grid)
            p.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(p), grid)
            logger.info("Batch grid saved -> %s", p)

        # Summary bar
        if save_summary and all_grades:
            bar = make_summary_bar(all_grades)
            p = Path(save_summary)
            p.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(p), bar)
            logger.info("Summary bar saved -> %s", p)

        return all_results

    # ── export ────────────────────────────────────────────

    def export_results(
        self,
        results: list[tuple[str, GradeResult]],
        *,
        json_path: str | Path | None = None,
        csv_path: str | Path | None = None,
    ) -> None:
        dicts = [result_to_dict(g, path=p) for p, g in results]
        if json_path:
            export_json(dicts, json_path)
            logger.info("JSON exported -> %s", json_path)
        if csv_path:
            export_csv(dicts, csv_path)
            logger.info("CSV exported -> %s", csv_path)
