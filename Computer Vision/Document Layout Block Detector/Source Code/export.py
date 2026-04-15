"""Export utilities for Document Layout Block Detector.

Supports:
- JSON export with block coordinates, classes, and page metadata.
- Optional crop saving (one image per detected block).

JSON output is structured for downstream OCR / document-parsing pipelines.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import LayoutConfig
from detector import PageResult

log = logging.getLogger("doc_layout.export")


class LayoutExporter:
    """Export layout detection results to JSON and optional crops."""

    def __init__(self, cfg: LayoutConfig) -> None:
        self.cfg = cfg
        self._json_pages: list[dict[str, Any]] = []
        self._crops_dir: Path | None = None

        if cfg.save_crops:
            self._crops_dir = Path(cfg.crops_dir)
            self._crops_dir.mkdir(parents=True, exist_ok=True)
            log.info("Crop export dir -> %s", self._crops_dir)

    def write(self, result: PageResult, image: np.ndarray, *,
              source_name: str = "") -> None:
        """Record one page result. Saves crops immediately if enabled."""
        record = self._to_json_record(result, source_name)

        if self.cfg.export_json:
            self._json_pages.append(record)

        if self._crops_dir is not None:
            self._save_crops(result, image, source_name)

    def close(self) -> None:
        """Flush JSON to disk."""
        if self.cfg.export_json and self._json_pages:
            out = Path(self.cfg.export_json)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(self._json_pages, indent=2), encoding="utf-8")
            log.info("JSON export -> %s (%d pages)", out, len(self._json_pages))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_json_record(result: PageResult, source_name: str) -> dict[str, Any]:
        blocks = []
        for blk in result.blocks:
            blocks.append({
                "block_id": blk.block_id,
                "class": blk.class_name,
                "confidence": round(blk.confidence, 4),
                "bbox": list(blk.bbox),
                "area": blk.area,
            })
        return {
            "source": source_name,
            "page_idx": result.page_idx,
            "image_height": result.image_shape[0],
            "image_width": result.image_shape[1],
            "total_blocks": result.total_blocks,
            "class_counts": result.class_counts,
            "blocks": blocks,
        }

    def _save_crops(self, result: PageResult, image: np.ndarray,
                    source_name: str) -> None:
        stem = Path(source_name).stem if source_name else f"page_{result.page_idx:04d}"
        page_dir = self._crops_dir / stem
        page_dir.mkdir(parents=True, exist_ok=True)

        for blk in result.blocks:
            x1, y1, x2, y2 = blk.bbox
            crop = image[max(0, y1):y2, max(0, x1):x2]
            fname = f"{blk.block_id:03d}_{blk.class_name}.png"
            cv2.imwrite(str(page_dir / fname), crop)

        log.debug("Saved %d crops for %s", len(result.blocks), stem)

    def __enter__(self) -> LayoutExporter:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
