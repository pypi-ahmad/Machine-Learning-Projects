"""Export utilities for Drone Ship OBB Detector.

Supports JSON and YOLO-OBB TXT export formats.

JSON: per-frame list of detections with corners, class, confidence, angle.
TXT:  YOLO-OBB format — one file per image, ``class x1 y1 x2 y2 x3 y3 x4 y4``.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import OBBConfig
from detector import FrameResult

log = logging.getLogger("drone_ship_obb.export")


class OBBExporter:
    """Write OBB detection results to JSON and/or TXT."""

    def __init__(self, cfg: OBBConfig, *, image_shape: tuple[int, int] | None = None) -> None:
        self.cfg = cfg
        self._json_records: list[dict[str, Any]] = []
        self._image_shape = image_shape  # (height, width) for TXT normalisation
        self._txt_dir: Path | None = None

        if cfg.export_txt:
            self._txt_dir = Path(cfg.export_txt)
            self._txt_dir.mkdir(parents=True, exist_ok=True)
            log.info("TXT export dir → %s", self._txt_dir)

    def write(self, result: FrameResult, *, image_name: str | None = None,
              image_shape: tuple[int, int] | None = None) -> None:
        """Append one frame result to all active sinks."""
        shape = image_shape or self._image_shape

        # JSON accumulation
        if self.cfg.export_json:
            self._json_records.append(self._to_json_record(result))

        # TXT per-image file
        if self._txt_dir is not None and shape is not None:
            fname = image_name or f"frame_{result.frame_idx:06d}"
            self._write_txt(result, fname, shape)

    def close(self) -> None:
        """Flush JSON to disk."""
        if self.cfg.export_json and self._json_records:
            out = Path(self.cfg.export_json)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(self._json_records, indent=2), encoding="utf-8")
            log.info("JSON export → %s (%d records)", out, len(self._json_records))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_json_record(result: FrameResult) -> dict[str, Any]:
        dets = []
        for d in result.detections:
            dets.append({
                "class": d.class_name,
                "confidence": round(d.confidence, 4),
                "corners": d.corners.tolist(),
                "centre": list(d.centre),
                "angle_deg": d.angle_deg,
            })
        return {
            "frame_idx": result.frame_idx,
            "total": result.total,
            "class_counts": result.class_counts,
            "detections": dets,
        }

    def _write_txt(self, result: FrameResult, name: str, shape: tuple[int, int]) -> None:
        """Write YOLO-OBB format: ``cls x1 y1 x2 y2 x3 y3 x4 y4`` (normalised)."""
        h, w = shape
        lines: list[str] = []
        # Build a reverse name→id map from model names
        for det in result.detections:
            corners_norm = det.corners.copy()
            corners_norm[:, 0] /= w
            corners_norm[:, 1] /= h
            coords = " ".join(f"{c:.6f}" for c in corners_norm.flatten())
            # Use class name as ID (consistent with YOLO-OBB format)
            lines.append(f"{det.class_name} {coords}")

        out_path = self._txt_dir / f"{Path(name).stem}.txt"
        out_path.write_text("\n".join(lines), encoding="utf-8")

    def __enter__(self) -> OBBExporter:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
