"""Export utilities for Face Clustering Photo Organizer.

Supports:
- JSON: full cluster details with face metadata.
- CSV: one row per face with cluster assignment.

Usage::

    from export import ClusterExporter

    with ClusterExporter(cfg) as exporter:
        exporter.write(result)
"""

from __future__ import annotations

import csv
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import FaceClusterConfig
from parser import ClusterResult

log = logging.getLogger("face_cluster.export")

_CSV_COLUMNS = [
    "cluster_id",
    "cluster_label",
    "source_photo",
    "box_x1",
    "box_y1",
    "box_x2",
    "box_y2",
    "det_confidence",
]


class ClusterExporter:
    """Write clustering results to JSON and/or CSV."""

    def __init__(self, cfg: FaceClusterConfig) -> None:
        self.cfg = cfg
        self._csv_writer: csv.DictWriter | None = None
        self._csv_fh = None
        self._json_records: list[dict[str, Any]] = []

        if cfg.export_csv:
            out = Path(cfg.export_csv)
            out.parent.mkdir(parents=True, exist_ok=True)
            self._csv_fh = open(out, "w", newline="", encoding="utf-8")
            self._csv_writer = csv.DictWriter(
                self._csv_fh, fieldnames=_CSV_COLUMNS,
            )
            self._csv_writer.writeheader()
            log.info("CSV export -> %s", out)

    def __enter__(self) -> ClusterExporter:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def write(self, result: ClusterResult) -> None:
        """Record clustering results."""
        for cluster in result.clusters:
            label = f"person_{cluster.cluster_id:03d}"
            for face in cluster.faces:
                if self._csv_writer is not None:
                    self._csv_writer.writerow({
                        "cluster_id": cluster.cluster_id,
                        "cluster_label": label,
                        "source_photo": face.source_path,
                        "box_x1": face.box[0],
                        "box_y1": face.box[1],
                        "box_x2": face.box[2],
                        "box_y2": face.box[3],
                        "det_confidence": round(face.confidence, 4),
                    })

        if self.cfg.export_json:
            self._json_records.append(
                self._to_json_record(result),
            )

    def close(self) -> None:
        """Flush and close file handles."""
        if self._csv_fh is not None:
            self._csv_fh.close()
            self._csv_fh = None
            self._csv_writer = None

        if self.cfg.export_json and self._json_records:
            out = Path(self.cfg.export_json)
            out.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "exported_at": datetime.now(timezone.utc).isoformat(),
                "results": self._json_records,
            }
            out.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            log.info("JSON export -> %s", out)

    @staticmethod
    def _to_json_record(result: ClusterResult) -> dict[str, Any]:
        clusters = []
        for cluster in result.clusters:
            faces = []
            for f in cluster.faces:
                faces.append({
                    "source_photo": f.source_path,
                    "box": list(f.box),
                    "det_confidence": round(f.confidence, 4),
                })
            clusters.append({
                "cluster_id": cluster.cluster_id,
                "label": f"person_{cluster.cluster_id:03d}",
                "num_faces": cluster.size,
                "faces": faces,
            })

        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "total_images": result.total_images,
            "total_faces": result.total_faces,
            "num_clusters": result.num_clusters,
            "backend": result.backend,
            "clusters": clusters,
        }
