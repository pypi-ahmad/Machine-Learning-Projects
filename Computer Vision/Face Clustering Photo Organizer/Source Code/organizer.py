"""File organizer module for Face Clustering Photo Organizer.

Exports clustering results as organized folder hierarchies:
each cluster gets a directory with its source photos copied in.
Also writes a JSON cluster manifest.
"""

from __future__ import annotations

import json
import logging
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from clusterer import Cluster
from config import FaceClusterConfig
from parser import ClusterResult

log = logging.getLogger("face_cluster.organizer")


class PhotoOrganizer:
    """Organize source photos into per-cluster directories."""

    def __init__(self, cfg: FaceClusterConfig) -> None:
        self.cfg = cfg

    def organize(
        self,
        result: ClusterResult,
        output_dir: str | Path | None = None,
    ) -> Path:
        """Create cluster directories and copy/link source photos.

        Parameters
        ----------
        result : ClusterResult
            Pipeline output.
        output_dir : str or Path, optional
            Override output directory.

        Returns
        -------
        Path
            Root output directory.
        """
        root = Path(output_dir) if output_dir else Path(self.cfg.output_dir)
        clusters_dir = root / "clusters"
        clusters_dir.mkdir(parents=True, exist_ok=True)

        for cluster in result.clusters:
            cluster_name = f"person_{cluster.cluster_id:03d}"
            cluster_dir = clusters_dir / cluster_name
            cluster_dir.mkdir(parents=True, exist_ok=True)

            # Collect unique source photos
            seen_sources: set[str] = set()
            for face in cluster.faces:
                src = Path(face.source_path)
                if str(src) in seen_sources or not src.exists():
                    continue
                seen_sources.add(str(src))

                dst = cluster_dir / src.name
                if dst.exists():
                    continue

                if self.cfg.symlink_photos:
                    try:
                        dst.symlink_to(src.resolve())
                    except OSError:
                        shutil.copy2(str(src), str(dst))
                elif self.cfg.copy_photos:
                    shutil.copy2(str(src), str(dst))

            log.info(
                "Cluster %s: %d faces from %d photos",
                cluster_name, cluster.size, len(seen_sources),
            )

        # Write manifest
        if self.cfg.save_manifest:
            self._write_manifest(result, root)

        log.info(
            "Organized %d clusters → %s",
            result.num_clusters, clusters_dir,
        )
        return root

    def _write_manifest(
        self, result: ClusterResult, root: Path,
    ) -> Path:
        """Write a JSON cluster manifest."""
        manifest_path = root / "cluster_manifest.json"

        clusters_data: list[dict[str, Any]] = []
        for cluster in result.clusters:
            sources = sorted(set(f.source_path for f in cluster.faces))
            clusters_data.append({
                "cluster_id": cluster.cluster_id,
                "label": f"person_{cluster.cluster_id:03d}",
                "num_faces": cluster.size,
                "num_photos": len(sources),
                "source_photos": sources,
            })

        manifest = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "total_images": result.total_images,
            "total_faces": result.total_faces,
            "num_clusters": result.num_clusters,
            "clusters": clusters_data,
        }

        manifest_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        log.info("Manifest → %s", manifest_path)
        return manifest_path
