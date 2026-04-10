"""Collage visualization module for Face Clustering Photo Organizer.

Generates a preview collage for each cluster — a grid of face
thumbnails showing the faces that belong to one identity group.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from clusterer import Cluster
from config import FaceClusterConfig
from parser import ClusterResult

log = logging.getLogger("face_cluster.collage")


def build_cluster_collage(
    cluster: Cluster,
    cfg: FaceClusterConfig,
) -> np.ndarray:
    """Build a grid collage of face thumbnails for one cluster.

    Parameters
    ----------
    cluster : Cluster
        Identity cluster.
    cfg : FaceClusterConfig
        Display config.

    Returns
    -------
    np.ndarray
        BGR collage image.
    """
    thumb = cfg.collage_thumb_size
    cols = cfg.collage_cols
    border = cfg.collage_border
    max_faces = cfg.collage_max_faces

    faces = cluster.faces[:max_faces]
    n = len(faces)
    if n == 0:
        return np.zeros((thumb, thumb, 3), dtype=np.uint8)

    rows = (n + cols - 1) // cols
    cell = thumb + border
    canvas_h = rows * cell + border
    canvas_w = cols * cell + border
    canvas = np.full((canvas_h, canvas_w, 3), 40, dtype=np.uint8)

    for idx, face in enumerate(faces):
        r, c = divmod(idx, cols)
        y0 = r * cell + border
        x0 = c * cell + border

        # Resize crop to thumbnail
        crop = face.crop
        if crop is None or crop.size == 0:
            continue
        resized = cv2.resize(crop, (thumb, thumb))
        canvas[y0 : y0 + thumb, x0 : x0 + thumb] = resized

    # Title bar
    title = f"Cluster {cluster.cluster_id} ({cluster.size} faces)"
    title_h = 32
    header = np.full((title_h, canvas_w, 3), 30, dtype=np.uint8)
    cv2.putText(
        header, title, (8, 22),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 0), 2,
    )
    return np.vstack([header, canvas])


def build_all_collages(
    result: ClusterResult,
    cfg: FaceClusterConfig,
) -> list[tuple[int, np.ndarray]]:
    """Build collages for all clusters.

    Returns
    -------
    list[tuple[int, np.ndarray]]
        [(cluster_id, collage_image), ...].
    """
    collages = []
    for cluster in result.clusters:
        img = build_cluster_collage(cluster, cfg)
        collages.append((cluster.cluster_id, img))
    return collages


def save_collages(
    result: ClusterResult,
    cfg: FaceClusterConfig,
    output_dir: str | Path | None = None,
) -> list[Path]:
    """Build and save collage images to disk.

    Returns
    -------
    list[Path]
        Paths to saved collage images.
    """
    root = Path(output_dir) if output_dir else Path(cfg.output_dir)
    collage_dir = root / "collages"
    collage_dir.mkdir(parents=True, exist_ok=True)

    paths: list[Path] = []
    for cluster in result.clusters:
        img = build_cluster_collage(cluster, cfg)
        out_path = collage_dir / f"cluster_{cluster.cluster_id:03d}.jpg"
        cv2.imwrite(str(out_path), img)
        paths.append(out_path)

    log.info("Saved %d collages → %s", len(paths), collage_dir)
    return paths


def show_collages(
    result: ClusterResult,
    cfg: FaceClusterConfig,
) -> None:
    """Display cluster collages in GUI windows."""
    for cluster in result.clusters:
        img = build_cluster_collage(cluster, cfg)
        title = f"Cluster {cluster.cluster_id} ({cluster.size} faces)"
        cv2.imshow(title, img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
