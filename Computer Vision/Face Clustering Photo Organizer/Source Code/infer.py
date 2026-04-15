"""CLI inference pipeline for Face Clustering Photo Organizer.

Scans a directory of photos, detects faces, extracts embeddings,
clusters by identity, and exports organized folders + collages.

Usage::

    python infer.py --source photos/
    python infer.py --source photos/ --algorithm dbscan --threshold 0.55
    python infer.py --source photos/ --export-json clusters.json --export-csv faces.csv
    python infer.py --source photos/ --no-display --save-collages
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from collage import save_collages, show_collages
from config import FaceClusterConfig, load_config
from export import ClusterExporter
from organizer import PhotoOrganizer
from parser import FaceClusterPipeline
from validator import ClusterValidator

log = logging.getLogger("face_cluster.infer")

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Face Clustering Photo Organizer -- Inference",
    )
    p.add_argument("--source", required=True,
                   help="Directory of photos or single image")
    p.add_argument("--config", default=None,
                   help="Path to YAML/JSON config")
    p.add_argument("--algorithm", choices=["agglomerative", "dbscan"],
                   default=None, help="Clustering algorithm")
    p.add_argument("--threshold", type=float, default=None,
                   help="Distance threshold (agglomerative) or eps (dbscan)")
    p.add_argument("--min-cluster-size", type=int, default=None,
                   help="Minimum faces per cluster")
    p.add_argument("--output-dir", default=None,
                   help="Output directory (default: output/)")
    p.add_argument("--no-display", action="store_true",
                   help="Disable GUI windows")
    p.add_argument("--no-copy", action="store_true",
                   help="Don't copy photos into cluster folders")
    p.add_argument("--save-collages", action="store_true",
                   help="Save collage images to disk")
    p.add_argument("--export-json", default=None,
                   help="JSON export path")
    p.add_argument("--export-csv", default=None,
                   help="CSV export path")
    p.add_argument("--force-download", action="store_true",
                   help="Force dataset re-download")
    return p.parse_args(argv)


def _apply_cli_overrides(
    cfg: FaceClusterConfig, args: argparse.Namespace,
) -> None:
    if args.algorithm:
        cfg.algorithm = args.algorithm
    if args.threshold is not None:
        if cfg.algorithm == "dbscan":
            cfg.dbscan_eps = args.threshold
        else:
            cfg.distance_threshold = args.threshold
    if args.min_cluster_size is not None:
        cfg.min_cluster_size = args.min_cluster_size
    if args.output_dir:
        cfg.output_dir = args.output_dir
    if args.no_display:
        cfg.show_display = False
    if args.no_copy:
        cfg.copy_photos = False
    if args.save_collages:
        cfg.save_collages = True
    if args.export_json:
        cfg.export_json = args.export_json
    if args.export_csv:
        cfg.export_csv = args.export_csv


def _collect_images(source: str) -> list[Path]:
    p = Path(source)
    if p.is_dir():
        files = []
        for ext in IMAGE_EXTS:
            files.extend(p.rglob(f"*{ext}"))
        files.sort()
        return files
    if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
        return [p]
    return []


def run(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s | %(name)s | %(message)s",
    )

    cfg = load_config(args.config) if args.config else FaceClusterConfig()
    _apply_cli_overrides(cfg, args)

    if args.force_download:
        from data_bootstrap import ensure_face_cluster_dataset
        ensure_face_cluster_dataset(force=True)

    # Collect images
    images = _collect_images(args.source)
    if not images:
        log.error("No images found at: %s", args.source)
        return
    log.info("Found %d images in %s", len(images), args.source)

    # Run pipeline
    pipeline = FaceClusterPipeline(cfg)
    pipeline.load()
    result = pipeline.process(images)

    # Validate
    validator = ClusterValidator(cfg)
    report = validator.validate(result)

    # Log summary
    log.info(
        "Results: %d images, %d faces, %d clusters",
        result.total_images, result.total_faces, result.num_clusters,
    )
    if report.warnings:
        for w in report.warnings:
            log.warning("  %s: %s", w.field_name, w.message)

    for cluster in result.clusters:
        sources = len(set(f.source_path for f in cluster.faces))
        log.info(
            "  Cluster %d: %d faces from %d photos",
            cluster.cluster_id, cluster.size, sources,
        )

    # Export
    with ClusterExporter(cfg) as exporter:
        exporter.write(result)

    # Organize into folders
    if cfg.copy_photos or cfg.save_manifest:
        organizer = PhotoOrganizer(cfg)
        organizer.organize(result)

    # Collages
    if cfg.save_collages:
        save_collages(result, cfg)

    if cfg.show_display and result.num_clusters > 0:
        show_collages(result, cfg)

    log.info("Done -- %d identity clusters found", result.num_clusters)


if __name__ == "__main__":
    run()
