"""Train / Evaluate Face Clustering Photo Organizer.

InsightFace models are pre-trained — this script downloads the
dataset and evaluates the clustering pipeline on a multi-identity
face dataset (LFW).

Usage::

    python train.py
    python train.py --data path/to/faces
    python train.py --force-download
    python train.py --max-identities 50
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from data_bootstrap import ensure_face_cluster_dataset


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description="Train/Evaluate Face Clustering Photo Organizer",
    )
    ap.add_argument("--data", type=str, default=None,
                    help="Path to face dataset (ImageFolder layout)")
    ap.add_argument("--max-identities", type=int, default=50,
                    help="Max identities to evaluate")
    ap.add_argument("--algorithm", choices=["agglomerative", "dbscan"],
                    default="agglomerative", help="Clustering algorithm")
    ap.add_argument("--threshold", type=float, default=None,
                    help="Distance threshold")
    ap.add_argument("--force-download", action="store_true",
                    help="Force re-download dataset")
    args = ap.parse_args(argv)

    if args.data is None:
        data_path = ensure_face_cluster_dataset(force=args.force_download)
        data_dir = str(data_path / "processed" / "identities")
        print(f"[INFO] Prepared dataset -> {data_path}")
    else:
        data_dir = args.data

    print(f"[INFO] Dataset ready at {data_dir}")
    print("[INFO] InsightFace models are pre-trained. Running clustering eval...")

    try:
        from config import FaceClusterConfig
        from parser import FaceClusterPipeline

        cfg = FaceClusterConfig()
        cfg.algorithm = args.algorithm
        if args.threshold is not None:
            if cfg.algorithm == "dbscan":
                cfg.dbscan_eps = args.threshold
            else:
                cfg.distance_threshold = args.threshold

        pipeline = FaceClusterPipeline(cfg)
        pipeline.load()

        if not pipeline.embedder.ready:
            print("[ERROR] InsightFace not available -- cannot evaluate")
            return

        data_root = Path(data_dir)

        # Find identity directories
        identity_dirs = _find_identity_dirs(data_root)
        if not identity_dirs:
            print("[WARN] No identity subdirectories found.")
            return

        # Select identities with >= 2 images
        usable = []
        all_images: list[str] = []
        ground_truth: dict[str, str] = {}  # image_path -> identity

        for id_dir in identity_dirs[: args.max_identities]:
            imgs = _find_image_files(id_dir)
            if len(imgs) < 2:
                continue
            usable.append(id_dir.name)
            for img in imgs[:5]:  # cap per identity
                all_images.append(str(img))
                ground_truth[str(img)] = id_dir.name

        print(f"[INFO] Using {len(usable)} identities, {len(all_images)} images")

        if len(all_images) < 4:
            print("[WARN] Not enough images for clustering evaluation")
            return

        # Run clustering
        result = pipeline.process(all_images)
        print(f"[INFO] Detected {result.total_faces} faces, formed {result.num_clusters} clusters")

        # Evaluate: check purity and completeness
        cluster_purities = []
        for cluster in result.clusters:
            identity_counts: dict[str, int] = {}
            for face in cluster.faces:
                gt_id = ground_truth.get(face.source_path, "unknown")
                identity_counts[gt_id] = identity_counts.get(gt_id, 0) + 1
            total = sum(identity_counts.values())
            dominant = max(identity_counts.values()) if identity_counts else 0
            purity = dominant / total if total > 0 else 0
            cluster_purities.append(purity)

            dominant_id = max(identity_counts, key=identity_counts.get)
            print(
                f"  Cluster {cluster.cluster_id}: {cluster.size} faces, "
                f"dominant={dominant_id} ({dominant}/{total}), "
                f"purity={purity:.2f}"
            )

        avg_purity = sum(cluster_purities) / len(cluster_purities) if cluster_purities else 0

        print(f"\n[SUMMARY]")
        print(f"  Identities:      {len(usable)}")
        print(f"  Images:          {len(all_images)}")
        print(f"  Faces detected:  {result.total_faces}")
        print(f"  Clusters formed: {result.num_clusters}")
        print(f"  Avg purity:      {avg_purity:.3f}")
        print(f"  Algorithm:       {cfg.algorithm}")

        # Save results
        out_path = Path(__file__).parent / "runs" / "eval_results.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps({
                "identities": len(usable),
                "images": len(all_images),
                "faces_detected": result.total_faces,
                "clusters_formed": result.num_clusters,
                "avg_purity": round(avg_purity, 4),
                "algorithm": cfg.algorithm,
                "cluster_sizes": [c.size for c in result.clusters],
            }, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"  Results -> {out_path}")

    except ImportError as exc:
        print(f"[WARN] Could not run evaluation: {exc}")
        print("[INFO] Install: pip install insightface onnxruntime scikit-learn")
    except Exception as exc:
        print(f"[ERROR] Evaluation failed: {exc}")
        raise


def _find_identity_dirs(data_root: Path) -> list[Path]:
    """Find identity subdirectories, trying one level deep."""
    dirs = sorted(d for d in data_root.iterdir() if d.is_dir())
    if dirs:
        for d in dirs:
            imgs = _find_image_files(d)
            if imgs:
                return dirs
    # Try one level deeper
    for sub in sorted(data_root.iterdir()):
        if sub.is_dir():
            deeper = sorted(d for d in sub.iterdir() if d.is_dir())
            if deeper:
                return deeper
    return []


def _find_image_files(identity_dir: Path) -> list[Path]:
    files = []
    for child in sorted(identity_dir.iterdir()):
        if child.is_file() and child.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
            files.append(child)
    return files


if __name__ == "__main__":
    main()
