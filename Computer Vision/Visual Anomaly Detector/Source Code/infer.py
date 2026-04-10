"""Visual Anomaly Detector — inference pipeline.

Supports single-image and folder-level inference with anomaly scoring,
heatmap generation, and result export.

Usage::

    python infer.py --source image.png
    python infer.py --source folder/ --export results.json
    python infer.py --source image.png --heatmap
    python infer.py --source folder/ --threshold 4.0 --no-display
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import cv2
import numpy as np

_SRC = Path(__file__).resolve().parent
_REPO = _SRC.parents[1]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_SRC))

from config import AnomalyConfig, load_config
from export import export_results
from feature_extractor import FeatureExtractor
from heatmap import HeatmapGenerator
from scorer import AnomalyScorer
from visualize import draw_batch_summary, draw_result

log = logging.getLogger("visual_anomaly.infer")

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def _load_model(cfg: AnomalyConfig) -> tuple[FeatureExtractor, AnomalyScorer, float]:
    """Load feature extractor and trained scorer."""
    extractor = FeatureExtractor(backbone=cfg.backbone, imgsz=cfg.imgsz)
    extractor.load()

    scorer = AnomalyScorer(
        method=cfg.scoring_method,
        knn_k=cfg.knn_k,
        regularization=cfg.regularization,
    )

    model_path = Path(cfg.model_save_path)
    if not model_path.is_absolute():
        model_path = _SRC / model_path

    if not model_path.exists():
        print(f"[ERROR] Trained model not found at {model_path}")
        print("[INFO] Run train.py first to train the anomaly model")
        sys.exit(1)

    scorer.load(str(model_path))

    # Load threshold from metadata
    threshold = cfg.anomaly_threshold
    meta_path = model_path.parent / "train_meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        threshold = meta.get("threshold", threshold)

    return extractor, scorer, threshold


def infer_image(
    path: str | Path,
    extractor: FeatureExtractor,
    scorer: AnomalyScorer,
    threshold: float,
    cfg: AnomalyConfig,
    *,
    show: bool = True,
    save_dir: str | None = None,
    heatmap: bool = False,
) -> dict:
    """Score a single image."""
    image = cv2.imread(str(path))
    if image is None:
        log.warning("Cannot read image: %s", path)
        return {"source": str(path), "error": "cannot read"}

    feat = extractor.extract(image)
    scores = scorer.score(feat)
    primary = scores.get(cfg.scoring_method, scores["mahalanobis"])
    is_anomaly = primary > threshold

    result = {
        "source": Path(path).name,
        "is_anomaly": is_anomaly,
        "anomaly_score": round(primary, 4),
        "mahalanobis": round(scores["mahalanobis"], 4),
        "knn": round(scores["knn"], 4),
        "threshold": round(threshold, 4),
        "label": "ANOMALY" if is_anomaly else "NORMAL",
    }

    if show or save_dir:
        annotated = draw_result(image, result, cfg)

        if heatmap and scorer.fitted:
            hm_gen = HeatmapGenerator(
                extractor, scorer,
                patch_size=cfg.patch_size,
                stride=cfg.patch_stride,
                alpha=cfg.heatmap_alpha,
            )
            _, hm_overlay = hm_gen.generate(image)
            # Stack original annotated and heatmap side by side
            h1, w1 = annotated.shape[:2]
            h2, w2 = hm_overlay.shape[:2]
            if h1 != h2 or w1 != w2:
                hm_overlay = cv2.resize(hm_overlay, (w1, h1))
            annotated = np.hstack([annotated, hm_overlay])

        if show:
            cv2.imshow("Visual Anomaly Detector", annotated)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        if save_dir:
            out_dir = Path(save_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"result_{Path(path).stem}.jpg"
            cv2.imwrite(str(out_path), annotated)
            log.info("Saved → %s", out_path)

    return result


def infer_folder(
    folder: str | Path,
    extractor: FeatureExtractor,
    scorer: AnomalyScorer,
    threshold: float,
    cfg: AnomalyConfig,
    *,
    show: bool = False,
    save_dir: str | None = None,
    heatmap: bool = False,
    limit: int = 0,
) -> list[dict]:
    """Score all images in a folder."""
    folder = Path(folder)
    images = sorted(
        p for p in folder.rglob("*") if p.suffix.lower() in IMAGE_EXTS
    )

    if limit:
        images = images[:limit]

    print(f"[INFO] Processing {len(images)} images from {folder}")

    results: list[dict] = []
    anomaly_count = 0
    for i, path in enumerate(images):
        result = infer_image(
            path, extractor, scorer, threshold, cfg,
            show=False, save_dir=save_dir, heatmap=heatmap,
        )
        results.append(result)
        if result.get("is_anomaly"):
            anomaly_count += 1

        if (i + 1) % 20 == 0 or i == len(images) - 1:
            print(f"  [{i + 1}/{len(images)}] anomalies so far: {anomaly_count}")

    print(f"\n[RESULT] {len(images)} images — "
          f"{anomaly_count} anomalies, "
          f"{len(images) - anomaly_count} normal")

    if show and results:
        summary_img = draw_batch_summary(results)
        cv2.imshow("Batch Summary", summary_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return results


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(
        description="Visual Anomaly Detector — image & folder inference",
    )
    parser.add_argument("--source", required=True,
                        help="Image path or folder path")
    parser.add_argument("--config", default=None,
                        help="YAML/JSON config file")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Override anomaly threshold")
    parser.add_argument("--heatmap", action="store_true",
                        help="Generate anomaly heatmap")
    parser.add_argument("--no-display", action="store_true",
                        help="Disable GUI display")
    parser.add_argument("--save-dir", default=None,
                        help="Save annotated images to directory")
    parser.add_argument("--export", default=None,
                        help="Export results to JSON/CSV file")
    parser.add_argument("--limit", type=int, default=0,
                        help="Max images to process (0=all)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    extractor, scorer, threshold = _load_model(cfg)

    if args.threshold is not None:
        threshold = args.threshold

    source = Path(args.source)
    show = not args.no_display

    if source.is_dir():
        results = infer_folder(
            source, extractor, scorer, threshold, cfg,
            show=show, save_dir=args.save_dir,
            heatmap=args.heatmap, limit=args.limit,
        )
    elif source.is_file():
        # Need numpy for heatmap stacking
        result = infer_image(
            source, extractor, scorer, threshold, cfg,
            show=show, save_dir=args.save_dir, heatmap=args.heatmap,
        )
        results = [result]
        print(f"[RESULT] {result['label']} — score={result.get('anomaly_score', 0):.4f}")
    else:
        print(f"[ERROR] Source not found: {source}")
        sys.exit(1)

    if args.export:
        fmt = "csv" if args.export.endswith(".csv") else "json"
        export_results(results, args.export, fmt=fmt)
        print(f"[INFO] Results exported to {args.export}")


if __name__ == "__main__":
    main()
