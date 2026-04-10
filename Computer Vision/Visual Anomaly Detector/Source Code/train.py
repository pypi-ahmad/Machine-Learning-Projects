"""Visual Anomaly Detector — training pipeline.

Trains a one-class anomaly model on normal (non-defective) images,
optionally evaluates on a test set, and auto-selects the threshold.

Usage::

    python train.py
    python train.py --data path/to/dataset
    python train.py --backbone resnet50 --scoring knn
    python train.py --force-download
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent
_REPO = _SRC.parents[1]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_SRC))

import numpy as np

from config import AnomalyConfig, load_config
from data_bootstrap import ensure_anomaly_dataset, find_normal_images, find_test_images
from feature_extractor import FeatureExtractor
from scorer import AnomalyScorer
from threshold import ThresholdSelector

log = logging.getLogger("visual_anomaly.train")


def train(
    data_root: Path,
    cfg: AnomalyConfig,
) -> tuple[AnomalyScorer, FeatureExtractor, float]:
    """Train the anomaly detector on normal images.

    Returns
    -------
    tuple
        ``(scorer, extractor, threshold)``
    """
    # Find normal images
    normal_paths = find_normal_images(data_root)
    if not normal_paths:
        print("[ERROR] No normal/good images found in dataset.")
        print("[INFO] Expected: data_root/train/good/ or data_root/good/")
        sys.exit(1)

    print(f"[INFO] Found {len(normal_paths)} normal training images")

    # Feature extraction
    extractor = FeatureExtractor(backbone=cfg.backbone, imgsz=cfg.imgsz)
    extractor.load()

    features = extractor.extract_from_paths(
        [str(p) for p in normal_paths], batch_size=32,
    )
    print(f"[INFO] Extracted features: {features.shape}")

    # Fit scorer
    scorer = AnomalyScorer(
        method=cfg.scoring_method,
        knn_k=cfg.knn_k,
        regularization=cfg.regularization,
    )
    summary = scorer.fit(features)
    print(f"[INFO] Scorer fitted: {summary}")

    # Threshold selection
    selector = ThresholdSelector()
    normal_scores = np.array([
        scorer.score_primary(f) for f in features
    ])

    threshold = cfg.anomaly_threshold

    test_normal_paths, test_anomaly_paths = find_test_images(data_root)

    if cfg.auto_threshold and (test_normal_paths or test_anomaly_paths):
        # Use validation data for optimal threshold
        if test_anomaly_paths:
            test_norm_feats = extractor.extract_from_paths(
                [str(p) for p in test_normal_paths], batch_size=32,
            ) if test_normal_paths else np.array([]).reshape(0, features.shape[1])

            test_anom_feats = extractor.extract_from_paths(
                [str(p) for p in test_anomaly_paths], batch_size=32,
            )

            test_norm_scores = np.array([
                scorer.score_primary(f) for f in test_norm_feats
            ]) if len(test_norm_feats) else normal_scores

            test_anom_scores = np.array([
                scorer.score_primary(f) for f in test_anom_feats
            ])

            threshold = selector.f1_optimal(test_norm_scores, test_anom_scores)
            auroc = selector.auroc(test_norm_scores, test_anom_scores)
            print(f"[INFO] F1-optimal threshold: {threshold:.4f}  AUROC: {auroc:.4f}")
        else:
            threshold = selector.percentile(normal_scores, cfg.auto_threshold_percentile)
    else:
        threshold = selector.percentile(normal_scores, cfg.auto_threshold_percentile)

    print(f"[INFO] Selected threshold: {threshold:.4f}")

    return scorer, extractor, threshold


def evaluate(
    data_root: Path,
    extractor: FeatureExtractor,
    scorer: AnomalyScorer,
    threshold: float,
) -> dict:
    """Evaluate the trained model on test data."""
    test_normal, test_anomaly = find_test_images(data_root)

    if not test_normal and not test_anomaly:
        print("[INFO] No test set found; skipping evaluation")
        return {}

    tp, tn, fp, fn = 0, 0, 0, 0

    for p in test_normal:
        import cv2
        img = cv2.imread(str(p))
        if img is None:
            continue
        feat = extractor.extract(img)
        score = scorer.score_primary(feat)
        if score > threshold:
            fp += 1
        else:
            tn += 1

    for p in test_anomaly:
        import cv2
        img = cv2.imread(str(p))
        if img is None:
            continue
        feat = extractor.extract(img)
        score = scorer.score_primary(feat)
        if score > threshold:
            tp += 1
        else:
            fn += 1

    total = tp + tn + fp + fn
    acc = (tp + tn) / total * 100 if total else 0
    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0

    print(f"\n{'=' * 50}")
    print(f"  TP={tp}  TN={tn}  FP={fp}  FN={fn}")
    print(f"  Accuracy:  {acc:.1f}%")
    print(f"  Precision: {precision:.3f}")
    print(f"  Recall:    {recall:.3f}")
    print(f"  F1 Score:  {f1:.3f}")
    print(f"{'=' * 50}")

    return {
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "accuracy": round(acc, 2),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(
        description="Visual Anomaly Detector — train on normal images",
    )
    parser.add_argument("--data", type=str, default=None,
                        help="Path to anomaly dataset root")
    parser.add_argument("--config", type=str, default=None,
                        help="YAML/JSON config file")
    parser.add_argument("--backbone", type=str, default=None,
                        help="Feature extractor backbone")
    parser.add_argument("--scoring", type=str, default=None,
                        choices=["mahalanobis", "knn", "combined"],
                        help="Scoring method")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Fixed anomaly threshold (disables auto)")
    parser.add_argument("--force-download", action="store_true",
                        help="Force re-download dataset")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.backbone:
        cfg.backbone = args.backbone
    if args.scoring:
        cfg.scoring_method = args.scoring
    if args.threshold is not None:
        cfg.anomaly_threshold = args.threshold
        cfg.auto_threshold = False

    # Resolve dataset
    if args.data:
        data_root = Path(args.data)
    else:
        data_root = ensure_anomaly_dataset(force=args.force_download)

    # Train
    scorer, extractor, threshold = train(data_root, cfg)

    # Evaluate
    eval_results = evaluate(data_root, extractor, scorer, threshold)

    # Save
    runs_dir = _SRC / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    scorer.save(str(runs_dir / "anomaly_model.npz"))

    meta = {
        "backbone": cfg.backbone,
        "scoring_method": cfg.scoring_method,
        "threshold": threshold,
        "auto_threshold": cfg.auto_threshold,
        **eval_results,
    }
    (runs_dir / "train_meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8",
    )
    print(f"[INFO] Model saved to {runs_dir / 'anomaly_model.npz'}")
    print(f"[INFO] Metadata saved to {runs_dir / 'train_meta.json'}")


if __name__ == "__main__":
    main()
