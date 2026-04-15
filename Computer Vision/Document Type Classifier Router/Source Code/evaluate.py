"""Document Type Classifier Router — model evaluation.

Computes overall and per-class accuracy on a validation set organised
in ImageFolder format, plus routing statistics.

Usage::

    python evaluate.py --data path/to/val
    python evaluate.py --data path/to/val --weights runs/document_cls/best_model.pt
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from classifier import DocumentClassifier
from config import DISPLAY_LABELS, RouterConfig, load_config
from router import DocumentRouter
from validator import collect_images, validate_image


def evaluate(
    val_dir: str | Path,
    clf: DocumentClassifier,
    router: DocumentRouter,
) -> dict:
    """Run evaluation over an ImageFolder validation set."""
    val_path = Path(val_dir)

    class_correct: dict[str, int] = defaultdict(int)
    class_total: dict[str, int] = defaultdict(int)
    routed_count = 0
    review_count = 0

    for class_dir in sorted(val_path.iterdir()):
        if not class_dir.is_dir():
            continue
        gt_class = class_dir.name

        for img_path in sorted(collect_images(class_dir)):
            try:
                img = validate_image(img_path)
            except (ValueError, FileNotFoundError):
                continue

            cr = clf.classify(img)
            rd = router.route(cr)

            class_total[gt_class] += 1
            if cr.class_name == gt_class:
                class_correct[gt_class] += 1

            if rd.routed:
                routed_count += 1
            else:
                review_count += 1

    total = sum(class_total.values())
    correct = sum(class_correct.values())
    overall_acc = correct / max(total, 1)

    per_class = {
        cls: class_correct[cls] / max(class_total[cls], 1)
        for cls in sorted(class_total)
    }

    return {
        "overall_accuracy": overall_acc,
        "total_images": total,
        "total_correct": correct,
        "routed": routed_count,
        "manual_review": review_count,
        "per_class_accuracy": per_class,
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Evaluate document type classifier + router"
    )
    ap.add_argument("--data", type=str, required=True,
                    help="Path to validation set (ImageFolder)")
    ap.add_argument("--weights", type=str, default=None)
    ap.add_argument("--config", type=str, default=None)
    ap.add_argument("--model", type=str, default=None)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--threshold", type=float, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.model:
        cfg.model_name = args.model
    if args.device:
        cfg.device = args.device
    if args.threshold is not None:
        cfg.confidence_threshold = args.threshold

    clf = DocumentClassifier(cfg)
    clf.load(args.weights)
    router = DocumentRouter(cfg)

    metrics = evaluate(args.data, clf, router)

    print(f"\n{'=' * 70}")
    print(f"  Document Type Classifier Router -- Evaluation")
    print(f"{'=' * 70}")
    print(f"  Overall accuracy: {metrics['overall_accuracy']:.2%} "
          f"({metrics['total_correct']}/{metrics['total_images']})")
    print(f"  Routed:           {metrics['routed']}")
    print(f"  Manual review:    {metrics['manual_review']}")

    print(f"\n  Per-class accuracy ({len(metrics['per_class_accuracy'])} classes):")
    for cls, acc in metrics["per_class_accuracy"].items():
        label = DISPLAY_LABELS.get(cls, cls)
        print(f"    {label:30s}  {acc:.2%}")
    print(f"{'=' * 70}\n")

    clf.close()


if __name__ == "__main__":
    main()
