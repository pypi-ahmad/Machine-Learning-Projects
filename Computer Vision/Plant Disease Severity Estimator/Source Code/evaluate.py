"""Plant Disease Severity Estimator — model evaluation.

Computes overall, per-class, and per-severity accuracy on a validation
set organised in ImageFolder format.

Usage::

    python evaluate.py --data path/to/val
    python evaluate.py --data path/to/val --weights runs/plant_disease_cls/best_model.pt
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from classifier import PlantDiseaseClassifier
from config import SEVERITY_NAMES, SeverityConfig, load_config, parse_class
from validator import collect_images, validate_image


def evaluate(
    val_dir: str | Path,
    clf: PlantDiseaseClassifier,
) -> dict:
    """Run evaluation over an ImageFolder validation set.

    Parameters
    ----------
    val_dir : Path
        Root directory with class sub-folders.
    clf : PlantDiseaseClassifier
        A loaded classifier instance.

    Returns
    -------
    dict
        Metrics including overall accuracy, per-class accuracy,
        and per-severity accuracy.
    """
    val_path = Path(val_dir)

    class_correct: dict[str, int] = defaultdict(int)
    class_total: dict[str, int] = defaultdict(int)
    sev_correct: dict[int, int] = defaultdict(int)
    sev_total: dict[int, int] = defaultdict(int)

    for class_dir in sorted(val_path.iterdir()):
        if not class_dir.is_dir():
            continue
        gt_class = class_dir.name
        _, _, gt_sev, _ = parse_class(gt_class)

        for img_path in sorted(collect_images(class_dir)):
            try:
                img = validate_image(img_path)
            except (ValueError, FileNotFoundError):
                continue

            result = clf.classify(img)
            correct = result.class_name == gt_class

            class_total[gt_class] += 1
            sev_total[gt_sev] += 1
            if correct:
                class_correct[gt_class] += 1
                sev_correct[gt_sev] += 1

    total = sum(class_total.values())
    correct = sum(class_correct.values())
    overall_acc = correct / max(total, 1)

    per_class = {
        cls: class_correct[cls] / max(class_total[cls], 1)
        for cls in sorted(class_total)
    }

    per_severity = {}
    for sev_idx in sorted(sev_total):
        name = SEVERITY_NAMES[sev_idx] if sev_idx < len(SEVERITY_NAMES) else f"sev_{sev_idx}"
        per_severity[name] = {
            "accuracy": sev_correct[sev_idx] / max(sev_total[sev_idx], 1),
            "total": sev_total[sev_idx],
            "correct": sev_correct[sev_idx],
        }

    return {
        "overall_accuracy": overall_acc,
        "total_images": total,
        "total_correct": correct,
        "per_class_accuracy": per_class,
        "per_severity": per_severity,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate plant disease classifier")
    ap.add_argument("--data", type=str, required=True,
                    help="Path to validation set (ImageFolder)")
    ap.add_argument("--weights", type=str, default=None)
    ap.add_argument("--config", type=str, default=None)
    ap.add_argument("--model", type=str, default=None)
    ap.add_argument("--device", type=str, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.model:
        cfg.model_name = args.model
    if args.device:
        cfg.device = args.device

    clf = PlantDiseaseClassifier(cfg)
    clf.load(args.weights)

    metrics = evaluate(args.data, clf)

    print(f"\n{'═' * 70}")
    print(f"  Plant Disease Severity Estimator — Evaluation")
    print(f"{'═' * 70}")
    print(f"  Overall accuracy: {metrics['overall_accuracy']:.2%} "
          f"({metrics['total_correct']}/{metrics['total_images']})")

    print(f"\n  Per-severity accuracy:")
    for sev_name, data in metrics["per_severity"].items():
        print(f"    {sev_name:10s}  {data['accuracy']:.2%}  "
              f"({data['correct']}/{data['total']})")

    print(f"\n  Per-class accuracy ({len(metrics['per_class_accuracy'])} classes):")
    for cls, acc in metrics["per_class_accuracy"].items():
        total = class_count if (class_count := int(acc * 100)) else ""
        print(f"    {cls:55s}  {acc:.2%}")
    print(f"{'═' * 70}\n")

    clf.close()


if __name__ == "__main__":
    main()
