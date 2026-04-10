"""Food Freshness Grader — evaluate classification quality.

Runs evaluation on the validation or test split of the dataset.

Usage::

    python evaluate.py --eval --data path/to/dataset
    python evaluate.py --eval --data path/to/dataset --weights runs/freshness_cls/best_model.pt
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from config import FreshnessConfig, load_config, parse_label
from grader import FreshnessGrader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def _find_eval_dir(data_dir: Path) -> Path:
    """Find the validation/test split directory."""
    for name in ("val", "validation", "valid", "test", "Val", "Test"):
        p = data_dir / name
        if p.is_dir():
            return p
    # Fall back to the data dir itself
    return data_dir


def _evaluate(args: argparse.Namespace) -> None:
    cfg = load_config(args.config) if args.config else FreshnessConfig()
    if args.weights:
        cfg.weights_path = args.weights
    if args.model:
        cfg.model_name = args.model

    data_path = Path(args.data)
    if not data_path.exists():
        print(f"[ERROR] Data directory not found: {data_path}")
        sys.exit(1)

    eval_dir = _find_eval_dir(data_path)
    print(f"Evaluating on: {eval_dir}")

    # Load dataset
    val_tf = transforms.Compose([
        transforms.Resize(cfg.imgsz + 32),
        transforms.CenterCrop(cfg.imgsz),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    val_ds = datasets.ImageFolder(str(eval_dir), transform=val_tf)
    class_names = val_ds.classes
    print(f"Classes ({len(class_names)}): {class_names}")
    print(f"Samples: {len(val_ds)}")

    # Load grader model
    grader = FreshnessGrader(cfg)
    grader.load(args.weights)

    loader = torch.utils.data.DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4
    )

    # Evaluate
    device = grader._device
    model = grader._model
    model.eval()

    correct = 0
    total = 0
    per_class_correct = np.zeros(len(class_names))
    per_class_total = np.zeros(len(class_names))
    freshness_correct = 0
    freshness_total = 0

    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            preds = outputs.argmax(dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

            for pred, label in zip(preds.cpu().numpy(), labels.cpu().numpy()):
                per_class_total[label] += 1
                if pred == label:
                    per_class_correct[label] += 1

                # Freshness-level accuracy (fresh vs stale)
                pred_freshness = parse_label(class_names[pred])[0]
                true_freshness = parse_label(class_names[label])[0]
                freshness_total += 1
                if pred_freshness == true_freshness:
                    freshness_correct += 1

    overall_acc = correct / total if total else 0
    freshness_acc = freshness_correct / freshness_total if freshness_total else 0

    print(f"\n{'=' * 55}")
    print(f"  EVALUATION RESULTS")
    print(f"{'=' * 55}")
    print(f"  Samples evaluated:     {total}")
    print(f"  Overall accuracy:      {overall_acc:.2%}")
    print(f"  Freshness accuracy:    {freshness_acc:.2%}  (fresh vs stale)")
    print(f"{'=' * 55}")

    print(f"\n{'Class':<25} {'Accuracy':<12} {'Correct':<10} {'Total'}")
    print("-" * 55)
    for i, name in enumerate(class_names):
        t = int(per_class_total[i])
        c = int(per_class_correct[i])
        acc = c / t if t > 0 else 0
        print(f"  {name:<23} {acc:<12.2%} {c:<10} {t}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate freshness grading quality")
    ap.add_argument("--eval", action="store_true", help="Run evaluation")
    ap.add_argument("--data", required=False, help="Dataset directory")
    ap.add_argument("--weights", default=None, help="Model weights path")
    ap.add_argument("--model", default=None, help="Model architecture")
    ap.add_argument("--batch-size", type=int, default=32, help="Batch size")
    ap.add_argument("--config", default=None, help="Config file path")
    args = ap.parse_args()

    if not args.eval:
        ap.print_help()
        return

    if not args.data:
        print("[ERROR] --data is required for evaluation")
        sys.exit(1)

    _evaluate(args)


if __name__ == "__main__":
    main()
