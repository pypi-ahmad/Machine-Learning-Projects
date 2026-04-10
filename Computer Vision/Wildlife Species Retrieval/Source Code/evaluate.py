"""Wildlife Species Retrieval — evaluation.

Performs leave-one-out retrieval evaluation: for each image in the
validation set, query the index (excluding itself) and check whether
the top-k results contain the correct species.

Usage::

    python evaluate.py --index-path index/wildlife_index.npz --val-dir data/val
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2

from config import WildlifeConfig, load_config
from embedder import WildlifeEmbedder
from index import WildlifeIndex
from validator import collect_images, infer_species

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def evaluate(
    index: WildlifeIndex,
    embedder: WildlifeEmbedder,
    val_dir: Path,
    top_k: int = 5,
) -> dict:
    """Run leave-one-out retrieval evaluation.

    For each validation image, embed it, search the index (excluding
    itself), and check whether the correct species appears in top-k.

    Returns dict with accuracy metrics.
    """
    images = collect_images(val_dir, recursive=True)
    if not images:
        raise FileNotFoundError(f"No images in {val_dir}")

    total = 0
    top1_correct = 0
    topk_correct = 0
    per_species_total: dict[str, int] = defaultdict(int)
    per_species_top1: dict[str, int] = defaultdict(int)

    for img_path in images:
        gt_species = infer_species(img_path, val_dir)
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        embedding = embedder.embed(img)
        hits = index.search(
            embedding, top_k=top_k,
            exclude_self=True, self_path=str(img_path),
        )
        if not hits:
            total += 1
            per_species_total[gt_species] += 1
            continue

        total += 1
        per_species_total[gt_species] += 1

        if hits[0].species == gt_species:
            top1_correct += 1
            per_species_top1[gt_species] += 1

        if any(h.species == gt_species for h in hits):
            topk_correct += 1

    per_species_acc = {
        sp: per_species_top1.get(sp, 0) / max(per_species_total[sp], 1)
        for sp in sorted(per_species_total)
    }

    return {
        "total": total,
        "top1_accuracy": top1_correct / max(total, 1),
        "top1_correct": top1_correct,
        f"top{top_k}_accuracy": topk_correct / max(total, 1),
        f"top{top_k}_correct": topk_correct,
        "num_species": len(per_species_total),
        "per_species_accuracy": per_species_acc,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate wildlife retrieval")
    ap.add_argument("--val-dir", required=True,
                    help="Validation set (ImageFolder)")
    ap.add_argument("--index-path", default=None, help="Index path")
    ap.add_argument("--backbone", default=None, help="Backbone name")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--config", default=None)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.index_path:
        cfg.index_path = args.index_path
    if args.backbone:
        cfg.backbone = args.backbone
    if args.device:
        cfg.device = args.device

    embedder = WildlifeEmbedder(cfg)
    embedder.load()

    index = WildlifeIndex.load(cfg.index_path)
    logger.info("Index loaded: %d entries", index.size)

    metrics = evaluate(index, embedder, Path(args.val_dir), top_k=args.top_k)

    print(f"\n{'═' * 65}")
    print(f"  Wildlife Species Retrieval — Evaluation")
    print(f"{'═' * 65}")
    print(f"  Total images:    {metrics['total']}")
    print(f"  Top-1 accuracy:  {metrics['top1_accuracy']:.2%} "
          f"({metrics['top1_correct']}/{metrics['total']})")
    print(f"  Top-{args.top_k} accuracy:  "
          f"{metrics[f'top{args.top_k}_accuracy']:.2%} "
          f"({metrics[f'top{args.top_k}_correct']}/{metrics['total']})")
    print(f"  Species:         {metrics['num_species']}")

    print(f"\n  Per-species top-1 accuracy:")
    for sp, acc in metrics["per_species_accuracy"].items():
        print(f"    {sp:25s}  {acc:.2%}")
    print(f"{'═' * 65}\n")

    embedder.close()


if __name__ == "__main__":
    main()
