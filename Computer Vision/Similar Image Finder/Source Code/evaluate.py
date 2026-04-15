"""Similar Image Finder — evaluate retrieval quality.
"""Similar Image Finder — evaluate retrieval quality.

Runs leave-one-out evaluation on the indexed dataset to measure
top-1 and top-k category accuracy.

Usage::

    python evaluate.py --eval
    python evaluate.py --eval --max-queries 200
"""
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


def _evaluate(args: argparse.Namespace) -> None:
    """Evaluate retrieval accuracy using leave-one-out over the index."""
    from config import SimilarityConfig, load_config
    from index import ImageIndex

    cfg = load_config(args.config) if args.config else SimilarityConfig()
    idx_path = args.index or cfg.index_path

    if not Path(idx_path).exists():
        print("[ERROR] Index not found. Run index_builder.py first.")
        sys.exit(1)

    idx = ImageIndex(metric=cfg.similarity_metric)
    idx.load(idx_path)

    n = len(idx)
    categories = idx._categories
    embeddings = idx._embeddings
    unique_cats = sorted(set(c for c in categories if c))
    print(f"Loaded index: {n} entries, {len(unique_cats)} categories")
    print(f"Categories: {', '.join(unique_cats)}")

    max_q = min(n, args.max_queries)
    indices = list(range(n))
    rng = np.random.RandomState(42)
    rng.shuffle(indices)
    query_indices = indices[:max_q]

    top_k = args.top_k
    top1_correct = 0
    topk_correct = 0
    total = 0

    print(f"\nEvaluating {max_q} queries (leave-one-out, k={top_k}) ...\n")

    for qi in query_indices:
        query_cat = categories[qi]
        if not query_cat:
            continue

        query_emb = embeddings[qi]

        # Cosine similarity (embeddings are L2-normalised)
        sims = (embeddings @ query_emb.reshape(-1, 1)).squeeze()
        sims[qi] = -1.0  # exclude self

        topk_idx = np.argsort(sims)[::-1][:top_k]
        topk_cats = [categories[i] for i in topk_idx]

        if topk_cats[0] == query_cat:
            top1_correct += 1
        if query_cat in topk_cats:
            topk_correct += 1
        total += 1

        if total % 50 == 0:
            print(f"  [{total}/{max_q}] top-1={top1_correct / total:.2%}  "
                  f"top-{top_k}={topk_correct / total:.2%}")

    top1_acc = top1_correct / total if total else 0
    topk_acc = topk_correct / total if total else 0

    print(f"\n{'=' * 50}")
    print(f"Queries evaluated:  {total}")
    print(f"Top-1 accuracy:     {top1_acc:.2%}")
    print(f"Top-{top_k} accuracy:     {topk_acc:.2%}")
    print(f"{'=' * 50}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate image retrieval quality")
    ap.add_argument("--eval", action="store_true", help="Run evaluation")
    ap.add_argument("--index", default=None, help="Override index path")
    ap.add_argument("--max-queries", type=int, default=500, help="Max queries to evaluate")
    ap.add_argument("--top-k", type=int, default=5, help="Top-k for category accuracy")
    ap.add_argument("--config", default=None, help="Path to config YAML")
    args = ap.parse_args()

    if not args.eval:
        ap.print_help()
        return

    _evaluate(args)


if __name__ == "__main__":
    main()
