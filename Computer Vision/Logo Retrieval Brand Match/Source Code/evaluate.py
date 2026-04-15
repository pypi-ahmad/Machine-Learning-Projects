"""Logo Retrieval Brand Match — evaluate retrieval quality.
"""Logo Retrieval Brand Match — evaluate retrieval quality.

Runs leave-one-out or held-out query evaluation on the indexed dataset
to measure top-1 and top-5 brand accuracy.

Usage::

    python evaluate.py --eval
    python evaluate.py --eval --max-queries 100
    python evaluate.py --eval --force-download
"""
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


def _evaluate(args: argparse.Namespace) -> None:
    """Evaluate retrieval accuracy using the index itself."""
    from config import LogoConfig, load_config
    from embedder import LogoEmbedder
    from index import LogoIndex

    cfg = load_config(args.config) if args.config else LogoConfig()
    idx_path = args.index or cfg.index_path

    if not Path(idx_path).exists():
        print("[ERROR] Index not found. Run index_builder.py first.")
        sys.exit(1)

    idx = LogoIndex.load(idx_path)
    print(f"Loaded index: {idx.size} entries, {len(idx.brands)} brands")

    embedder = LogoEmbedder(cfg)
    embedder.load()

    max_q = min(idx.size, args.max_queries)
    indices = list(range(idx.size))
    rng = np.random.RandomState(42)
    rng.shuffle(indices)
    query_indices = indices[:max_q]

    top1_correct = 0
    top5_correct = 0
    total = 0

    print(f"\nEvaluating {max_q} queries (leave-one-out) ...\n")

    for qi in query_indices:
        query_path = idx._paths[qi]
        query_brand = idx._brands[qi]
        query_embedding = idx._embeddings[qi]

        # Search excluding the query itself
        sims = (idx._embeddings @ query_embedding.reshape(-1, 1)).squeeze()
        sims[qi] = -1.0  # exclude self

        top5_idx = np.argsort(sims)[::-1][:5]
        top5_brands = [idx._brands[i] for i in top5_idx]

        if top5_brands[0] == query_brand:
            top1_correct += 1
        if query_brand in top5_brands:
            top5_correct += 1
        total += 1

        if total % 20 == 0:
            print(f"  [{total}/{max_q}] top-1={top1_correct / total:.2%}  "
                  f"top-5={top5_correct / total:.2%}")

    embedder.close()

    top1_acc = top1_correct / total if total else 0
    top5_acc = top5_correct / total if total else 0

    print(f"\n{'=' * 50}")
    print(f"Queries evaluated:  {total}")
    print(f"Top-1 accuracy:     {top1_acc:.2%}")
    print(f"Top-5 accuracy:     {top5_acc:.2%}")
    print(f"{'=' * 50}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Logo Retrieval Brand Match -- evaluation",
    )
    parser.add_argument("--eval", action="store_true",
                        help="Run retrieval evaluation")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--index", type=str, default=None,
                        help="Path to logo index (.npz)")
    parser.add_argument("--max-queries", type=int, default=200,
                        help="Max queries to evaluate (default: 200)")
    parser.add_argument("--force-download", action="store_true")
    args = parser.parse_args()

    if args.eval:
        _evaluate(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
