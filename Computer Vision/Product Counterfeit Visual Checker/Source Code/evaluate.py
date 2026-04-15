"""Product Counterfeit Visual Checker — evaluate screening quality.
"""Product Counterfeit Visual Checker — evaluate screening quality.

Runs cross-product evaluation: for each product image, screen it
against references from the same and different products, and measure
how well the system separates same-product from different-product.

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
    """Evaluate screening accuracy using leave-one-out over the reference store."""
    from config import CounterfeitConfig, load_config
    from reference_store import ReferenceStore

    cfg = load_config(args.config) if args.config else CounterfeitConfig()
    ref_path = args.ref_path or cfg.reference_path

    if not Path(ref_path).exists():
        print("[ERROR] Reference store not found. Run reference_builder.py first.")
        sys.exit(1)

    store = ReferenceStore(metric=cfg.similarity_metric)
    store.load(ref_path)

    n = len(store)
    products = store._products
    embeddings = store._embeddings
    unique_products = sorted(set(p for p in products if p))
    print(f"Loaded references: {n} entries, {len(unique_products)} products")

    max_q = min(n, args.max_queries)
    indices = list(range(n))
    rng = np.random.RandomState(42)
    rng.shuffle(indices)
    query_indices = indices[:max_q]

    top1_correct = 0
    top3_correct = 0
    total = 0

    print(f"\nEvaluating {max_q} queries (leave-one-out) ...\n")

    for qi in query_indices:
        query_product = products[qi]
        if not query_product:
            continue

        query_emb = embeddings[qi]
        sims = (embeddings @ query_emb.reshape(-1, 1)).squeeze()
        sims[qi] = -1.0  # exclude self

        top3_idx = np.argsort(sims)[::-1][:3]
        top3_products = [products[i] for i in top3_idx]

        if top3_products[0] == query_product:
            top1_correct += 1
        if query_product in top3_products:
            top3_correct += 1
        total += 1

        if total % 50 == 0:
            print(f"  [{total}/{max_q}] top-1={top1_correct / total:.2%}  "
                  f"top-3={top3_correct / total:.2%}")

    top1_acc = top1_correct / total if total else 0
    top3_acc = top3_correct / total if total else 0

    print(f"\n{'=' * 50}")
    print(f"Queries evaluated:  {total}")
    print(f"Top-1 product acc:  {top1_acc:.2%}")
    print(f"Top-3 product acc:  {top3_acc:.2%}")
    print(f"{'=' * 50}")
    print(f"\nHigher accuracy -> better at matching products to their")
    print(f"correct references -> more reliable mismatch flagging.")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Evaluate counterfeit screening quality"
    )
    ap.add_argument("--eval", action="store_true", help="Run evaluation")
    ap.add_argument("--ref-path", default=None, help="Override reference store path")
    ap.add_argument("--max-queries", type=int, default=500,
                    help="Max queries to evaluate")
    ap.add_argument("--config", default=None, help="Path to config YAML/JSON")
    args = ap.parse_args()

    if not args.eval:
        ap.print_help()
        return

    _evaluate(args)


if __name__ == "__main__":
    main()
