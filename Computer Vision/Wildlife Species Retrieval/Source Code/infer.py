"""Wildlife Species Retrieval — CLI query entry point.

Usage::

    python infer.py --source photo.jpg --top-k 8
    python infer.py --source photo.jpg --save-grid output/grid.jpg
    python infer.py --source photo.jpg --export-json results.json
    python infer.py --source photo.jpg --rerank
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import WildlifeConfig, load_config
from controller import WildlifeController

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Wildlife species retrieval — find similar images"
    )
    ap.add_argument("--source", required=True, help="Query image path")
    ap.add_argument("--index-path", default=None, help="Override index path")
    ap.add_argument("--top-k", type=int, default=None, help="Number of results")
    ap.add_argument("--backbone", default=None, help="Backbone model name")
    ap.add_argument("--rerank", action="store_true",
                    help="Enable classifier reranking")
    ap.add_argument("--save-grid", default=None,
                    help="Save result grid to path")
    ap.add_argument("--show", action="store_true",
                    help="Display result grid")
    ap.add_argument("--export-json", default=None,
                    help="Export results to JSON")
    ap.add_argument("--export-csv", default=None,
                    help="Export results to CSV")
    ap.add_argument("--config", default=None, help="Path to config YAML")
    ap.add_argument("--device", default=None, help="Compute device")
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    cfg = load_config(args.config)
    if args.index_path:
        cfg.index_path = args.index_path
    if args.backbone:
        cfg.backbone = args.backbone
    if args.top_k:
        cfg.top_k = args.top_k
    if args.device:
        cfg.device = args.device
    if args.rerank:
        cfg.enable_rerank = True

    ctrl = WildlifeController(cfg)
    ctrl.load()

    result = ctrl.query_and_visualise(
        args.source,
        save_path=args.save_grid,
        show=args.show,
    )

    # Print results
    print(f"\nQuery: {result.query_path}")
    print(f"Top match: {result.top_path}  (score={result.top_score:.4f})")
    print(f"\n{'Rank':<6} {'Score':<10} {'Species':<20} Path")
    print("-" * 75)
    for h in result.hits:
        print(f"{h.rank:<6} {h.score:<10.4f} {h.species or '-':<20} {h.path}")

    if result.species_votes:
        print(f"\nSpecies votes:")
        for sp, score in sorted(result.species_votes.items(),
                                key=lambda x: -x[1]):
            print(f"  {sp:<25} {score:.4f}")

    # Export
    ctrl.export_results(
        result,
        json_path=args.export_json,
        csv_path=args.export_csv,
    )

    ctrl.close()


if __name__ == "__main__":
    main()
