"""Food Freshness Grader — CLI entry point.

Grade food images for freshness (single or batch).

Usage:
    python infer.py --source apple.jpg
    python infer.py --source photos/ --batch
    python infer.py --source apple.jpg --save output/graded.jpg
    python infer.py --source photos/ --batch --export-json output/results.json
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import FreshnessConfig, load_config
from controller import FreshnessController

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Grade food freshness from images")
    ap.add_argument("--source", required=True,
                    help="Image path (single) or directory (batch)")
    ap.add_argument("--batch", action="store_true",
                    help="Grade all images in --source directory")
    ap.add_argument("--weights", default=None, help="Override model weights path")
    ap.add_argument("--model", default=None, help="Model architecture name")
    ap.add_argument("--save", default=None,
                    help="Save annotated image (single mode)")
    ap.add_argument("--save-grid", default=None,
                    help="Save batch result grid")
    ap.add_argument("--show", action="store_true", help="Display result")
    ap.add_argument("--export-json", default=None, help="Export results to JSON")
    ap.add_argument("--export-csv", default=None, help="Export results to CSV")
    ap.add_argument("--batch-size", type=int, default=16,
                    help="Batch size for batch mode")
    ap.add_argument("--config", default=None, help="Path to config YAML/JSON")
    ap.add_argument("--device", default=None, help="Compute device (cpu/cuda)")
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    cfg = load_config(args.config) if args.config else FreshnessConfig()
    if args.weights:
        cfg.weights_path = args.weights
    if args.model:
        cfg.model_name = args.model
    if args.device:
        cfg.device = args.device

    ctrl = FreshnessController(cfg)
    ctrl.load()

    if args.batch:
        # ── Batch mode ────────────────────────────────────
        results = ctrl.grade_batch(
            args.source,
            batch_size=args.batch_size,
            save_grid=args.save_grid,
        )

        # Print summary
        n_fresh = sum(1 for _, g in results if g.freshness == "fresh")
        n_stale = sum(1 for _, g in results if g.freshness == "stale")
        print(f"\n{'=' * 60}")
        print(f"  BATCH GRADING RESULTS")
        print(f"{'=' * 60}")
        print(f"  Total:  {len(results)}")
        print(f"  Fresh:  {n_fresh}  ({n_fresh / len(results):.0%})")
        print(f"  Stale:  {n_stale}  ({n_stale / len(results):.0%})")
        print(f"{'=' * 60}")

        # Top results
        print(f"\n{'#':<4} {'Grade':<8} {'Produce':<15} {'Conf':<8} Path")
        print("-" * 65)
        for i, (path, g) in enumerate(results[:20], 1):
            print(f"{i:<4} {g.freshness:<8} {g.produce:<15} "
                  f"{g.confidence:<8.1%} {path}")
        if len(results) > 20:
            print(f"  … and {len(results) - 20} more")

        # Export
        ctrl.export_results(
            results,
            json_path=args.export_json,
            csv_path=args.export_csv,
        )

    else:
        # ── Single mode ───────────────────────────────────
        result = ctrl.grade_and_annotate(
            args.source,
            save_path=args.save,
            show=args.show,
        )

        print(f"\n{'=' * 40}")
        print(f"  Freshness:  {result.freshness.upper()}")
        print(f"  Produce:    {result.produce}")
        print(f"  Confidence: {result.confidence:.1%}")
        print(f"  Class:      {result.class_name}")
        print(f"{'=' * 40}")

        if args.export_json or args.export_csv:
            ctrl.export_results(
                [(args.source, result)],
                json_path=args.export_json,
                csv_path=args.export_csv,
            )

    ctrl.close()


if __name__ == "__main__":
    main()
