"""Plant Disease Severity Estimator — CLI inference.
"""Plant Disease Severity Estimator — CLI inference.

Usage::

    # Single image
    python infer.py --source leaf.jpg

    # Directory batch
    python infer.py --source images/ --batch

    # Save annotated + grid + exports
    python infer.py --source images/ --batch --save --save-grid \\
        --export-json results.json --export-csv results.csv
"""
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from classifier import PlantDiseaseClassifier, PredictionResult
from config import SeverityConfig, load_config
from export import export_csv, export_json
from validator import collect_images, validate_image
from visualize import save_annotated, save_grid


def _print_result(path: Path | str, r: PredictionResult) -> None:
    sev = r.severity_name.upper()
    lr = f"  lesion={r.lesion_ratio:.1%}" if r.lesion_ratio is not None else ""
    print(
        f"  {Path(path).name:40s} "
        f"{r.plant:12s} {r.disease:30s} "
        f"[{sev:8s}]  conf={r.confidence:.2%}{lr}"
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Plant Disease Severity Estimator -- inference"
    )
    ap.add_argument("--source", required=True,
                    help="Image path or directory")
    ap.add_argument("--batch", action="store_true",
                    help="Process all images in source directory")
    ap.add_argument("--weights", type=str, default=None,
                    help="Path to model weights")
    ap.add_argument("--config", type=str, default=None,
                    help="Path to config file (JSON/YAML)")
    ap.add_argument("--model", type=str, default=None,
                    help="Model architecture override")
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--save", action="store_true",
                    help="Save annotated images")
    ap.add_argument("--save-grid", action="store_true",
                    help="Save batch thumbnail grid")
    ap.add_argument("--export-json", type=str, default=None,
                    help="Export results to JSON file")
    ap.add_argument("--export-csv", type=str, default=None,
                    help="Export results to CSV file")
    ap.add_argument("--output-dir", type=str, default="output",
                    help="Output directory for saved files")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.model:
        cfg.model_name = args.model
    if args.device:
        cfg.device = args.device

    clf = PlantDiseaseClassifier(cfg)
    clf.load(args.weights)

    paths = collect_images(args.source) if args.batch else [Path(args.source)]
    images = [validate_image(p) for p in paths]
    results = clf.classify_batch(images)

    print(f"\n{'-' * 80}")
    print(f"  Plant Disease Severity Estimator -- {len(results)} image(s)")
    print(f"{'-' * 80}")
    for p, r in zip(paths, results):
        _print_result(p, r)
    print(f"{'-' * 80}\n")

    # ── Optional outputs ──────────────────────────────────
    if args.save:
        for p, img, r in zip(paths, images, results):
            out = save_annotated(img, r, args.output_dir, f"annotated_{p.stem}.jpg", cfg)
            print(f"  Saved: {out}")

    if args.save_grid and len(images) > 1:
        out = save_grid(images, results, args.output_dir, "grid.jpg", cfg)
        print(f"  Grid:  {out}")

    if args.export_json:
        out = export_json(results, args.export_json, [str(p) for p in paths])
        print(f"  JSON:  {out}")

    if args.export_csv:
        out = export_csv(results, args.export_csv, [str(p) for p in paths])
        print(f"  CSV:   {out}")

    clf.close()


if __name__ == "__main__":
    main()
