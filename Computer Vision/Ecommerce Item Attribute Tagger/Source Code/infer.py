"""Ecommerce Item Attribute Tagger — inference CLI.

Supports single-image and folder batch inference with optional
catalog export (JSON / CSV).

Usage::

    python infer.py --source product.jpg
    python infer.py --source products_folder/ --export catalog.json
    python infer.py --source img.jpg --weights best_model.pt --overlay
    python infer.py --source folder/ --export catalog.csv --no-display
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import cv2
import numpy as np

_SRC = Path(__file__).resolve().parent
_REPO = _SRC.parents[1]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_SRC))

from config import TaggerConfig, load_config
from detector import ItemDetector
from attribute_predictor import AttributePredictor
from visualize import draw_attributes, draw_label_overlay
from export import export_catalog, export_csv

log = logging.getLogger("attribute_tagger.infer")

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _collect_images(source: Path) -> list[Path]:
    """Collect image paths from file or directory."""
    if source.is_file():
        return [source]
    if source.is_dir():
        return sorted(p for p in source.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    raise FileNotFoundError(f"Source not found: {source}")


def run_inference(
    source: Path,
    cfg: TaggerConfig,
    weights: str | None = None,
    *,
    save_dir: Path | None = None,
    export_path: Path | None = None,
    show: bool = True,
    overlay: bool = False,
) -> list[dict]:
    """Run attribute prediction on one or more images.

    Returns list of result dicts.
    """
    # Load models
    detector = ItemDetector(cfg)
    detector.load()

    predictor = AttributePredictor(cfg)
    predictor.load(weights)

    images = _collect_images(source)
    log.info("Processing %d image(s)", len(images))

    results: list[dict] = []

    for img_path in images:
        frame = cv2.imread(str(img_path))
        if frame is None:
            log.warning("Cannot read: %s", img_path)
            continue

        # Isolate item
        crop, box = detector.isolate(frame)

        # Predict attributes
        prediction = predictor.predict_proba(crop)

        result = {
            "source": str(img_path),
            "prediction": prediction,
            "box": box,
        }
        results.append(result)

        # Console output
        print(f"\n{'='*60}")
        print(f"  {img_path.name}")
        print(f"{'='*60}")
        for attr, info in prediction.items():
            conf = info["confidence"]
            marker = " ⚠" if conf < cfg.confidence_threshold else ""
            print(f"  {attr:20s}  {info['label']:20s}  {conf:.1%}{marker}")

        # Visualisation
        if show or save_dir:
            if overlay:
                vis = draw_label_overlay(frame, prediction, cfg)
            else:
                vis = draw_attributes(frame, prediction, cfg, source_name=img_path.name)

            if save_dir:
                save_dir.mkdir(parents=True, exist_ok=True)
                out_path = save_dir / f"tagged_{img_path.name}"
                cv2.imwrite(str(out_path), vis)
                log.info("Saved → %s", out_path)

            if show:
                cv2.imshow("Attribute Tagger", vis)
                key = cv2.waitKey(0) & 0xFF
                if key in (27, ord("q")):
                    break

    if show:
        cv2.destroyAllWindows()

    # Export
    if export_path:
        ext = export_path.suffix.lower()
        if ext == ".csv":
            export_csv(results, export_path)
        else:
            export_catalog(results, export_path)
        print(f"\nExported {len(results)} items → {export_path}")

    # Print JSON summary
    summary = json.dumps(
        [{"source": r["source"],
          "attributes": {k: v["label"] for k, v in r["prediction"].items()}}
         for r in results],
        indent=2,
    )
    print(f"\nStructured output:\n{summary}")

    return results


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(
        description="Ecommerce Item Attribute Tagger — inference",
    )
    parser.add_argument("--source", type=str, required=True,
                        help="Image path or directory")
    parser.add_argument("--weights", type=str, default=None,
                        help="Model checkpoint path")
    parser.add_argument("--config", type=str, default=None,
                        help="YAML/JSON config file")
    parser.add_argument("--export", type=str, default=None,
                        help="Export path (.json or .csv)")
    parser.add_argument("--save-dir", type=str, default=None,
                        help="Directory to save annotated images")
    parser.add_argument("--overlay", action="store_true",
                        help="Compact overlay instead of sidebar panel")
    parser.add_argument("--no-display", action="store_true",
                        help="Skip GUI display")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.device:
        cfg.device = args.device

    run_inference(
        source=Path(args.source),
        cfg=cfg,
        weights=args.weights,
        save_dir=Path(args.save_dir) if args.save_dir else None,
        export_path=Path(args.export) if args.export else None,
        show=not args.no_display,
        overlay=args.overlay,
    )


if __name__ == "__main__":
    main()
