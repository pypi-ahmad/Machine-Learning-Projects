"""CLI inference pipeline for Document Layout Block Detector.

Supports image files, PDFs (multi-page), and directories of images.

Usage::

    python infer.py --source scan.jpg
    python infer.py --source document.pdf --config layout_config.yaml
    python infer.py --source pages/ --export-json results.json --save-crops
    python infer.py --source scan.png --no-display --save-annotated
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import LayoutConfig, load_config
from detector import LayoutDetector
from export import LayoutExporter
from pdf_utils import is_pdf, pdf_to_images
from visualize import draw_overlay

log = logging.getLogger("doc_layout.infer")

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Document Layout Block Detector — Inference")
    p.add_argument("--source", required=True, help="Image, PDF, or directory path")
    p.add_argument("--config", default=None, help="Path to YAML/JSON config")
    p.add_argument("--model", default=None, help="Override model weights")
    p.add_argument("--conf", type=float, default=None, help="Override confidence threshold")
    p.add_argument("--imgsz", type=int, default=None, help="Override image size")
    p.add_argument("--no-display", action="store_true", help="Disable GUI window")
    p.add_argument("--export-json", default=None, help="JSON export path")
    p.add_argument("--save-crops", action="store_true", help="Save cropped block regions")
    p.add_argument("--crops-dir", default=None, help="Crop output directory")
    p.add_argument("--save-annotated", action="store_true", help="Save annotated images")
    p.add_argument("--output-dir", default=None, help="Output directory for annotated images")
    p.add_argument("--force-download", action="store_true", help="Force dataset re-download")
    return p.parse_args(argv)


def run(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")

    cfg = load_config(args.config) if args.config else LayoutConfig()

    # CLI overrides
    if args.model:
        cfg.model = args.model
    if args.conf is not None:
        cfg.conf_threshold = args.conf
    if args.imgsz is not None:
        cfg.imgsz = args.imgsz
    if args.no_display:
        cfg.show_display = False
    if args.export_json:
        cfg.export_json = args.export_json
    if args.save_crops:
        cfg.save_crops = True
    if args.crops_dir:
        cfg.crops_dir = args.crops_dir
    if args.save_annotated:
        cfg.save_annotated = True
    if args.output_dir:
        cfg.output_dir = args.output_dir

    if args.force_download:
        from data_bootstrap import ensure_layout_dataset
        ensure_layout_dataset(force=True)

    detector = LayoutDetector(cfg)
    source = Path(args.source)

    # Collect images to process
    pages: list[tuple[str, np.ndarray]] = []

    if source.is_dir():
        for img_path in sorted(source.iterdir()):
            if img_path.suffix.lower() in IMAGE_EXTS:
                img = cv2.imread(str(img_path))
                if img is not None:
                    pages.append((img_path.name, img))
    elif is_pdf(source):
        pdf_pages = pdf_to_images(source, dpi=cfg.pdf_dpi)
        for i, page_img in enumerate(pdf_pages):
            pages.append((f"{source.stem}_page_{i:04d}.png", page_img))
    else:
        img = cv2.imread(str(source))
        if img is None:
            log.error("Cannot read image: %s", source)
            return
        pages.append((source.name, img))

    if not pages:
        log.error("No images found in source: %s", source)
        return

    log.info("Processing %d page(s) from %s", len(pages), source)

    with LayoutExporter(cfg) as exporter:
        for page_idx, (name, image) in enumerate(pages):
            result = detector.process(image, page_idx=page_idx)
            exporter.write(result, image, source_name=name)

            vis = draw_overlay(image, result, cfg)

            if cfg.save_annotated:
                out_dir = Path(cfg.output_dir)
                out_dir.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(out_dir / f"annotated_{name}"), vis)

            if cfg.show_display:
                cv2.imshow("Document Layout Block Detector", vis)
                key = cv2.waitKey(0) & 0xFF
                if key == ord("q"):
                    break

            log.info(
                "Page %d (%s): %d blocks  %s",
                page_idx, name, result.total_blocks, result.class_counts,
            )

    if cfg.show_display:
        cv2.destroyAllWindows()

    log.info("Done.")


if __name__ == "__main__":
    run()
