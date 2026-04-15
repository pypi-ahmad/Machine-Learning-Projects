"""CLI inference pipeline for Invoice Field Extractor.

Supports single images, directories of images, and PDF files.

Usage::

    python infer.py --source invoice.jpg
    python infer.py --source invoices/ --export-json results.json
    python infer.py --source invoice.pdf --export-csv results.csv --no-display
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import InvoiceConfig, load_config
from export import InvoiceExporter
from ocr_engine import OCREngine
from parser import InvoiceParser
from validator import InvoiceValidator
from visualize import draw_overlay

log = logging.getLogger("invoice_extractor.infer")

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Invoice Field Extractor -- Inference")
    p.add_argument("--source", required=True, help="Image/PDF path or directory")
    p.add_argument("--config", default=None, help="Path to YAML/JSON config")
    p.add_argument("--lang", default=None, help="OCR language (default: en)")
    p.add_argument("--gpu", action="store_true", help="Enable GPU for OCR")
    p.add_argument("--no-display", action="store_true", help="Disable GUI window")
    p.add_argument("--export-json", default=None, help="JSON export path")
    p.add_argument("--export-csv", default=None, help="CSV export path")
    p.add_argument("--save-annotated", action="store_true", help="Save annotated images")
    p.add_argument("--output-dir", default="output", help="Output directory")
    p.add_argument("--force-download", action="store_true", help="Force dataset re-download")
    return p.parse_args(argv)


def _collect_sources(source: str) -> list[Path]:
    """Resolve *source* to a list of image/PDF paths."""
    p = Path(source)
    if p.is_dir():
        files = []
        for ext in IMAGE_EXTS | {".pdf"}:
            files.extend(p.glob(f"*{ext}"))
        files.sort()
        return files
    if p.is_file():
        return [p]
    log.error("Source not found: %s", source)
    return []


def _load_pdf_pages(pdf_path: Path, dpi: int = 300) -> list:
    """Convert PDF pages to BGR images.

    Uses pdf2image (poppler) if available, falls back to PyMuPDF.
    """
    try:
        from pdf2image import convert_from_path
        import numpy as np

        pil_images = convert_from_path(str(pdf_path), dpi=dpi)
        pages = []
        for pil_img in pil_images:
            arr = np.array(pil_img)
            pages.append(cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))
        return pages
    except ImportError:
        pass

    try:
        import fitz  # PyMuPDF
        import numpy as np

        doc = fitz.open(str(pdf_path))
        pages = []
        for page in doc:
            pix = page.get_pixmap(dpi=dpi)
            arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                pix.h, pix.w, pix.n
            )
            if pix.n == 4:
                arr = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
            else:
                arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            pages.append(arr)
        doc.close()
        return pages
    except ImportError:
        log.error("PDF support requires pdf2image or PyMuPDF — install one of them")
        return []


def run(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")

    cfg = load_config(args.config) if args.config else InvoiceConfig()

    # CLI overrides
    if args.lang:
        cfg.ocr_lang = args.lang
    if args.gpu:
        cfg.use_gpu = True
    if args.no_display:
        cfg.show_display = False
    if args.export_json:
        cfg.export_json = args.export_json
    if args.export_csv:
        cfg.export_csv = args.export_csv
    if args.save_annotated:
        cfg.save_annotated = True
    if args.output_dir:
        cfg.output_dir = args.output_dir

    if args.force_download:
        from data_bootstrap import ensure_invoice_dataset
        ensure_invoice_dataset(force=True)

    engine = OCREngine(cfg)
    parser = InvoiceParser()
    validator = InvoiceValidator(cfg)

    sources = _collect_sources(args.source)
    if not sources:
        log.error("No images found at: %s", args.source)
        return

    with InvoiceExporter(cfg) as exporter:
        for src_path in sources:
            if src_path.suffix.lower() == ".pdf":
                pages = _load_pdf_pages(src_path, dpi=cfg.pdf_dpi)
                for i, page_img in enumerate(pages):
                    label = f"{src_path.name}[p{i + 1}]"
                    _process_image(page_img, label, engine, parser, validator, exporter, cfg)
            else:
                img = cv2.imread(str(src_path))
                if img is None:
                    log.warning("Cannot read: %s", src_path)
                    continue
                _process_image(img, src_path.name, engine, parser, validator, exporter, cfg)

    log.info("Done — processed %d source(s)", len(sources))


def _process_image(
    image,
    label: str,
    engine: OCREngine,
    parser: InvoiceParser,
    validator: InvoiceValidator,
    exporter: InvoiceExporter,
    cfg: InvoiceConfig,
) -> None:
    """Process a single image through the full pipeline."""
    blocks = engine.run(image)
    result = parser.parse(blocks)
    report = validator.validate(result)

    exporter.write(result, report=report, source=label)

    # Log summary
    found = [f"{n}={ef.value}" for n, ef in result.fields.items()]
    log.info("%s: %d blocks, fields=[%s]", label, len(blocks), ", ".join(found))
    if report.warnings:
        for w in report.warnings:
            log.warning("  %s: %s", w.field_name, w.message)

    # Visualize
    vis = draw_overlay(image, result, cfg, ocr_blocks=blocks)

    if cfg.save_annotated:
        out_dir = Path(cfg.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        safe_name = label.replace("[", "_").replace("]", "")
        out_path = out_dir / f"annotated_{safe_name}.jpg"
        cv2.imwrite(str(out_path), vis)
        log.info("  Saved -> %s", out_path)

    if cfg.show_display:
        cv2.imshow(f"Invoice: {label}", vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run()
