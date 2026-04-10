"""PDF-to-image conversion utility.

Converts PDF pages to BGR numpy arrays for detection.
Supports two backends (auto-detected):
  1. ``PyMuPDF`` (fitz) — fast, no system dependencies
  2. ``pdf2image`` + Poppler — fallback

Usage::

    from pdf_utils import pdf_to_images
    pages = pdf_to_images("document.pdf", dpi=300)
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

log = logging.getLogger("doc_layout.pdf_utils")


def pdf_to_images(pdf_path: str | Path, *, dpi: int = 300) -> list[np.ndarray]:
    """Convert all pages of a PDF to BGR numpy arrays.

    Tries PyMuPDF first, falls back to pdf2image.

    Parameters
    ----------
    pdf_path : str | Path
        Path to the PDF file.
    dpi : int
        Resolution for rasterisation.

    Returns
    -------
    list[np.ndarray]
        One BGR image per page.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    try:
        return _pymupdf_convert(pdf_path, dpi)
    except ImportError:
        pass

    try:
        return _pdf2image_convert(pdf_path, dpi)
    except ImportError:
        pass

    raise ImportError(
        "PDF conversion requires either PyMuPDF (`pip install PyMuPDF`) "
        "or pdf2image (`pip install pdf2image`) + Poppler."
    )


def _pymupdf_convert(pdf_path: Path, dpi: int) -> list[np.ndarray]:
    import fitz  # PyMuPDF

    import cv2

    doc = fitz.open(str(pdf_path))
    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)
    pages: list[np.ndarray] = []

    for page in doc:
        pix = page.get_pixmap(matrix=matrix)
        # pix.samples is RGB; convert to BGR
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, 3)
        bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        pages.append(bgr)

    doc.close()
    log.info("Converted %d pages via PyMuPDF (dpi=%d)", len(pages), dpi)
    return pages


def _pdf2image_convert(pdf_path: Path, dpi: int) -> list[np.ndarray]:
    from pdf2image import convert_from_path

    import cv2

    pil_images = convert_from_path(str(pdf_path), dpi=dpi)
    pages: list[np.ndarray] = []

    for pil_img in pil_images:
        rgb = np.array(pil_img)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        pages.append(bgr)

    log.info("Converted %d pages via pdf2image (dpi=%d)", len(pages), dpi)
    return pages


def is_pdf(path: str | Path) -> bool:
    """Check if a file path is a PDF."""
    return Path(path).suffix.lower() == ".pdf"
