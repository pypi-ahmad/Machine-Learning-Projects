"""Dataset bootstrap for Invoice Field Extractor.

Downloads a public invoice/receipt dataset via DatasetResolver.
Falls back to generating synthetic invoice images with text-like
elements for demo and CI purposes.

Usage::

    from data_bootstrap import ensure_invoice_dataset

    data_root = ensure_invoice_dataset()            # idempotent
    data_root = ensure_invoice_dataset(force=True)  # force re-download
"""

from __future__ import annotations

import json
import logging
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

log = logging.getLogger("invoice_extractor.data_bootstrap")

PROJECT_KEY = "invoice_field_extractor"
DATA_ROOT = REPO_ROOT / "data" / PROJECT_KEY

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def ensure_invoice_dataset(*, force: bool = False) -> Path:
    """Download or generate the invoice dataset.

    Returns the project data root containing invoice images.
    """
    ready_marker = DATA_ROOT / ".ready"
    if ready_marker.exists() and not force:
        log.info("[%s] Dataset ready at %s -- skipping", PROJECT_KEY, DATA_ROOT)
        return DATA_ROOT

    # Try real download first
    try:
        from scripts.download_data import ensure_dataset as _ensure
        data_path = _ensure(PROJECT_KEY, force=force)
        images = list(data_path.rglob("*.jpg")) + list(data_path.rglob("*.png"))
        if images:
            log.info("[%s] Real dataset found at %s (%d images)", PROJECT_KEY, data_path, len(images))
            ready_marker.parent.mkdir(parents=True, exist_ok=True)
            ready_marker.write_text(time.strftime("%Y-%m-%dT%H:%M:%S"), encoding="utf-8")
            return data_path
    except Exception as exc:
        log.warning("[%s] Real download failed (%s) -- generating synthetic data", PROJECT_KEY, exc)

    # Synthetic fallback
    _generate_synthetic(DATA_ROOT)

    # Metadata
    info = {
        "dataset_key": PROJECT_KEY,
        "source_type": "synthetic",
        "description": "Auto-generated invoice images for OCR testing",
        "prepared_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    (DATA_ROOT / "dataset_info.json").write_text(
        json.dumps(info, indent=2), encoding="utf-8"
    )

    ready_marker.write_text(time.strftime("%Y-%m-%dT%H:%M:%S"), encoding="utf-8")
    log.info("[%s] Synthetic dataset ready at %s", PROJECT_KEY, DATA_ROOT)
    return DATA_ROOT


# ---------------------------------------------------------------------------
# Synthetic invoice generator
# ---------------------------------------------------------------------------

_VENDORS = [
    "Acme Corp", "Global Tech Inc", "Summit Services LLC",
    "Prime Supplies Co", "Metro Logistics", "Atlas Industries",
]

_ITEMS = [
    ("Widget A", "10", "25.00", "250.00"),
    ("Service B", "5", "100.00", "500.00"),
    ("Part C-100", "20", "12.50", "250.00"),
    ("License D", "1", "999.00", "999.00"),
    ("Consulting", "8", "150.00", "1200.00"),
    ("Shipping", "1", "45.00", "45.00"),
]

FONT = cv2.FONT_HERSHEY_SIMPLEX


def _generate_synthetic(root: Path) -> None:
    """Create synthetic invoice images for OCR testing."""
    random.seed(42)
    np.random.seed(42)

    images_dir = root / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    for idx in range(20):
        img = _make_invoice(idx)
        cv2.imwrite(str(images_dir / f"invoice_{idx:04d}.png"), img)


def _make_invoice(idx: int) -> np.ndarray:
    """Render a single synthetic invoice image."""
    h, w = 1100, 850
    img = np.full((h, w, 3), 250, dtype=np.uint8)

    vendor = random.choice(_VENDORS)
    inv_num = f"INV-{random.randint(1000, 9999)}"
    inv_date = f"{random.randint(1,12):02d}/{random.randint(1,28):02d}/2025"
    due_date = f"{random.randint(1,12):02d}/{random.randint(1,28):02d}/2025"

    y = 50
    # Vendor header
    cv2.putText(img, vendor, (50, y), FONT, 1.0, (30, 30, 30), 2)
    y += 50
    cv2.putText(img, "123 Business Street, City, ST 12345", (50, y), FONT, 0.45, (80, 80, 80), 1)
    y += 50

    # Invoice details
    cv2.line(img, (50, y), (w - 50, y), (180, 180, 180), 1)
    y += 30
    cv2.putText(img, f"Invoice Number: {inv_num}", (50, y), FONT, 0.55, (30, 30, 30), 1)
    y += 30
    cv2.putText(img, f"Invoice Date: {inv_date}", (50, y), FONT, 0.55, (30, 30, 30), 1)
    y += 30
    cv2.putText(img, f"Due Date: {due_date}", (50, y), FONT, 0.55, (30, 30, 30), 1)
    y += 30
    cv2.putText(img, "Bill To: Customer Corp", (50, y), FONT, 0.55, (30, 30, 30), 1)
    y += 50

    # Line items header
    cv2.line(img, (50, y), (w - 50, y), (180, 180, 180), 1)
    y += 25
    cv2.putText(img, "Description", (50, y), FONT, 0.5, (60, 60, 60), 1)
    cv2.putText(img, "Qty", (400, y), FONT, 0.5, (60, 60, 60), 1)
    cv2.putText(img, "Price", (500, y), FONT, 0.5, (60, 60, 60), 1)
    cv2.putText(img, "Amount", (650, y), FONT, 0.5, (60, 60, 60), 1)
    y += 10
    cv2.line(img, (50, y), (w - 50, y), (180, 180, 180), 1)
    y += 25

    # Random line items
    n_items = random.randint(2, 5)
    items = random.sample(_ITEMS, min(n_items, len(_ITEMS)))
    subtotal = 0.0
    for desc, qty, price, amount in items:
        cv2.putText(img, desc, (50, y), FONT, 0.45, (30, 30, 30), 1)
        cv2.putText(img, qty, (420, y), FONT, 0.45, (30, 30, 30), 1)
        cv2.putText(img, f"${price}", (500, y), FONT, 0.45, (30, 30, 30), 1)
        cv2.putText(img, f"${amount}", (650, y), FONT, 0.45, (30, 30, 30), 1)
        subtotal += float(amount)
        y += 28

    # Totals
    y += 20
    cv2.line(img, (500, y), (w - 50, y), (180, 180, 180), 1)
    y += 25
    tax = round(subtotal * 0.08, 2)
    total = round(subtotal + tax, 2)
    cv2.putText(img, f"Subtotal: ${subtotal:.2f}", (500, y), FONT, 0.5, (30, 30, 30), 1)
    y += 28
    cv2.putText(img, f"Tax: ${tax:.2f}", (500, y), FONT, 0.5, (30, 30, 30), 1)
    y += 28
    cv2.putText(img, f"Total: ${total:.2f}", (500, y), FONT, 0.6, (20, 20, 20), 2)

    return img
