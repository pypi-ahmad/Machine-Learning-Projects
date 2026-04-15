"""Dataset bootstrap for Receipt Digitizer.

Downloads and prepares a public receipt OCR dataset for testing
and benchmarking the extraction pipeline.  Falls back to synthetic
receipt images when the Kaggle/HuggingFace download is unavailable.

Dataset: jinhybr/OCR-receipt (Hugging Face) via DatasetResolver.

Usage::

    from data_bootstrap import ensure_receipt_dataset

    data_root = ensure_receipt_dataset()            # idempotent
    data_root = ensure_receipt_dataset(force=True)  # force re-download
"""

from __future__ import annotations

import json
import logging
import random
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

log = logging.getLogger("receipt_digitizer.data_bootstrap")

PROJECT_KEY = "receipt_digitizer"
DATA_ROOT = REPO_ROOT / "data" / PROJECT_KEY


def ensure_receipt_dataset(*, force: bool = False) -> Path:
    """Download and prepare the receipt dataset (idempotent).

    Falls back to synthetic receipt images when download fails.
    """
    images_dir = DATA_ROOT / "images"
    ready_marker = DATA_ROOT / ".ready"

    if ready_marker.exists() and not force:
        log.info("[%s] Dataset ready at %s -- skipping", PROJECT_KEY, DATA_ROOT)
        return DATA_ROOT

    # Try downloading real dataset first
    try:
        from scripts.download_data import ensure_dataset as _ensure
        data_path = _ensure(PROJECT_KEY, force=force)

        # Collect images from download
        images_dir.mkdir(parents=True, exist_ok=True)
        count = 0
        for img in data_path.rglob("*"):
            if img.suffix.lower() in {".jpg", ".jpeg", ".png"} and img.parent != images_dir:
                dst = images_dir / img.name
                if not dst.exists():
                    shutil.copy2(str(img), str(dst))
                    count += 1
        if count > 0:
            log.info("[%s] Collected %d receipt images", PROJECT_KEY, count)
            ready_marker.write_text(time.strftime("%Y-%m-%dT%H:%M:%S"), encoding="utf-8")
            return DATA_ROOT
    except Exception as exc:
        log.warning("[%s] Download failed (%s), generating synthetic receipts", PROJECT_KEY, exc)

    # Fallback: generate synthetic receipt images
    _generate_synthetic_receipts(images_dir, n=20)
    ready_marker.write_text(time.strftime("%Y-%m-%dT%H:%M:%S"), encoding="utf-8")
    log.info("[%s] Synthetic dataset ready at %s", PROJECT_KEY, DATA_ROOT)
    return DATA_ROOT


# ---------------------------------------------------------------------------
# Synthetic receipt generator
# ---------------------------------------------------------------------------

_MERCHANTS = [
    "COFFEE HOUSE", "GROCERY MART", "PIZZA PALACE", "TECH STORE",
    "BOOKSHOP", "DELI CORNER", "BURGER JOINT", "PHARMA PLUS",
    "GAS STATION", "FLOWER SHOP", "BAKERY FRESH", "SUSHI BAR",
    "HARDWARE HUB", "PET SUPPLY", "SPORTS GEAR",
]

_ITEMS = [
    ("Coffee Latte", 4.50), ("Espresso", 3.25), ("Muffin", 2.99),
    ("Sandwich", 7.50), ("Orange Juice", 3.75), ("Salad Bowl", 8.99),
    ("Water Bottle", 1.50), ("Chips", 2.25), ("Cookie", 1.99),
    ("Bagel", 3.50), ("Soup", 5.99), ("Tea", 2.50),
    ("Pasta", 9.99), ("Bread", 4.25), ("Milk 1L", 3.99),
    ("Eggs 12pk", 5.49), ("Yogurt", 1.89), ("Cheese", 6.75),
    ("Apples 1kg", 3.29), ("Banana", 0.99),
]

_PAYMENTS = ["Visa ****1234", "Mastercard ****5678", "Cash", "Debit ****9012"]


def _generate_synthetic_receipts(out_dir: Path, n: int = 20) -> None:
    """Render synthetic receipt images with known fields for testing."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(42)

    for i in range(n):
        merchant = rng.choice(_MERCHANTS)
        date_str = f"{rng.randint(1,12):02d}/{rng.randint(1,28):02d}/2025"
        time_str = f"{rng.randint(8,21):02d}:{rng.randint(0,59):02d}"
        payment = rng.choice(_PAYMENTS)
        num_items = rng.randint(2, 6)
        items = rng.sample(_ITEMS, min(num_items, len(_ITEMS)))

        # Build text lines
        lines = []
        lines.append(merchant)
        lines.append(f"123 Main St, City")
        lines.append(f"Tel: 555-{rng.randint(1000,9999)}")
        lines.append("")
        lines.append(f"Date: {date_str}  Time: {time_str}")
        lines.append("-" * 34)

        subtotal = 0.0
        for desc, price in items:
            qty = rng.randint(1, 3)
            amount = price * qty
            subtotal += amount
            if qty > 1:
                lines.append(f"{qty} x ${price:.2f}  {desc}")
            else:
                lines.append(f"  {desc:<22s} ${amount:.2f}")
        lines.append(f"  {'':22s} ------")

        tax = round(subtotal * 0.08, 2)
        total = round(subtotal + tax, 2)

        lines.append(f"  {'Subtotal':<22s} ${subtotal:.2f}")
        lines.append(f"  {'Tax':<22s} ${tax:.2f}")
        lines.append(f"  {'TOTAL':<22s} ${total:.2f}")
        lines.append("")
        lines.append(f"  Payment: {payment}")
        lines.append(f"  Currency: USD")
        lines.append("")
        lines.append("  Thank you!")

        # Render to image
        font = cv2.FONT_HERSHEY_SIMPLEX
        fs = 0.5
        thick = 1
        line_h = 22
        pad = 20
        w = 400
        h = pad * 2 + line_h * len(lines)

        img = np.ones((h, w, 3), dtype=np.uint8) * 245  # off-white
        # Add slight noise
        noise = np.random.RandomState(i).randint(0, 12, img.shape, dtype=np.uint8)
        img = np.clip(img.astype(np.int16) - noise, 0, 255).astype(np.uint8)

        y = pad + line_h
        for line in lines:
            cv2.putText(img, line, (pad, y), font, fs, (10, 10, 10), thick)
            y += line_h

        cv2.imwrite(str(out_dir / f"receipt_{i:04d}.png"), img)

    log.info("[%s] Generated %d synthetic receipts in %s", PROJECT_KEY, n, out_dir)
