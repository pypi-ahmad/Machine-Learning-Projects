"""Dataset bootstrap for Form OCR Checkbox Extractor.

Downloads and prepares a public form/document dataset for testing
and benchmarking the checkbox + OCR extraction pipeline.  Falls back
to synthetic form images when the download is unavailable.

Usage::

    from data_bootstrap import ensure_form_dataset

    data_root = ensure_form_dataset()              # idempotent
    data_root = ensure_form_dataset(force=True)    # force re-download
"""

from __future__ import annotations

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

log = logging.getLogger("form_checkbox.data_bootstrap")

PROJECT_KEY = "form_ocr_checkbox_extractor"
DATA_ROOT = REPO_ROOT / "data" / PROJECT_KEY


def ensure_form_dataset(*, force: bool = False) -> Path:
    """Download and prepare the form dataset (idempotent).

    Falls back to synthetic form images when download fails.
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

        images_dir.mkdir(parents=True, exist_ok=True)
        count = 0
        for img in data_path.rglob("*"):
            if img.suffix.lower() in {".jpg", ".jpeg", ".png"} and img.parent != images_dir:
                dst = images_dir / img.name
                if not dst.exists():
                    shutil.copy2(str(img), str(dst))
                    count += 1
        if count > 0:
            log.info("[%s] Collected %d form images", PROJECT_KEY, count)
            ready_marker.write_text(time.strftime("%Y-%m-%dT%H:%M:%S"), encoding="utf-8")
            return DATA_ROOT
    except Exception as exc:
        log.warning("[%s] Download failed (%s), generating synthetic forms", PROJECT_KEY, exc)

    # Fallback: generate synthetic form images
    _generate_synthetic_forms(images_dir, n=20)
    ready_marker.write_text(time.strftime("%Y-%m-%dT%H:%M:%S"), encoding="utf-8")
    log.info("[%s] Synthetic dataset ready at %s", PROJECT_KEY, DATA_ROOT)
    return DATA_ROOT


# ---------------------------------------------------------------------------
# Synthetic form generator
# ---------------------------------------------------------------------------

_FORM_TITLES = [
    "APPLICATION FORM", "REGISTRATION FORM", "SURVEY FORM",
    "CONSENT FORM", "FEEDBACK FORM", "ORDER FORM",
    "MEMBERSHIP FORM", "ENROLLMENT FORM", "REQUEST FORM",
    "INTAKE FORM",
]

_NAMES = [
    "John Smith", "Maria Garcia", "David Chen", "Sarah Johnson",
    "Michael Brown", "Emily Davis", "James Wilson", "Lisa Anderson",
    "Robert Taylor", "Jennifer Martinez",
]

_CHECKBOX_LABELS = [
    "I agree to the terms and conditions",
    "Subscribe to newsletter",
    "Receive email notifications",
    "I am over 18 years old",
    "I have read the privacy policy",
    "Opt-in for marketing communications",
    "I consent to data processing",
    "Enable two-factor authentication",
]

_RADIO_GROUPS = [
    ("Gender", ["Male", "Female", "Other"]),
    ("Preferred Contact", ["Email", "Phone", "Mail"]),
    ("Experience Level", ["Beginner", "Intermediate", "Advanced"]),
    ("Payment Method", ["Credit Card", "PayPal", "Bank Transfer"]),
]


def _generate_synthetic_forms(out_dir: Path, n: int = 20) -> None:
    """Render synthetic form images with checkboxes and text fields."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(42)

    for i in range(n):
        title = rng.choice(_FORM_TITLES)
        name = rng.choice(_NAMES)
        email = name.split()[0].lower() + "@example.com"
        phone = f"({rng.randint(200,999)}) {rng.randint(100,999)}-{rng.randint(1000,9999)}"
        date_str = f"{rng.randint(1,12):02d}/{rng.randint(1,28):02d}/2025"

        w, h = 800, 1100
        img = np.ones((h, w, 3), dtype=np.uint8) * 248

        # Subtle noise
        noise = np.random.RandomState(i).randint(0, 6, img.shape, dtype=np.uint8)
        img = np.clip(img.astype(np.int16) - noise, 0, 255).astype(np.uint8)

        font = cv2.FONT_HERSHEY_SIMPLEX
        dark = (20, 20, 20)
        grey = (100, 100, 100)
        line_colour = (180, 180, 180)

        # Title
        cv2.putText(img, title, (30, 50), font, 0.9, dark, 2)
        cv2.line(img, (30, 60), (w - 30, 60), dark, 2)

        y = 110

        # Text fields with labels and underlines
        text_fields = [
            ("Name:", name),
            ("Email:", email),
            ("Phone:", phone),
            ("Date:", date_str),
        ]
        for label, value in text_fields:
            cv2.putText(img, label, (30, y), font, 0.5, dark, 1)
            cv2.putText(img, value, (150, y), font, 0.5, grey, 1)
            cv2.line(img, (150, y + 5), (w - 50, y + 5), line_colour, 1)
            y += 45

        y += 20

        # Checkboxes
        num_cb = rng.randint(2, 4)
        cb_labels = rng.sample(_CHECKBOX_LABELS, num_cb)
        for cb_label in cb_labels:
            checked = rng.random() > 0.5
            # Draw checkbox rectangle
            bx, by = 40, y - 14
            bw, bh = 18, 18
            cv2.rectangle(img, (bx, by), (bx + bw, by + bh), dark, 2)
            if checked:
                # Draw X mark inside
                cv2.line(img, (bx + 3, by + 3), (bx + bw - 3, by + bh - 3), dark, 2)
                cv2.line(img, (bx + bw - 3, by + 3), (bx + 3, by + bh - 3), dark, 2)
            cv2.putText(img, cb_label, (70, y), font, 0.42, dark, 1)
            y += 35

        y += 15

        # Radio button group
        group_name, options = rng.choice(_RADIO_GROUPS)
        cv2.putText(img, f"{group_name}:", (30, y), font, 0.5, dark, 1)
        y += 30
        selected_idx = rng.randint(0, len(options) - 1)
        for j, option in enumerate(options):
            cx, cy = 50, y - 5
            radius = 9
            cv2.circle(img, (cx, cy), radius, dark, 2)
            if j == selected_idx:
                cv2.circle(img, (cx, cy), 5, dark, -1)
            cv2.putText(img, option, (70, y), font, 0.42, dark, 1)
            y += 30

        y += 20

        # Signature line
        cv2.putText(img, "Signature:", (30, y), font, 0.5, dark, 1)
        cv2.line(img, (150, y + 5), (400, y + 5), dark, 1)

        cv2.imwrite(str(out_dir / f"form_{i:04d}.png"), img)

    log.info("[%s] Generated %d synthetic forms in %s", PROJECT_KEY, n, out_dir)
