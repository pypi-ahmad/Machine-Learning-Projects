"""Dataset bootstrap for ID Card KYC Parser.

Downloads and prepares a public ID card / document OCR dataset
for testing and benchmarking the extraction pipeline.  Falls back
to synthetic ID card images when the download is unavailable.

Usage::

    from data_bootstrap import ensure_idcard_dataset

    data_root = ensure_idcard_dataset()            # idempotent
    data_root = ensure_idcard_dataset(force=True)  # force re-download
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

log = logging.getLogger("id_card_kyc.data_bootstrap")

PROJECT_KEY = "id_card_kyc_parser"
DATA_ROOT = REPO_ROOT / "data" / PROJECT_KEY


def ensure_idcard_dataset(*, force: bool = False) -> Path:
    """Download and prepare the ID card dataset (idempotent).

    Falls back to synthetic ID card images when download fails.
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
            log.info("[%s] Collected %d card images", PROJECT_KEY, count)
            ready_marker.write_text(time.strftime("%Y-%m-%dT%H:%M:%S"), encoding="utf-8")
            return DATA_ROOT
    except Exception as exc:
        log.warning("[%s] Download failed (%s), generating synthetic ID cards", PROJECT_KEY, exc)

    # Fallback: generate synthetic ID card images
    _generate_synthetic_cards(images_dir, n=20)
    ready_marker.write_text(time.strftime("%Y-%m-%dT%H:%M:%S"), encoding="utf-8")
    log.info("[%s] Synthetic dataset ready at %s", PROJECT_KEY, DATA_ROOT)
    return DATA_ROOT


# ---------------------------------------------------------------------------
# Synthetic ID card generator
# ---------------------------------------------------------------------------

_NAMES = [
    "JOHN SMITH", "MARIA GARCIA", "DAVID CHEN", "SARAH JOHNSON",
    "MICHAEL BROWN", "EMILY DAVIS", "JAMES WILSON", "LISA ANDERSON",
    "ROBERT TAYLOR", "JENNIFER MARTINEZ", "WILLIAM LEE", "AMANDA THOMAS",
    "CHRISTOPHER HARRIS", "JESSICA ROBINSON", "DANIEL CLARK",
]

_NATIONALITIES = [
    "USA", "GBR", "CAN", "AUS", "FRA", "DEU", "ESP", "ITA",
    "JPN", "BRA", "MEX", "NLD", "SWE", "NOR", "CHE",
]

_GENDERS = ["M", "F"]

_DOC_TYPES = ["NATIONAL ID", "DRIVER LICENSE", "RESIDENCE PERMIT"]


def _generate_synthetic_cards(out_dir: Path, n: int = 20) -> None:
    """Render synthetic ID card images with known KYC fields."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(42)

    for i in range(n):
        name = rng.choice(_NAMES)
        nationality = rng.choice(_NATIONALITIES)
        gender = rng.choice(_GENDERS)
        doc_type = rng.choice(_DOC_TYPES)
        dob_y = rng.randint(1960, 2000)
        dob = f"{rng.randint(1,28):02d}/{rng.randint(1,12):02d}/{dob_y}"
        issue_y = rng.randint(2020, 2024)
        issue_date = f"{rng.randint(1,28):02d}/{rng.randint(1,12):02d}/{issue_y}"
        expiry_date = f"{rng.randint(1,28):02d}/{rng.randint(1,12):02d}/{issue_y + 10}"
        id_num = f"{nationality[0]}{rng.randint(100000000, 999999999)}"

        # ISO ID-1 proportions (856x540)
        w, h = 856, 540
        img = np.ones((h, w, 3), dtype=np.uint8) * 240

        # Noise for realism
        noise = np.random.RandomState(i).randint(0, 10, img.shape, dtype=np.uint8)
        img = np.clip(img.astype(np.int16) - noise, 0, 255).astype(np.uint8)

        # Header bar
        header_colour = (
            rng.randint(40, 100),
            rng.randint(40, 100),
            rng.randint(120, 200),
        )
        cv2.rectangle(img, (0, 0), (w, 65), header_colour, -1)

        font = cv2.FONT_HERSHEY_SIMPLEX
        white = (255, 255, 255)
        dark = (20, 20, 20)
        grey = (90, 90, 90)

        # Header text
        cv2.putText(img, doc_type, (20, 45), font, 0.8, white, 2)

        # Photo placeholder
        cv2.rectangle(img, (30, 90), (210, 320), (180, 180, 180), -1)
        cv2.putText(img, "PHOTO", (80, 215), font, 0.6, grey, 1)

        # Field labels and values
        x_label = 240
        x_val = 240
        y = 120

        fields = [
            ("Full Name:", name),
            ("Date of Birth:", dob),
            ("Gender:", gender),
            ("Nationality:", nationality),
            ("ID Number:", id_num),
            ("Issue Date:", issue_date),
            ("Expiry Date:", expiry_date),
        ]

        for label, value in fields:
            cv2.putText(img, label, (x_label, y), font, 0.45, grey, 1)
            cv2.putText(img, value, (x_val + 150, y), font, 0.50, dark, 1)
            y += 35

        # Footer
        cv2.line(img, (20, h - 60), (w - 20, h - 60), grey, 1)
        cv2.putText(img, f"Card No: {id_num}", (20, h - 30), font, 0.40, grey, 1)

        cv2.imwrite(str(out_dir / f"idcard_{i:04d}.png"), img)

    log.info("[%s] Generated %d synthetic ID cards in %s", PROJECT_KEY, n, out_dir)
