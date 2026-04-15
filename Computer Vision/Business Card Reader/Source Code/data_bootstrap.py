"""Dataset bootstrap for Business Card Reader.

Downloads and prepares a public business card OCR dataset for
testing and benchmarking the extraction pipeline.  Falls back to
synthetic business card images when the download is unavailable.

Usage::

    from data_bootstrap import ensure_card_dataset

    data_root = ensure_card_dataset()            # idempotent
    data_root = ensure_card_dataset(force=True)  # force re-download
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

log = logging.getLogger("business_card.data_bootstrap")

PROJECT_KEY = "business_card_reader"
DATA_ROOT = REPO_ROOT / "data" / PROJECT_KEY


def ensure_card_dataset(*, force: bool = False) -> Path:
    """Download and prepare the business card dataset (idempotent).

    Falls back to synthetic business card images when download fails.
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
        log.warning("[%s] Download failed (%s), generating synthetic cards", PROJECT_KEY, exc)

    # Fallback: generate synthetic business card images
    _generate_synthetic_cards(images_dir, n=20)
    ready_marker.write_text(time.strftime("%Y-%m-%dT%H:%M:%S"), encoding="utf-8")
    log.info("[%s] Synthetic dataset ready at %s", PROJECT_KEY, DATA_ROOT)
    return DATA_ROOT


# ---------------------------------------------------------------------------
# Synthetic business card generator
# ---------------------------------------------------------------------------

_NAMES = [
    "John Smith", "Maria Garcia", "David Chen", "Sarah Johnson",
    "Michael Brown", "Emily Davis", "James Wilson", "Lisa Anderson",
    "Robert Taylor", "Jennifer Martinez", "William Lee", "Amanda Thomas",
    "Christopher Harris", "Jessica Robinson", "Daniel Clark",
]

_TITLES = [
    "Software Engineer", "Marketing Director", "CEO", "Product Manager",
    "Sales Representative", "CTO", "UX Designer", "Data Analyst",
    "VP of Operations", "Senior Consultant", "Account Executive",
    "Creative Director", "HR Manager", "Lead Developer", "CFO",
]

_COMPANIES = [
    "Acme Corp.", "TechVision Inc.", "GlobalSync Solutions",
    "Pinnacle Systems Ltd.", "NexGen Technologies", "BlueStar Consulting",
    "Summit Partners LLC", "DataFlow Analytics", "CloudBridge Co.",
    "Vertex Industries", "Horizon Group", "Catalyst Labs",
    "Sterling Associates", "Quantum Enterprises", "Atlas International",
]

_DOMAINS = [
    "acmecorp.com", "techvision.io", "globalsync.com", "pinnaclesys.com",
    "nexgentech.com", "bluestar.co", "summitpartners.com", "dataflow.ai",
    "cloudbridge.co", "vertexind.com", "horizongrp.com", "catalystlabs.io",
    "sterlingassoc.com", "quantument.com", "atlasinternational.com",
]

_STREETS = [
    "123 Main Street", "456 Oak Avenue", "789 Pine Road",
    "321 Elm Boulevard", "654 Maple Drive", "987 Cedar Lane",
    "111 First Street", "222 Park Avenue", "333 Broadway",
    "444 Market Street",
]

_CITIES = [
    "New York, NY 10001", "San Francisco, CA 94102", "Chicago, IL 60601",
    "Austin, TX 78701", "Seattle, WA 98101", "Boston, MA 02101",
    "Denver, CO 80201", "Atlanta, GA 30301", "Miami, FL 33101",
    "Portland, OR 97201",
]


def _generate_synthetic_cards(out_dir: Path, n: int = 20) -> None:
    """Render synthetic business card images with known contact fields."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(42)

    for i in range(n):
        name = rng.choice(_NAMES)
        title = rng.choice(_TITLES)
        company = rng.choice(_COMPANIES)
        idx = rng.randint(0, len(_DOMAINS) - 1)
        domain = _DOMAINS[idx]
        first = name.split()[0].lower()
        last = name.split()[-1].lower()
        email = f"{first}.{last}@{domain}"
        phone = f"+1 ({rng.randint(200,999)}) {rng.randint(100,999)}-{rng.randint(1000,9999)}"
        website = f"www.{domain}"
        street = rng.choice(_STREETS)
        city = rng.choice(_CITIES)

        # Card dimensions (standard business card aspect ratio ~3.5:2)
        w, h = 700, 400
        img = np.ones((h, w, 3), dtype=np.uint8) * 250  # off-white

        # Slight noise for realism
        noise = np.random.RandomState(i).randint(0, 8, img.shape, dtype=np.uint8)
        img = np.clip(img.astype(np.int16) - noise, 0, 255).astype(np.uint8)

        # Add a subtle accent line
        accent = (rng.randint(40, 120), rng.randint(40, 120), rng.randint(100, 200))
        cv2.line(img, (30, 75), (w - 30, 75), accent, 2)

        font = cv2.FONT_HERSHEY_SIMPLEX
        dark = (20, 20, 20)
        grey = (80, 80, 80)

        # Company (top)
        cv2.putText(img, company, (30, 55), font, 0.65, dark, 2)

        # Name (prominent)
        cv2.putText(img, name, (30, 130), font, 0.85, dark, 2)

        # Title
        cv2.putText(img, title, (30, 165), font, 0.50, grey, 1)

        # Contact info (lower half)
        y = 220
        cv2.putText(img, phone, (30, y), font, 0.45, dark, 1)
        y += 30
        cv2.putText(img, email, (30, y), font, 0.45, dark, 1)
        y += 30
        cv2.putText(img, website, (30, y), font, 0.45, grey, 1)
        y += 35
        cv2.putText(img, street, (30, y), font, 0.40, grey, 1)
        y += 25
        cv2.putText(img, city, (30, y), font, 0.40, grey, 1)

        cv2.imwrite(str(out_dir / f"card_{i:04d}.png"), img)

    log.info("[%s] Generated %d synthetic business cards in %s", PROJECT_KEY, n, out_dir)
