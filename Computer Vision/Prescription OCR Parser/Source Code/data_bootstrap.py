"""Dataset bootstrap for Prescription OCR Parser.

Downloads and prepares a public medical document dataset for v1
evaluation and falls back to synthetic prescription images when the
public source is unavailable.

Usage::

    from data_bootstrap import ensure_prescription_dataset

    data_root = ensure_prescription_dataset()              # idempotent
    data_root = ensure_prescription_dataset(force=True)    # force re-download
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

log = logging.getLogger("prescription_ocr.data_bootstrap")

PROJECT_KEY = "prescription_ocr_parser"
DATA_ROOT = REPO_ROOT / "data" / PROJECT_KEY


def ensure_prescription_dataset(*, force: bool = False) -> Path:
    """Download and prepare the prescription dataset (idempotent).

    Attempts the configured public source first, then generates a
    synthetic prescription set so the parser can still be validated.
    """
    images_dir = DATA_ROOT / "images"
    ready_marker = DATA_ROOT / ".ready"

    if ready_marker.exists() and not force:
        log.info("[%s] Dataset ready at %s -- skipping", PROJECT_KEY, DATA_ROOT)
        return DATA_ROOT

    try:
        from scripts.download_data import ensure_dataset as _ensure

        data_path = _ensure(PROJECT_KEY, force=force)
        images_dir.mkdir(parents=True, exist_ok=True)

        count = 0
        for img in data_path.rglob("*"):
            if img.suffix.lower() in IMAGE_EXTS and img.parent != images_dir:
                dst = images_dir / img.name
                if not dst.exists():
                    shutil.copy2(str(img), str(dst))
                    count += 1

        if count > 0:
            _write_dataset_info(
                source="nielsr/funsd",
                license_note="See Hugging Face dataset page for current license terms.",
                fallback_used=False,
            )
            ready_marker.write_text(time.strftime("%Y-%m-%dT%H:%M:%S"), encoding="utf-8")
            log.info("[%s] Collected %d public document images", PROJECT_KEY, count)
            return DATA_ROOT
    except Exception as exc:
        log.warning(
            "[%s] Download failed (%s); generating synthetic prescription images",
            PROJECT_KEY,
            exc,
        )

    _generate_synthetic_prescriptions(images_dir, n=12)
    _write_dataset_info(
        source="synthetic-prescription-v1",
        license_note="Generated locally for testing; no clinical content.",
        fallback_used=True,
    )
    ready_marker.write_text(time.strftime("%Y-%m-%dT%H:%M:%S"), encoding="utf-8")
    log.info("[%s] Synthetic dataset ready at %s", PROJECT_KEY, DATA_ROOT)
    return DATA_ROOT


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

_MEDICINES = [
    ("Amoxicillin", "500 mg", "Twice daily", "For 7 days", "Oral", "Take after meals"),
    ("Paracetamol", "650 mg", "Every 8 hours", "For 3 days", "Oral", "Take if fever persists"),
    ("Cetirizine", "10 mg", "Once daily", "For 5 days", "Oral", "Take at bedtime"),
    ("Azithromycin", "250 mg", "Once daily", "For 3 days", "Oral", "Finish the full course"),
    ("Ibuprofen", "400 mg", "Thrice daily", "For 4 days", "Oral", "Take with food"),
    ("Pantoprazole", "40 mg", "Once daily", "For 10 days", "Oral", "Take before breakfast"),
]

_PATIENTS = [
    "Riya Sharma", "Daniel Brooks", "Mina Patel", "Harish Rao", "Sara Khan", "Liam Carter",
]

_DOCTORS = [
    "Dr. Meera Shah", "Dr. Alan Joseph", "Dr. Priya Menon", "Dr. David Cole",
]


def _generate_synthetic_prescriptions(out_dir: Path, n: int = 12) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(42)

    for index in range(n):
        width = 1200
        height = 1500
        paper = np.full((height, width, 3), 248, dtype=np.uint8)
        noise = np.random.RandomState(index).randint(0, 6, paper.shape, dtype=np.uint8)
        image = np.clip(paper.astype(np.int16) - noise, 0, 255).astype(np.uint8)

        doctor = rng.choice(_DOCTORS)
        patient = rng.choice(_PATIENTS)
        date = f"{rng.randint(1,28):02d}/{rng.randint(1,12):02d}/2026"
        meds = rng.sample(_MEDICINES, 3)

        cv2.putText(image, doctor, (70, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (25, 25, 25), 2)
        cv2.putText(image, "Prescription", (420, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (25, 25, 25), 2)
        cv2.line(image, (60, 115), (1140, 115), (80, 80, 80), 2)
        cv2.putText(image, f"Patient: {patient}", (70, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (35, 35, 35), 2)
        cv2.putText(image, f"Date: {date}", (820, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (35, 35, 35), 2)

        y = 260
        for med_name, dosage, frequency, duration, route, instruction in meds:
            cv2.putText(image, med_name, (110, y), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (20, 20, 20), 2)
            cv2.putText(image, dosage, (520, y), cv2.FONT_HERSHEY_SIMPLEX, 0.82, (20, 20, 20), 2)
            cv2.putText(image, frequency, (720, y), cv2.FONT_HERSHEY_SIMPLEX, 0.74, (20, 20, 20), 2)
            cv2.putText(image, duration, (110, y + 48), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (35, 35, 35), 2)
            cv2.putText(image, route, (470, y + 48), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (35, 35, 35), 2)
            cv2.putText(image, instruction, (650, y + 48), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (35, 35, 35), 2)
            cv2.line(image, (80, y + 78), (1120, y + 78), (210, 210, 210), 1)
            y += 145

        cv2.putText(
            image,
            "Informational sample only - verify with a licensed clinician.",
            (70, height - 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (60, 60, 160),
            2,
        )
        cv2.imwrite(str(out_dir / f"prescription_{index:04d}.png"), image)

    log.info("[%s] Generated %d synthetic prescriptions in %s", PROJECT_KEY, n, out_dir)


def _write_dataset_info(*, source: str, license_note: str, fallback_used: bool) -> None:
    info_path = DATA_ROOT / "dataset_info.json"
    payload = {
        "dataset_key": PROJECT_KEY,
        "source": source,
        "license_note": license_note,
        "fallback_used": fallback_used,
        "prepared_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    info_path.write_text(__import__("json").dumps(payload, indent=2), encoding="utf-8")
