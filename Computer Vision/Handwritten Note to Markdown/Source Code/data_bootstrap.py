"""Dataset bootstrap for Handwritten Note to Markdown.

Downloads and prepares a public handwriting recognition dataset
for testing and benchmarking the TrOCR pipeline.  Falls back to
synthetic handwritten-style images when the download is unavailable.

Usage::

    from data_bootstrap import ensure_handwriting_dataset

    data_root = ensure_handwriting_dataset()              # idempotent
    data_root = ensure_handwriting_dataset(force=True)    # force re-download
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

log = logging.getLogger("handwritten_note.data_bootstrap")

PROJECT_KEY = "handwritten_note_to_markdown"
DATA_ROOT = REPO_ROOT / "data" / PROJECT_KEY


def ensure_handwriting_dataset(*, force: bool = False) -> Path:
    """Download and prepare the handwriting dataset (idempotent).

    Falls back to synthetic handwriting images when download fails.
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
            log.info("[%s] Collected %d handwriting images", PROJECT_KEY, count)
            ready_marker.write_text(time.strftime("%Y-%m-%dT%H:%M:%S"), encoding="utf-8")
            return DATA_ROOT
    except Exception as exc:
        log.warning(
            "[%s] Download failed (%s), generating synthetic handwriting images",
            PROJECT_KEY, exc,
        )

    # Fallback: generate synthetic handwriting-style images
    _generate_synthetic_notes(images_dir, n=20)
    ready_marker.write_text(time.strftime("%Y-%m-%dT%H:%M:%S"), encoding="utf-8")
    log.info("[%s] Synthetic dataset ready at %s", PROJECT_KEY, DATA_ROOT)
    return DATA_ROOT


# ---------------------------------------------------------------------------
# Synthetic handwritten note generator
# ---------------------------------------------------------------------------

_SENTENCES = [
    "The quick brown fox jumps over the lazy dog",
    "Meeting notes from Monday morning standup",
    "Remember to buy milk eggs and bread",
    "Call dentist office to reschedule appointment",
    "Project deadline is next Friday at five pm",
    "Ideas for the new product launch campaign",
    "Review the quarterly financial report today",
    "Schedule lunch with the marketing team",
    "Pick up dry cleaning before six pm",
    "Brainstorm session for website redesign",
    "Read chapter seven of the textbook tonight",
    "Submit expense reports by end of month",
    "Water the plants in the office kitchen",
    "Update resume and cover letter draft",
    "Practice piano scales for thirty minutes",
    "Fix the leaking faucet in the bathroom",
    "Prepare presentation slides for Thursday",
    "Send thank you cards to wedding guests",
    "Check train schedule for weekend trip",
    "Organize desk drawers and file cabinet",
    "Write feedback for team performance review",
    "Confirm hotel reservation for conference",
    "Draft agenda for the board meeting",
    "Review pull request from the dev team",
    "Take dog to vet for annual checkup",
]

_TITLES = [
    "TODO LIST", "Meeting Notes", "Shopping List", "Daily Journal",
    "Project Ideas", "Action Items", "Reminders", "Quick Notes",
    "Weekly Goals", "Study Notes",
]


def _generate_synthetic_notes(out_dir: Path, n: int = 20) -> None:
    """Render synthetic handwriting-style note images."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(42)

    # Use FONT_HERSHEY_SCRIPT fonts to approximate handwriting
    hand_fonts = [
        cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
        cv2.FONT_HERSHEY_SCRIPT_COMPLEX,
    ]

    for i in range(n):
        title = rng.choice(_TITLES)
        num_lines = rng.randint(4, 8)
        lines = rng.sample(_SENTENCES, min(num_lines, len(_SENTENCES)))

        w, h = 800, 120 + num_lines * 55
        # Slightly off-white with variation
        bg_val = rng.randint(235, 250)
        img = np.ones((h, w, 3), dtype=np.uint8) * bg_val

        # Add subtle paper texture noise
        noise = np.random.RandomState(i).randint(0, 8, img.shape, dtype=np.uint8)
        img = np.clip(img.astype(np.int16) - noise, 0, 255).astype(np.uint8)

        # Optional faint ruled lines
        if rng.random() > 0.3:
            for ly in range(90, h, 55):
                cv2.line(img, (30, ly), (w - 30, ly), (210, 210, 230), 1)

        font = rng.choice(hand_fonts)
        # Dark ink colour with slight variation
        ink = (
            rng.randint(10, 40),
            rng.randint(10, 40),
            rng.randint(60, 100),
        )

        # Title (larger, bolder)
        cv2.putText(img, title, (40, 55), font, 0.8, ink, 2)

        # Lines
        y = 120
        for line in lines:
            # Slight random x-offset and y-jitter for handwriting feel
            x_off = rng.randint(35, 55)
            y_jitter = rng.randint(-3, 3)
            fs = rng.uniform(0.48, 0.58)
            cv2.putText(img, line, (x_off, y + y_jitter), font, fs, ink, 1)
            y += 55

        cv2.imwrite(str(out_dir / f"note_{i:04d}.png"), img)

    log.info("[%s] Generated %d synthetic notes in %s", PROJECT_KEY, n, out_dir)
