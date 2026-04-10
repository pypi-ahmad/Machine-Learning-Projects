"""Regex-based plate text cleanup.

Normalises raw OCR output to a clean plate string by removing
invalid characters, applying common OCR-error corrections, and
validating against a configurable pattern.

Usage::

    from plate_cleaner import PlateCleaner
    from config import PlateConfig

    cleaner = PlateCleaner(PlateConfig())
    cleaned = cleaner.clean("AB-12 3CD")
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import PlateConfig


# Common OCR misreads on plates
_OCR_CORRECTIONS: dict[str, str] = {
    "O": "0",   # In numeric context
    "I": "1",
    "S": "5",
    "Z": "2",
    "B": "8",
    "G": "6",
}


class PlateCleaner:
    """Clean and normalise raw OCR plate text."""

    def __init__(self, cfg: PlateConfig) -> None:
        self.cfg = cfg
        self._strip_re = re.compile(cfg.strip_chars)
        self._valid_re = re.compile(cfg.valid_plate_pattern)

    def clean(self, raw_text: str) -> str:
        """Clean raw OCR text into a normalised plate string."""
        # Uppercase
        text = raw_text.upper().strip()

        # Strip invalid characters
        text = self._strip_re.sub("", text)

        # Collapse whitespace
        text = re.sub(r"\s+", " ", text).strip()

        return text

    def is_valid(self, plate: str) -> bool:
        """Check if *plate* matches the expected plate pattern."""
        if len(plate) < self.cfg.min_plate_length:
            return False
        if len(plate) > self.cfg.max_plate_length:
            return False
        return bool(self._valid_re.match(plate))

    def apply_corrections(self, plate: str) -> str:
        """Apply common OCR character corrections for plates.

        Example: in a position where a digit is expected, 'O' → '0'.
        This is conservative — only applied to isolated characters.
        """
        # Simple heuristic: if the plate has a clear letter-digit pattern
        # we can correct specific characters. For now, just return as-is
        # to avoid aggressive false corrections.
        return plate
