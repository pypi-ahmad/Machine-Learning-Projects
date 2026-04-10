"""Markdown formatter for recognised handwritten text.

Converts a list of recognised lines (with geometry metadata) into
structured Markdown, optionally detecting headers, list items,
and paragraph breaks.

Usage::

    from markdown_formatter import MarkdownFormatter
    from config import NoteConfig

    fmt = MarkdownFormatter(NoteConfig())
    md_text = fmt.format(recognised_lines)
"""

from __future__ import annotations

import logging
import statistics
from dataclasses import dataclass

log = logging.getLogger("handwritten_note.formatter")


@dataclass
class RecognisedLine:
    """A single recognised text line with layout metadata."""

    text: str
    confidence: float
    y_start: int
    y_end: int
    height: int
    x_offset: int
    line_index: int


class MarkdownFormatter:
    """Convert recognised lines into Markdown text."""

    def __init__(self, cfg) -> None:
        self.cfg = cfg

    def format(self, lines: list[RecognisedLine]) -> str:
        """Produce Markdown from *lines*."""
        if not lines:
            return ""

        median_height = self._median_height(lines)
        median_gap = self._median_gap(lines)

        md_parts: list[str] = []

        for i, line in enumerate(lines):
            text = line.text.strip()
            if not text:
                continue

            # Detect header
            if self.cfg.detect_headers and self._is_header(line, median_height):
                text = f"## {text}"

            # Detect list item
            elif self.cfg.detect_lists and self._is_list_item(line):
                # Preserve existing bullet/number or add one
                if not text.startswith(("-", "*", "•")) and not self._starts_with_number(text):
                    text = f"- {text}"

            # Paragraph break (extra blank line before this line)
            if i > 0 and median_gap > 0:
                gap = line.y_start - lines[i - 1].y_end
                if gap >= median_gap * self.cfg.paragraph_gap_ratio:
                    md_parts.append("")  # blank line → new paragraph

            md_parts.append(text)

        return "\n".join(md_parts) + "\n"

    def format_plain(self, lines: list[RecognisedLine]) -> str:
        """Produce plain text (no Markdown decoration)."""
        if not lines:
            return ""

        median_gap = self._median_gap(lines)
        parts: list[str] = []

        for i, line in enumerate(lines):
            text = line.text.strip()
            if not text:
                continue
            if i > 0 and median_gap > 0:
                gap = line.y_start - lines[i - 1].y_end
                if gap >= median_gap * self.cfg.paragraph_gap_ratio:
                    parts.append("")
            parts.append(text)

        return "\n".join(parts) + "\n"

    def format_with_confidence(self, lines: list[RecognisedLine]) -> str:
        """Markdown with inline confidence annotations for low-confidence lines."""
        if not lines:
            return ""

        median_height = self._median_height(lines)
        median_gap = self._median_gap(lines)
        threshold = self.cfg.confidence_threshold

        md_parts: list[str] = []

        for i, line in enumerate(lines):
            text = line.text.strip()
            if not text:
                continue

            # Paragraph break
            if i > 0 and median_gap > 0:
                gap = line.y_start - lines[i - 1].y_end
                if gap >= median_gap * self.cfg.paragraph_gap_ratio:
                    md_parts.append("")

            # Header
            if self.cfg.detect_headers and self._is_header(line, median_height):
                text = f"## {text}"
            elif self.cfg.detect_lists and self._is_list_item(line):
                if not text.startswith(("-", "*", "•")) and not self._starts_with_number(text):
                    text = f"- {text}"

            # Confidence annotation
            if line.confidence < threshold:
                text = f"{text}  <!-- low confidence: {line.confidence:.2f} -->"

            md_parts.append(text)

        return "\n".join(md_parts) + "\n"

    # ------------------------------------------------------------------
    # Internal heuristics
    # ------------------------------------------------------------------

    def _is_header(self, line: RecognisedLine, median_height: float) -> bool:
        if median_height <= 0:
            return False
        return line.height >= median_height * self.cfg.header_height_ratio

    def _is_list_item(self, line: RecognisedLine) -> bool:
        return line.x_offset >= self.cfg.list_indent_px

    @staticmethod
    def _starts_with_number(text: str) -> bool:
        parts = text.split(".", 1)
        return len(parts) > 1 and parts[0].strip().isdigit()

    @staticmethod
    def _median_height(lines: list[RecognisedLine]) -> float:
        heights = [ln.height for ln in lines if ln.height > 0]
        return statistics.median(heights) if heights else 0.0

    @staticmethod
    def _median_gap(lines: list[RecognisedLine]) -> float:
        if len(lines) < 2:
            return 0.0
        gaps = []
        for i in range(1, len(lines)):
            gap = lines[i].y_start - lines[i - 1].y_end
            if gap > 0:
                gaps.append(gap)
        return statistics.median(gaps) if gaps else 0.0
