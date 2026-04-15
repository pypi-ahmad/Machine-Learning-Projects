"""High-level parser for Handwritten Note to Markdown.

Orchestrates line segmentation -> TrOCR recognition -> markdown
formatting into a single :class:`NoteParseResult`.

Usage::

    from parser import NoteParser

    parser = NoteParser(cfg)
    result = parser.parse(image)
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import NoteConfig
from line_segmenter import LineRegion, LineSegmenter
from markdown_formatter import MarkdownFormatter, RecognisedLine
from ocr_engine import RecognitionResult, TrOCREngine


@dataclass
class NoteParseResult:
    """Complete note extraction result."""

    markdown: str = ""
    plain_text: str = ""
    confidence_markdown: str = ""
    lines: list[RecognisedLine] = field(default_factory=list)
    regions: list[LineRegion] = field(default_factory=list)
    recognition_results: list[RecognitionResult] = field(default_factory=list)
    num_lines: int = 0
    mean_confidence: float = 0.0


class NoteParser:
    """Orchestrate segmentation + OCR + formatting."""

    def __init__(self, cfg: NoteConfig) -> None:
        self.cfg = cfg
        self._segmenter = LineSegmenter(cfg)
        self._engine = TrOCREngine(cfg)
        self._formatter = MarkdownFormatter(cfg)

    def parse(self, image: np.ndarray) -> NoteParseResult:
        """Full pipeline: segment -> recognise -> format."""
        # 1. Segment into lines
        regions = self._segmenter.segment(image)

        # 2. Recognise each line
        rec_results: list[RecognitionResult] = []
        recognised_lines: list[RecognisedLine] = []

        for i, region in enumerate(regions):
            rec = self._engine.recognise(region.crop)
            rec_results.append(rec)

            recognised_lines.append(RecognisedLine(
                text=rec.text,
                confidence=rec.confidence,
                y_start=region.y_start,
                y_end=region.y_end,
                height=region.height,
                x_offset=region.x_offset,
                line_index=i,
            ))

        # 3. Format
        markdown = self._formatter.format(recognised_lines)
        plain_text = self._formatter.format_plain(recognised_lines)
        conf_md = self._formatter.format_with_confidence(recognised_lines)

        # 4. Aggregate confidence
        confidences = [r.confidence for r in rec_results if r.text.strip()]
        mean_conf = (
            float(np.mean(confidences)) if confidences else 0.0
        )

        return NoteParseResult(
            markdown=markdown,
            plain_text=plain_text,
            confidence_markdown=conf_md,
            lines=recognised_lines,
            regions=regions,
            recognition_results=rec_results,
            num_lines=len(recognised_lines),
            mean_confidence=mean_conf,
        )
