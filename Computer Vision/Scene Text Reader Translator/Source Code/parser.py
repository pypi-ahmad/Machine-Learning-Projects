"""High-level pipeline for Scene Text Reader Translator.

Orchestrates PaddleOCR → optional translation → structured result.

Usage::

    from parser import SceneTextPipeline

    pipeline = SceneTextPipeline(cfg)
    result = pipeline.process(image)
    result, blocks = pipeline.process_with_blocks(image)
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import SceneTextConfig
from ocr_engine import OCRBlock, OCREngine
from translator import Translator


@dataclass
class TextRead:
    """A single text region read from the scene."""

    text: str
    translated_text: str
    confidence: float
    bbox: list[list[int]]       # [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
    centre: tuple[int, int]


@dataclass
class SceneTextResult:
    """Result from processing a single image/frame."""

    reads: list[TextRead] = field(default_factory=list)
    raw_text: str = ""
    num_blocks: int = 0
    mean_confidence: float = 0.0
    frame_index: int = 0
    translation_enabled: bool = False
    translation_provider: str = ""


class SceneTextPipeline:
    """Full scene text pipeline: OCR → translate → structure."""

    def __init__(self, cfg: SceneTextConfig) -> None:
        self.cfg = cfg
        self._engine = OCREngine(cfg)
        self._translator = Translator(cfg)
        self._frame_count = 0

    def process(self, image: np.ndarray) -> SceneTextResult:
        """Process a single image through the full pipeline."""
        blocks = self._engine.run(image)
        return self._build_result(blocks)

    def process_with_blocks(
        self, image: np.ndarray,
    ) -> tuple[SceneTextResult, list[OCRBlock]]:
        """Process and also return raw OCR blocks for overlay."""
        blocks = self._engine.run(image)
        result = self._build_result(blocks)
        return result, blocks

    def _build_result(self, blocks: list[OCRBlock]) -> SceneTextResult:
        self._frame_count += 1

        if not blocks:
            return SceneTextResult(frame_index=self._frame_count)

        reads: list[TextRead] = []
        for blk in blocks:
            translated = self._translator.translate(blk.text)
            reads.append(TextRead(
                text=blk.text,
                translated_text=translated,
                confidence=blk.confidence,
                bbox=blk.bbox,
                centre=blk.centre,
            ))

        confs = [b.confidence for b in blocks]
        mean_conf = float(np.mean(confs)) if confs else 0.0
        raw_text = self._engine.full_text(blocks)

        return SceneTextResult(
            reads=reads,
            raw_text=raw_text,
            num_blocks=len(reads),
            mean_confidence=mean_conf,
            frame_index=self._frame_count,
            translation_enabled=self._translator.is_enabled,
            translation_provider=self._translator.provider_name,
        )
