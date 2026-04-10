"""High-level parser for ID Card KYC Parser.

Orchestrates template selection and field extraction from OCR
blocks.  The parser itself is template-agnostic — it delegates
to the appropriate :class:`templates.IDTemplate`.

Usage::

    from parser import IDCardParser
    from ocr_engine import OCRBlock

    parser = IDCardParser(template="generic")
    result = parser.parse(blocks)
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ocr_engine import OCRBlock
from templates import ExtractedField, get_template


@dataclass
class ParseResult:
    """Complete extraction result."""

    fields: dict[str, ExtractedField] = field(default_factory=dict)
    template_used: str = ""
    raw_text: str = ""
    num_blocks: int = 0


class IDCardParser:
    """Parse OCR blocks into structured ID card fields."""

    def __init__(self, template: str = "generic") -> None:
        self._template = get_template(template)

    @property
    def template_name(self) -> str:
        return self._template.name

    def parse(self, blocks: list[OCRBlock]) -> ParseResult:
        if not blocks:
            return ParseResult(template_used=self._template.name)

        full_text = "\n".join(b.text for b in blocks)
        fields = self._template.extract(blocks)

        return ParseResult(
            fields=fields,
            template_used=self._template.name,
            raw_text=full_text,
            num_blocks=len(blocks),
        )
