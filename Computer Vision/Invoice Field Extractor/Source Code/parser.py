"""Field extraction from OCR output for Invoice Field Extractor.

Uses regex patterns and heuristics to extract structured fields
from OCR text blocks.  Each extracted field carries a confidence
score derived from the underlying OCR detection confidence.

Usage::

    from parser import InvoiceParser
    from ocr_engine import OCREngine
    from config import InvoiceConfig

    engine = OCREngine(InvoiceConfig())
    blocks = engine.run(image)
    result = InvoiceParser().parse(blocks)
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ocr_engine import OCRBlock


# ------------------------------------------------------------------
# Result data structures
# ------------------------------------------------------------------

@dataclass
class ExtractedField:
    """Single extracted field with provenance."""

    name: str
    value: str
    confidence: float
    source_text: str        # raw OCR text that matched
    source_block_idx: int   # index into the blocks list


@dataclass
class LineItem:
    """Single line item row."""

    description: str
    quantity: str | None = None
    unit_price: str | None = None
    amount: str | None = None
    confidence: float = 0.0


@dataclass
class ParseResult:
    """Complete extraction result."""

    fields: dict[str, ExtractedField] = field(default_factory=dict)
    line_items: list[LineItem] = field(default_factory=list)
    raw_text: str = ""
    num_blocks: int = 0


# ------------------------------------------------------------------
# Patterns
# ------------------------------------------------------------------

_INVOICE_NUM_PAT = re.compile(
    r"(?:invoice|inv|bill)\s*(?:#|no\.?|number)?\s*[:\-]?\s*"
    r"([A-Z0-9][\w\-/]{2,})",
    re.IGNORECASE,
)

_DATE_PAT = re.compile(
    r"(?:date|dated|invoice\s*date|issue\s*date|inv\.\s*date)"
    r"\s*[:\-]?\s*"
    r"(\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4}|\w+\s+\d{1,2},?\s+\d{4})",
    re.IGNORECASE,
)

_DUE_DATE_PAT = re.compile(
    r"(?:due\s*date|payment\s*due)\s*[:\-]?\s*"
    r"(\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4}|\w+\s+\d{1,2},?\s+\d{4})",
    re.IGNORECASE,
)

_TOTAL_PAT = re.compile(
    r"(?:total|grand\s*total|amount\s*due|balance\s*due)"
    r"\s*[:\-]?\s*[\$€£¥]?\s*([\d,]+\.?\d*)",
    re.IGNORECASE,
)

_SUBTOTAL_PAT = re.compile(
    r"(?:subtotal|sub\s*total|sub-total)"
    r"\s*[:\-]?\s*[\$€£¥]?\s*([\d,]+\.?\d*)",
    re.IGNORECASE,
)

_TAX_PAT = re.compile(
    r"(?:tax|vat|gst|hst)\s*[:\-]?\s*[\$€£¥]?\s*([\d,]+\.?\d*)",
    re.IGNORECASE,
)

_BILL_TO_PAT = re.compile(
    r"(?:bill\s*to|billed\s*to|sold\s*to|customer)\s*[:\-]?\s*(.+)",
    re.IGNORECASE,
)

_CURRENCY_SYMBOLS: dict[str, str] = {
    "$": "USD",
    "€": "EUR",
    "£": "GBP",
    "¥": "JPY",
}


# ------------------------------------------------------------------
# Parser
# ------------------------------------------------------------------

class InvoiceParser:
    """Extract structured fields from OCR blocks."""

    def parse(self, blocks: list[OCRBlock]) -> ParseResult:
        """Parse *blocks* and return an :class:`ParseResult`."""
        if not blocks:
            return ParseResult()

        full_text = "\n".join(b.text for b in blocks)
        result = ParseResult(raw_text=full_text, num_blocks=len(blocks))

        # Join every block text with index for source tracking
        indexed = [(i, b) for i, b in enumerate(blocks)]

        self._extract_invoice_number(indexed, result)
        self._extract_dates(indexed, result)
        self._extract_totals(indexed, result)
        self._extract_currency(indexed, result)
        self._extract_vendor(indexed, result)
        self._extract_bill_to(indexed, result)
        self._extract_line_items(indexed, result)

        return result

    # -- helpers -------------------------------------------------------

    def _match_field(
        self,
        name: str,
        pattern: re.Pattern[str],
        indexed: list[tuple[int, OCRBlock]],
        result: ParseResult,
    ) -> None:
        """Search blocks for *pattern* and store the first match."""
        for idx, blk in indexed:
            m = pattern.search(blk.text)
            if m:
                result.fields[name] = ExtractedField(
                    name=name,
                    value=m.group(1).strip(),
                    confidence=blk.confidence,
                    source_text=blk.text,
                    source_block_idx=idx,
                )
                return
        # Fall back to full text scan (multi-line matches)
        full = result.raw_text
        m = pattern.search(full)
        if m:
            result.fields[name] = ExtractedField(
                name=name,
                value=m.group(1).strip(),
                confidence=0.5,
                source_text=m.group(0),
                source_block_idx=-1,
            )

    # -- field extractors ----------------------------------------------

    def _extract_invoice_number(
        self,
        indexed: list[tuple[int, OCRBlock]],
        result: ParseResult,
    ) -> None:
        self._match_field("invoice_number", _INVOICE_NUM_PAT, indexed, result)

    def _extract_dates(
        self,
        indexed: list[tuple[int, OCRBlock]],
        result: ParseResult,
    ) -> None:
        self._match_field("invoice_date", _DATE_PAT, indexed, result)
        self._match_field("due_date", _DUE_DATE_PAT, indexed, result)

    def _extract_totals(
        self,
        indexed: list[tuple[int, OCRBlock]],
        result: ParseResult,
    ) -> None:
        self._match_field("total", _TOTAL_PAT, indexed, result)
        self._match_field("subtotal", _SUBTOTAL_PAT, indexed, result)
        self._match_field("tax", _TAX_PAT, indexed, result)

    def _extract_currency(
        self,
        indexed: list[tuple[int, OCRBlock]],
        result: ParseResult,
    ) -> None:
        for idx, blk in indexed:
            for sym, code in _CURRENCY_SYMBOLS.items():
                if sym in blk.text:
                    result.fields["currency"] = ExtractedField(
                        name="currency",
                        value=code,
                        confidence=blk.confidence,
                        source_text=blk.text,
                        source_block_idx=idx,
                    )
                    return

    def _extract_vendor(
        self,
        indexed: list[tuple[int, OCRBlock]],
        result: ParseResult,
    ) -> None:
        skip = re.compile(
            r"^(invoice|date|bill|tax|page|total|subtotal|qty|amount|"
            r"description|item|no\.|#)",
            re.IGNORECASE,
        )
        for idx, blk in indexed[:5]:
            text = blk.text.strip()
            if len(text) > 3 and not skip.match(text):
                result.fields["vendor_name"] = ExtractedField(
                    name="vendor_name",
                    value=text,
                    confidence=blk.confidence,
                    source_text=blk.text,
                    source_block_idx=idx,
                )
                return

    def _extract_bill_to(
        self,
        indexed: list[tuple[int, OCRBlock]],
        result: ParseResult,
    ) -> None:
        self._match_field("bill_to", _BILL_TO_PAT, indexed, result)

    def _extract_line_items(
        self,
        indexed: list[tuple[int, OCRBlock]],
        result: ParseResult,
    ) -> None:
        """Heuristic line-item extraction.

        Looks for blocks that contain a monetary amount pattern and
        are not header/total lines.
        """
        total_kw = re.compile(
            r"^(total|subtotal|sub\s*total|tax|vat|gst|amount\s*due|"
            r"balance|grand|discount)",
            re.IGNORECASE,
        )
        amount_pat = re.compile(r"[\$€£¥]?\s*\d[\d,]*\.\d{2}")

        for idx, blk in indexed:
            text = blk.text.strip()
            if total_kw.match(text) or len(text) < 4:
                continue
            amounts = amount_pat.findall(text)
            if not amounts:
                continue
            # Treat last amount as the line total
            desc = amount_pat.sub("", text).strip().rstrip("$€£¥ \t")
            if len(desc) < 2:
                continue
            result.line_items.append(
                LineItem(
                    description=desc,
                    amount=amounts[-1].strip(),
                    confidence=blk.confidence,
                )
            )
