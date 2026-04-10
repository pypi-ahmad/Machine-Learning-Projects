"""Receipt field extraction from OCR output.

Uses regex patterns and heuristics to extract structured fields
from OCR text blocks.  Each extracted field carries a confidence
score derived from the underlying OCR detection.

Usage::

    from parser import ReceiptParser
    from ocr_engine import OCRBlock

    parser = ReceiptParser()
    result = parser.parse(blocks)
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
    source_text: str
    source_block_idx: int


@dataclass
class LineItem:
    """Single receipt line item."""

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

_DATE_PAT = re.compile(
    r"(?:date|dated)?\s*[:\-]?\s*"
    r"(\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4}|\w+\s+\d{1,2},?\s+\d{4})",
    re.IGNORECASE,
)

_TIME_PAT = re.compile(
    r"(?:time)?\s*[:\-]?\s*(\d{1,2}:\d{2}(?::\d{2})?\s*(?:AM|PM)?)",
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
    r"(?:tax|vat|gst|hst|sales\s*tax)"
    r"\s*[:\-]?\s*[\$€£¥]?\s*([\d,]+\.?\d*)",
    re.IGNORECASE,
)

_TIP_PAT = re.compile(
    r"(?:tip|gratuity)\s*[:\-]?\s*[\$€£¥]?\s*([\d,]+\.?\d*)",
    re.IGNORECASE,
)

_PAYMENT_PAT = re.compile(
    r"(?:payment|paid\s*(?:by|with)|card\s*type|tender)"
    r"\s*[:\-]?\s*(.+)",
    re.IGNORECASE,
)

_CARD_HINT_PAT = re.compile(
    r"(visa|mastercard|amex|discover|debit|credit|cash|apple\s*pay|"
    r"google\s*pay|paypal|\*{4}\s*\d{4})",
    re.IGNORECASE,
)

_CURRENCY_SYMBOLS: dict[str, str] = {
    "$": "USD",
    "€": "EUR",
    "£": "GBP",
    "¥": "JPY",
}

_AMOUNT_PAT = re.compile(r"[\$€£¥]?\s*\d[\d,]*\.\d{2}")

_QTY_LINE_PAT = re.compile(
    r"(\d+)\s*[xX@]\s*[\$€£¥]?\s*([\d,]+\.?\d*)\s+(.*)",
)

# Lines that are summary labels — not items
_SUMMARY_KW = re.compile(
    r"^(total|subtotal|sub\s*total|tax|vat|gst|hst|tip|gratuity|"
    r"change|cash|visa|master|amex|debit|credit|payment|paid|"
    r"balance|amount\s*due|discount|coupon|savings|card|tender|"
    r"grand|net|round)",
    re.IGNORECASE,
)


# ------------------------------------------------------------------
# Parser
# ------------------------------------------------------------------


class ReceiptParser:
    """Extract structured fields from OCR blocks."""

    def parse(self, blocks: list[OCRBlock]) -> ParseResult:
        if not blocks:
            return ParseResult()

        full_text = "\n".join(b.text for b in blocks)
        result = ParseResult(raw_text=full_text, num_blocks=len(blocks))
        indexed = list(enumerate(blocks))

        self._extract_merchant(indexed, result)
        self._extract_date(indexed, result)
        self._extract_time(indexed, result)
        self._extract_totals(indexed, result)
        self._extract_tip(indexed, result)
        self._extract_currency(indexed, result)
        self._extract_payment(indexed, result)
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

    def _extract_merchant(
        self,
        indexed: list[tuple[int, OCRBlock]],
        result: ParseResult,
    ) -> None:
        """Merchant name: first substantive line that is not a keyword."""
        skip = re.compile(
            r"^(date|time|total|subtotal|tax|receipt|invoice|order|"
            r"cashier|server|register|store|table|check|bill|"
            r"tel|phone|fax|www\.|http|qty|item|#|\d{1,2}[/\-])",
            re.IGNORECASE,
        )
        for idx, blk in indexed[:6]:
            text = blk.text.strip()
            if len(text) > 2 and not skip.match(text):
                result.fields["merchant_name"] = ExtractedField(
                    name="merchant_name",
                    value=text,
                    confidence=blk.confidence,
                    source_text=blk.text,
                    source_block_idx=idx,
                )
                return

    def _extract_date(
        self,
        indexed: list[tuple[int, OCRBlock]],
        result: ParseResult,
    ) -> None:
        self._match_field("date", _DATE_PAT, indexed, result)

    def _extract_time(
        self,
        indexed: list[tuple[int, OCRBlock]],
        result: ParseResult,
    ) -> None:
        self._match_field("time", _TIME_PAT, indexed, result)

    def _extract_totals(
        self,
        indexed: list[tuple[int, OCRBlock]],
        result: ParseResult,
    ) -> None:
        self._match_field("total", _TOTAL_PAT, indexed, result)
        self._match_field("subtotal", _SUBTOTAL_PAT, indexed, result)
        self._match_field("tax", _TAX_PAT, indexed, result)

    def _extract_tip(
        self,
        indexed: list[tuple[int, OCRBlock]],
        result: ParseResult,
    ) -> None:
        self._match_field("tip", _TIP_PAT, indexed, result)

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

    def _extract_payment(
        self,
        indexed: list[tuple[int, OCRBlock]],
        result: ParseResult,
    ) -> None:
        # Try explicit "payment:" label first
        self._match_field("payment_method", _PAYMENT_PAT, indexed, result)
        if "payment_method" in result.fields:
            return
        # Fall back to card keyword scan
        for idx, blk in indexed:
            m = _CARD_HINT_PAT.search(blk.text)
            if m:
                result.fields["payment_method"] = ExtractedField(
                    name="payment_method",
                    value=m.group(1).strip(),
                    confidence=blk.confidence,
                    source_text=blk.text,
                    source_block_idx=idx,
                )
                return

    def _extract_line_items(
        self,
        indexed: list[tuple[int, OCRBlock]],
        result: ParseResult,
    ) -> None:
        """Heuristic line-item extraction.

        Recognises two patterns:
        - ``QTY x PRICE  DESCRIPTION`` (qty-first)
        - ``DESCRIPTION  AMOUNT`` (amount-only)
        """
        for idx, blk in indexed:
            text = blk.text.strip()
            if _SUMMARY_KW.match(text) or len(text) < 3:
                continue

            # Try qty x price first
            qm = _QTY_LINE_PAT.match(text)
            if qm:
                result.line_items.append(
                    LineItem(
                        description=qm.group(3).strip(),
                        quantity=qm.group(1),
                        unit_price=qm.group(2),
                        confidence=blk.confidence,
                    )
                )
                continue

            # Fall back to amount-bearing lines
            amounts = _AMOUNT_PAT.findall(text)
            if not amounts:
                continue
            desc = _AMOUNT_PAT.sub("", text).strip().rstrip("$€£¥ \t")
            if len(desc) < 2:
                continue
            result.line_items.append(
                LineItem(
                    description=desc,
                    amount=amounts[-1].strip(),
                    confidence=blk.confidence,
                )
            )
