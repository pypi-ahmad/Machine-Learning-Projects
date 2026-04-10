"""Contact field extraction from OCR output for Business Card Reader.

Uses regex patterns and heuristics to classify each OCR text block
into a contact field category.  Each extracted field carries a
confidence score derived from the underlying OCR detection.

Strategy:
    1. Pattern-first: email, phone, website, and address are matched
       by strong regex patterns against every block.
    2. Contextual: title and company are matched via keyword lists and
       positional heuristics (title usually near name, company often
       appears with logo text or in upper/lower portion).
    3. Residual: the remaining prominent block near the top is treated
       as the person's name.

Usage::

    from parser import CardParser
    from ocr_engine import OCRBlock

    parser = CardParser()
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
class ParseResult:
    """Complete extraction result."""

    fields: dict[str, ExtractedField] = field(default_factory=dict)
    raw_text: str = ""
    num_blocks: int = 0


# ------------------------------------------------------------------
# Patterns
# ------------------------------------------------------------------

_EMAIL_PAT = re.compile(
    r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}",
)

_PHONE_PAT = re.compile(
    r"(?:\+?\d{1,3}[\s\-.]?)?"         # country code
    r"(?:\(?\d{1,4}\)?[\s\-.]?)?"      # area code
    r"\d[\d\s\-.]{5,}\d",              # core number
)

_WEBSITE_PAT = re.compile(
    r"(?:https?://)?(?:www\.)?[a-zA-Z0-9\-]+\.[a-zA-Z]{2,}(?:/\S*)?",
    re.IGNORECASE,
)

_ADDRESS_INDICATORS = re.compile(
    r"(?:street|st\.|ave\.?|avenue|blvd\.?|boulevard|road|rd\.|"
    r"drive|dr\.|lane|ln\.|suite|ste\.?|floor|fl\.|"
    r"building|bldg\.?|p\.?o\.?\s*box|"
    r"\d{5}(?:\-\d{4})?|"              # US zip
    r"[A-Z]\d[A-Z]\s*\d[A-Z]\d)",     # CA postal
    re.IGNORECASE,
)

_TITLE_KEYWORDS = re.compile(
    r"(?:^|\b)(?:ceo|cto|cfo|coo|cio|vp|svp|evp|"
    r"president|director|manager|engineer|developer|"
    r"designer|architect|analyst|consultant|specialist|"
    r"coordinator|administrator|assistant|associate|"
    r"partner|founder|co-founder|owner|principal|"
    r"professor|prof\.|dr\.|attorney|counsel|"
    r"supervisor|lead|head|chief|officer|executive|"
    r"representative|agent|broker|advisor|strategist|"
    r"intern|editor|writer|photographer|accountant|"
    r"sales|marketing|operations|finance|"
    r"human\s*resources|hr\b|it\b|"
    r"senior|junior|sr\.|jr\.)\b",
    re.IGNORECASE,
)

# Blocks that are clearly noise / not a name
_NOISE_PAT = re.compile(
    r"^[\d\s\-\.\(\)\+]+$|"           # pure digits/symbols
    r"^.{1,2}$|"                       # too short
    r"@|www\.|\.com|\.org|\.net|"      # email/url fragments
    r"^\d+\s",                         # starts with number (address-like)
    re.IGNORECASE,
)


# ------------------------------------------------------------------
# Parser
# ------------------------------------------------------------------

class CardParser:
    """Extract structured contact fields from OCR blocks."""

    def parse(self, blocks: list[OCRBlock]) -> ParseResult:
        if not blocks:
            return ParseResult()

        full_text = "\n".join(b.text for b in blocks)
        result = ParseResult(raw_text=full_text, num_blocks=len(blocks))

        # Track which blocks are consumed
        used: set[int] = set()

        # 1. Pattern-based fields (order matters — most specific first)
        self._extract_emails(blocks, result, used)
        self._extract_phones(blocks, result, used)
        self._extract_websites(blocks, result, used)
        self._extract_addresses(blocks, result, used)

        # 2. Context-based fields
        self._extract_title(blocks, result, used)
        self._extract_company(blocks, result, used)

        # 3. Residual — name
        self._extract_name(blocks, result, used)

        return result

    # -- pattern extractors --------------------------------------------

    def _extract_emails(
        self,
        blocks: list[OCRBlock],
        result: ParseResult,
        used: set[int],
    ) -> None:
        emails: list[str] = []
        best_idx = -1
        best_conf = 0.0
        best_src = ""

        for idx, blk in enumerate(blocks):
            matches = _EMAIL_PAT.findall(blk.text)
            if matches:
                emails.extend(matches)
                if blk.confidence > best_conf:
                    best_conf = blk.confidence
                    best_idx = idx
                    best_src = blk.text
                used.add(idx)

        if emails:
            result.fields["email"] = ExtractedField(
                name="email",
                value=emails[0],
                confidence=best_conf,
                source_text=best_src,
                source_block_idx=best_idx,
            )

    def _extract_phones(
        self,
        blocks: list[OCRBlock],
        result: ParseResult,
        used: set[int],
    ) -> None:
        for idx, blk in enumerate(blocks):
            if idx in used:
                continue
            m = _PHONE_PAT.search(blk.text)
            if m:
                # Validate: at least 7 digits
                digits = re.sub(r"\D", "", m.group())
                if len(digits) < 7:
                    continue
                result.fields["phone"] = ExtractedField(
                    name="phone",
                    value=m.group().strip(),
                    confidence=blk.confidence,
                    source_text=blk.text,
                    source_block_idx=idx,
                )
                used.add(idx)
                return

    def _extract_websites(
        self,
        blocks: list[OCRBlock],
        result: ParseResult,
        used: set[int],
    ) -> None:
        for idx, blk in enumerate(blocks):
            if idx in used:
                continue
            m = _WEBSITE_PAT.search(blk.text)
            if m:
                url = m.group()
                # Skip if it looks like an email domain fragment
                if "@" in blk.text:
                    continue
                result.fields["website"] = ExtractedField(
                    name="website",
                    value=url,
                    confidence=blk.confidence,
                    source_text=blk.text,
                    source_block_idx=idx,
                )
                used.add(idx)
                return

    def _extract_addresses(
        self,
        blocks: list[OCRBlock],
        result: ParseResult,
        used: set[int],
    ) -> None:
        addr_parts: list[str] = []
        best_idx = -1
        best_conf = 0.0
        best_src = ""

        for idx, blk in enumerate(blocks):
            if idx in used:
                continue
            if _ADDRESS_INDICATORS.search(blk.text):
                addr_parts.append(blk.text.strip())
                if blk.confidence > best_conf:
                    best_conf = blk.confidence
                    best_idx = idx
                    best_src = blk.text
                used.add(idx)

        if addr_parts:
            result.fields["address"] = ExtractedField(
                name="address",
                value=", ".join(addr_parts),
                confidence=best_conf,
                source_text=best_src,
                source_block_idx=best_idx,
            )

    # -- context extractors --------------------------------------------

    def _extract_title(
        self,
        blocks: list[OCRBlock],
        result: ParseResult,
        used: set[int],
    ) -> None:
        for idx, blk in enumerate(blocks):
            if idx in used:
                continue
            if _TITLE_KEYWORDS.search(blk.text):
                result.fields["title"] = ExtractedField(
                    name="title",
                    value=blk.text.strip(),
                    confidence=blk.confidence,
                    source_text=blk.text,
                    source_block_idx=idx,
                )
                used.add(idx)
                return

    def _extract_company(
        self,
        blocks: list[OCRBlock],
        result: ParseResult,
        used: set[int],
    ) -> None:
        """Company heuristic: look for LLC/Inc/Corp/Ltd suffixes, or
        pick the most prominent unused block that is not a name candidate."""
        corp_pat = re.compile(
            r"(?:inc\.?|llc|ltd\.?|corp\.?|corporation|company|co\.|"
            r"group|enterprises|solutions|technologies|consulting|"
            r"services|partners|associates|international|global|"
            r"industries|systems|labs?|studio|agency|foundation)\b",
            re.IGNORECASE,
        )
        for idx, blk in enumerate(blocks):
            if idx in used:
                continue
            if corp_pat.search(blk.text):
                result.fields["company"] = ExtractedField(
                    name="company",
                    value=blk.text.strip(),
                    confidence=blk.confidence,
                    source_text=blk.text,
                    source_block_idx=idx,
                )
                used.add(idx)
                return

        # Fallback: largest-font heuristic — the block with the tallest
        # bounding box among unused upper-half blocks
        h_mid = max(
            (b.centre[1] for b in blocks), default=0
        ) / 2
        candidates = [
            (idx, blk) for idx, blk in enumerate(blocks)
            if idx not in used
            and blk.centre[1] < h_mid
            and not _NOISE_PAT.search(blk.text)
            and len(blk.text.strip()) > 2
        ]
        if not candidates:
            return

        # Pick the one with the tallest bbox (proxy for font size)
        def _bbox_height(item: tuple[int, OCRBlock]) -> int:
            bb = item[1].bbox
            ys = [p[1] for p in bb]
            return max(ys) - min(ys)

        best = max(candidates, key=_bbox_height)
        idx, blk = best
        result.fields["company"] = ExtractedField(
            name="company",
            value=blk.text.strip(),
            confidence=blk.confidence * 0.8,  # lower confidence for heuristic
            source_text=blk.text,
            source_block_idx=idx,
        )
        used.add(idx)

    def _extract_name(
        self,
        blocks: list[OCRBlock],
        result: ParseResult,
        used: set[int],
    ) -> None:
        """Name: most prominent unused block that looks like a person's name."""
        for idx, blk in enumerate(blocks):
            if idx in used:
                continue
            text = blk.text.strip()
            if _NOISE_PAT.search(text):
                continue
            if len(text) < 2:
                continue
            # A name should be mostly alphabetical
            alpha_ratio = sum(c.isalpha() or c.isspace() for c in text) / max(len(text), 1)
            if alpha_ratio < 0.7:
                continue
            result.fields["name"] = ExtractedField(
                name="name",
                value=text,
                confidence=blk.confidence,
                source_text=blk.text,
                source_block_idx=idx,
            )
            used.add(idx)
            return
