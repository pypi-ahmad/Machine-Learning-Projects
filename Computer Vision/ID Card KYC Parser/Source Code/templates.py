"""Template definitions for ID Card KYC Parser.

Each template encapsulates field-extraction logic for a specific
document type (e.g. generic ID, US driver licence, EU national ID,
passport MRZ).  Templates are isolated so adding new ones does not
affect existing logic.

Usage::

    from templates import get_template

    tmpl = get_template("generic")
    fields = tmpl.extract(blocks)
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ocr_engine import OCRBlock


# ------------------------------------------------------------------
# Result types
# ------------------------------------------------------------------

@dataclass
class ExtractedField:
    """Single extracted field with provenance."""

    name: str
    value: str
    confidence: float
    source_text: str
    source_block_idx: int


# ------------------------------------------------------------------
# Template protocol
# ------------------------------------------------------------------

class IDTemplate(Protocol):
    """Interface every template must satisfy."""

    name: str

    def extract(self, blocks: list[OCRBlock]) -> dict[str, ExtractedField]:
        """Return extracted fields dict keyed by field name."""
        ...


# ------------------------------------------------------------------
# Shared helpers
# ------------------------------------------------------------------

def _match_label_value(
    label_pat: re.Pattern[str],
    blocks: list[OCRBlock],
    field_name: str,
    used: set[int],
) -> ExtractedField | None:
    """Match blocks where the label and value appear on the same line,
    or where the value is on the next line below the label."""
    for idx, blk in enumerate(blocks):
        if idx in used:
            continue
        m = label_pat.search(blk.text)
        if m:
            # Value on same line after the label
            value = blk.text[m.end():].strip().strip(":").strip()
            if value:
                used.add(idx)
                return ExtractedField(
                    name=field_name,
                    value=value,
                    confidence=blk.confidence,
                    source_text=blk.text,
                    source_block_idx=idx,
                )
            # Value might be on the next block
            if idx + 1 < len(blocks) and (idx + 1) not in used:
                nxt = blocks[idx + 1]
                if nxt.text.strip():
                    used.add(idx)
                    used.add(idx + 1)
                    return ExtractedField(
                        name=field_name,
                        value=nxt.text.strip(),
                        confidence=nxt.confidence,
                        source_text=nxt.text,
                        source_block_idx=idx + 1,
                    )
    return None


# ------------------------------------------------------------------
# Generic template
# ------------------------------------------------------------------

_NAME_LABELS = re.compile(
    r"(?:full\s*name|name|nom|nombre|given\s*name|surname)",
    re.IGNORECASE,
)
_DOB_LABELS = re.compile(
    r"(?:date\s*of\s*birth|d\.?o\.?b\.?|birth\s*date|born|"
    r"fecha\s*de\s*nacimiento|date\s*de\s*naissance)",
    re.IGNORECASE,
)
_ID_NUM_LABELS = re.compile(
    r"(?:id\s*(?:no\.?|number|#)|document\s*(?:no\.?|number)|"
    r"license\s*(?:no\.?|number)|passport\s*(?:no\.?|number)|"
    r"card\s*(?:no\.?|number)|n[uú]mero)",
    re.IGNORECASE,
)
_NATIONALITY_LABELS = re.compile(
    r"(?:nationality|nationalit[eé]|ciudadan[ií]a|country)",
    re.IGNORECASE,
)
_GENDER_LABELS = re.compile(
    r"(?:sex|gender|sexe|sexo)",
    re.IGNORECASE,
)
_EXPIRY_LABELS = re.compile(
    r"(?:expiry|expiration|exp\.?\s*date|valid\s*(?:until|thru|through)|"
    r"date\s*d'expiration|fecha\s*de\s*vencimiento)",
    re.IGNORECASE,
)
_ISSUE_LABELS = re.compile(
    r"(?:issue\s*date|date\s*(?:of\s*)?issue|issued|"
    r"date\s*de\s*d[eé]livrance|fecha\s*de\s*emisi[oó]n)",
    re.IGNORECASE,
)
_ADDRESS_LABELS = re.compile(
    r"(?:address|addr\.?|domicile|direcci[oó]n|adresse)",
    re.IGNORECASE,
)
_DOC_TYPE_LABELS = re.compile(
    r"(?:identity\s*card|national\s*id|driver.?s?\s*licen[cs]e|"
    r"passport|residence\s*permit|carte\s*d'identit[eé]|"
    r"permis\s*de\s*conduire|tarjeta\s*de\s*identidad)",
    re.IGNORECASE,
)

_DATE_PAT = re.compile(
    r"\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4}|\d{4}[/\-\.]\d{1,2}[/\-\.]\d{1,2}|"
    r"\w+\s+\d{1,2},?\s+\d{4}|\d{1,2}\s+\w+\s+\d{4}",
)

_GENDER_VALUE = re.compile(r"\b(male|female|m|f)\b", re.IGNORECASE)


class GenericTemplate:
    """Catch-all template — extracts fields using label-value patterns."""

    name = "generic"

    def extract(self, blocks: list[OCRBlock]) -> dict[str, ExtractedField]:
        fields: dict[str, ExtractedField] = {}
        used: set[int] = set()

        label_map: list[tuple[str, re.Pattern[str]]] = [
            ("full_name", _NAME_LABELS),
            ("date_of_birth", _DOB_LABELS),
            ("id_number", _ID_NUM_LABELS),
            ("nationality", _NATIONALITY_LABELS),
            ("gender", _GENDER_LABELS),
            ("expiry_date", _EXPIRY_LABELS),
            ("issue_date", _ISSUE_LABELS),
            ("address", _ADDRESS_LABELS),
        ]

        for fname, pat in label_map:
            ef = _match_label_value(pat, blocks, fname, used)
            if ef:
                fields[fname] = ef

        # Document type — scan all text
        self._extract_doc_type(blocks, fields, used)

        # Fallbacks for date_of_birth / id_number if not yet found
        self._fallback_dates(blocks, fields, used)
        self._fallback_id_number(blocks, fields, used)

        return fields

    def _extract_doc_type(
        self,
        blocks: list[OCRBlock],
        fields: dict[str, ExtractedField],
        used: set[int],
    ) -> None:
        for idx, blk in enumerate(blocks):
            m = _DOC_TYPE_LABELS.search(blk.text)
            if m:
                fields["document_type"] = ExtractedField(
                    name="document_type",
                    value=m.group().strip(),
                    confidence=blk.confidence,
                    source_text=blk.text,
                    source_block_idx=idx,
                )
                return

    def _fallback_dates(
        self,
        blocks: list[OCRBlock],
        fields: dict[str, ExtractedField],
        used: set[int],
    ) -> None:
        """Pick up un-labelled date strings."""
        if "date_of_birth" in fields:
            return
        for idx, blk in enumerate(blocks):
            if idx in used:
                continue
            m = _DATE_PAT.search(blk.text)
            if m:
                fields["date_of_birth"] = ExtractedField(
                    name="date_of_birth",
                    value=m.group().strip(),
                    confidence=blk.confidence * 0.7,
                    source_text=blk.text,
                    source_block_idx=idx,
                )
                used.add(idx)
                return

    def _fallback_id_number(
        self,
        blocks: list[OCRBlock],
        fields: dict[str, ExtractedField],
        used: set[int],
    ) -> None:
        """Heuristic: long alphanumeric string is likely an ID number."""
        if "id_number" in fields:
            return
        id_pat = re.compile(r"[A-Z0-9]{6,20}")
        for idx, blk in enumerate(blocks):
            if idx in used:
                continue
            m = id_pat.search(blk.text)
            if m:
                fields["id_number"] = ExtractedField(
                    name="id_number",
                    value=m.group(),
                    confidence=blk.confidence * 0.6,
                    source_text=blk.text,
                    source_block_idx=idx,
                )
                used.add(idx)
                return


# ------------------------------------------------------------------
# MRZ passport template
# ------------------------------------------------------------------

_MRZ_LINE = re.compile(r"[A-Z0-9<]{30,44}")


class PassportMRZTemplate:
    """Passport template — parses the Machine Readable Zone (MRZ)."""

    name = "passport"

    def extract(self, blocks: list[OCRBlock]) -> dict[str, ExtractedField]:
        fields: dict[str, ExtractedField] = {}
        mrz_lines: list[tuple[int, OCRBlock]] = []

        for idx, blk in enumerate(blocks):
            clean = blk.text.replace(" ", "").upper()
            if _MRZ_LINE.fullmatch(clean):
                mrz_lines.append((idx, blk))

        if len(mrz_lines) >= 2:
            self._parse_td3(mrz_lines, fields)
        else:
            # Fall back to generic
            generic = GenericTemplate()
            return generic.extract(blocks)

        fields["document_type"] = ExtractedField(
            name="document_type",
            value="Passport",
            confidence=0.95,
            source_text="MRZ",
            source_block_idx=mrz_lines[0][0],
        )
        return fields

    def _parse_td3(
        self,
        mrz_lines: list[tuple[int, OCRBlock]],
        fields: dict[str, ExtractedField],
    ) -> None:
        """Parse TD-3 (passport) MRZ — two lines of 44 characters."""
        idx1, blk1 = mrz_lines[-2]
        idx2, blk2 = mrz_lines[-1]
        line1 = blk1.text.replace(" ", "").upper()
        line2 = blk2.text.replace(" ", "").upper()

        # Line 1: P<ISSUING_COUNTRY<SURNAME<<GIVEN_NAMES<<<
        if len(line1) >= 5:
            name_part = line1[5:].replace("<", " ").strip()
            parts = [p for p in name_part.split("  ") if p.strip()]
            full_name = " ".join(parts)
            if full_name:
                fields["full_name"] = ExtractedField(
                    name="full_name",
                    value=full_name,
                    confidence=blk1.confidence,
                    source_text=blk1.text,
                    source_block_idx=idx1,
                )
            if len(line1) >= 5:
                nationality_code = line1[2:5].replace("<", "")
                if nationality_code:
                    fields["nationality"] = ExtractedField(
                        name="nationality",
                        value=nationality_code,
                        confidence=blk1.confidence,
                        source_text=blk1.text,
                        source_block_idx=idx1,
                    )

        # Line 2: passport number, DOB, gender, expiry
        if len(line2) >= 28:
            passport_no = line2[0:9].replace("<", "").strip()
            if passport_no:
                fields["id_number"] = ExtractedField(
                    name="id_number",
                    value=passport_no,
                    confidence=blk2.confidence,
                    source_text=blk2.text,
                    source_block_idx=idx2,
                )

            dob_raw = line2[13:19]
            if dob_raw.isdigit():
                dob = f"{dob_raw[4:6]}/{dob_raw[2:4]}/{dob_raw[0:2]}"
                fields["date_of_birth"] = ExtractedField(
                    name="date_of_birth",
                    value=dob,
                    confidence=blk2.confidence,
                    source_text=blk2.text,
                    source_block_idx=idx2,
                )

            gender_char = line2[20] if len(line2) > 20 else ""
            if gender_char in ("M", "F"):
                fields["gender"] = ExtractedField(
                    name="gender",
                    value="Male" if gender_char == "M" else "Female",
                    confidence=blk2.confidence,
                    source_text=blk2.text,
                    source_block_idx=idx2,
                )

            exp_raw = line2[21:27]
            if exp_raw.isdigit():
                exp = f"{exp_raw[4:6]}/{exp_raw[2:4]}/{exp_raw[0:2]}"
                fields["expiry_date"] = ExtractedField(
                    name="expiry_date",
                    value=exp,
                    confidence=blk2.confidence,
                    source_text=blk2.text,
                    source_block_idx=idx2,
                )


# ------------------------------------------------------------------
# US driver licence template
# ------------------------------------------------------------------

class USDLTemplate:
    """US Driver Licence — common field labels."""

    name = "us_dl"

    def extract(self, blocks: list[OCRBlock]) -> dict[str, ExtractedField]:
        fields: dict[str, ExtractedField] = {}
        used: set[int] = set()

        dl_labels: list[tuple[str, re.Pattern[str]]] = [
            ("full_name", re.compile(r"(?:name|fn|ln)\b", re.IGNORECASE)),
            ("date_of_birth", re.compile(r"(?:dob|date\s*of\s*birth)", re.IGNORECASE)),
            ("id_number", re.compile(r"(?:dl\s*(?:no\.?|number|#)|license\s*(?:no\.?|number))", re.IGNORECASE)),
            ("expiry_date", re.compile(r"(?:exp|expiry|expiration)", re.IGNORECASE)),
            ("issue_date", re.compile(r"(?:iss|issued|issue\s*date)", re.IGNORECASE)),
            ("address", re.compile(r"(?:addr|address)", re.IGNORECASE)),
            ("gender", re.compile(r"(?:sex|gender)", re.IGNORECASE)),
        ]

        for fname, pat in dl_labels:
            ef = _match_label_value(pat, blocks, fname, used)
            if ef:
                fields[fname] = ef

        fields["document_type"] = ExtractedField(
            name="document_type",
            value="Driver Licence",
            confidence=0.9,
            source_text="template",
            source_block_idx=-1,
        )
        return fields


# ------------------------------------------------------------------
# EU national ID template
# ------------------------------------------------------------------

class EUIDTemplate:
    """EU National ID — common field labels (multi-language)."""

    name = "eu_id"

    def extract(self, blocks: list[OCRBlock]) -> dict[str, ExtractedField]:
        fields: dict[str, ExtractedField] = {}
        used: set[int] = set()

        eu_labels: list[tuple[str, re.Pattern[str]]] = [
            ("full_name", re.compile(r"(?:name|nom|nombre|naam|vorname|nachname|surname|given)", re.IGNORECASE)),
            ("date_of_birth", re.compile(r"(?:date\s*of\s*birth|date\s*de\s*naissance|fecha\s*de\s*nacimiento|geburtsdatum)", re.IGNORECASE)),
            ("id_number", re.compile(r"(?:card\s*(?:no\.?|number)|document\s*(?:no\.?|number)|numéro|número|nummer)", re.IGNORECASE)),
            ("nationality", re.compile(r"(?:nationality|nationalité|ciudadanía|staatsangehörigkeit)", re.IGNORECASE)),
            ("gender", re.compile(r"(?:sex|sexe|sexo|geschlecht)", re.IGNORECASE)),
            ("expiry_date", re.compile(r"(?:expiry|expiration|valid|gültig|validité|vencimiento)", re.IGNORECASE)),
            ("issue_date", re.compile(r"(?:issue|délivrance|emisión|ausstellungsdatum)", re.IGNORECASE)),
        ]

        for fname, pat in eu_labels:
            ef = _match_label_value(pat, blocks, fname, used)
            if ef:
                fields[fname] = ef

        fields["document_type"] = ExtractedField(
            name="document_type",
            value="National ID",
            confidence=0.9,
            source_text="template",
            source_block_idx=-1,
        )
        return fields


# ------------------------------------------------------------------
# Template registry
# ------------------------------------------------------------------

_TEMPLATES: dict[str, IDTemplate] = {
    "generic": GenericTemplate(),
    "passport": PassportMRZTemplate(),
    "us_dl": USDLTemplate(),
    "eu_id": EUIDTemplate(),
}


def get_template(name: str) -> IDTemplate:
    """Retrieve a template by name.

    Raises
    ------
    KeyError
        If no template with *name* exists.
    """
    if name not in _TEMPLATES:
        available = ", ".join(sorted(_TEMPLATES.keys()))
        raise KeyError(
            f"Unknown template '{name}'. Available: {available}"
        )
    return _TEMPLATES[name]


def list_templates() -> list[str]:
    """Return sorted list of available template names."""
    return sorted(_TEMPLATES.keys())
