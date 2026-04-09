"""Core conversion logic: XML string/bytes -> Python dict -> JSON string."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import xmltodict

logger = logging.getLogger("convert_xml_to_json")

# ---------------------------------------------------------------------------
# XML -> dict
# ---------------------------------------------------------------------------


def xml_to_dict(xml_text: str | bytes, *, encoding: str = "utf-8") -> dict:
    """Parse an XML string (or bytes) into an ordered dict via *xmltodict*.

    Parameters
    ----------
    xml_text:
        Raw XML content.  If *bytes*, decoded using *encoding* first.
    encoding:
        Character encoding used when *xml_text* is bytes.

    Returns
    -------
    dict
        The parsed XML as a nested Python dictionary.

    Raises
    ------
    ValueError
        If the XML cannot be parsed.
    """
    if isinstance(xml_text, bytes):
        xml_text = xml_text.decode(encoding)

    try:
        parsed = xmltodict.parse(xml_text)
    except Exception as exc:
        msg = f"Failed to parse XML: {exc}"
        raise ValueError(msg) from exc

    if parsed is None:
        msg = "XML produced an empty result"
        raise ValueError(msg)

    return dict(parsed)


# ---------------------------------------------------------------------------
# dict -> JSON
# ---------------------------------------------------------------------------


def dict_to_json(data: dict, *, indent: int | None = 2, sort_keys: bool = False) -> str:
    """Serialise a dict to a JSON string.

    Parameters
    ----------
    data:
        The dictionary to serialise.
    indent:
        Number of spaces for pretty-printing.  *None* produces compact JSON.
    sort_keys:
        Whether to sort dictionary keys in output.
    """
    return json.dumps(data, indent=indent, ensure_ascii=False, sort_keys=sort_keys)


# ---------------------------------------------------------------------------
# High-level: XML -> JSON
# ---------------------------------------------------------------------------


def convert_xml_to_json(
    xml_text: str | bytes,
    *,
    encoding: str = "utf-8",
    indent: int | None = 2,
    sort_keys: bool = False,
) -> str:
    """Convert raw XML content to a JSON string.

    This is the main API entry point combining parsing and serialisation.
    """
    data = xml_to_dict(xml_text, encoding=encoding)
    return dict_to_json(data, indent=indent, sort_keys=sort_keys)


# ---------------------------------------------------------------------------
# File helpers
# ---------------------------------------------------------------------------


def read_xml_file(path: Path, *, encoding: str = "utf-8") -> str:
    """Read an XML file and return its text content."""
    if not path.exists():
        msg = f"File not found: {path}"
        raise FileNotFoundError(msg)
    if not path.is_file():
        msg = f"Not a file: {path}"
        raise FileNotFoundError(msg)
    return path.read_text(encoding=encoding)


def write_json_file(path: Path, content: str, *, encoding: str = "utf-8") -> None:
    """Write a JSON string to a file, creating parent dirs if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content + "\n", encoding=encoding)
    logger.info("Written %d bytes to %s", len(content), path)
