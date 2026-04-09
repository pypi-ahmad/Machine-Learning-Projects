"""Data I/O utilities for loading and saving project data.

Handles CSV, JSON, JSONL, TXT, and common encodings automatically.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from utils.logger import get_logger

logger = get_logger(__name__)

# ---- Encoding detection (lightweight) -----------------------------------

_ENCODINGS = ["utf-8", "utf-8-sig", "latin-1", "cp1252", "iso-8859-1"]


def _detect_encoding(path: Path) -> str:
    for enc in _ENCODINGS:
        try:
            with open(path, "r", encoding=enc) as f:
                f.read(4096)
            return enc
        except (UnicodeDecodeError, UnicodeError):
            continue
    return "utf-8"


# ---- Loaders ------------------------------------------------------------


def load_csv(path: str | Path, **kwargs) -> pd.DataFrame:
    path = Path(path)
    enc = kwargs.pop("encoding", _detect_encoding(path))
    logger.info("Loading CSV: %s (encoding=%s)", path.name, enc)
    return pd.read_csv(path, encoding=enc, **kwargs)


def load_json(path: str | Path, lines: bool = False) -> pd.DataFrame | list | dict:
    path = Path(path)
    logger.info("Loading JSON: %s (lines=%s)", path.name, lines)
    if lines:
        return pd.read_json(path, lines=True)
    enc = _detect_encoding(path)
    with open(path, "r", encoding=enc) as f:
        content = f.read().strip()
    if content.startswith("[") or content.startswith("{"):
        try:
            return pd.read_json(path)
        except ValueError:
            return json.loads(content)
    # Fallback: try JSON lines
    return pd.read_json(path, lines=True)


def load_text(path: str | Path) -> str:
    path = Path(path)
    enc = _detect_encoding(path)
    return path.read_text(encoding=enc)


def load_text_lines(path: str | Path) -> list[str]:
    return load_text(path).splitlines()


def load_labeled_text(path: str | Path, sep: str = "\t") -> pd.DataFrame:
    """Load a text file with tab-separated label/text pairs (e.g. review files)."""
    path = Path(path)
    lines = load_text_lines(path)
    records = []
    for line in lines:
        parts = line.rsplit(sep, 1)
        if len(parts) == 2:
            records.append({"text": parts[0].strip(), "label": parts[1].strip()})
    logger.info("Loaded %d labeled records from %s", len(records), path.name)
    return pd.DataFrame(records)


def load_data_auto(path: str | Path) -> pd.DataFrame:
    """Auto-detect format and load as DataFrame."""
    path = Path(path)
    ext = path.suffix.lower()
    if ext == ".csv":
        return load_csv(path)
    elif ext == ".json":
        result = load_json(path)
        if isinstance(result, pd.DataFrame):
            return result
        return pd.DataFrame(result)
    elif ext == ".jsonl":
        return load_json(path, lines=True)
    elif ext in (".txt", ".tsv"):
        try:
            return load_csv(path, sep="\t")
        except Exception:
            return load_labeled_text(path)
    else:
        raise ValueError(f"Unsupported file extension: {ext}")


# ---- Savers --------------------------------------------------------------


def save_csv(df: pd.DataFrame, path: str | Path, **kwargs) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, **kwargs)
    logger.info("Saved CSV: %s (%d rows)", path, len(df))
    return path


def save_json(data: dict | list, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)
    logger.info("Saved JSON: %s", path)
    return path
