#!/usr/bin/env python3
"""
Centralised dataset loader for all ML projects.

Provides a single entry-point — ``load_dataset(project_key)`` — that
resolves the correct data source from ``dataset_registry.json``, handles
local files, URL downloads, and API fetches (yfinance), and returns a
pandas DataFrame ready for downstream processing.

Usage in any pipeline::

    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    from core.data_loader import load_dataset

    df = load_dataset('boston_house_classification')
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import urllib.request
from pathlib import Path
from typing import Optional

import pandas as pd

# ─── Logging ────────────────────────────────────────────────────────
logger = logging.getLogger("core.data_loader")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(name)s] %(levelname)s  %(message)s",
        datefmt="%H:%M:%S",
    ))
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)

# ─── Constants ──────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent          # workspace root
_REGISTRY_PATH = _ROOT / "dataset_registry.json"
_CACHE_DIR = _ROOT / ".dataset_cache"

# ─── Registry singleton ────────────────────────────────────────────
_registry: dict | None = None


def _load_registry() -> dict:
    """Load and cache the dataset registry."""
    global _registry
    if _registry is None:
        if not _REGISTRY_PATH.exists():
            raise FileNotFoundError(
                f"dataset_registry.json not found at {_REGISTRY_PATH}"
            )
        with open(_REGISTRY_PATH, encoding="utf-8") as f:
            _registry = json.load(f)
        logger.debug("Loaded registry with %d entries", len(_registry))
    return _registry


# ════════════════════════════════════════════════════════════════════
# PUBLIC API
# ════════════════════════════════════════════════════════════════════

def load_dataset(project_key: str, **read_kwargs) -> pd.DataFrame:
    """Load the primary dataset for *project_key*.

    Resolution order (from ``source_type``):
      * **local** → ``pd.read_csv`` / ``read_excel`` / ``read_json`` / etc.
      * **url**   → download to cache, then load locally.
      * **api**   → call the appropriate fetcher (yfinance), cache result.

    Parameters
    ----------
    project_key : str
        Slug key matching an entry in ``dataset_registry.json``.
    **read_kwargs
        Extra keyword arguments forwarded to the pandas reader
        (e.g. ``encoding``, ``sep``, ``parse_dates``).

    Returns
    -------
    pd.DataFrame

    Raises
    ------
    KeyError
        If *project_key* is not found in the registry.
    FileNotFoundError
        If the resolved dataset file does not exist on disk.
    """
    reg = _load_registry()

    if project_key not in reg:
        raise KeyError(
            f"Unknown project key '{project_key}'. "
            f"Available: {', '.join(sorted(reg.keys())[:10])} …"
        )

    info = reg[project_key]
    source_type = info.get("source_type", "local")
    rel_path = info.get("path", "")
    logger.info(
        "Resolving dataset for '%s' [source=%s, path=%s]",
        project_key, source_type, rel_path,
    )

    # ── Local ──────────────────────────────────────────────────
    if source_type == "local":
        return _load_local(rel_path, info, **read_kwargs)

    # ── URL download ───────────────────────────────────────────
    if source_type == "url":
        url = info.get("url") or info.get("fallback", "")
        if not url:
            raise ValueError(
                f"source_type='url' for '{project_key}' but no 'url' or "
                "'fallback' field in registry."
            )
        cached = cache_dataset(url)
        return _read_file(cached, **read_kwargs)

    # ── API fetch ──────────────────────────────────────────────
    if source_type == "api":
        return _load_api(project_key, info, **read_kwargs)

    raise ValueError(
        f"Unknown source_type '{source_type}' for project '{project_key}'"
    )


def detect_dataset_type(df: pd.DataFrame) -> str:
    """Heuristically classify a DataFrame as text / tabular / timeseries.

    Rules (applied in order):
      1. ``text``       — >40 % of columns are ``object`` dtype AND the
                          longest string column has median length > 50 chars.
      2. ``timeseries`` — index is ``DatetimeIndex``, **or** there is at
                          least one column with dtype ``datetime64``.
      3. ``tabular``    — everything else.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    str
        One of ``'text'``, ``'timeseries'``, ``'tabular'``.
    """
    if df.empty:
        return "tabular"

    obj_cols = df.select_dtypes(include=["object", "string"]).columns
    obj_ratio = len(obj_cols) / max(len(df.columns), 1)

    # Text detection
    if obj_ratio > 0.4 and len(obj_cols) > 0:
        max_median_len = max(
            df[c].dropna().astype(str).str.len().median()
            for c in obj_cols
        )
        if max_median_len > 50:
            return "text"

    # Time-series detection
    if isinstance(df.index, pd.DatetimeIndex):
        return "timeseries"
    dt_cols = df.select_dtypes(include=["datetime64", "datetimetz"]).columns
    if len(dt_cols) > 0:
        return "timeseries"
    # Check for columns with 'date' or 'time' in name that could be parsed
    date_like = [
        c for c in df.columns
        if any(tok in c.lower() for tok in ("date", "time", "timestamp"))
    ]
    if date_like:
        for c in date_like:
            try:
                pd.to_datetime(df[c].head(20), format="mixed", dayfirst=False)
                return "timeseries"
            except (ValueError, TypeError):
                continue

    return "tabular"


def handle_missing_data(
    df: pd.DataFrame,
    *,
    numeric_strategy: str = "median",
    categorical_strategy: str = "mode",
    drop_threshold: float = 0.5,
) -> pd.DataFrame:
    """Handle missing values in *df* using safe, conservative defaults.

    Only acts **if** there *are* missing values — otherwise returns
    the DataFrame untouched.

    Parameters
    ----------
    df : pd.DataFrame
    numeric_strategy : str
        ``'mean'``, ``'median'``, or ``'zero'``.
    categorical_strategy : str
        ``'mode'`` or ``'unknown'``.
    drop_threshold : float
        Drop columns whose missing ratio exceeds this (0–1).

    Returns
    -------
    pd.DataFrame
        A copy with missing data handled.
    """
    if not df.isnull().any().any():
        logger.debug("No missing data — returning untouched")
        return df

    df = df.copy()
    total = len(df)

    # 1. Drop columns with too many missing values
    missing_ratio = df.isnull().sum() / total
    drop_cols = missing_ratio[missing_ratio > drop_threshold].index.tolist()
    if drop_cols:
        logger.info("Dropping %d columns (>%.0f%% missing): %s",
                     len(drop_cols), drop_threshold * 100, drop_cols)
        df = df.drop(columns=drop_cols)

    # 2. Numeric columns
    num_cols = df.select_dtypes(include="number").columns
    for col in num_cols:
        if df[col].isnull().any():
            if numeric_strategy == "median":
                fill = df[col].median()
            elif numeric_strategy == "mean":
                fill = df[col].mean()
            else:
                fill = 0
            df[col] = df[col].fillna(fill)

    # 3. Categorical / object columns
    cat_cols = df.select_dtypes(include=["object", "category", "string"]).columns
    for col in cat_cols:
        if df[col].isnull().any():
            if categorical_strategy == "mode" and not df[col].mode().empty:
                fill = df[col].mode().iloc[0]
            else:
                fill = "Unknown"
            df[col] = df[col].fillna(fill)

    remaining = df.isnull().sum().sum()
    if remaining:
        logger.warning("%d missing values remain after imputation", remaining)
    else:
        logger.info("All missing values handled")

    return df


def cache_dataset(source: str, *, dest_name: str | None = None) -> Path:
    """Download *source* (URL) into the local cache and return the path.

    If the file is already cached (same filename), the cached copy is
    returned without re-downloading.

    Parameters
    ----------
    source : str
        A URL to download.
    dest_name : str, optional
        Override filename in cache.  Defaults to the URL's basename.

    Returns
    -------
    pathlib.Path
        Absolute path to the cached file.
    """
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)

    if dest_name is None:
        # Use a hash prefix to avoid collisions
        url_hash = hashlib.md5(source.encode()).hexdigest()[:8]
        basename = Path(source.split("?")[0]).name or "dataset"
        dest_name = f"{url_hash}_{basename}"

    dest = _CACHE_DIR / dest_name
    if dest.exists():
        logger.info("Using cached dataset: %s", dest)
        return dest

    logger.info("Downloading %s → %s", source, dest)
    try:
        urllib.request.urlretrieve(source, dest)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to download dataset from {source}: {exc}"
        ) from exc

    logger.info("Cached dataset: %s (%.1f KB)", dest,
                dest.stat().st_size / 1024)
    return dest


# ════════════════════════════════════════════════════════════════════
# INTERNAL HELPERS
# ════════════════════════════════════════════════════════════════════

def _resolve_path(rel_path: str) -> Path:
    """Resolve a registry-relative path to an absolute path."""
    p = _ROOT / rel_path
    if p.exists():
        return p
    # Try forward-slash normalisation
    p2 = _ROOT / rel_path.replace("\\", "/")
    if p2.exists():
        return p2
    return p  # return as-is; caller will raise


def _read_csv_robust(path: Path, **kwargs) -> pd.DataFrame:
    """Read a CSV/text file with encoding fallback and bad-line tolerance."""
    encodings = ["utf-8", "latin-1", "cp1252", "iso-8859-1"]
    # Allow caller to override encoding; if so, try only that one
    if "encoding" in kwargs:
        encodings = [kwargs.pop("encoding")]
    # Default to tolerant line handling
    kwargs.setdefault("on_bad_lines", "warn")
    last_err: Exception | None = None
    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc, **kwargs)
        except UnicodeDecodeError as e:
            last_err = e
            continue
    # If all encodings failed, raise the last error
    raise last_err  # type: ignore[misc]


def _read_file(path: Path, **kwargs) -> pd.DataFrame:
    """Auto-detect file format and read into DataFrame."""
    suffix = path.suffix.lower()
    if suffix in (".csv", ".txt", ".data"):
        return _read_csv_robust(path, **kwargs)
    if suffix in (".xls", ".xlsx"):
        return pd.read_excel(path, **kwargs)
    if suffix == ".json":
        return pd.read_json(path, **kwargs)
    if suffix == ".parquet":
        return pd.read_parquet(path, **kwargs)
    if suffix == ".tsv":
        return _read_csv_robust(path, sep="\t", **kwargs)
    # Default fallback — try CSV with robust encoding
    return _read_csv_robust(path, **kwargs)


def _load_local(rel_path: str, info: dict, **kwargs) -> pd.DataFrame:
    """Load a local dataset from the workspace data/ directory."""
    abs_path = _resolve_path(rel_path)
    if not abs_path.exists():
        # Try fallback
        fb = info.get("fallback")
        if fb and fb != "manual_required":
            logger.warning(
                "Primary path %s missing — trying fallback %s",
                abs_path, fb,
            )
            abs_path = _resolve_path(fb)
        if not abs_path.exists():
            raise FileNotFoundError(
                f"Dataset not found: {abs_path}\n"
                f"  project: {info.get('project_name', '?')}\n"
                f"  registry path: {rel_path}"
            )
    logger.info("Loading local dataset: %s", abs_path)
    return _read_file(abs_path, **kwargs)


def _load_api(project_key: str, info: dict, **kwargs) -> pd.DataFrame:
    """Fetch data via an API (currently supports yfinance)."""
    rel_path = info.get("path", "")

    # If we have a cached local copy, use it
    cached = _resolve_path(rel_path)
    if cached.exists():
        logger.info("Using existing cached API data: %s", cached)
        return _read_file(cached, **kwargs)

    # yfinance fetch
    try:
        import yfinance as yf
    except ImportError:
        raise ImportError(
            "yfinance is required for API dataset fetching. "
            "Install with: pip install yfinance"
        )

    # Extract ticker info from evidence or default
    evidence = info.get("evidence", [])
    ticker = "AAPL"  # default
    for ev in evidence:
        if "yf.download" in ev:
            # Try to extract ticker from evidence string
            import re
            m = re.search(r"yf\.download\(['\"]([^'\"]+)['\"]", ev)
            if m:
                ticker = m.group(1)
                break

    logger.info("Fetching %s data via yfinance", ticker)
    data = yf.download(ticker, period="max", progress=False)

    # Cache for next time
    if rel_path:
        cache_path = _ROOT / rel_path
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        data.to_csv(cache_path)
        logger.info("Cached API data to %s", cache_path)

    return data
