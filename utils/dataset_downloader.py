"""Kaggle dataset downloader for the NLP Projects workspace.

Usage from any notebook:

    from utils.dataset_downloader import download_dataset

    # Download by project key (matches config/datasets.json)
    data_dir = download_dataset("e-commerce-clothing-reviews")

    # Then load files from data_dir:
    import pandas as pd
    df = pd.read_csv(data_dir / "Womens Clothing E-Commerce Reviews.csv")

The function is **idempotent** — if the data already exists locally it
returns immediately without re-downloading.

Requirements:
    pip install kaggle
    Set KAGGLE_USERNAME + KAGGLE_KEY in .env (or ~/.kaggle/kaggle.json).
"""

from __future__ import annotations

import json
import os
import zipfile
from pathlib import Path
from typing import Literal

from utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
_DATASETS_JSON = _WORKSPACE_ROOT / "config" / "datasets.json"
_DATA_ROOT = _WORKSPACE_ROOT / "data"


def _load_registry() -> dict:
    """Load the datasets registry from config/datasets.json."""
    with open(_DATASETS_JSON, "r", encoding="utf-8") as f:
        return json.load(f)


def _ensure_kaggle_credentials() -> None:
    """Try to load Kaggle credentials from .env or env vars.

    Supports three authentication methods (checked in order):
      1. KAGGLE_API_TOKEN env var  (newer KGAT_* token format, kaggle>=1.6)
      2. KAGGLE_USERNAME + KAGGLE_KEY env vars  (classic)
      3. ~/.kaggle/kaggle.json file
    """
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"

    # 1. Newer single-token auth (KGAT_* format)
    if os.environ.get("KAGGLE_API_TOKEN"):
        return  # kaggle>=1.6 picks this up automatically

    # 2. Classic username + key
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return  # already in env

    # Try loading from .env file at workspace root
    env_file = _WORKSPACE_ROOT / ".env"
    if env_file.exists():
        try:
            from dotenv import load_dotenv

            load_dotenv(env_file)
        except ImportError:
            pass

    if os.environ.get("KAGGLE_API_TOKEN"):
        return
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return

    # 3. kaggle.json on disk
    if kaggle_json.exists():
        return  # kaggle API will pick this up

    raise EnvironmentError(
        "Kaggle credentials not found. Set KAGGLE_API_TOKEN (KGAT_* token), "
        "or KAGGLE_USERNAME + KAGGLE_KEY in .env / env vars, "
        "or place kaggle.json in ~/.kaggle/. "
        "See https://www.kaggle.com/docs/api#getting-started-installation-&-authentication"
    )


def _extract_zips(directory: Path) -> None:
    """Extract all .zip files in *directory* and remove the archives."""
    for zp in list(directory.glob("*.zip")):
        logger.info("Extracting %s ...", zp.name)
        with zipfile.ZipFile(zp, "r") as zf:
            zf.extractall(directory)
        zp.unlink()
        logger.info("Extracted and removed %s", zp.name)


def _download_kaggle_dataset(slug: str, dest: Path) -> None:
    """Download a Kaggle *dataset* (owner/dataset-name) into *dest*."""
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    logger.info("Downloading dataset '%s' -> %s", slug, dest)
    api.dataset_download_files(slug, path=str(dest), unzip=False)
    _extract_zips(dest)


def _download_kaggle_competition(slug: str, dest: Path) -> None:
    """Download a Kaggle *competition* dataset into *dest*."""
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    logger.info("Downloading competition '%s' -> %s", slug, dest)
    api.competition_download_files(slug, path=str(dest))
    _extract_zips(dest)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def download_dataset(
    project_key: str,
    *,
    force: bool = False,
    dest: Path | str | None = None,
) -> Path:
    """Download the dataset for *project_key* and return the path to the raw data.

    Parameters
    ----------
    project_key : str
        Key matching an entry in ``config/datasets.json``
        (e.g. ``"e-commerce-clothing-reviews"``).
    force : bool
        If *True*, re-download even if data already exists.
    dest : Path | str | None
        Override destination directory.  Defaults to
        ``data/<project_key>/raw``.

    Returns
    -------
    Path
        The directory containing the downloaded (and extracted) files.
    """
    registry = _load_registry()
    if project_key not in registry:
        available = ", ".join(sorted(registry))
        raise KeyError(
            f"Unknown project key '{project_key}'. "
            f"Available keys: {available}"
        )

    entry = registry[project_key]
    slug: str = entry["slug"]
    dtype: Literal["dataset", "competition"] = entry.get("type", "dataset")

    raw_dir = Path(dest) if dest else _DATA_ROOT / project_key / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    # Idempotency — skip if we already have files (beyond .gitkeep)
    existing = [f for f in raw_dir.iterdir() if f.name != ".gitkeep"]
    if existing and not force:
        logger.info(
            "Data for '%s' already present (%d files). Use force=True to re-download.",
            project_key,
            len(existing),
        )
        return raw_dir

    _ensure_kaggle_credentials()

    if dtype == "competition":
        _download_kaggle_competition(slug, raw_dir)
    else:
        _download_kaggle_dataset(slug, raw_dir)

    # Log what we got
    files = list(raw_dir.rglob("*"))
    file_count = sum(1 for f in files if f.is_file() and f.name != ".gitkeep")
    logger.info("Done — %d file(s) in %s", file_count, raw_dir)
    return raw_dir


def download_all(*, force: bool = False) -> dict[str, Path]:
    """Download datasets for ALL projects. Returns {project_key: raw_dir}.

    Useful for a one-shot bulk download.
    """
    registry = _load_registry()
    results: dict[str, Path] = {}
    for key in sorted(registry):
        try:
            results[key] = download_dataset(key, force=force)
        except Exception as exc:
            logger.error("Failed to download '%s': %s", key, exc)
    return results


def list_projects() -> list[str]:
    """Return a sorted list of all available project keys."""
    return sorted(_load_registry())
