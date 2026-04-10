#!/usr/bin/env python3
"""Unified dataset download & preparation script.

Every project in this repo can bootstrap its dataset automatically via this
script.  It wraps :class:`utils.datasets.DatasetResolver` with:

- ``data/<project>/raw/``      — original downloaded files
- ``data/<project>/processed/`` — prepared / split data ready for training
- ``data/<project>/dataset_info.json`` — provenance metadata

Usage::

    # Download + prepare a single project
    python scripts/download_data.py emotion_recognition

    # Force re-download (ignores cache)
    python scripts/download_data.py emotion_recognition --force-download

    # Download all registered datasets
    python scripts/download_data.py --all

    # List available datasets and their status
    python scripts/download_data.py --list

    # Download-only mode (skip prepare step)
    python scripts/download_data.py emotion_recognition --download-only

Source preference order:
  1. Hugging Face Datasets (programmatic, streaming)
  2. Direct public ZIP / TAR URLs
  3. Roboflow public exports (YOLO-labeled CV data)
  4. Kaggle (requires credentials — fails clearly if missing)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from utils.datasets import DatasetResolver

log = logging.getLogger("download_data")
logging.basicConfig(
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    level=logging.INFO,
)


# ---------------------------------------------------------------------------
# ensure_dataset — the single entry point
# ---------------------------------------------------------------------------
def ensure_dataset(
    project_key: str,
    *,
    force: bool = False,
    download_only: bool = False,
) -> Path:
    """Download and prepare a dataset, returning the data root.

    Idempotent: skips download if ``data/<key>/raw/.ready`` exists
    (unless *force* is True).

    Parameters
    ----------
    project_key : str
        Registry key matching a YAML config or built-in entry.
    force : bool
        Re-download even if data is already cached.
    download_only : bool
        Skip the preparation step; only download raw data.

    Returns
    -------
    Path
        The project data root (``data/<project_key>/``).
    """
    resolver = DatasetResolver()

    # Resolve triggers download if needed
    data_path = resolver.resolve(project_key, force=force)

    # Ensure raw/ and processed/ subdirectories exist
    raw_dir = data_path / "raw"
    processed_dir = data_path / "processed"
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    # Write/update dataset_info.json at the project data root
    info_path = data_path / "dataset_info.json"
    if not info_path.exists():
        entry = resolver.registry.get(project_key, {})
        info = {
            "dataset_key": project_key,
            "source_type": entry.get("type", "unknown"),
            "description": entry.get("description", ""),
            "downloaded_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "data_root": str(data_path),
            "raw_dir": str(raw_dir),
            "processed_dir": str(processed_dir),
        }
        for k in ("id", "url", "workspace", "project", "version"):
            if k in entry:
                info[f"source_{k}"] = entry[k]
        info_path.write_text(json.dumps(info, indent=2), encoding="utf-8")
        log.info("[%s] Wrote dataset_info.json", project_key)

    if not download_only:
        log.info("[%s] Data ready at %s", project_key, data_path)

    return data_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download and prepare datasets for CV projects",
        epilog=(
            "Examples:\n"
            "  python scripts/download_data.py emotion_recognition\n"
            "  python scripts/download_data.py --all\n"
            "  python scripts/download_data.py emotion_recognition --force-download\n"
            "  python scripts/download_data.py --list\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "project", nargs="?",
        help="Project key to download (e.g. emotion_recognition)",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Download all registered datasets",
    )
    parser.add_argument(
        "--list", action="store_true",
        help="List all datasets and their download status",
    )
    parser.add_argument(
        "--force-download", action="store_true",
        help="Force re-download even if data already exists",
    )
    parser.add_argument(
        "--download-only", action="store_true",
        help="Download raw data only; skip preparation step",
    )
    args = parser.parse_args()

    # -- List mode --
    if args.list:
        resolver = DatasetResolver()
        print(resolver.status())
        return

    # -- Validate args --
    if not args.project and not args.all:
        parser.print_help()
        print("\nError: specify a project key or use --all")
        sys.exit(1)

    resolver = DatasetResolver()

    # -- Collect keys --
    if args.all:
        keys = sorted(resolver.registry.keys())
    else:
        if args.project not in resolver.registry:
            available = ", ".join(sorted(resolver.registry.keys()))
            print(f"Error: '{args.project}' not found.\nAvailable: {available}")
            sys.exit(1)
        keys = [args.project]

    # -- Download loop --
    results: list[dict] = []
    for key in keys:
        entry = resolver.registry[key]
        ds_type = entry.get("type", "")

        # Skip local-only in --all mode (nothing to download)
        if args.all and ds_type == "local_only":
            continue

        print(f"\n{'=' * 60}")
        print(f"  {key}  ({ds_type})")
        print(f"{'=' * 60}")

        try:
            path = ensure_dataset(
                key,
                force=args.force_download,
                download_only=args.download_only,
            )
            results.append({"key": key, "ok": True, "path": str(path)})
            print(f"  => OK: {path}")
        except Exception as exc:
            results.append({"key": key, "ok": False, "error": str(exc)})
            print(f"  => FAILED: {exc}")

    # -- Summary --
    ok = sum(1 for r in results if r["ok"])
    failed = sum(1 for r in results if not r["ok"])
    print(f"\n{'=' * 60}")
    print(f"  SUMMARY: {ok} succeeded, {failed} failed, {len(results)} total")
    print(f"{'=' * 60}")

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
