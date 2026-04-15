"""Logo Retrieval Brand Match — index builder script.
"""Logo Retrieval Brand Match — index builder script.

Build or update the embedding index from the dataset (or any directory
of logo images organised in brand sub-folders).

Usage::

    # Build index from auto-downloaded dataset
    python index_builder.py

    # Build from custom directory
    python index_builder.py --data path/to/logos/

    # Force re-download dataset
    python index_builder.py --force-download

    # Update an existing index with new images
    python index_builder.py --data new_logos/ --update
"""
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def _discover_logos(data_dir: Path) -> list[tuple[str, Path]]:
    """Find logo images organised as brand_name/image.ext."""
    entries: list[tuple[str, Path]] = []
    for brand_dir in sorted(data_dir.iterdir()):
        if not brand_dir.is_dir():
            continue
        brand = brand_dir.name
        for img_path in sorted(brand_dir.iterdir()):
            if img_path.suffix.lower() in _IMAGE_EXTS:
                entries.append((brand, img_path))
    return entries


def _build_index(args: argparse.Namespace) -> None:
    import cv2
    import numpy as np

    from config import LogoConfig, load_config
    from embedder import LogoEmbedder
    from index import LogoIndex

    cfg = load_config(args.config) if args.config else LogoConfig()

    # Resolve data directory
    if args.data:
        data_dir = Path(args.data)
    else:
        from data_bootstrap import ensure_logo_dataset
        raw = ensure_logo_dataset(force=args.force_download)
        # Look for Logos sub-directory (dataset structure)
        logos_dir = raw / "processed" / "brands"
        if logos_dir.exists():
            data_dir = logos_dir
        else:
            data_dir = raw

    print(f"Scanning logos in: {data_dir}")
    entries = _discover_logos(data_dir)
    if not entries:
        print("[ERROR] No brand/image folders found.")
        sys.exit(1)

    print(f"Found {len(entries)} logo image(s) across "
          f"{len(set(b for b, _ in entries))} brand(s)")

    # Load or create index
    idx_path = args.index or cfg.index_path
    if args.update and Path(idx_path).exists():
        print(f"Loading existing index from {idx_path} (update mode)")
        idx = LogoIndex.load(idx_path)
        existing = set(idx._paths)
        entries = [(b, p) for b, p in entries if str(p) not in existing]
        if not entries:
            print("No new images to add -- index is up to date.")
            return
        print(f"  Adding {len(entries)} new image(s)")
    else:
        idx = LogoIndex()

    # Embed
    embedder = LogoEmbedder(cfg)
    embedder.load()

    batch_size = 32
    for start in range(0, len(entries), batch_size):
        batch_entries = entries[start:start + batch_size]
        images = []
        paths = []
        brands = []
        for brand, img_path in batch_entries:
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"  [SKIP] Cannot read: {img_path}")
                continue
            images.append(img)
            paths.append(str(img_path))
            brands.append(brand)

        if not images:
            continue

        embeddings = embedder.embed_batch(images)
        idx.add_batch(embeddings, paths, brands)

        done = min(start + batch_size, len(entries))
        print(f"  Embedded [{done}/{len(entries)}]")

    embedder.close()

    # Save
    saved_path = idx.save(idx_path)
    summary = idx.summary()
    print(f"\nIndex saved to: {saved_path}")
    print(f"  Total entries:   {summary['total_entries']}")
    print(f"  Embedding dim:   {summary['embedding_dim']}")
    print(f"  Unique brands:   {summary['num_brands']}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Logo Retrieval Brand Match -- build embedding index",
    )
    parser.add_argument("--data", type=str, default=None,
                        help="Directory of brand/image sub-folders")
    parser.add_argument("--config", type=str, default=None,
                        help="Config JSON/YAML file")
    parser.add_argument("--index", type=str, default=None,
                        help="Output index path (default: from config)")
    parser.add_argument("--update", action="store_true",
                        help="Add new images to existing index")
    parser.add_argument("--force-download", action="store_true",
                        help="Force re-download dataset")
    args = parser.parse_args()
    _build_index(args)


if __name__ == "__main__":
    main()
