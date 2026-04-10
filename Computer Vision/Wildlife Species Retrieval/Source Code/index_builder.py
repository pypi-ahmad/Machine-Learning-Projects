"""Wildlife Species Retrieval — index builder CLI.

Build or rebuild the embedding index from a wildlife image directory.

Usage::

    python index_builder.py --image-dir data/animals
    python index_builder.py --image-dir data/animals --force --batch-size 64
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import WildlifeConfig, load_config
from controller import WildlifeController

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Build the wildlife embedding index"
    )
    ap.add_argument("--image-dir", required=True,
                    help="Directory of images (species sub-folders)")
    ap.add_argument("--index-path", default=None,
                    help="Override index save path")
    ap.add_argument("--backbone", default=None,
                    help="Backbone model name")
    ap.add_argument("--batch-size", type=int, default=32,
                    help="Embedding batch size")
    ap.add_argument("--force", action="store_true",
                    help="Rebuild even if index exists")
    ap.add_argument("--config", default=None,
                    help="Path to config YAML")
    ap.add_argument("--device", default=None,
                    help="Compute device (cpu/cuda)")
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    cfg = load_config(args.config)
    if args.index_path:
        cfg.index_path = args.index_path
    if args.backbone:
        cfg.backbone = args.backbone
    if args.device:
        cfg.device = args.device

    ctrl = WildlifeController(cfg)
    ctrl.load()

    t0 = time.perf_counter()
    idx = ctrl.build_index(
        args.image_dir, batch_size=args.batch_size, force=args.force
    )
    elapsed = time.perf_counter() - t0

    logger.info("Done in %.1fs  —  %s", elapsed, idx.summary())
    ctrl.close()


if __name__ == "__main__":
    main()
