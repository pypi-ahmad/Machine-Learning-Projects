"""Product Counterfeit Visual Checker — reference builder CLI.

Build the approved-reference embedding store from a directory of
product images organised into product sub-folders.

Usage:
    python reference_builder.py --image-dir data/grocery/processed/products
    python reference_builder.py --image-dir data/grocery/processed/products --force
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import CounterfeitConfig, load_config
from controller import CounterfeitController

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build the product reference store")
    ap.add_argument("--image-dir", required=True,
                    help="Directory of approved product images (product sub-folders)")
    ap.add_argument("--ref-path", default=None,
                    help="Override reference store save path")
    ap.add_argument("--backbone", default=None, help="Backbone model name")
    ap.add_argument("--batch-size", type=int, default=32, help="Embedding batch size")
    ap.add_argument("--force", action="store_true",
                    help="Rebuild even if reference store exists")
    ap.add_argument("--config", default=None, help="Path to config YAML/JSON")
    ap.add_argument("--device", default=None, help="Compute device (cpu/cuda)")
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    cfg = load_config(args.config) if args.config else CounterfeitConfig()
    if args.ref_path:
        cfg.reference_path = args.ref_path
    if args.backbone:
        cfg.backbone = args.backbone
    if args.device:
        cfg.device = args.device

    ctrl = CounterfeitController(cfg)
    ctrl.load()

    t0 = time.perf_counter()
    store = ctrl.build_references(args.image_dir, batch_size=args.batch_size,
                                  force=args.force)
    elapsed = time.perf_counter() - t0

    logger.info("Done in %.1fs  —  %s", elapsed, store.summary())
    ctrl.close()


if __name__ == "__main__":
    main()
