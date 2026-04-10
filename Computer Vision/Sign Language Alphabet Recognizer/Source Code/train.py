"""Sign Language Alphabet Recognizer — training / evaluation entry point.

Delegates to :mod:`trainer` for the full pipeline:
  dataset download → feature extraction → training → evaluation.
"""

from __future__ import annotations

import argparse
import sys


def main() -> None:
    p = argparse.ArgumentParser(description="Train the sign-language classifier")
    p.add_argument("--force-download", action="store_true", help="Re-download dataset")
    p.add_argument("--model-out", default="model/sign_lang_clf.pkl", help="Model save path")
    p.add_argument("--test-size", type=float, default=0.2, help="Test split ratio")
    p.add_argument("--max-iter", type=int, default=500, help="MLP max iterations")
    p.add_argument(
        "--max-images-per-class", type=int, default=0,
        help="Limit images per class (0 = all)",
    )
    args = p.parse_args()

    from trainer import main as trainer_main

    trainer_main(
        force_download=args.force_download,
        model_out=args.model_out,
        test_size=args.test_size,
        max_iter=args.max_iter,
        max_images_per_class=args.max_images_per_class,
    )


if __name__ == "__main__":
    main()
