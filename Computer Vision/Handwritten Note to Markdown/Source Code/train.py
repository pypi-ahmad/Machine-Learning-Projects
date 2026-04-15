"""Train / Evaluate Handwritten Note to Markdown.

This project uses TrOCR (pre-trained) and projection-profile
line segmentation, so no training is needed.  This script
downloads the dataset and evaluates the full pipeline on
sample images.

Usage::

    python train.py
    python train.py --data path/to/dataset
    python train.py --force-download
    python train.py --max-samples 30
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from data_bootstrap import ensure_handwriting_dataset


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description="Train/Evaluate Handwritten Note to Markdown",
    )
    ap.add_argument("--data", type=str, default=None,
                    help="Path to handwriting dataset")
    ap.add_argument("--max-samples", type=int, default=20,
                    help="Max samples to evaluate")
    ap.add_argument("--force-download", action="store_true",
                    help="Force re-download dataset")
    ap.add_argument("--model", type=str, default=None,
                    help="TrOCR model name override")
    args = ap.parse_args(argv)

    if args.data is None:
        data_path = ensure_handwriting_dataset(force=args.force_download)
        data_dir = str(data_path)
        print(f"[INFO] Resolved dataset -> {data_path}")
    else:
        data_dir = args.data

    print(f"[INFO] Dataset ready at {data_dir}")
    print("[INFO] TrOCR is pre-trained; evaluating OCR on handwriting images.")

    try:
        from modern import HandwrittenNoteToMarkdown

        proj = HandwrittenNoteToMarkdown()
        setup_kwargs: dict = {}
        if args.model:
            setup_kwargs["model"] = args.model
        proj.setup(**setup_kwargs)

        data_root = Path(data_dir)
        images = sorted(
            list(data_root.rglob("*.jpg"))
            + list(data_root.rglob("*.png"))
            + list(data_root.rglob("*.jpeg"))
        )
        if not images:
            print("[WARN] No images found in dataset directory.")
            return

        results = []
        for img_path in images[: args.max_samples]:
            out = proj.predict(str(img_path))
            result = out["result"]
            report = out["report"]
            entry = {
                "file": img_path.name,
                "num_lines": result.num_lines,
                "mean_confidence": round(result.mean_confidence, 3),
                "valid": report.valid,
                "warnings": len(report.warnings),
                "preview": result.plain_text[:80].replace("\n", " "),
            }
            results.append(entry)
            nonempty = sum(1 for ln in result.lines if ln.text.strip())
            print(
                f"  {img_path.name}: "
                f"{nonempty}/{result.num_lines} lines, "
                f"conf={result.mean_confidence:.2f}, "
                f"text='{result.plain_text[:50].strip()}'..."
            )

        # Summary
        out_path = Path(__file__).parent / "runs" / "eval_results.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

        avg_lines = sum(r["num_lines"] for r in results) / max(len(results), 1)
        avg_conf = sum(r["mean_confidence"] for r in results) / max(len(results), 1)
        print(f"\n[INFO] Evaluated {len(results)} notes")
        print(f"[INFO] Avg lines per note: {avg_lines:.1f}")
        print(f"[INFO] Avg confidence: {avg_conf:.2f}")
        print(f"[INFO] Results saved to {out_path}")

    except ImportError as e:
        print(f"[WARN] Could not run evaluation: {e}")
        print("[INFO] Install deps: pip install transformers torch pillow")


if __name__ == "__main__":
    main()
