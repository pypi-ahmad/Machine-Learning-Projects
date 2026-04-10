"""Train / Evaluate Exam Sheet Parser.

This project uses PaddleOCR (pre-trained) and rule-based layout
parsing, so no training is needed.  This script downloads the
dataset and evaluates the full pipeline on sample images.

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

from utils.datasets import DatasetResolver


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description="Train/Evaluate Exam Sheet Parser",
    )
    ap.add_argument("--data", type=str, default=None,
                    help="Path to exam sheet dataset")
    ap.add_argument("--max-samples", type=int, default=20,
                    help="Max samples to evaluate")
    ap.add_argument("--force-download", action="store_true",
                    help="Force re-download dataset")
    args = ap.parse_args(argv)

    if args.data is None:
        data_path = DatasetResolver().resolve(
            "exam_sheet_parser", force=args.force_download,
        )
        data_dir = str(data_path)
        print(f"[INFO] Resolved dataset → {data_path}")
    else:
        data_dir = args.data

    print(f"[INFO] Dataset ready at {data_dir}")
    print("[INFO] PaddleOCR is pre-trained; evaluating layout parsing on images.")

    try:
        from modern import ExamSheetParserModern

        proj = ExamSheetParserModern()
        proj.setup()

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
                "num_blocks": result.num_blocks,
                "num_questions": result.num_questions,
                "total_marks": result.total_marks,
                "mean_confidence": round(result.mean_confidence, 3),
                "valid": report.valid,
                "headings": result.headings[:3],
            }
            results.append(entry)
            print(
                f"  {img_path.name}: {result.num_blocks} blocks, "
                f"{result.num_questions} questions, "
                f"conf={result.mean_confidence:.2f}"
            )

        # Summary
        total = len(results)
        avg_blocks = sum(r["num_blocks"] for r in results) / max(total, 1)
        avg_q = sum(r["num_questions"] for r in results) / max(total, 1)
        avg_conf = sum(r["mean_confidence"] for r in results) / max(total, 1)

        print(f"\n[SUMMARY] {total} images evaluated")
        print(f"  avg blocks/image:    {avg_blocks:.1f}")
        print(f"  avg questions/image: {avg_q:.1f}")
        print(f"  avg confidence:      {avg_conf:.3f}")

        out_path = Path("eval_results.json")
        out_path.write_text(
            json.dumps(results, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"  Results → {out_path}")

    except Exception as exc:
        print(f"[ERROR] Evaluation failed: {exc}")
        raise


if __name__ == "__main__":
    main()
