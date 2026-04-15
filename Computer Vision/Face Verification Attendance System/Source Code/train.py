"""Train / Evaluate Face Verification Attendance System.

InsightFace models are pre-trained. This script prepares the
dataset and evaluates the enrollment + verification pipeline
on a multi-identity face dataset.

Usage::

    python train.py
    python train.py --data path/to/faces
    python train.py --force-download
    python train.py --max-identities 50 --threshold 0.45
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from data_bootstrap import ensure_face_attendance_dataset


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description="Train/Evaluate Face Verification Attendance System",
    )
    ap.add_argument("--data", type=str, default=None,
                    help="Path to face dataset (ImageFolder layout)")
    ap.add_argument("--max-identities", type=int, default=50,
                    help="Max identities to evaluate")
    ap.add_argument("--threshold", type=float, default=0.45,
                    help="Cosine similarity threshold")
    ap.add_argument("--force-download", action="store_true",
                    help="Force re-download dataset")
    args = ap.parse_args(argv)

    if args.data is None:
        data_path = ensure_face_attendance_dataset(force=args.force_download)
        data_dir = str(data_path / "processed" / "identities")
        print(f"[INFO] Prepared dataset -> {data_path}")
    else:
        data_dir = args.data

    print(f"[INFO] Dataset ready at {data_dir}")
    print("[INFO] InsightFace models are pre-trained. Running enrollment/verification eval...")

    try:
        from config import FaceAttendanceConfig
        from parser import FaceAttendancePipeline

        cfg = FaceAttendanceConfig()
        cfg.similarity_threshold = args.threshold

        pipeline = FaceAttendancePipeline(cfg)
        pipeline.load()

        if not pipeline.embedder.ready:
            print("[ERROR] InsightFace not available -- cannot evaluate")
            return

        data_root = Path(data_dir)

        # Find identity directories (ImageFolder layout)
        identity_dirs = _find_identity_dirs(data_root)
        if not identity_dirs:
            print("[WARN] No identity subdirectories found. Expected ImageFolder layout.")
            return

        print(f"[INFO] Found {len(identity_dirs)} identities")

        # Filter to identities with at least 2 images
        # Select identities that have enough images and can actually enroll.
        usable = []
        for id_dir in identity_dirs:
            imgs = _find_image_files(id_dir)
            if len(imgs) < 2:
                continue
            if not pipeline.enroll_single(id_dir.name, str(imgs[0])):
                print(f"[WARN] Skipping '{id_dir.name}' -- enrollment failed")
                continue
            usable.append((id_dir, imgs))
            if len(usable) >= args.max_identities:
                break

        if not usable:
            print("[WARN] No identities with >= 2 images found")
            return

        print(f"[INFO] Evaluating {len(usable)} identities with >= 2 images")

        # Verify on remaining images
        correct = 0
        total = 0
        results = []

        for id_dir, imgs in usable:
            name = id_dir.name
            for img_path in imgs[1:3]:  # test on up to 2 images
                out = pipeline.process(
                    __import__("cv2").imread(str(img_path)),
                )
                for m in out.matches:
                    total += 1
                    is_correct = m.identity == name
                    if is_correct:
                        correct += 1
                    results.append({
                        "identity": name,
                        "predicted": m.identity,
                        "similarity": round(m.similarity, 4),
                        "correct": is_correct,
                        "file": img_path.name,
                    })

        # Summary
        acc = correct / total * 100 if total else 0
        print(f"\n[SUMMARY] {len(usable)} identities evaluated")
        print(f"  Verification accuracy: {correct}/{total} ({acc:.1f}%)")
        print(f"  Threshold:             {args.threshold}")

        # Save results
        out_path = Path(__file__).parent / "runs" / "eval_results.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps({
                "identities_evaluated": len(usable),
                "threshold": args.threshold,
                "correct": correct,
                "total": total,
                "accuracy": round(acc, 2),
                "per_sample": results,
            }, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"  Results -> {out_path}")

    except ImportError as exc:
        print(f"[WARN] Could not run evaluation: {exc}")
        print("[INFO] Install: pip install insightface onnxruntime")
    except Exception as exc:
        print(f"[ERROR] Evaluation failed: {exc}")
        raise


def _find_identity_dirs(data_root: Path) -> list[Path]:
    """Find identity subdirectories, trying one level deep."""
    dirs = [d for d in sorted(data_root.iterdir()) if d.is_dir()]
    if dirs:
        # Check if these are identity dirs (contain images)
        for d in dirs:
            imgs = _find_image_files(d)
            if imgs:
                return dirs
    # Try one level deeper
    for sub in sorted(data_root.iterdir()):
        if sub.is_dir():
            deeper = [d for d in sorted(sub.iterdir()) if d.is_dir()]
            if deeper:
                return deeper
    return []


def _find_image_files(identity_dir: Path) -> list[Path]:
    files = []
    for child in sorted(identity_dir.iterdir()):
        if child.is_file() and child.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
            files.append(child)
    return files


if __name__ == "__main__":
    main()
