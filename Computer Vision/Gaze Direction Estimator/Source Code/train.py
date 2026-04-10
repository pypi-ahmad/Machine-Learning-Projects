"""Train / Evaluate Gaze Direction Estimator.

MediaPipe is pre-trained — this script downloads an evaluation
dataset and runs the gaze pipeline to report direction
distribution and ratio statistics.

Usage::

    python train.py
    python train.py --data path/to/images
    python train.py --force-download
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils.datasets import DatasetResolver

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description="Evaluate Gaze Direction Estimator",
    )
    ap.add_argument("--data", type=str, default=None,
                    help="Path to evaluation data (images/videos)")
    ap.add_argument("--max-frames", type=int, default=500,
                    help="Max frames per video to process")
    ap.add_argument("--force-download", action="store_true",
                    help="Force re-download dataset")
    args = ap.parse_args(argv)

    if args.data is None:
        data_path = DatasetResolver().resolve(
            "gaze_direction_estimator", force=args.force_download,
        )
        data_dir = str(data_path)
        print(f"[INFO] Resolved dataset → {data_path}")
    else:
        data_dir = args.data

    print(f"[INFO] Dataset at {data_dir}")
    print("[INFO] MediaPipe is pre-trained. Running gaze evaluation...")

    try:
        import cv2

        from analyzer import GazePipeline
        from config import GazeConfig

        cfg = GazeConfig(enable_smoothing=False)  # raw for evaluation
        pipeline = GazePipeline(cfg)
        pipeline.load()

        data_root = Path(data_dir)

        images = sorted(
            f for f in data_root.rglob("*")
            if f.suffix.lower() in IMAGE_EXTS
        )
        videos = sorted(
            f for f in data_root.rglob("*")
            if f.suffix.lower() in VIDEO_EXTS
        )

        print(f"[INFO] Found {len(images)} images, {len(videos)} videos")

        total_frames = 0
        faces_detected = 0
        direction_counts: dict[str, int] = {}
        h_ratios: list[float] = []
        v_ratios: list[float] = []

        # Process images
        for img_path in images:
            frame = cv2.imread(str(img_path))
            if frame is None:
                continue
            result = pipeline.process(frame)
            total_frames += 1
            if result.face_detected and result.iris.detected:
                faces_detected += 1
                d = result.direction
                direction_counts[d] = direction_counts.get(d, 0) + 1
                h_ratios.append(result.raw_gaze.h_ratio)
                v_ratios.append(result.raw_gaze.v_ratio)

        # Process videos
        for vid_path in videos:
            pipeline.reset()
            cap = cv2.VideoCapture(str(vid_path))
            if not cap.isOpened():
                continue

            frame_count = 0
            while frame_count < args.max_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                result = pipeline.process(frame)
                total_frames += 1
                frame_count += 1
                if result.face_detected and result.iris.detected:
                    faces_detected += 1
                    d = result.direction
                    direction_counts[d] = direction_counts.get(d, 0) + 1
                    h_ratios.append(result.raw_gaze.h_ratio)
                    v_ratios.append(result.raw_gaze.v_ratio)

            cap.release()

        if total_frames == 0:
            print("[WARN] No frames processed")
            return

        avg_h = sum(h_ratios) / len(h_ratios) if h_ratios else 0
        avg_v = sum(v_ratios) / len(v_ratios) if v_ratios else 0

        print(f"\n[SUMMARY]")
        print(f"  Total frames:      {total_frames}")
        print(f"  Faces w/ iris:     {faces_detected}")
        print(f"  Avg H ratio:       {avg_h:.3f}")
        print(f"  Avg V ratio:       {avg_v:.3f}")
        print(f"  Direction distribution:")
        for d in ["CENTER", "LEFT", "RIGHT", "UP", "DOWN"]:
            cnt = direction_counts.get(d, 0)
            pct = 100.0 * cnt / faces_detected if faces_detected else 0
            print(f"    {d:8s}: {cnt:5d}  ({pct:5.1f}%)")

        out_path = Path(__file__).parent / "runs" / "eval_results.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps({
                "total_frames": total_frames,
                "faces_detected": faces_detected,
                "avg_h_ratio": round(avg_h, 4),
                "avg_v_ratio": round(avg_v, 4),
                "direction_counts": direction_counts,
            }, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"  Results → {out_path}")

    except ImportError as exc:
        print(f"[WARN] Could not run evaluation: {exc}")
        print("[INFO] Install: pip install mediapipe opencv-python numpy")
    except Exception as exc:
        print(f"[ERROR] Evaluation failed: {exc}")
        raise


if __name__ == "__main__":
    main()
