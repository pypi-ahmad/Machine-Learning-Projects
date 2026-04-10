"""Train / Evaluate Driver Drowsiness Monitor.

MediaPipe is pre-trained — this script downloads a drowsiness
evaluation dataset and runs the detection pipeline to measure
EAR/MAR metrics and alert statistics.

Usage::

    python train.py
    python train.py --data path/to/videos
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
        description="Evaluate Driver Drowsiness Monitor",
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
            "driver_drowsiness_monitor", force=args.force_download,
        )
        data_dir = str(data_path)
        print(f"[INFO] Resolved dataset → {data_path}")
    else:
        data_dir = args.data

    print(f"[INFO] Dataset at {data_dir}")
    print("[INFO] MediaPipe is pre-trained. Running drowsiness eval...")

    try:
        import cv2

        from config import DrowsinessConfig
        from parser import DrowsinessPipeline

        cfg = DrowsinessConfig()
        pipeline = DrowsinessPipeline(cfg)
        pipeline.load()

        data_root = Path(data_dir)

        # Collect evaluation files
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
        total_blinks = 0
        total_yawns = 0
        total_alerts = 0
        ear_values: list[float] = []
        mar_values: list[float] = []

        # Process images
        for img_path in images:
            frame = cv2.imread(str(img_path))
            if frame is None:
                continue
            result = pipeline.process(frame)
            total_frames += 1
            if result.face_detected:
                ear_values.append(result.blink.ear)
                mar_values.append(result.yawn.mar)
            total_alerts += len(result.alerts)

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

                if result.face_detected:
                    ear_values.append(result.blink.ear)
                    mar_values.append(result.yawn.mar)
                total_alerts += len(result.alerts)

            total_blinks += pipeline.blink_tracker._total_blinks
            total_yawns += pipeline.yawn_tracker._total_yawns
            cap.release()

        if total_frames == 0:
            print("[WARN] No frames processed")
            return

        avg_ear = sum(ear_values) / len(ear_values) if ear_values else 0
        avg_mar = sum(mar_values) / len(mar_values) if mar_values else 0

        print(f"\n[SUMMARY]")
        print(f"  Total frames:    {total_frames}")
        print(f"  Faces detected:  {len(ear_values)}")
        print(f"  Avg EAR:         {avg_ear:.3f}")
        print(f"  Avg MAR:         {avg_mar:.3f}")
        print(f"  Total blinks:    {total_blinks}")
        print(f"  Total yawns:     {total_yawns}")
        print(f"  Total alerts:    {total_alerts}")

        out_path = Path(__file__).parent / "runs" / "eval_results.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps({
                "total_frames": total_frames,
                "faces_detected": len(ear_values),
                "avg_ear": round(avg_ear, 4),
                "avg_mar": round(avg_mar, 4),
                "total_blinks": total_blinks,
                "total_yawns": total_yawns,
                "total_alerts": total_alerts,
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
