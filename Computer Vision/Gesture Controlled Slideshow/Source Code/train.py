"""Train / Evaluate Gesture Controlled Slideshow.

MediaPipe Hands is pre-trained — this script downloads a sample
hand-gesture dataset and runs the gesture recognition pipeline
to report gesture distribution and detection statistics.

Usage::

    python train.py
    python train.py --data path/to/hand_images
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
        description="Evaluate Gesture Controlled Slideshow",
    )
    ap.add_argument("--data", type=str, default=None,
                    help="Path to evaluation data (hand images/videos)")
    ap.add_argument("--max-frames", type=int, default=500,
                    help="Max frames per video to process")
    ap.add_argument("--force-download", action="store_true",
                    help="Force re-download dataset")
    args = ap.parse_args(argv)

    if args.data is None:
        data_path = DatasetResolver().resolve(
            "gesture_controlled_slideshow", force=args.force_download,
        )
        data_dir = str(data_path)
        print(f"[INFO] Resolved dataset → {data_path}")
    else:
        data_dir = args.data

    print(f"[INFO] Dataset at {data_dir}")
    print("[INFO] MediaPipe is pre-trained. Running gesture evaluation...")

    try:
        import cv2

        from config import GestureConfig
        from controller import SlideshowController

        cfg = GestureConfig()
        controller = SlideshowController(cfg)
        controller.load()

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
        hands_detected = 0
        gesture_counts: dict[str, int] = {}
        action_counts: dict[str, int] = {}

        for img_path in images:
            frame = cv2.imread(str(img_path))
            if frame is None:
                continue
            result = controller.process(frame)
            total_frames += 1
            if result.hand_detected:
                hands_detected += 1
                g = result.gesture.gesture
                gesture_counts[g] = gesture_counts.get(g, 0) + 1
            if result.debounced.triggered:
                a = result.debounced.action
                action_counts[a] = action_counts.get(a, 0) + 1

        for vid_path in videos:
            controller.reset()
            cap = cv2.VideoCapture(str(vid_path))
            if not cap.isOpened():
                continue

            frame_count = 0
            while frame_count < args.max_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                result = controller.process(frame)
                total_frames += 1
                frame_count += 1
                if result.hand_detected:
                    hands_detected += 1
                    g = result.gesture.gesture
                    gesture_counts[g] = gesture_counts.get(g, 0) + 1
                if result.debounced.triggered:
                    a = result.debounced.action
                    action_counts[a] = action_counts.get(a, 0) + 1

            cap.release()

        if total_frames == 0:
            print("[WARN] No frames processed")
            return

        print(f"\n[SUMMARY]")
        print(f"  Total frames:      {total_frames}")
        print(f"  Hands detected:    {hands_detected}")
        print(f"  Gesture distribution:")
        for g, cnt in sorted(gesture_counts.items(), key=lambda x: -x[1]):
            pct = 100.0 * cnt / hands_detected if hands_detected else 0
            print(f"    {g:15s}: {cnt:5d}  ({pct:5.1f}%)")
        if action_counts:
            print(f"  Actions triggered:")
            for a, cnt in sorted(action_counts.items(), key=lambda x: -x[1]):
                print(f"    {a:15s}: {cnt}")

        out_path = Path(__file__).parent / "runs" / "eval_results.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps({
                "total_frames": total_frames,
                "hands_detected": hands_detected,
                "gesture_counts": gesture_counts,
                "action_counts": action_counts,
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
