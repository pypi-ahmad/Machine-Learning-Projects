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
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from data_bootstrap import ensure_gesture_dataset

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}


def _collect_media_files(data_dir: Path) -> tuple[list[Path], list[Path]]:
    media_dir = data_dir / "processed" / "media"
    search_root = media_dir if media_dir.exists() else data_dir
    images = sorted(
        f for f in search_root.rglob("*")
        if f.suffix.lower() in IMAGE_EXTS
    )
    videos = sorted(
        f for f in search_root.rglob("*")
        if f.suffix.lower() in VIDEO_EXTS
    )
    return images, videos


def _load_manifest_labels(data_dir: Path) -> dict[Path, str]:
    manifest_path = data_dir / "processed" / "manifest.csv"
    if not manifest_path.exists():
        return {}

    labels: dict[Path, str] = {}
    with open(manifest_path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            media_rel = row.get("media_path", "")
            label = row.get("expected_gesture", "")
            if not media_rel or not label:
                continue
            labels[data_dir / media_rel] = label
    return labels


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
        data_path = ensure_gesture_dataset(force=args.force_download)
        data_dir = str(data_path)
        print(f"[INFO] Resolved dataset -> {data_path}")
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
        images, videos = _collect_media_files(data_root)
        manifest_labels = _load_manifest_labels(data_root)

        print(f"[INFO] Found {len(images)} images, {len(videos)} videos")

        total_frames = 0
        hands_detected = 0
        gesture_counts: dict[str, int] = {}
        action_counts: dict[str, int] = {}
        labeled_frames = 0
        labeled_correct = 0
        expected_counts: dict[str, int] = {}

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
                expected = manifest_labels.get(img_path)
                if expected:
                    labeled_frames += 1
                    expected_counts[expected] = expected_counts.get(expected, 0) + 1
                    if g == expected:
                        labeled_correct += 1
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
                    expected = manifest_labels.get(vid_path)
                    if expected:
                        labeled_frames += 1
                        expected_counts[expected] = expected_counts.get(expected, 0) + 1
                        if g == expected:
                            labeled_correct += 1
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
        if labeled_frames:
            accuracy = labeled_correct / labeled_frames
            print(f"  Gesture label acc: {accuracy:.3f} ({labeled_correct}/{labeled_frames})")
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
                "expected_counts": expected_counts,
                "labeled_frames": labeled_frames,
                "labeled_correct": labeled_correct,
                "gesture_label_accuracy": round(labeled_correct / labeled_frames, 4) if labeled_frames else None,
            }, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"  Results -> {out_path}")

    except ImportError as exc:
        print(f"[WARN] Could not run evaluation: {exc}")
        print("[INFO] Install: pip install mediapipe opencv-python numpy")
    except Exception as exc:
        print(f"[ERROR] Evaluation failed: {exc}")
        raise


if __name__ == "__main__":
    main()
