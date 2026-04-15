"""Train / Evaluate Blink Headpose Analyzer.

MediaPipe is pre-trained — this script downloads an evaluation
dataset and runs the blink + head-pose pipeline to report
EAR statistics and blink counts.

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

from data_bootstrap import ensure_blink_headpose_dataset

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


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description="Evaluate Blink Headpose Analyzer",
    )
    ap.add_argument("--data", type=str, default=None,
                    help="Path to evaluation data (images/videos)")
    ap.add_argument("--max-frames", type=int, default=500,
                    help="Max frames per video to process")
    ap.add_argument("--force-download", action="store_true",
                    help="Force re-download dataset")
    args = ap.parse_args(argv)

    if args.data is None:
        data_path = ensure_blink_headpose_dataset(force=args.force_download)
        data_dir = str(data_path)
        print(f"[INFO] Resolved dataset -> {data_path}")
    else:
        data_dir = args.data

    print(f"[INFO] Dataset at {data_dir}")
    print("[INFO] MediaPipe is pre-trained. Running evaluation...")

    try:
        import cv2

        from analyzer import AnalyzerPipeline
        from config import AnalyzerConfig

        cfg = AnalyzerConfig()
        pipeline = AnalyzerPipeline(cfg)
        pipeline.load()

        data_root = Path(data_dir)
        images, videos = _collect_media_files(data_root)

        print(f"[INFO] Found {len(images)} images, {len(videos)} videos")

        total_frames = 0
        total_blinks = 0
        ear_values: list[float] = []
        yaw_values: list[float] = []
        pitch_values: list[float] = []
        faces_detected = 0

        # Process images
        for img_path in images:
            frame = cv2.imread(str(img_path))
            if frame is None:
                continue
            result = pipeline.process(frame)
            total_frames += 1
            if result.face_detected:
                faces_detected += 1
                ear_values.append(result.blink.ear)
                yaw_values.append(result.head_pose.yaw)
                pitch_values.append(result.head_pose.pitch)

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
                    faces_detected += 1
                    ear_values.append(result.blink.ear)
                    yaw_values.append(result.head_pose.yaw)
                    pitch_values.append(result.head_pose.pitch)

            total_blinks += pipeline.blink_counter._total_blinks
            cap.release()

        if total_frames == 0:
            print("[WARN] No frames processed")
            return

        avg_ear = sum(ear_values) / len(ear_values) if ear_values else 0
        avg_yaw = sum(abs(y) for y in yaw_values) / len(yaw_values) if yaw_values else 0
        avg_pitch = sum(abs(p) for p in pitch_values) / len(pitch_values) if pitch_values else 0

        print(f"\n[SUMMARY]")
        print(f"  Total frames:    {total_frames}")
        print(f"  Faces detected:  {faces_detected}")
        print(f"  Avg EAR:         {avg_ear:.3f}")
        print(f"  Avg |Yaw|:       {avg_yaw:.1f}°")
        print(f"  Avg |Pitch|:     {avg_pitch:.1f}°")
        print(f"  Total blinks:    {total_blinks}")

        out_path = Path(__file__).parent / "runs" / "eval_results.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps({
                "total_frames": total_frames,
                "faces_detected": faces_detected,
                "avg_ear": round(avg_ear, 4),
                "avg_abs_yaw": round(avg_yaw, 2),
                "avg_abs_pitch": round(avg_pitch, 2),
                "total_blinks": total_blinks,
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
