"""Exercise Rep Counter -- dataset download + evaluation."""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main(force_download: bool = False) -> None:
    """Download evaluation data and run exercise analysis evaluation."""
    import cv2

    from controller import ExerciseController
    from data_bootstrap import ensure_exercise_dataset

    data_dir = ensure_exercise_dataset(force=force_download)
    print(f"Dataset ready: {data_dir}")

    media_dir = data_dir / "processed" / "media"
    if not media_dir.exists():
        print("No processed media found -- skipping evaluation.")
        return

    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    images = sorted(
        p for p in media_dir.iterdir() if p.suffix.lower() in exts
    )
    videos = sorted(
        p for p in media_dir.iterdir()
        if p.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv"}
    )

    print(f"Found {len(images)} images, {len(videos)} videos")

    # Evaluate on images (pose detection check)
    if images:
        ctrl = ExerciseController()
        ctrl.load()
        detected = 0
        for img_path in images:
            frame = cv2.imread(str(img_path))
            if frame is None:
                continue
            r = ctrl.process(frame)
            if r.pose.detected:
                detected += 1
        ctrl.close()
        print(f"\nPose detection rate: {detected}/{len(images)} "
              f"({detected/len(images)*100:.1f}%)")

    # Evaluate on videos (rep counting)
    for vid_path in videos[:5]:  # limit to first 5 videos
        ctrl = ExerciseController()
        ctrl.load()
        cap = cv2.VideoCapture(str(vid_path))
        frames = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            ctrl.process(frame)
            frames += 1
        cap.release()
        print(f"  {vid_path.name}: {frames} frames, {ctrl.reps} reps detected")
        ctrl.close()

    results = {
        "total_images": len(images),
        "total_videos": len(videos),
        "pose_detection_images": detected if images else 0,
    }
    out_path = Path("eval_results.json")
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    force = "--force-download" in sys.argv
    main(force_download=force)
