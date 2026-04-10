"""Yoga Pose Correction Coach — dataset download + evaluation."""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main(force_download: bool = False) -> None:
    """Download evaluation data and run pose classification evaluation."""
    import cv2

    from controller import YogaCoachController
    from data_bootstrap import ensure_yoga_dataset

    data_dir = ensure_yoga_dataset(force=force_download)
    print(f"Dataset ready: {data_dir}")

    media_dir = data_dir / "processed" / "media"
    if not media_dir.exists():
        print("No processed media found — skipping evaluation.")
        return

    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    images = sorted(
        p for p in media_dir.iterdir() if p.suffix.lower() in exts
    )
    if not images:
        print("No images in processed/media — skipping evaluation.")
        return

    print(f"Evaluating on {len(images)} images …")

    ctrl = YogaCoachController()
    ctrl.load()

    pose_dist: dict[str, int] = {}
    detected = 0
    total_corrections = 0

    for img_path in images:
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue
        r = ctrl.process(frame)
        if r.pose.detected:
            detected += 1
        label = r.smoothed_pose
        pose_dist[label] = pose_dist.get(label, 0) + 1
        total_corrections += len(r.corrections)

    ctrl.close()

    print(f"\nPose detection rate: {detected}/{len(images)} "
          f"({detected / max(len(images), 1) * 100:.1f}%)")
    print("\nPose distribution:")
    for k in sorted(pose_dist):
        print(f"  {k}: {pose_dist[k]}")
    print(f"\nTotal correction hints generated: {total_corrections}")

    results = {
        "total_images": len(images),
        "detected": detected,
        "pose_distribution": pose_dist,
        "total_corrections": total_corrections,
    }
    out_path = Path("eval_results.json")
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    force = "--force-download" in sys.argv
    main(force_download=force)
