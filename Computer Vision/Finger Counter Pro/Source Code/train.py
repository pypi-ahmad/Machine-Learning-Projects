"""Finger Counter Pro — dataset download + evaluation."""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main(force_download: bool = False) -> None:
    """Download evaluation data and run finger-counting evaluation."""
    import cv2

    from controller import CountingController
    from data_bootstrap import ensure_finger_counter_dataset

    data_dir = ensure_finger_counter_dataset(force=force_download)
    print(f"Dataset ready: {data_dir}")

    ctrl = CountingController()
    ctrl.load()

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
    count_dist: dict[int, int] = {}
    hand_counts: dict[int, int] = {}
    processed = 0

    for img_path in images:
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue
        r = ctrl.process(frame)
        total = r.smoothed_total
        count_dist[total] = count_dist.get(total, 0) + 1
        hc = r.multi.count
        hand_counts[hc] = hand_counts.get(hc, 0) + 1
        processed += 1

    ctrl.close()

    print(f"\nProcessed: {processed}/{len(images)}")
    print("\nFinger-count distribution:")
    for k in sorted(count_dist):
        print(f"  {k} fingers: {count_dist[k]}")
    print("\nHands-detected distribution:")
    for k in sorted(hand_counts):
        print(f"  {k} hands: {hand_counts[k]}")

    results = {
        "total_images": len(images),
        "processed": processed,
        "count_distribution": {str(k): v for k, v in sorted(count_dist.items())},
        "hand_distribution": {str(k): v for k, v in sorted(hand_counts.items())},
    }
    out_path = Path("eval_results.json")
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    force = "--force-download" in sys.argv
    main(force_download=force)
