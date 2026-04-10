"""Interactive Video Object Cutout Studio — benchmark on DAVIS 2017.

Evaluates SAM 2 segmentation quality on DAVIS sequences by comparing
predicted masks (from first-frame GT prompt) against ground-truth
annotations.

Usage::

    python benchmark.py --eval
    python benchmark.py --eval --max-sequences 5
    python benchmark.py --eval --force-download
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


def _iou(pred: np.ndarray, gt: np.ndarray) -> float:
    """Compute intersection-over-union between two binary masks."""
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    return float(inter / union) if union > 0 else 0.0


def _gt_to_box(mask: np.ndarray) -> np.ndarray | None:
    """Extract bounding box from a ground-truth mask."""
    ys, xs = np.where(mask > 0)
    if len(ys) == 0:
        return None
    return np.array([xs.min(), ys.min(), xs.max(), ys.max()], dtype=np.float32)


def _evaluate(args: argparse.Namespace) -> None:
    """Run evaluation on DAVIS 2017 sequences."""
    from config import CutoutConfig
    from controller import CutoutController
    from data_bootstrap import (
        _find_davis_root,
        ensure_davis_dataset,
        get_sequence_annotations,
        get_sequence_frames,
        list_sequences,
    )

    data_root = ensure_davis_dataset(force=args.force_download)
    davis_root = _find_davis_root(data_root)
    if davis_root is None:
        print("[ERROR] DAVIS structure not found in dataset.")
        return

    sequences = list_sequences(davis_root)
    if not sequences:
        print("[ERROR] No sequences found.")
        return

    max_seq = min(len(sequences), args.max_sequences)
    sequences = sequences[:max_seq]
    print(f"Evaluating on {max_seq} DAVIS sequence(s) ...\n")

    cfg = CutoutConfig()
    ctrl = CutoutController(cfg)
    ctrl.load_image_engine()

    all_ious: list[float] = []
    seq_results: list[dict] = []

    for seq_name in sequences:
        frames = get_sequence_frames(davis_root, seq_name)
        annots = get_sequence_annotations(davis_root, seq_name)
        if not frames or not annots:
            continue

        # Use first-frame GT mask to derive a box prompt
        gt_first = cv2.imread(str(annots[0]), cv2.IMREAD_GRAYSCALE)
        box = _gt_to_box(gt_first)
        if box is None:
            continue

        seq_ious: list[float] = []
        for frame_path, annot_path in zip(frames, annots):
            frame = cv2.imread(str(frame_path))
            gt = cv2.imread(str(annot_path), cv2.IMREAD_GRAYSCALE) > 0

            result = ctrl.segment_image(frame, box=box)
            iou = _iou(result.best_mask, gt)
            seq_ious.append(iou)

        mean_iou = float(np.mean(seq_ious)) if seq_ious else 0.0
        all_ious.extend(seq_ious)
        seq_results.append({"sequence": seq_name, "frames": len(seq_ious), "mean_iou": mean_iou})
        print(f"  {seq_name:25s}  frames={len(seq_ious):3d}  mean IoU={mean_iou:.4f}")

    ctrl.close()

    overall = float(np.mean(all_ious)) if all_ious else 0.0
    print(f"\n{'=' * 55}")
    print(f"Sequences evaluated: {len(seq_results)}")
    print(f"Total frames:        {len(all_ious)}")
    print(f"Overall mean IoU:    {overall:.4f}")
    print(f"{'=' * 55}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Interactive Video Object Cutout Studio — benchmark",
    )
    parser.add_argument("--eval", action="store_true",
                        help="Run DAVIS benchmark evaluation")
    parser.add_argument("--max-sequences", type=int, default=10,
                        help="Max DAVIS sequences to evaluate (default: 10)")
    parser.add_argument("--force-download", action="store_true",
                        help="Force re-download dataset")
    args = parser.parse_args()

    if args.eval:
        _evaluate(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
