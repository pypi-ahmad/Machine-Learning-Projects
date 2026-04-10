"""Interactive Video Object Cutout Studio — interactive demo.

Provides a guided, step-by-step experience:
  1. Load image or video.
  2. Collect prompts interactively (point clicks / box drags).
  3. Run SAM 2 segmentation (+ video propagation if applicable).
  4. Display results and export.

Usage::

    # Interactive image segmentation
    python demo.py --source photo.jpg

    # Interactive video segmentation
    python demo.py --source clip.mp4

    # With pre-set output options
    python demo.py --source clip.mp4 --save-cutout --save-mask
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

from config import CutoutConfig, load_config
from controller import CutoutController
from export import VideoExporter, draw_overlay, save_alpha_mask, save_cutout, save_overlay
from prompt_ui import PromptCollector
from validator import validate_source


def _demo_image(image_path: str, cfg: CutoutConfig, args: argparse.Namespace) -> None:
    """Interactive single-image segmentation demo."""
    image = cv2.imread(image_path)
    if image is None:
        print(f"[ERROR] Cannot read image: {image_path}")
        return

    print("=== Image Segmentation Demo ===")
    print("Annotate the object you want to cut out, then press Enter.\n")

    collector = PromptCollector(window_name="SAM 2 — Click to Prompt")
    prompts = collector.collect(image)
    if prompts is None or prompts.is_empty:
        print("[INFO] Cancelled — no prompts provided.")
        return

    print("Running SAM 2 segmentation ...")
    ctrl = CutoutController(cfg)
    result = ctrl.segment_image(
        image,
        points=prompts.points_array(),
        labels=prompts.labels_array(),
        box=prompts.box_array(),
    )

    print(f"  Best mask score: {result.score:.4f}")
    print(f"  Mask area: {result.best_mask.sum()} px")

    # Show all candidate masks if multimask_output
    mr = result.mask_result
    if mr.masks.shape[0] > 1:
        print(f"\n  {mr.masks.shape[0]} mask candidates:")
        for i in range(mr.masks.shape[0]):
            tag = " ← best" if i == mr.best_idx else ""
            print(f"    [{i}] score={mr.scores[i]:.4f}  area={mr.masks[i].sum()} px{tag}")

    # Display result
    overlay = draw_overlay(image, result.best_mask, cfg.overlay_color, cfg.overlay_alpha)
    cv2.imshow("Segmentation Result", overlay)
    print("\nPress any key to continue (or 'q' to quit without saving) ...")
    key = cv2.waitKey(0) & 0xFF
    cv2.destroyAllWindows()

    if key == ord("q"):
        ctrl.close()
        return

    # Export
    out_dir = Path(cfg.output_dir)
    stem = Path(image_path).stem

    save_overlay(image, result.best_mask, out_dir / f"{stem}_overlay.png",
                 cfg.overlay_color, cfg.overlay_alpha)

    if args.save_mask:
        save_alpha_mask(result.best_mask, out_dir / f"{stem}_mask.png")
    if args.save_cutout:
        save_cutout(image, result.best_mask, out_dir / f"{stem}_cutout.png")

    ctrl.close()
    print(f"\n[DONE] Results saved to {out_dir}/")


def _demo_video(video_path: str, cfg: CutoutConfig, args: argparse.Namespace) -> None:
    """Interactive video segmentation demo with propagation."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {video_path}")
        return

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"=== Video Segmentation Demo ===")
    print(f"Video: {video_path}  ({total} frames, {fps:.1f} FPS)")

    frame_idx = args.frame_idx
    for _ in range(frame_idx + 1):
        ok, first_frame = cap.read()
    cap.release()

    if not ok:
        print(f"[ERROR] Cannot read frame {frame_idx}.")
        return

    print(f"\nPrompt on frame {frame_idx}. Annotate the object, then press Enter.\n")
    collector = PromptCollector(window_name="SAM 2 — Prompt on Frame")
    prompts = collector.collect(first_frame)
    if prompts is None or prompts.is_empty:
        print("[INFO] Cancelled — no prompts provided.")
        return

    print("Propagating masks across all frames ...")
    ctrl = CutoutController(cfg)
    video_result = ctrl.process_video(
        video_path, frame_idx, obj_id=1,
        points=prompts.points_array(),
        labels=prompts.labels_array(),
        box=prompts.box_array(),
    )

    print(f"  Propagated to {video_result.frame_count} frames.")

    # Preview a few frames
    preview_indices = list(sorted(video_result.propagation.frame_masks.keys()))
    step = max(1, len(preview_indices) // 5)
    sample = preview_indices[::step][:6]

    print(f"\nPreviewing {len(sample)} sample frames (press any key to advance) ...")
    for fidx in sample:
        mask = video_result.propagation.get_mask(fidx, obj_id=1)
        frame = ctrl.get_frame(video_result.frames_dir, fidx)
        if frame is None or mask is None:
            continue
        vis = draw_overlay(frame, mask, cfg.overlay_color, cfg.overlay_alpha)
        label = f"Frame {fidx}/{video_result.frame_count}"
        cv2.putText(vis, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow("Video Propagation Preview", vis)
        cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Export
    out_dir = Path(cfg.output_dir)
    exporter = VideoExporter(
        out_dir,
        save_masks=args.save_mask,
        save_cutouts=args.save_cutout,
        save_overlays=True,
        overlay_color=cfg.overlay_color,
        overlay_alpha=cfg.overlay_alpha,
    )

    for fidx in sorted(video_result.propagation.frame_masks.keys()):
        mask = video_result.propagation.get_mask(fidx, obj_id=1)
        frame = ctrl.get_frame(video_result.frames_dir, fidx)
        if frame is not None and mask is not None:
            exporter.export_frame(fidx, frame, mask)

    stats = exporter.finalize()
    ctrl.close()
    print(f"\n[DONE] {stats['frames_exported']} frames exported to {out_dir}/")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Interactive Video Object Cutout Studio — interactive demo",
    )
    parser.add_argument("--source", type=str, required=True,
                        help="Image or video path")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--frame-idx", type=int, default=0,
                        help="Video frame to prompt on (default: 0)")
    parser.add_argument("--save-mask", action="store_true",
                        help="Export binary masks")
    parser.add_argument("--save-cutout", action="store_true",
                        help="Export transparent RGBA cutouts")
    parser.add_argument("--force-download", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)

    report = validate_source(args.source)
    if not report.ok:
        for w in report.warnings:
            print(f"[ERROR] {w}")
        sys.exit(1)

    if report.source_type == "image":
        _demo_image(args.source, cfg, args)
    elif report.source_type == "video":
        _demo_video(args.source, cfg, args)
    else:
        print(f"[ERROR] Demo supports image or video. Got: {report.source_type}")
        sys.exit(1)


if __name__ == "__main__":
    main()
