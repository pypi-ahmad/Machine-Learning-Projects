"""Interactive Video Object Cutout Studio — CLI entry point.
"""Interactive Video Object Cutout Studio — CLI entry point.

Batch (non-interactive) inference.  For interactive prompt collection,
use ``demo.py`` instead.

Usage::

    # Image with point prompt
    python infer.py --source image.jpg --point 320,240

    # Image with box prompt
    python infer.py --source image.jpg --box 10,10,300,200

    # Video (prompt on frame 0, propagate)
    python infer.py --source video.mp4 --point 400,300 --frame-idx 0

    # Directory of images
    python infer.py --source images/ --point 320,240

    # Webcam (opens interactive prompt on first frame)
    python infer.py --source 0
"""
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


def _parse_points(raw: str | None) -> tuple[list[tuple[int, int]], list[int]] | tuple[None, None]:
    """Parse 'x1,y1;x2,y2' or 'x1,y1' into points + labels (all fg=1)."""
    if not raw:
        return None, None
    pts, lbls = [], []
    for tok in raw.split(";"):
        parts = tok.strip().split(",")
        if len(parts) == 2:
            pts.append((int(parts[0]), int(parts[1])))
            lbls.append(1)
        elif len(parts) == 3:
            pts.append((int(parts[0]), int(parts[1])))
            lbls.append(int(parts[2]))
    return pts, lbls


def _parse_box(raw: str | None) -> tuple[int, int, int, int] | None:
    """Parse 'x1,y1,x2,y2'."""
    if not raw:
        return None
    parts = [int(x) for x in raw.split(",")]
    if len(parts) != 4:
        raise ValueError(f"Box must be x1,y1,x2,y2 -- got: {raw}")
    return tuple(parts)


def _run_image(args: argparse.Namespace) -> None:
    import cv2
    import numpy as np
    from config import load_config
    from controller import CutoutController
    from export import draw_overlay, save_alpha_mask, save_cutout, save_overlay

    cfg = load_config(args.config)
    ctrl = CutoutController(cfg)
    image = cv2.imread(args.source)
    if image is None:
        print(f"[ERROR] Cannot read image: {args.source}")
        return

    pts_list, lbls = _parse_points(args.point)
    box = _parse_box(args.box)

    pts_arr = np.array(pts_list, dtype=np.float32) if pts_list else None
    lbl_arr = np.array(lbls, dtype=np.int32) if lbls else None
    box_arr = np.array(box, dtype=np.float32) if box else None

    if pts_arr is None and box_arr is None:
        # Interactive fallback
        from prompt_ui import PromptCollector
        prompts = PromptCollector().collect(image)
        if prompts is None or prompts.is_empty:
            print("[INFO] No prompts provided. Exiting.")
            return
        pts_arr = prompts.points_array()
        lbl_arr = prompts.labels_array()
        box_arr = prompts.box_array()

    result = ctrl.segment_image(image, points=pts_arr, labels=lbl_arr, box=box_arr)
    print(f"Score: {result.score:.4f}  Mask pixels: {result.best_mask.sum()}")

    out_dir = Path(cfg.output_dir)
    stem = Path(args.source).stem

    if args.save_mask:
        save_alpha_mask(result.best_mask, out_dir / f"{stem}_mask.png")
    if args.save_cutout:
        save_cutout(image, result.best_mask, out_dir / f"{stem}_cutout.png")
    if args.save_overlay or not args.no_display:
        overlay = draw_overlay(image, result.best_mask, cfg.overlay_color, cfg.overlay_alpha)
        if args.save_overlay:
            save_overlay(image, result.best_mask, out_dir / f"{stem}_overlay.png",
                         cfg.overlay_color, cfg.overlay_alpha)

    if not args.no_display:
        overlay = draw_overlay(image, result.best_mask, cfg.overlay_color, cfg.overlay_alpha)
        cv2.imshow("Cutout Result", overlay)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    ctrl.close()
    print(f"[DONE] Results saved to {out_dir}")


def _run_directory(args: argparse.Namespace) -> None:
    import cv2
    import numpy as np
    from config import load_config
    from controller import CutoutController
    from export import save_alpha_mask, save_cutout, save_overlay
    from validator import is_image

    cfg = load_config(args.config)
    ctrl = CutoutController(cfg)
    out_dir = Path(cfg.output_dir)

    pts_list, lbls = _parse_points(args.point)
    box = _parse_box(args.box)
    pts_arr = np.array(pts_list, dtype=np.float32) if pts_list else None
    lbl_arr = np.array(lbls, dtype=np.int32) if lbls else None
    box_arr = np.array(box, dtype=np.float32) if box else None

    if pts_arr is None and box_arr is None:
        print("[ERROR] Point or box prompt required for directory mode.")
        return

    images = sorted(
        f for f in Path(args.source).iterdir() if is_image(f)
    )
    print(f"Processing {len(images)} image(s) ...")

    for idx, img_path in enumerate(images):
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue
        result = ctrl.segment_image(frame, points=pts_arr, labels=lbl_arr, box=box_arr)
        stem = img_path.stem

        if args.save_mask:
            save_alpha_mask(result.best_mask, out_dir / "masks" / f"{stem}.png")
        if args.save_cutout:
            save_cutout(frame, result.best_mask, out_dir / "cutouts" / f"{stem}.png")
        if args.save_overlay:
            save_overlay(frame, result.best_mask, out_dir / "overlays" / f"{stem}.png",
                         cfg.overlay_color, cfg.overlay_alpha)

        if (idx + 1) % 5 == 0 or idx == len(images) - 1:
            print(f"  [{idx + 1}/{len(images)}] {img_path.name}: score={result.score:.4f}")

    ctrl.close()
    print(f"[DONE] Results saved to {out_dir}")


def _run_video(args: argparse.Namespace) -> None:
    import cv2
    import numpy as np
    from config import load_config
    from controller import CutoutController
    from export import VideoExporter

    cfg = load_config(args.config)
    ctrl = CutoutController(cfg)
    out_dir = Path(cfg.output_dir)

    pts_list, lbls = _parse_points(args.point)
    box = _parse_box(args.box)

    frame_idx = args.frame_idx

    # If no prompts given, open interactive prompt on the first frame
    if not pts_list and not box:
        cap = cv2.VideoCapture(args.source)
        for _ in range(frame_idx + 1):
            ok, first_frame = cap.read()
        cap.release()
        if not ok:
            print(f"[ERROR] Cannot read frame {frame_idx}")
            return

        from prompt_ui import PromptCollector
        prompts = PromptCollector().collect(first_frame)
        if prompts is None or prompts.is_empty:
            print("[INFO] No prompts provided. Exiting.")
            return
        pts_arr = prompts.points_array()
        lbl_arr = prompts.labels_array()
        box_arr = prompts.box_array()
    else:
        pts_arr = np.array(pts_list, dtype=np.float32) if pts_list else None
        lbl_arr = np.array(lbls, dtype=np.int32) if lbls else None
        box_arr = np.array(box, dtype=np.float32) if box else None

    print(f"Processing video: {args.source}")
    video_result = ctrl.process_video(
        args.source, frame_idx, obj_id=1,
        points=pts_arr, labels=lbl_arr, box=box_arr,
    )

    exporter = VideoExporter(
        out_dir,
        save_masks=args.save_mask,
        save_cutouts=args.save_cutout,
        save_overlays=args.save_overlay,
        overlay_color=cfg.overlay_color,
        overlay_alpha=cfg.overlay_alpha,
    )

    for fidx in sorted(video_result.propagation.frame_masks.keys()):
        mask = video_result.propagation.get_mask(fidx, obj_id=1)
        if mask is None:
            continue
        frame = ctrl.get_frame(video_result.frames_dir, fidx)
        if frame is None:
            continue
        exporter.export_frame(fidx, frame, mask)

        if (fidx + 1) % 10 == 0 or fidx == video_result.frame_count - 1:
            print(f"  [{fidx + 1}/{video_result.frame_count}] exported")

    stats = exporter.finalize()
    ctrl.close()
    print(f"[DONE] {stats['frames_exported']} frames exported to {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Interactive Video Object Cutout Studio -- batch inference",
    )
    parser.add_argument("--source", type=str, default="0",
                        help="Image, directory, video path, or webcam index")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to JSON/YAML config file")
    parser.add_argument("--point", type=str, default=None,
                        help="Point prompt(s): 'x,y' or 'x,y;x2,y2' or 'x,y,label'")
    parser.add_argument("--box", type=str, default=None,
                        help="Box prompt: 'x1,y1,x2,y2'")
    parser.add_argument("--frame-idx", type=int, default=0,
                        help="Frame index for video prompt (default: 0)")
    parser.add_argument("--no-display", action="store_true",
                        help="Suppress GUI window")
    parser.add_argument("--save-mask", action="store_true",
                        help="Save alpha masks")
    parser.add_argument("--save-cutout", action="store_true",
                        help="Save transparent cutouts")
    parser.add_argument("--save-overlay", action="store_true",
                        help="Save overlay visualisations")
    parser.add_argument("--force-download", action="store_true",
                        help="Force re-download dataset")
    args = parser.parse_args()

    from validator import validate_source
    report = validate_source(args.source)

    if not report.ok:
        for w in report.warnings:
            print(f"[ERROR] {w}")
        sys.exit(1)

    if report.source_type == "image":
        _run_image(args)
    elif report.source_type == "directory":
        _run_directory(args)
    elif report.source_type in ("video", "webcam"):
        _run_video(args)
    else:
        print(f"[ERROR] Unsupported source type: {report.source_type}")
        sys.exit(1)


if __name__ == "__main__":
    main()
