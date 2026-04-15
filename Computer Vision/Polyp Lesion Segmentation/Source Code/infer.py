"""Polyp Lesion Segmentation — CLI entry point.
"""Polyp Lesion Segmentation — CLI entry point.

Usage::

    # Single image
    python infer.py --source polyp.jpg

    # Directory of images
    python infer.py --source images/ --save-annotated --save-masks

    # Video / webcam
    python infer.py --source 0
    python infer.py --source endoscopy.mp4 --save-annotated

    # Full export
    python infer.py --source polyp.jpg \\
        --export-json report.json --export-csv stats.csv \\
        --save-annotated --save-masks

    # Switch backend (if MedSAM installed)
    python infer.py --source polyp.jpg --backend medsam

    # Evaluate with ground-truth masks
    python infer.py --source images/ --gt-dir masks/ --save-annotated

DISCLAIMER: This tool is for research purposes only.
It is NOT a medical device and must NOT be used for clinical diagnosis.
"""
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Polyp Lesion Segmentation -- polyp detection and segmentation",
        epilog="DISCLAIMER: Research only. NOT for clinical diagnosis.",
    )
    p.add_argument(
        "--source", default="0",
        help="'0' for webcam, or path to image/video/directory (default: 0)",
    )
    p.add_argument("--config", default=None, help="YAML/JSON config path")
    p.add_argument("--backend", default="yolo",
                    help="Segmentation backend: 'yolo' (default) or 'medsam'")
    p.add_argument("--gt-dir", default=None,
                    help="Directory with ground-truth masks for Dice/IoU evaluation")
    p.add_argument("--no-display", action="store_true", help="Headless mode")
    p.add_argument("--output-dir", default="output", help="Output directory")
    p.add_argument("--export-json", default=None, help="JSON export path")
    p.add_argument("--export-csv", default=None, help="CSV export path")
    p.add_argument("--save-annotated", action="store_true",
                    help="Save annotated images/video")
    p.add_argument("--save-masks", action="store_true",
                    help="Save binary mask PNGs")
    p.add_argument("--force-download", action="store_true",
                    help="Re-download dataset")
    return p


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)

    print("DISCLAIMER: This tool is for research purposes only.")
    print("It is NOT a medical device and must NOT be used for clinical diagnosis.\n")

    import cv2

    from config import PolypConfig, load_config
    from controller import PolypController
    from export import CSVExporter, export_frame_json, save_mask
    from validator import is_image, is_video, validate_source
    from visualize import draw_polyp_overlay

    if args.force_download:
        from data_bootstrap import ensure_polyp_dataset
        ensure_polyp_dataset(force=True)

    cfg = load_config(args.config) if args.config else PolypConfig()
    cfg.output_dir = args.output_dir
    cfg.backend = args.backend
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    vr = validate_source(args.source)
    if not vr.ok:
        for w in vr.warnings:
            print(f"ERROR: {w}", file=sys.stderr)
        sys.exit(1)

    ctrl = PolypController(cfg)
    ctrl.load()

    csv_exp = CSVExporter(args.export_csv) if args.export_csv else None

    # Load GT masks if provided
    gt_masks: dict[str, Path] = {}
    if args.gt_dir:
        gt_dir = Path(args.gt_dir)
        if gt_dir.is_dir():
            for f in gt_dir.iterdir():
                if f.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}:
                    gt_masks[f.stem] = f

    try:
        source = args.source
        if source.isdigit():
            _run_video(ctrl, int(source), cfg, args, out_dir, csv_exp,
                       cv2, draw_polyp_overlay, export_frame_json, save_mask,
                       is_webcam=True)
        elif Path(source).is_dir():
            _run_directory(ctrl, Path(source), cfg, args, out_dir, csv_exp,
                           cv2, draw_polyp_overlay, export_frame_json,
                           save_mask, gt_masks)
        elif is_video(source):
            _run_video(ctrl, source, cfg, args, out_dir, csv_exp,
                       cv2, draw_polyp_overlay, export_frame_json, save_mask,
                       is_webcam=False)
        elif is_image(source):
            _run_image(ctrl, Path(source), cfg, args, out_dir, csv_exp,
                       cv2, draw_polyp_overlay, export_frame_json, save_mask,
                       gt_masks)
        else:
            print(f"Unsupported source: {source}", file=sys.stderr)
            sys.exit(1)
    finally:
        if csv_exp:
            csv_exp.close()
        ctrl.close()
        if not args.no_display:
            cv2.destroyAllWindows()


# ── source handlers ────────────────────────────────────────


def _load_gt(path: Path, gt_masks: dict[str, Path], cv2_mod) -> "np.ndarray | None":
    """Load a ground-truth mask by matching stem to gt_masks dict."""
    gt_path = gt_masks.get(path.stem)
    if gt_path is None:
        return None
    gt = cv2_mod.imread(str(gt_path), cv2_mod.IMREAD_GRAYSCALE)
    return gt


def _run_image(ctrl, path, cfg, args, out_dir, csv_exp,
               cv2, draw_polyp_overlay, export_frame_json, save_mask_fn,
               gt_masks):
    import numpy as np

    frame = cv2.imread(str(path))
    if frame is None:
        print(f"Cannot read image: {path}", file=sys.stderr)
        return

    gt = _load_gt(path, gt_masks, cv2)
    result = ctrl.process(frame, source=str(path), gt_mask=gt)
    m = result.metrics
    dice_str = f"  Dice={m.dice:.4f} IoU={m.iou:.4f}" if m.dice is not None else ""
    print(f"{path.name}: polyp={m.polyp_coverage:.2%} ({m.polyp_count} region(s), "
          f"{m.polyp_area_px:,}px){dice_str}")

    vis = draw_polyp_overlay(frame, result.segmentation, m, cfg)

    if not args.no_display:
        cv2.imshow("Polyp Segmentation", vis)
        cv2.waitKey(0)

    if args.save_annotated:
        cv2.imwrite(str(out_dir / f"{path.stem}_annotated.jpg"), vis)

    if args.save_masks:
        save_mask_fn(out_dir / f"{path.stem}_mask.png",
                     result.segmentation.combined_mask)

    if args.export_json:
        jp = args.export_json
        if Path(jp).is_dir() or jp.endswith("/") or jp.endswith("\\"):
            jp = str(Path(jp) / f"{path.stem}.json")
        export_frame_json(jp, result.segmentation, m, source=str(path),
                          backend=result.backend_name)

    if csv_exp:
        csv_exp.write_row(_csv_row(result, str(path)))


def _run_directory(ctrl, dir_path, cfg, args, out_dir, csv_exp,
                   cv2, draw_polyp_overlay, export_frame_json,
                   save_mask_fn, gt_masks):
    from validator import _IMAGE_EXTS
    images = sorted(f for f in dir_path.iterdir() if f.suffix.lower() in _IMAGE_EXTS)
    print(f"Processing {len(images)} image(s) from {dir_path}")

    total_dice = 0.0
    total_iou = 0.0
    n_gt = 0

    for idx, img_path in enumerate(images):
        frame = cv2.imread(str(img_path))
        if frame is None:
            print(f"  [SKIP] Cannot read: {img_path.name}")
            continue

        gt = _load_gt(img_path, gt_masks, cv2)
        result = ctrl.process(frame, source=str(img_path), gt_mask=gt)
        m = result.metrics

        dice_str = ""
        if m.dice is not None:
            dice_str = f"  Dice={m.dice:.4f}"
            total_dice += m.dice
            total_iou += m.iou
            n_gt += 1

        print(f"  [{idx + 1}/{len(images)}] {img_path.name}: "
              f"polyp={m.polyp_coverage:.2%} ({m.polyp_area_px:,}px){dice_str}")

        if args.save_annotated:
            vis = draw_polyp_overlay(frame, result.segmentation, m, cfg)
            cv2.imwrite(str(out_dir / f"{img_path.stem}_annotated.jpg"), vis)

        if args.save_masks:
            save_mask_fn(out_dir / f"{img_path.stem}_mask.png",
                         result.segmentation.combined_mask)

        if args.export_json:
            jp = args.export_json
            if Path(jp).is_dir() or jp.endswith("/") or jp.endswith("\\"):
                Path(jp).mkdir(parents=True, exist_ok=True)
                jp = str(Path(jp) / f"{img_path.stem}.json")
            else:
                jp = str(out_dir / f"{img_path.stem}.json")
            export_frame_json(jp, result.segmentation, m, source=str(img_path),
                              backend=result.backend_name)

        if csv_exp:
            csv_exp.write_row(_csv_row(result, str(img_path)))

    print(f"\nDone -- processed {len(images)} image(s)")

    if n_gt > 0:
        print(f"Mean Dice: {total_dice / n_gt:.4f}  "
              f"Mean IoU:  {total_iou / n_gt:.4f}  "
              f"(over {n_gt} image(s) with GT)")


def _run_video(ctrl, source, cfg, args, out_dir, csv_exp,
               cv2, draw_polyp_overlay, export_frame_json, save_mask_fn,
               is_webcam):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Cannot open source: {source}", file=sys.stderr)
        return

    writer = None
    if args.save_annotated:
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        vid_name = "webcam" if is_webcam else Path(str(source)).stem
        writer = cv2.VideoWriter(
            str(out_dir / f"{vid_name}_annotated.mp4"), fourcc, fps, (w, h),
        )

    idx = 0
    last_result = None
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if is_webcam:
                frame = cv2.flip(frame, 1)

            result = ctrl.process(frame, source=str(source))
            last_result = result
            m = result.metrics

            if idx % 30 == 0:
                print(f"  [frame {idx}] polyp={m.polyp_coverage:.2%} "
                      f"({m.polyp_count} region(s))")

            vis = draw_polyp_overlay(frame, result.segmentation, m, cfg)

            if not args.no_display:
                cv2.imshow("Polyp Segmentation", vis)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break

            if writer:
                writer.write(vis)

            if csv_exp:
                csv_exp.write_row(_csv_row(result, str(source), frame_idx=idx))

            idx += 1
    finally:
        cap.release()
        if writer:
            writer.release()

    if args.export_json and last_result is not None:
        export_frame_json(
            args.export_json, last_result.segmentation, last_result.metrics,
            source=str(source), frame_idx=idx - 1,
            backend=last_result.backend_name,
        )

    print(f"\nDone -- processed {idx} frame(s)")


# ── helpers ────────────────────────────────────────────────


def _csv_row(result, source, frame_idx=None):
    m = result.metrics
    row = {
        "source": source,
        "backend": result.backend_name,
        "polyp_area_px": m.polyp_area_px,
        "polyp_coverage": round(m.polyp_coverage, 6),
        "polyp_count": m.polyp_count,
        "mean_confidence": round(m.mean_confidence, 4),
        "largest_polyp_px": m.largest_polyp_px,
    }
    if m.dice is not None:
        row["dice"] = m.dice
        row["iou"] = m.iou
    if frame_idx is not None:
        row["frame_idx"] = frame_idx
    return row


if __name__ == "__main__":
    main()
