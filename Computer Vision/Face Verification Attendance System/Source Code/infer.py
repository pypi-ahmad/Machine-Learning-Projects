"""CLI inference pipeline for Face Verification Attendance System.

Supports enrollment and verification modes with webcam, video, and
image inputs.

Usage::

    # Enroll identities
    python infer.py --mode enroll --identity Alice --source alice.jpg
    python infer.py --mode enroll --source faces/ --gallery-dir gallery

    # Verify / take attendance
    python infer.py --mode verify --source test.jpg --gallery-dir gallery
    python infer.py --mode verify --source 0 --gallery-dir gallery   # webcam
    python infer.py --mode verify --source video.mp4 --export-csv attendance.csv
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import FaceAttendanceConfig, load_config
from export import AttendanceExporter
from parser import FaceAttendancePipeline
from validator import AttendanceValidator
from visualize import draw_overlay

log = logging.getLogger("face_attendance.infer")

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv"}


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Face Verification Attendance System — Inference",
    )
    p.add_argument(
        "--mode", choices=["enroll", "verify"], default="verify",
        help="Pipeline mode: enroll new identities or verify/attend",
    )
    p.add_argument("--source", required=True,
                   help="Image path, directory, video file, or '0' for webcam")
    p.add_argument("--identity", default=None,
                   help="Identity name (required for single-image enrollment)")
    p.add_argument("--config", default=None, help="Path to YAML/JSON config")
    p.add_argument("--gallery-dir", default=None,
                   help="Gallery directory for enrolled embeddings")
    p.add_argument("--threshold", type=float, default=None,
                   help="Cosine similarity threshold")
    p.add_argument("--no-display", action="store_true",
                   help="Disable GUI windows")
    p.add_argument("--export-json", default=None, help="JSON export path")
    p.add_argument("--export-csv", default=None, help="CSV export path")
    p.add_argument("--save-annotated", action="store_true",
                   help="Save annotated images")
    p.add_argument("--output-dir", default="output",
                   help="Output directory for saved images")
    p.add_argument("--force-download", action="store_true",
                   help="Force dataset re-download")
    return p.parse_args(argv)


def _apply_cli_overrides(
    cfg: FaceAttendanceConfig, args: argparse.Namespace,
) -> None:
    if args.gallery_dir:
        cfg.gallery_dir = args.gallery_dir
    if args.threshold is not None:
        cfg.similarity_threshold = args.threshold
    if args.no_display:
        cfg.show_display = False
    if args.export_json:
        cfg.export_json = args.export_json
    if args.export_csv:
        cfg.export_csv = args.export_csv
    if args.save_annotated:
        cfg.save_annotated = True
    if args.output_dir:
        cfg.output_dir = args.output_dir


def _collect_images(source: str) -> list[Path]:
    p = Path(source)
    if p.is_dir():
        files = []
        for ext in IMAGE_EXTS:
            files.extend(p.glob(f"*{ext}"))
        files.sort()
        return files
    if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
        return [p]
    return []


def _is_webcam(source: str) -> bool:
    return source.isdigit()


def _is_video(source: str) -> bool:
    return Path(source).suffix.lower() in VIDEO_EXTS


# ── Enrollment mode ───────────────────────────────────────


def _run_enroll(
    pipeline: FaceAttendancePipeline,
    cfg: FaceAttendanceConfig,
    args: argparse.Namespace,
) -> None:
    """Enroll identities from images or a directory."""
    source = Path(args.source)

    if source.is_dir():
        # Each subdirectory = one identity
        identity_dirs = [d for d in sorted(source.iterdir()) if d.is_dir()]
        if not identity_dirs:
            # Flat directory: all images under one identity
            if not args.identity:
                log.error(
                    "Flat directory enrollment requires --identity NAME"
                )
                return
            images = _collect_images(str(source))
            if not images:
                log.error("No images found in %s", source)
                return
            ok = pipeline.enroll(args.identity, [str(p) for p in images])
            log.info(
                "Enrolled '%s': %s (%d images)",
                args.identity, "OK" if ok else "FAILED", len(images),
            )
        else:
            for id_dir in identity_dirs:
                name = id_dir.name
                imgs = _collect_images(str(id_dir))
                if not imgs:
                    continue
                ok = pipeline.enroll(name, [str(p) for p in imgs])
                log.info(
                    "Enrolled '%s': %s (%d images)",
                    name, "OK" if ok else "FAILED", len(imgs),
                )
    elif source.is_file():
        if not args.identity:
            log.error("Single-image enrollment requires --identity NAME")
            return
        ok = pipeline.enroll_single(args.identity, str(source))
        log.info(
            "Enrolled '%s': %s", args.identity, "OK" if ok else "FAILED",
        )
    else:
        log.error("Source not found: %s", source)
        return

    # Save gallery
    gallery_path = pipeline.enrollment.save()
    log.info(
        "Gallery saved: %d identities → %s",
        pipeline.enrollment.size, gallery_path,
    )


# ── Verification mode ─────────────────────────────────────


def _run_verify(
    pipeline: FaceAttendancePipeline,
    validator: AttendanceValidator,
    cfg: FaceAttendanceConfig,
    args: argparse.Namespace,
) -> None:
    """Run verification pipeline on images, video, or webcam."""
    # Load gallery
    if not pipeline.load_gallery():
        log.warning("No gallery loaded — all faces will be Unknown")

    source = args.source
    out_dir = Path(cfg.output_dir)

    if _is_webcam(source):
        _verify_webcam(pipeline, validator, cfg, int(source))
    elif _is_video(source):
        _verify_video(pipeline, validator, cfg, source, out_dir)
    else:
        _verify_images(pipeline, validator, cfg, source, out_dir)


def _verify_images(
    pipeline: FaceAttendancePipeline,
    validator: AttendanceValidator,
    cfg: FaceAttendanceConfig,
    source: str,
    out_dir: Path,
) -> None:
    images = _collect_images(source)
    if not images:
        log.error("No images found at: %s", source)
        return

    with AttendanceExporter(cfg) as exporter:
        for img_path in images:
            img = cv2.imread(str(img_path))
            if img is None:
                log.warning("Cannot read: %s", img_path)
                continue

            result = pipeline.process(img)
            report = validator.validate(
                result, gallery_size=pipeline.matcher.gallery_size,
            )
            exporter.write(result, source=img_path.name)

            # Log summary
            log.info(
                "%s: %d faces, %d matched, %d unknown",
                img_path.name,
                result.num_faces,
                result.num_matched,
                result.num_unknown,
            )
            if report.warnings:
                for w in report.warnings:
                    log.warning("  %s: %s", w.field_name, w.message)

            # Visualize
            vis = draw_overlay(
                img, result, cfg,
                recent_attendance=pipeline.logger.recent_identities(),
            )
            if cfg.save_annotated:
                out_dir.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(out_dir / f"annotated_{img_path.name}"), vis)

            if cfg.show_display:
                cv2.imshow(f"Attendance: {img_path.name}", vis)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

    # Save attendance log
    if pipeline.logger.count > 0:
        pipeline.logger.save_csv()
        pipeline.logger.save_json()


def _verify_webcam(
    pipeline: FaceAttendancePipeline,
    validator: AttendanceValidator,
    cfg: FaceAttendanceConfig,
    device: int,
) -> None:
    cap = cv2.VideoCapture(device)
    if not cap.isOpened():
        log.error("Cannot open webcam device %d", device)
        return

    log.info("Webcam verification — press 'q' to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        result = pipeline.process(frame)
        vis = draw_overlay(
            frame, result, cfg,
            recent_attendance=pipeline.logger.recent_identities(),
        )
        cv2.imshow("Face Attendance — Webcam", vis)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Save logs
    if pipeline.logger.count > 0:
        pipeline.logger.save_csv()
        pipeline.logger.save_json()
        log.info("Attendance logged: %d entries", pipeline.logger.count)


def _verify_video(
    pipeline: FaceAttendancePipeline,
    validator: AttendanceValidator,
    cfg: FaceAttendanceConfig,
    video_path: str,
    out_dir: Path,
) -> None:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        log.error("Cannot open video: %s", video_path)
        return

    frame_idx = 0
    log.info("Processing video: %s", video_path)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        result = pipeline.process(frame)

        if cfg.show_display:
            vis = draw_overlay(
                frame, result, cfg,
                recent_attendance=pipeline.logger.recent_identities(),
            )
            cv2.imshow("Face Attendance — Video", vis)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()

    # Save logs
    if pipeline.logger.count > 0:
        pipeline.logger.save_csv()
        pipeline.logger.save_json()
        log.info(
            "Video done: %d frames, %d attendance entries",
            frame_idx, pipeline.logger.count,
        )


# ── Main ──────────────────────────────────────────────────


def run(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s | %(name)s | %(message)s",
    )

    cfg = load_config(args.config) if args.config else FaceAttendanceConfig()
    _apply_cli_overrides(cfg, args)

    if args.force_download:
        from data_bootstrap import ensure_face_attendance_dataset
        ensure_face_attendance_dataset(force=True)

    pipeline = FaceAttendancePipeline(cfg)
    pipeline.load()

    if args.mode == "enroll":
        _run_enroll(pipeline, cfg, args)
    else:
        validator = AttendanceValidator(cfg)
        _run_verify(pipeline, validator, cfg, args)


if __name__ == "__main__":
    run()
