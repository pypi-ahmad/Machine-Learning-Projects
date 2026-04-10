#!/usr/bin/env python3
"""
Smoke Test — Phase 1B
=====================
Quick sanity check that importable projects can at least be parsed without
SyntaxErrors and that relocated model/data files exist at their expected paths.

Usage:
    python scripts/smoke_test.py
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


# ---------------------------------------------------------------------------
# 1. Verify relocated files exist
# ---------------------------------------------------------------------------

RELOCATED_FILES: list[tuple[str, Path]] = [
    ("Project  3 — caffemodel",   REPO_ROOT / "models" / "project_03" / "res10_300x300_ssd_iter_140000.caffemodel"),
    ("Project  4 — shape_pred",   REPO_ROOT / "models" / "project_04" / "shape_predictor_68_face_landmarks.dat"),
    ("Project 12 — caffemodel",   REPO_ROOT / "models" / "project_12" / "MobileNetSSD_deploy.caffemodel"),
    ("Project 17 — shape_pred",   REPO_ROOT / "models" / "project_17" / "shape_predictor_68_face_landmarks.dat"),
    ("Project 13 — Test.gif",     REPO_ROOT / "data"   / "project_13" / "Test.gif"),
    ("Project 16 — carv.mp4",     REPO_ROOT / "data"   / "project_16" / "carv.mp4"),
    ("Project 18 — ball video",   REPO_ROOT / "data"   / "project_18" / "ball_tracking_example.mp4"),
]


def check_relocated_files() -> int:
    """Return count of missing relocated files."""
    missing = 0
    print("=" * 60)
    print("  RELOCATED FILE CHECK")
    print("=" * 60)
    for label, path in RELOCATED_FILES:
        exists = path.exists()
        status = "OK" if exists else "MISSING"
        icon = "\u2713" if exists else "\u2717"
        print(f"  [{icon}] {status:>7}  {label}")
        if not exists:
            print(f"           Expected: {path}")
            missing += 1
    print()
    return missing


# ---------------------------------------------------------------------------
# 2. Syntax-check key Python files (ast.parse)
# ---------------------------------------------------------------------------

SYNTAX_CHECK_FILES: list[tuple[str, Path]] = [
    ("Project  3 — detect_faces.py",        REPO_ROOT / "CV Project 3 - Real Time Face detector Image" / "detect_faces.py"),
    ("Project  3 — detect_faces_video.py",   REPO_ROOT / "CV Project 3 - Real Time Face detector Image" / "detect_faces_video.py"),
    ("Project  4 — facial_landmarking.py",   REPO_ROOT / "CV Project 4 - Facial Landmarking" / "facial_landmarking.py"),
    ("Project  5 — fingerCount.py",          REPO_ROOT / "CV Projects 5 - fingerCounter" / "fingerCount.py"),
    ("Project 10 — main.py",                 REPO_ROOT / "CV Project 10 - Live PoseDetector" / "main.py"),
    ("Project 12 — real_time_object_det.py", REPO_ROOT / "CV Project 12 - Real Time Object Detection-fine" / "real_time_object_detection.py"),
    ("Project 13 — Sudoku.py",               REPO_ROOT / "CV Project 13 - Real Time Sudoku Solver" / "Sudoku.py"),
    ("Project 13 — main.py",                 REPO_ROOT / "CV Project 13 - Real Time Sudoku Solver" / "main.py"),
    ("Project 14 — Warp.py",                 REPO_ROOT / "CV Project 14 - click-detect on image" / "Warp.py"),
    ("Project 15 — cartoon.py",              REPO_ROOT / "CV Project 15 - Live Image Cartoonifier" / "cartoon.py"),
    ("Project 16 — Vehicles_detection.py",   REPO_ROOT / "CV Project 16 -Live Car-Detection" / "Vehicles_detection.py"),
    ("Project 17 — blink_detector.py",       REPO_ROOT / "CV Project 17 - Blink Detection" / "blink_detector.py"),
    ("Project 18 — ballTracking.py",         REPO_ROOT / "CV Project 18 - Live Ball Tracking" / "ballTracking.py"),
    ("Project 26 — LiveHSVAdjustor.py",      REPO_ROOT / "CV Projects 26 - Real Time Color Detection" / "LiveHSVAdjustor.py"),
    ("Project 49 — TextSimple.py",           REPO_ROOT / "CV Projects 49- Real Time TextDetection" / "TextSimple.py"),
    ("Project 49 — TextMoreExamples.py",     REPO_ROOT / "CV Projects 49- Real Time TextDetection" / "TextMoreExamples.py"),
]


def check_syntax() -> int:
    """Return count of files that fail ast.parse."""
    failures = 0
    print("=" * 60)
    print("  SYNTAX CHECK (ast.parse)")
    print("=" * 60)
    for label, path in SYNTAX_CHECK_FILES:
        if not path.exists():
            print(f"  [\u2717] MISSING  {label}")
            failures += 1
            continue
        try:
            source = path.read_text(encoding="utf-8", errors="replace")
            ast.parse(source, filename=str(path))
            print(f"  [\u2713]      OK  {label}")
        except SyntaxError as exc:
            print(f"  [\u2717]   FAIL  {label}  —  {exc.msg} (line {exc.lineno})")
            failures += 1
    print()
    return failures


# ---------------------------------------------------------------------------
# 3. Verify utils package imports
# ---------------------------------------------------------------------------

def check_utils() -> int:
    """Return count of import failures from the utils package."""
    failures = 0
    print("=" * 60)
    print("  UTILS PACKAGE IMPORT CHECK")
    print("=" * 60)
    modules = [
        ("utils.paths",  "PathResolver"),
        ("utils.device", "get_device"),
        ("utils.logger", "get_logger"),
    ]
    for mod_name, attr in modules:
        try:
            mod = __import__(mod_name, fromlist=[attr])
            obj = getattr(mod, attr)
            print(f"  [\u2713]      OK  {mod_name}.{attr}")
        except Exception as exc:
            print(f"  [\u2717]   FAIL  {mod_name}.{attr}  —  {exc}")
            failures += 1
    print()
    return failures


# ---------------------------------------------------------------------------
# 4. Summary
# ---------------------------------------------------------------------------

def main() -> None:
    print()
    print("  PHASE 1B — SMOKE TEST")
    print("  " + "=" * 58)
    print()

    total_issues = 0
    total_issues += check_relocated_files()
    total_issues += check_syntax()
    total_issues += check_utils()

    print("=" * 60)
    if total_issues == 0:
        print("  ALL CHECKS PASSED  \u2714")
    else:
        print(f"  {total_issues} ISSUE(S) DETECTED  \u2718")
    print("=" * 60)
    sys.exit(1 if total_issues else 0)


if __name__ == "__main__":
    main()
