#!/usr/bin/env python3
"""
Phase 1B smoke test — validates that projects are importable, paths resolve,
and critical assets exist.  Does NOT launch GUIs, webcams, or training loops.

Usage:
    python scripts/smoke_test.py           # run all checks
    python scripts/smoke_test.py --verbose # extra detail
"""
from __future__ import annotations

import argparse
import importlib
import sys
import textwrap
import traceback
from pathlib import Path

# ── Bootstrap repo root ────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from utils.paths import (
    PathResolver,
    MODELS_DIR,
    DATA_DIR,
    get_project_dir,
)

paths = PathResolver()

# ═══════════════════════════════════════════════════════════
#  Test definitions
# ═══════════════════════════════════════════════════════════

RESULTS: list[dict] = []


def _record(pid: str, name: str, passed: bool, detail: str = ""):
    RESULTS.append({"pid": pid, "name": name, "passed": passed, "detail": detail})


# ── 1. PathResolver sanity ─────────────────────────────────
def test_path_resolver():
    """PathResolver methods return existing directories for known slugs."""
    checks = [
        ("models", "age_gender_recognition"),
        ("models", "celebrity_face_recognition"),
        ("models", "face_emotion_recognition"),
        ("models", "face_landmark_detection"),
        ("models", "fire_and_smoke_detection"),
        ("models", "food_object_detection"),
        ("models", "licence_plate_detector"),
        ("models", "realtime_object_tracking"),
        ("models", "handwriting_recognition"),
        ("data", "pedestrian_detection"),
        ("data", "road_lane_detection"),
        ("data", "emotion_recognition"),
    ]
    for method, slug in checks:
        p = getattr(paths, method)(slug)
        ok = p.is_dir()
        _record("--", f"PathResolver.{method}('{slug}')", ok,
                str(p) if ok else f"MISSING: {p}")


# ── 2. Critical model files exist ─────────────────────────
def test_model_files():
    """Verify key model weights exist in /models/."""
    expected = {
        "P3":  ("age_gender_recognition", ["age_net.caffemodel", "gender_net.caffemodel"]),
        "P7":  ("celebrity_face_recognition", ["Weights Face Recognitions.h5"]),
        "P11": ("face_emotion_recognition", ["model.h5"]),
        "P12": ("face_landmark_detection", ["shape_predictor_68_face_landmarks.dat"]),
        "P14": ("fire_and_smoke_detection", ["best.pt"]),
        "P16": ("food_object_detection", ["inception_food_rec_50epochs.h5"]),
        "P18": ("licence_plate_detector", ["best.pt"]),
        "P24": ("realtime_object_tracking", ["best.pt"]),
    }
    for pid, (slug, files) in expected.items():
        for fname in files:
            fp = MODELS_DIR / slug / fname
            _record(pid, f"model file: {slug}/{fname}", fp.is_file(),
                    f"size={fp.stat().st_size:,}" if fp.is_file() else f"MISSING: {fp}")


# ── 3. Critical data files exist ──────────────────────────
def test_data_files():
    """Verify relocated media assets exist in /data/."""
    expected = {
        "P9":  ("emotion_recognition", ["tvid.mp4"]),
        "P22": ("pedestrian_detection", ["vid.mp4"]),
        "P25": ("road_lane_detection", ["lane_vid2.mp4"]),
    }
    for pid, (slug, files) in expected.items():
        for fname in files:
            fp = DATA_DIR / slug / fname
            _record(pid, f"data file: {slug}/{fname}", fp.is_file(),
                    f"size={fp.stat().st_size:,}" if fp.is_file() else f"MISSING: {fp}")


# ── 4. get_project_dir resolves for all 30 projects ───────
def test_project_dirs():
    """Every project folder can be resolved via get_project_dir."""
    projects = [
        ("P1", "Aerial Cactus Identification"),
        ("P2", "Aerial Imagery Segmentation"),
        ("P3", "Age Gender Recognition"),
        ("P4", "Brain Tumour Detection"),
        ("P5", "Building Footprint Segmentation"),
        ("P6", "Cartoonize The Image"),
        ("P7", "Celebrity Face Recognition"),
        ("P8", "Cell Nuclei Segmentation"),
        ("P9", "Emotion Recognition from facial expression"),
        ("P10", "Face Anti Spoofing Detection"),
        ("P11", "Face Emotion Recognition"),
        ("P12", "Face Landmark Detection"),
        ("P13", "Face Mask Detection"),
        ("P14", "Fire and Smoke Detection"),
        ("P15", "Food Image Recognition & Calories Estimation"),
        ("P16", "Food Object Detection"),
        ("P17", "Handwriting Recognition"),
        ("P18", "Licence Plate Detector"),
        ("P19", "Logo Detection and Brand Recognition"),
        ("P20", "Lung Segmentation From Chest X-Ray"),
        ("P21", "Medical Image Segmentaion for Tumour Detection"),
        ("P22", "Pedestrian Detection"),
        ("P23", "Plant Disease Predicton"),
        ("P24", "Real-time Object Tracking"),
        ("P25", "Road Lane Detection"),
        ("P26", "Road segmentation for autonomous vechicles"),
        ("P27", "Sign Language Recognition"),
        ("P28", "Skin Cancer Detection"),
        ("P29", "Traffic Sign Recognition"),
        ("P30", "Wildlife Image Classification"),
    ]
    for pid, name in projects:
        try:
            d = get_project_dir(name)
            _record(pid, f"project_dir({name})", d.is_dir(), str(d))
        except FileNotFoundError as e:
            _record(pid, f"project_dir({name})", False, str(e))


# ── 5. Entry point files exist ─────────────────────────────
def test_entry_points():
    """Verify entry point scripts exist for projects that should have them."""
    entry_points = {
        "P3":  ("Age Gender Recognition", "Source Code", "main.py"),
        "P6":  ("Cartoonize The Image", "Source Code", "main.py"),
        "P9":  ("Emotion Recognition from facial expression", "Source Code", "main.py"),
        "P11": ("Face Emotion Recognition", "Source Code", "main.py"),
        "P12": ("Face Landmark Detection", "Source Code", "main.py"),
        "P14": ("Fire and Smoke Detection", "Source Code", "main.py"),
        "P16": ("Food Object Detection", "Source Code", "run.py"),
        "P17": ("Handwriting Recognition", "Source Code", "main.py"),
        "P18": ("Licence Plate Detector", "Source Code", "main.py"),
        "P22": ("Pedestrian Detection", "Source Code", "main.py"),
        "P24": ("Real-time Object Tracking", "Souce Code", "webapp.py"),
        "P25": ("Road Lane Detection", "Source Code", "main.py"),
        "P27": ("Sign Language Recognition", "Souce Code", "app.py"),
        "P29": ("Traffic Sign Recognition", "Souce Code", "run.py"),
    }
    for pid, (proj, src_dir, entry) in entry_points.items():
        fp = REPO_ROOT / proj / src_dir / entry
        _record(pid, f"entry_point: {entry}", fp.is_file(),
                str(fp) if fp.is_file() else f"MISSING: {fp}")


# ── 6. Quick import test for utils ────────────────────────
def test_utils_import():
    """Verify the shared utils package imports cleanly."""
    modules = ["utils", "utils.paths", "utils.device", "utils.logger"]
    for mod_name in modules:
        try:
            importlib.import_module(mod_name)
            _record("--", f"import {mod_name}", True)
        except Exception as e:
            _record("--", f"import {mod_name}", False, str(e))


# ── 7. project_meta.yaml exists for all 30 ────────────────
def test_meta_files():
    """All 30 projects have a project_meta.yaml."""
    count = 0
    for item in REPO_ROOT.iterdir():
        meta = item / "project_meta.yaml"
        if meta.is_file():
            count += 1
    _record("--", f"project_meta.yaml count (expect 30)", count == 30, f"found={count}")


# ═══════════════════════════════════════════════════════════
#  Runner
# ═══════════════════════════════════════════════════════════

def run_all_tests(verbose: bool = False):
    """Execute all smoke tests and print summary."""
    test_funcs = [
        test_utils_import,
        test_path_resolver,
        test_meta_files,
        test_project_dirs,
        test_entry_points,
        test_model_files,
        test_data_files,
    ]

    for fn in test_funcs:
        category = fn.__doc__ or fn.__name__
        print(f"\n{'='*60}")
        print(f"  {category}")
        print(f"{'='*60}")
        before = len(RESULTS)
        try:
            fn()
        except Exception:
            _record("--", fn.__name__, False, traceback.format_exc())
        after = len(RESULTS)
        for r in RESULTS[before:after]:
            icon = "PASS" if r["passed"] else "FAIL"
            line = f"  [{icon}] {r['pid']:>4} | {r['name']}"
            if verbose and r["detail"]:
                line += f"  ->  {r['detail']}"
            elif not r["passed"] and r["detail"]:
                line += f"  ->  {r['detail']}"
            print(line)

    # ── Summary ────────────────────────────────────────────
    total = len(RESULTS)
    passed = sum(1 for r in RESULTS if r["passed"])
    failed = total - passed
    print(f"\n{'='*60}")
    print(f"  SMOKE TEST SUMMARY: {passed}/{total} passed, {failed} failed")
    print(f"{'='*60}")

    if failed:
        print("\nFailed checks:")
        for r in RESULTS:
            if not r["passed"]:
                print(f"  - [{r['pid']}] {r['name']}: {r['detail']}")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 1B smoke test")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show details for passing tests too")
    args = parser.parse_args()
    sys.exit(run_all_tests(verbose=args.verbose))
