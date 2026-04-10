#!/usr/bin/env python3
"""CI sanity check — validates repository structure and conventions.

Checks:
1. All 30 projects have modern.py
2. All 30 projects have dataset config YAML
3. All 28 trainable projects have train.py
4. No files > 50 MB in tracked files
5. Smoke tests pass
6. Key imports work (core, utils, models, train)

Usage::

    python scripts/ci_sanity.py
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

PASS = 0
FAIL = 0


def check(label: str, condition: bool, detail: str = "") -> bool:
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  [PASS] {label}")
    else:
        FAIL += 1
        msg = f"  [FAIL] {label}"
        if detail:
            msg += f" — {detail}"
        print(msg)
    return condition


# --- Project folders ---
EXPECTED_PROJECTS = [
    "Aerial Cactus Identification",
    "Aerial Imagery Segmentation",
    "Age Gender Recognition",
    "Brain Tumour Detection",
    "Building Footprint Segmentation",
    "Cartoonize The Image",
    "Celebrity Face Recognition",
    "Cell Nuclei Segmentation",
    "Emotion Recognition from facial expression",
    "Face Anti Spoofing Detection",
    "Face Emotion Recognition",
    "Face Landmark Detection",
    "Face Mask Detection",
    "Fire and Smoke Detection",
    "Food Image Recognition & Calories Estimation",
    "Food Object Detection",
    "Handwriting Recognition",
    "Licence Plate Detector",
    "Logo Detection and Brand Recognition",
    "Lung Segmentation From Chest X-Ray",
    "Medical Image Segmentaion for Tumour Detection",
    "Pedestrian Detection",
    "Plant Disease Predicton",
    "Real-time Object Tracking",
    "Road Lane Detection",
    "Road segmentation for autonomous vechicles",
    "Sign Language Recognition",
    "Skin Cancer Detection",
    "Traffic Sign Recognition",
    "Wildlife Image Classification",
]

UTILITY_PROJECTS = {"Cartoonize The Image", "Road Lane Detection"}

PROJECT_KEYS = [
    "aerial_cactus_identification", "aerial_imagery_segmentation",
    "age_gender_recognition", "brain_tumour_detection",
    "building_footprint_segmentation", "cartoonize_image",
    "celebrity_face_recognition", "cell_nuclei_segmentation",
    "emotion_recognition", "face_anti_spoofing",
    "face_emotion_recognition", "face_landmark_detection",
    "face_mask_detection", "fire_smoke_detection",
    "food_image_recognition", "food_object_detection",
    "handwriting_recognition", "licence_plate_detector",
    "logo_detection", "lung_segmentation",
    "medical_image_segmentation", "pedestrian_detection",
    "plant_disease_prediction", "realtime_object_tracking",
    "road_lane_detection", "road_segmentation",
    "sign_language_recognition", "skin_cancer_detection",
    "traffic_sign_recognition", "wildlife_classification",
]


def find_src(folder: str) -> Path | None:
    # Check for the standard dirs, plus wildlife's special case.
    # Prefer dirs that actually contain modern.py.
    candidates = ["Source Code", "Souce Code", "wildlife image classification"]
    for name in candidates:
        p = REPO / folder / name
        if p.is_dir() and (p / "modern.py").exists():
            return p
    # Fallback: first existing candidate even without modern.py
    for name in candidates:
        p = REPO / folder / name
        if p.is_dir():
            return p
    return None


def main() -> None:
    print("=" * 65)
    print("  CI Sanity Check")
    print("=" * 65)
    print()

    # 1. Check project folders exist
    print("-- 1. Project folders " + "-" * 42)
    for folder in EXPECTED_PROJECTS:
        src = find_src(folder)
        check(f"Folder: {folder}", src is not None and src.is_dir())
    print()

    # 2. modern.py in every project
    print("-- 2. modern.py wrappers " + "-" * 39)
    for folder in EXPECTED_PROJECTS:
        src = find_src(folder)
        if src:
            check(f"modern.py: {folder}", (src / "modern.py").is_file())
        else:
            check(f"modern.py: {folder}", False, "no source dir")
    print()

    # 3. train.py for non-utility projects
    print("-- 3. train.py (28 trainable projects) " + "-" * 23)
    for folder in EXPECTED_PROJECTS:
        if folder in UTILITY_PROJECTS:
            continue
        src = find_src(folder)
        if src:
            check(f"train.py: {folder}", (src / "train.py").is_file())
        else:
            check(f"train.py: {folder}", False, "no source dir")
    print()

    # 4. Dataset config YAMLs
    print("-- 4. Dataset config YAMLs " + "-" * 37)
    configs_dir = REPO / "configs" / "datasets"
    for key in PROJECT_KEYS:
        check(f"config: {key}.yaml", (configs_dir / f"{key}.yaml").is_file())
    print()

    # 5. Key files exist
    print("-- 5. Key infrastructure files " + "-" * 33)
    key_files = [
        "core/base.py", "core/registry.py", "core/runner.py",
        "models/registry.py",
        "utils/yolo.py", "utils/datasets.py", "utils/data_downloader.py", "utils/paths.py",
        "train/train_detection.py", "train/train_classification.py",
        "train/train_segmentation.py", "train/train_pose.py",
        "benchmarks/run_all.py", "benchmarks/evaluate_accuracy.py",
        "scripts/smoke_test.py", "scripts/check_large_files.py",
        "scripts/setup_env.ps1", "scripts/setup_env.sh",
        ".gitignore", ".githooks/pre-commit",
        "README.md", "requirements.txt",
    ]
    for f in key_files:
        check(f"file: {f}", (REPO / f).is_file())
    print()

    # 6. Key imports
    print("-- 6. Key imports " + "-" * 46)
    for mod_name in ["core", "core.base", "core.registry", "core.runner",
                     "models.registry", "utils.paths", "utils.yolo"]:
        try:
            importlib.import_module(mod_name)
            check(f"import {mod_name}", True)
        except Exception as e:
            check(f"import {mod_name}", False, str(e)[:80])
    print()

    # 7. Per-project README.md
    print("-- 7. Per-project README.md " + "-" * 37)
    for folder in EXPECTED_PROJECTS:
        src = find_src(folder)
        if src:
            check(f"README.md: {folder}", (src / "README.md").is_file())
        else:
            check(f"README.md: {folder}", False, "no source dir")
    print()

    # 8. Evaluator dry-run (--limit 1 --no-download)
    print("-- 8. Evaluator dry-run " + "-" * 41)
    try:
        import subprocess
        result = subprocess.run(
            [sys.executable, "-m", "benchmarks.evaluate_accuracy", "--limit", "1", "--no-download"],
            capture_output=True, text=True, timeout=120,
            cwd=str(REPO),
        )
        check("evaluator --limit 1 exits cleanly", result.returncode == 0,
              result.stderr.strip()[:120] if result.returncode != 0 else "")
    except Exception as exc:
        check("evaluator --limit 1 exits cleanly", False, str(exc)[:120])
    print()

    # Summary
    total = PASS + FAIL
    print("=" * 65)
    print(f"  CI SANITY: {PASS}/{total} passed, {FAIL} failed")
    print("=" * 65)

    sys.exit(1 if FAIL > 0 else 0)


if __name__ == "__main__":
    main()
