#!/usr/bin/env python
"""Extend all dataset YAML configs with download.enabled, auto_download, and expected fields.

Run once:
    python scripts/_extend_dataset_configs.py
"""
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent.parent
CONFIGS = ROOT / "configs" / "datasets"

# Map: project_key → extra download fields
EXTENSIONS = {
    "ball_tracking": {
        "enabled": True,
        "auto_download": True,
        "archive_type": "zip",
        "dest": "data/ball_tracking",
    },
    "blink_detection": {
        "enabled": True,
        "auto_download": False,
        "archive_type": "zip",
        "dest": "data/blink_detection",
    },
    "car_detection": {
        "enabled": True,
        "auto_download": False,
        "archive_type": "zip",
        "dest": "data/car_detection",
    },
    "custom_object_detection": {
        "enabled": True,
        "auto_download": True,
        "archive_type": "zip",
        "dest": "data/custom_object_detection",
    },
    "face_attributes": {
        "enabled": True,
        "auto_download": False,
        "archive_type": "zip",
        "dest": "data/face_attributes",
    },
    "face_detection": {
        "enabled": True,
        "auto_download": True,
        "archive_type": "zip",
        "dest": "data/face_detection",
    },
    "face_detection_haar": {
        "enabled": True,
        "auto_download": True,
        "archive_type": "tar.gz",
        "dest": "data/face_detection_haar",
    },
    "face_mask_detection": {
        "enabled": True,
        "auto_download": True,
        "archive_type": "zip",
        "dest": "data/face_mask_detection",
    },
    "facial_landmarks": {
        "enabled": True,
        "auto_download": False,
        "archive_type": "zip",
        "dest": "data/facial_landmarks",
    },
    "finger_counter": {
        "enabled": True,
        "auto_download": False,
        "archive_type": "zip",
        "dest": "data/finger_counter",
    },
    "hand_tracking": {
        "enabled": True,
        "auto_download": False,
        "archive_type": "zip",
        "dest": "data/hand_tracking",
    },
    "image_segmentation": {
        "enabled": True,
        "auto_download": False,
        "archive_type": "zip",
        "dest": "data/image_segmentation",
    },
    "object_detection": {
        "enabled": True,
        "auto_download": True,
        "archive_type": "zip",
        "dest": "data/object_detection",
    },
    "pose_detector": {
        "enabled": True,
        "auto_download": True,
        "archive_type": "zip",
        "dest": "data/pose_detector",
    },
    "sudoku_solver": {
        "enabled": True,
        "auto_download": True,
        "archive_type": "gz",
        "dest": "data/sudoku_solver",
    },
    "text_detection": {
        "enabled": True,
        "auto_download": True,
        "archive_type": "zip",
        "dest": "data/text_detection",
    },
    "volume_controller": {
        "enabled": True,
        "auto_download": False,
        "archive_type": "zip",
        "dest": "data/volume_controller",
    },
}

# Expected marker files per project
EXPECTED_FILES = {
    "ball_tracking": ["data.yaml", "train/images", "valid/images"],
    "blink_detection": ["data.yaml", "train/images", "valid/images"],
    "car_detection": ["data.yaml", "train/images", "valid/images"],
    "custom_object_detection": ["data.yaml", "train/images", "valid/images"],
    "face_attributes": ["data.yaml", "train/images", "valid/images"],
    "face_detection": ["data.yaml", "train/images", "valid/images"],
    "face_detection_haar": ["data.yaml", "train/images", "valid/images"],
    "face_mask_detection": ["data.yaml", "train/images", "valid/images"],
    "facial_landmarks": ["data.yaml", "train/images", "valid/images"],
    "finger_counter": ["data.yaml", "train/images", "valid/images"],
    "hand_tracking": ["data.yaml", "train/images", "valid/images"],
    "image_segmentation": ["data.yaml", "train/images", "valid/images"],
    "object_detection": ["data.yaml", "train/images", "valid/images"],
    "pose_detector": ["data.yaml", "train/images", "valid/images"],
    "sudoku_solver": ["train/0", "train/1", "val/0", "val/1"],
    "text_detection": ["data.yaml", "train/images", "valid/images"],
    "volume_controller": ["data.yaml", "train/images", "valid/images"],
}


def extend_yaml(path: Path, ext: dict, expected: list[str]) -> None:
    text = path.read_text(encoding="utf-8")

    # If already extended, skip
    if "auto_download:" in text:
        print(f"  SKIP (already extended): {path.name}")
        return

    # Build extension block
    lines = [
        f"  enabled: {str(ext['enabled']).lower()}",
        f"  auto_download: {str(ext['auto_download']).lower()}",
        f"  archive_type: \"{ext['archive_type']}\"",
        f"  dest: \"{ext['dest']}\"",
    ]

    # Add expected files
    expected_lines = [f"  expected:"]
    for e in expected:
        expected_lines.append(f"    - \"{e}\"")

    # Insert after the last line of the download.sources block
    # Find the last line of the file and append
    if not text.endswith("\n"):
        text += "\n"

    extension = "\n".join(lines) + "\n" + "\n".join(expected_lines) + "\n"
    text += extension

    path.write_text(text, encoding="utf-8")
    print(f"  OK: {path.name}")


def main():
    print("Extending dataset configs with download metadata...")
    for pk, ext in EXTENSIONS.items():
        yaml_path = CONFIGS / f"{pk}.yaml"
        if not yaml_path.exists():
            print(f"  MISSING: {yaml_path.name}")
            continue
        expected = EXPECTED_FILES.get(pk, [])
        extend_yaml(yaml_path, ext, expected)
    print("\nDone.")


if __name__ == "__main__":
    main()
