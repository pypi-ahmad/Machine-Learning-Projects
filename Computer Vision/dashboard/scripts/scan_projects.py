"""Scan all modern.py files to build the project manifest for the dashboard.

This script introspects every registered CVProject subclass and extracts:
- registry key, project_type, description, legacy_tech, modern_tech
- available files (train.py, infer.py, config.py, README.md)
- dataset config info from configs/datasets/*.yaml
- folder path, input modes, model family

Outputs ``dashboard/public/data/projects.json`` consumed by the frontend.

Usage::

    python dashboard/scripts/scan_projects.py
    python dashboard/scripts/scan_projects.py --pretty
"""

from __future__ import annotations

import argparse
import ast
import json
import logging
import re
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

log = logging.getLogger("scan_projects")

# ── Category normalisation ──────────────────────────────────

CATEGORY_MAP = {
    "detection": "Detection",
    "classification": "Classification",
    "segmentation": "Segmentation",
    "pose": "Pose & Landmarks",
    "tracking": "Tracking",
    "retrieval": "Retrieval & Search",
    "ocr": "OCR & Document AI",
    "anomaly": "Anomaly Detection",
    "utility": "OpenCV Utilities",
}

# Heuristic sub-categories based on keywords in project key/description
TAG_RULES: list[tuple[str, list[str]]] = [
    ("Medical", ["tumour", "tumor", "lesion", "polyp", "wound", "lung", "cell", "skin_cancer", "medical"]),
    ("Industrial", ["defect", "scratch", "crack", "conveyor", "anomaly"]),
    ("Safety", ["ppe", "fire", "smoke", "drowsiness", "compliance"]),
    ("Traffic", ["traffic", "parking", "vehicle", "pothole", "road_lane", "licence_plate", "number_plate"]),
    ("Agriculture", ["crop", "weed", "plant_disease", "cactus", "food_freshness"]),
    ("Face", ["face", "emotion", "age_gender", "celebrity", "blink", "gaze", "spoofing"]),
    ("Document AI", ["invoice", "receipt", "business_card", "id_card", "form_ocr", "exam_sheet", "prescription", "document", "handwrit"]),
    ("Gesture & Pose", ["gesture", "finger", "sign_language", "exercise", "yoga", "pose"]),
    ("Aerial & Remote", ["aerial", "drone", "ship", "building_footprint", "waterbody", "flood"]),
    ("Retail & Ecommerce", ["retail", "shelf", "ecommerce", "product", "logo", "counterfeit"]),
]

MODEL_FAMILIES = {
    "yolo": "YOLO",
    "resnet": "ResNet",
    "mobilenet": "MobileNet",
    "mediapipe": "MediaPipe",
    "deepface": "DeepFace",
    "insightface": "InsightFace",
    "paddleocr": "PaddleOCR",
    "easyocr": "EasyOCR",
    "trocr": "TrOCR",
    "sam": "SAM",
    "slic": "SLIC",
    "autoencoder": "AutoEncoder",
    "opencv": "OpenCV",
}

INPUT_MODES = {
    "image": True,   # almost all support image
    "video": False,
    "webcam": False,
    "folder": False,  # batch
    "text": False,
    "pdf": False,
}


def _infer_tags(key: str, desc: str) -> list[str]:
    """Assign sub-category tags based on project key and description."""
    combined = f"{key} {desc}".lower()
    tags = []
    for tag, keywords in TAG_RULES:
        if any(kw in combined for kw in keywords):
            tags.append(tag)
    return tags


def _infer_model_family(modern_tech: str) -> list[str]:
    """Extract model family labels from the modern_tech string."""
    tech_lower = modern_tech.lower()
    families = []
    for needle, label in MODEL_FAMILIES.items():
        if needle in tech_lower:
            families.append(label)
    return families or ["Custom"]


def _infer_input_modes(key: str, folder: Path) -> list[str]:
    """Guess supported input modes from project files."""
    modes = ["image"]  # baseline

    # Check infer.py or modern.py for video/webcam/folder clues
    for fname in ("infer.py", "modern.py", "train.py"):
        fpath = folder / fname
        if fpath.exists():
            try:
                text = fpath.read_text(encoding="utf-8", errors="replace")
                if "video" in text.lower() or "cap" in text.lower():
                    if "video" not in modes:
                        modes.append("video")
                if "webcam" in text.lower() or "camera" in text.lower() or "cap(0)" in text.lower():
                    if "webcam" not in modes:
                        modes.append("webcam")
                if "folder" in text.lower() or "batch" in text.lower() or "glob" in text.lower():
                    if "folder" not in modes:
                        modes.append("folder")
                if "pdf" in text.lower():
                    if "pdf" not in modes:
                        modes.append("pdf")
            except Exception:
                pass

    return modes


def _extract_register_metadata(modern_py: Path) -> list[dict]:
    """Parse modern.py AST to extract @register keys and class metadata."""
    try:
        source = modern_py.read_text(encoding="utf-8", errors="replace")
        tree = ast.parse(source)
    except (SyntaxError, UnicodeDecodeError):
        return []

    results = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue

        # Find @register decorators
        keys = []
        for dec in node.decorator_list:
            if isinstance(dec, ast.Call) and isinstance(dec.func, ast.Name) and dec.func.id == "register":
                if dec.args and isinstance(dec.args[0], ast.Constant):
                    keys.append(dec.args[0].value)

        if not keys:
            continue

        # Extract class-level attributes
        meta = {
            "class_name": node.name,
            "keys": keys,
            "project_type": "",
            "description": "",
            "legacy_tech": "",
            "modern_tech": "",
        }
        for item in node.body:
            if isinstance(item, ast.Assign):
                for target in item.targets:
                    if isinstance(target, ast.Name) and isinstance(item.value, ast.Constant):
                        if target.id in meta:
                            meta[target.id] = item.value.value

        results.append(meta)

    return results


def _load_dataset_configs() -> dict[str, dict]:
    """Load all configs/datasets/*.yaml files."""
    configs_dir = REPO_ROOT / "configs" / "datasets"
    datasets = {}
    if not configs_dir.exists():
        return datasets

    for yaml_path in sorted(configs_dir.glob("*.yaml")):
        try:
            import yaml
            data = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
            key = yaml_path.stem
            datasets[key] = data
        except Exception:
            pass

    return datasets


def _check_data_ready(key: str) -> bool:
    """Check if dataset is already downloaded."""
    data_dir = REPO_ROOT / "data" / key
    marker = data_dir / "processed" / ".ready"
    return marker.exists() or (data_dir / "dataset_info.json").exists()


def scan_all_projects(*, pretty: bool = False) -> dict:
    """Scan the repo and generate the full project manifest."""
    dataset_configs = _load_dataset_configs()

    projects = []
    seen_keys: set[str] = set()

    # Scan all modern.py files
    for pattern in (
        "*/Source Code/modern.py",
        "*/Souce Code/modern.py",
    ):
        for modern_py in sorted(REPO_ROOT.glob(pattern)):
            src_dir = modern_py.parent
            project_dir = src_dir.parent

            entries = _extract_register_metadata(modern_py)
            if not entries:
                continue

            for entry in entries:
                primary_key = entry["keys"][0]
                if primary_key in seen_keys:
                    continue
                seen_keys.add(primary_key)

                # Files available
                has_train = (src_dir / "train.py").exists()
                has_infer = (src_dir / "infer.py").exists()
                has_config = (src_dir / "config.py").exists()
                has_readme = (project_dir / "README.md").exists() or (src_dir / "README.md").exists()

                # Dataset info
                ds_cfg = dataset_configs.get(primary_key, {})
                # Try alternate keys
                for alt_key in entry["keys"]:
                    if alt_key in dataset_configs:
                        ds_cfg = dataset_configs[alt_key]
                        break

                data_ready = _check_data_ready(primary_key)

                # Normalise category
                ptype = entry["project_type"].lower()
                category = CATEGORY_MAP.get(ptype, "Other")

                tags = _infer_tags(primary_key, entry["description"])
                model_family = _infer_model_family(entry["modern_tech"])
                input_modes = _infer_input_modes(primary_key, src_dir)

                project = {
                    "key": primary_key,
                    "aliases": entry["keys"][1:] if len(entry["keys"]) > 1 else [],
                    "name": project_dir.name,
                    "className": entry["class_name"],
                    "category": category,
                    "projectType": ptype,
                    "description": entry["description"],
                    "legacyTech": entry["legacy_tech"],
                    "modernTech": entry["modern_tech"],
                    "tags": tags,
                    "modelFamily": model_family,
                    "inputModes": input_modes,
                    "hasTraining": has_train,
                    "hasInference": has_infer,
                    "hasConfig": has_config,
                    "hasReadme": has_readme,
                    "folderPath": str(project_dir.relative_to(REPO_ROOT)),
                    "sourcePath": str(src_dir.relative_to(REPO_ROOT)),
                    "dataset": {
                        "configured": bool(ds_cfg),
                        "ready": data_ready,
                        "type": ds_cfg.get("type") or ds_cfg.get("source", {}).get("type", ""),
                        "id": ds_cfg.get("id") or ds_cfg.get("source", {}).get("url", ""),
                    },
                }
                projects.append(project)

    # Also scan for legacy numbered projects (CV Project N)
    for legacy_dir in sorted(REPO_ROOT.glob("CV Project*")):
        if not legacy_dir.is_dir():
            continue
        key = re.sub(r"[^a-z0-9]+", "_", legacy_dir.name.lower()).strip("_")
        if key in seen_keys:
            continue
        seen_keys.add(key)

        # Try to find Python files
        py_files = list(legacy_dir.rglob("*.py"))
        has_main = any("main" in p.stem.lower() or p.stem.lower() == "app" for p in py_files)

        projects.append({
            "key": key,
            "aliases": [],
            "name": legacy_dir.name,
            "className": "",
            "category": "OpenCV Utilities",
            "projectType": "utility",
            "description": f"Legacy OpenCV project: {legacy_dir.name}",
            "legacyTech": "OpenCV",
            "modernTech": "OpenCV",
            "tags": ["Legacy"],
            "modelFamily": ["OpenCV"],
            "inputModes": ["image", "webcam"],
            "hasTraining": False,
            "hasInference": has_main,
            "hasConfig": False,
            "hasReadme": (legacy_dir / "README.md").exists(),
            "folderPath": str(legacy_dir.relative_to(REPO_ROOT)),
            "sourcePath": str(legacy_dir.relative_to(REPO_ROOT)),
            "dataset": {"configured": False, "ready": False, "type": "", "id": ""},
        })

    # Also scan CV Projects N (plural)
    for legacy_dir in sorted(REPO_ROOT.glob("CV Projects*")):
        if not legacy_dir.is_dir():
            continue
        key = re.sub(r"[^a-z0-9]+", "_", legacy_dir.name.lower()).strip("_")
        if key in seen_keys:
            continue
        seen_keys.add(key)

        py_files = list(legacy_dir.rglob("*.py"))
        has_main = any("main" in p.stem.lower() or p.stem.lower() == "app" for p in py_files)

        projects.append({
            "key": key,
            "aliases": [],
            "name": legacy_dir.name,
            "className": "",
            "category": "OpenCV Utilities",
            "projectType": "utility",
            "description": f"Legacy OpenCV project: {legacy_dir.name}",
            "legacyTech": "OpenCV",
            "modernTech": "OpenCV",
            "tags": ["Legacy"],
            "modelFamily": ["OpenCV"],
            "inputModes": ["image", "webcam"],
            "hasTraining": False,
            "hasInference": has_main,
            "hasConfig": False,
            "hasReadme": (legacy_dir / "README.md").exists(),
            "folderPath": str(legacy_dir.relative_to(REPO_ROOT)),
            "sourcePath": str(legacy_dir.relative_to(REPO_ROOT)),
            "dataset": {"configured": False, "ready": False, "type": "", "id": ""},
        })

    # Sort by category then name
    projects.sort(key=lambda p: (p["category"], p["name"]))

    # Build summary stats
    categories = {}
    for p in projects:
        cat = p["category"]
        categories[cat] = categories.get(cat, 0) + 1

    manifest = {
        "version": "1.0.0",
        "generatedAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "repoVersion": "2.0.0",
        "stats": {
            "totalProjects": len(projects),
            "trainable": sum(1 for p in projects if p["hasTraining"]),
            "withInference": sum(1 for p in projects if p["hasInference"]),
            "withDataset": sum(1 for p in projects if p["dataset"]["configured"]),
            "dataReady": sum(1 for p in projects if p["dataset"]["ready"]),
            "categories": categories,
        },
        "projects": projects,
    }

    return manifest


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Scan projects for dashboard manifest")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON")
    parser.add_argument("--output", type=str, default=None, help="Output path")
    args = parser.parse_args()

    manifest = scan_all_projects(pretty=args.pretty)

    # Default output
    if args.output:
        out = Path(args.output)
    else:
        out = REPO_ROOT / "dashboard" / "public" / "data" / "projects.json"

    out.parent.mkdir(parents=True, exist_ok=True)
    indent = 2 if args.pretty else None
    out.write_text(json.dumps(manifest, indent=indent, ensure_ascii=False), encoding="utf-8")
    log.info("Wrote %d projects → %s", manifest["stats"]["totalProjects"], out)
    print(f"✓ {manifest['stats']['totalProjects']} projects → {out}")
    print(f"  Categories: {manifest['stats']['categories']}")


if __name__ == "__main__":
    main()
