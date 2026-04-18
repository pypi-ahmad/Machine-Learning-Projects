"""Build Face Mask Detection notebook — YOLO26m detection, face mask dataset."""
import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell
from pathlib import Path

cells = []

cells.append(new_markdown_cell("""\
# Face Mask Detection with YOLO26m (Notebook-First, End-to-End)

## 1. Title and Project Overview

This notebook is a real, executable face mask detection pipeline trained on the
**face-mask-detection** dataset.

What this notebook does:
- downloads `andrewmvd/face-mask-detection` from Kaggle in notebook cells
- converts Pascal VOC XML annotations to YOLO format
- verifies images, labels, and splits before training
- trains **YOLO26m** mask detector (2–3 classes) with OOM fallback to YOLO26s
- evaluates with mAP50, mAP50-95, precision, recall, and per-class AP
- saves `best.pt`, `best.onnx`, and `metrics.json`"""))

cells.append(new_markdown_cell("""\
## 2. Problem Statement

Detect and classify faces in images by mask-wearing status: with_mask, without_mask,
and optionally mask_weared_incorrect.

- Input: RGB images from diverse sources (social media, surveillance-style frames)
- Output: bounding boxes + class label per face
- Dataset: face-mask-detection — ~850 images with ~3,800 annotated face bboxes
- Challenges: mask occlusion variations, lighting, angle, multiple faces per image"""))

cells.append(new_markdown_cell("""\
## 3. Why the Chosen Method Is Correct

**Task family:** face-level object detection.

- Per saved CV rules: face detection defaults to YOLO26m
- Face mask classification is per-face (not per-image), so bounding-box detection + label is the right approach
- YOLO26m is the April 2026 standard for multi-class face detection; fallback to YOLO26s on OOM"""))

cells.append(new_markdown_cell("## 4. Hardware / Environment Check"))

cells.append(new_code_cell("""\
import os, platform, random
import numpy as np
import torch

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Python  : {platform.python_version()}")
print(f"PyTorch : {torch.__version__}")
print(f"CUDA    : {torch.cuda.is_available()} — {getattr(torch.version, 'cuda', 'N/A')}")
print(f"Device  : {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU     : {torch.cuda.get_device_name(0)}")
    print(f"VRAM    : {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")"""))

cells.append(new_markdown_cell("## 5. Dependency Installation"))

cells.append(new_code_cell("""\
import subprocess, sys, importlib

def ensure_package(import_name: str, pip_name: str | None = None) -> None:
    pip_name = pip_name or import_name
    try:
        importlib.import_module(import_name)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pip_name])

ensure_package("ultralytics")
ensure_package("cv2", "opencv-python")
ensure_package("matplotlib")
ensure_package("pandas")
ensure_package("kagglehub")
ensure_package("lxml")
ensure_package("albumentations")
print("All dependencies satisfied.")"""))

cells.append(new_markdown_cell("## 6. Imports and Configuration"))

cells.append(new_code_cell("""\
import json, os, shutil, random
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

PROJECT_DIR   = Path(os.path.dirname(os.path.abspath("__file__")))
DATA_ROOT     = PROJECT_DIR.parents[2] / "data"
RUNS_DIR      = PROJECT_DIR.parents[2] / "runs"
ARTIFACTS_DIR = PROJECT_DIR / "artifacts"

for d in (DATA_ROOT, RUNS_DIR, ARTIFACTS_DIR):
    d.mkdir(parents=True, exist_ok=True)

print(f"DATA_ROOT    : {DATA_ROOT}")
print(f"ARTIFACTS_DIR: {ARTIFACTS_DIR}")"""))

cells.append(new_markdown_cell("""\
## 7. Dataset Source Explanation

**Dataset:** Kaggle — `andrewmvd/face-mask-detection`

- Source URL: https://www.kaggle.com/datasets/andrewmvd/face-mask-detection
- ~850 images, ~3,800 annotated faces with mask-wearing status
- Annotations: Pascal VOC XML format in `annotations/` folder
- Classes: `with_mask`, `without_mask`, and sometimes `mask_weared_incorrect`
- This notebook converts XML → YOLO format

**Credential requirements:**
- Set `KAGGLE_USERNAME` + `KAGGLE_KEY` env vars or place `kaggle.json` in `~/.kaggle/`
- Raises a clear error if credentials are missing — no synthetic fallback"""))

cells.append(new_code_cell("""\
import subprocess

KAGGLE_DATASET = "andrewmvd/face-mask-detection"
DATASET_DIR = DATA_ROOT / "face_mask_detection"
DATASET_DIR.mkdir(parents=True, exist_ok=True)


def check_kaggle_credentials() -> None:
    has_env  = os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY")
    has_file = (Path.home() / ".kaggle" / "kaggle.json").exists()
    if not has_env and not has_file:
        raise RuntimeError(
            "Kaggle credentials not found.\\n"
            "Set KAGGLE_USERNAME and KAGGLE_KEY env vars, or place kaggle.json in ~/.kaggle/"
        )


def download_dataset() -> Path:
    check_kaggle_credentials()
    try:
        import kagglehub
        path = Path(kagglehub.dataset_download(KAGGLE_DATASET))
        return path
    except Exception:
        pass
    subprocess.check_call([
        "kaggle", "datasets", "download", "-d", KAGGLE_DATASET,
        "-p", str(DATASET_DIR), "--unzip",
    ])
    return DATASET_DIR


print("Downloading face-mask-detection from Kaggle...")
raw_root = Path(download_dataset())
print(f"Raw dataset at: {raw_root}")

print("\\nTop-level contents:")
for p in sorted(raw_root.iterdir()):
    print(f"  {p.name}{'/' if p.is_dir() else ''}")"""))

cells.append(new_markdown_cell("""\
## 8. Pascal VOC XML → YOLO Format Conversion

The face-mask-detection dataset uses Pascal VOC XML annotations.
This cell parses XML and converts bounding boxes to YOLO format."""))

cells.append(new_code_cell("""\
import xml.etree.ElementTree as ET

def find_annotations_root(raw_root: Path) -> Path:
    \"\"\"Find the folder containing annotations/ and images/ subdirs.\"\"\"
    for candidate in [raw_root, *raw_root.rglob("annotations")]:
        folder = candidate if candidate.is_dir() and candidate.name == "annotations" else None
        if folder:
            parent = folder.parent
            if (parent / "images").exists():
                return parent
    raise RuntimeError(f"Dataset structure not found under {raw_root}")


dataset_root = find_annotations_root(raw_root)
print(f"Dataset root: {dataset_root}")

ann_dir = dataset_root / "annotations"
img_dir = dataset_root / "images"

print(f"Annotations: {len(list(ann_dir.glob('*.xml')))} XML files")
print(f"Images     : {len(list(img_dir.glob('*.png')))} + {len(list(img_dir.glob('*.jpg')))} images")

# Parse all XML files to infer class names robustly
classes_found = set()
for xml_path in sorted(ann_dir.glob("*.xml")):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    for obj in root.findall("object"):
        cls = obj.find("name")
        if cls is not None and cls.text:
            classes_found.add(cls.text.strip())

class_list = sorted(list(classes_found))
class_id_map = {name: idx for idx, name in enumerate(class_list)}
print(f"\\nDetected classes: {class_list}")"""))

cells.append(new_code_cell("""\
def parse_voc_xml(xml_path: Path) -> list[tuple[str, float, float, float, float]]:
    \"\"\"Parse VOC XML; return list of (class_name, x1, y1, x2, y2) in absolute pixels.\"\"\"
    tree = ET.parse(xml_path)
    root = tree.getroot()
    bboxes = []
    for obj in root.findall("object"):
        name = obj.find("name").text.strip()
        bbox_elem = obj.find("bndbox")
        x1 = float(bbox_elem.find("xmin").text)
        y1 = float(bbox_elem.find("ymin").text)
        x2 = float(bbox_elem.find("xmax").text)
        y2 = float(bbox_elem.find("ymax").text)
        bboxes.append((name, x1, y1, x2, y2))
    return bboxes


def convert_to_yolo(split_name: str, dataset_root: Path, yolo_root: Path,
                    max_images: int = 10000) -> tuple[int, int]:
    \"\"\"Convert VOC XML → YOLO; return (n_ok, n_skip).\"\"\"
    img_src = dataset_root / "images"
    ann_src = dataset_root / "annotations"
    img_dst = yolo_root / "images" / split_name
    lbl_dst = yolo_root / "labels" / split_name
    img_dst.mkdir(parents=True, exist_ok=True)
    lbl_dst.mkdir(parents=True, exist_ok=True)

    xml_files = sorted(ann_src.glob("*.xml"))
    random.Random(SEED).shuffle(xml_files)
    # Deterministic split: 70% train, 15% val, rest test
    n_total = len(xml_files)
    if split_name == "train":
        xml_files = xml_files[:int(0.70 * n_total)]
    elif split_name == "val":
        xml_files = xml_files[int(0.70 * n_total): int(0.85 * n_total)]
    else:  # test
        xml_files = xml_files[int(0.85 * n_total):]

    n_ok = n_skip = 0
    for xml_path in xml_files[:max_images]:
        try:
            bboxes = parse_voc_xml(xml_path)
            if not bboxes:
                n_skip += 1
                continue

            # Find corresponding image
            stem = xml_path.stem
            img_path = None
            for ext in [".jpg", ".png", ".JPG", ".PNG"]:
                candidate = img_src / f"{stem}{ext}"
                if candidate.exists():
                    img_path = candidate
                    break
            if img_path is None:
                # Try alternate naming
                for candidate in img_src.glob(f"{stem}*"):
                    if candidate.suffix.lower() in [".jpg", ".png"]:
                        img_path = candidate
                        break

            if img_path is None or not img_path.exists():
                n_skip += 1
                continue

            img = cv2.imread(str(img_path))
            if img is None:
                n_skip += 1
                continue
            H, W = img.shape[:2]

            # Convert bboxes to YOLO format
            yolo_lines = []
            for class_name, x1, y1, x2, y2 in bboxes:
                if class_name not in class_id_map:
                    continue
                cls_id = class_id_map[class_name]
                xc = (x1 + x2) / 2 / W
                yc = (y1 + y2) / 2 / H
                bw = (x2 - x1) / W
                bh = (y2 - y1) / H
                xc, yc = max(0.0, min(1.0, xc)), max(0.0, min(1.0, yc))
                bw, bh = max(0.001, min(1.0, bw)), max(0.001, min(1.0, bh))
                yolo_lines.append(f"{cls_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")

            if not yolo_lines:
                n_skip += 1
                continue

            # Copy image, write label
            dst_img = img_dst / img_path.name
            dst_lbl = lbl_dst / f"{stem}.txt"
            shutil.copy2(img_path, dst_img)
            dst_lbl.write_text("\\n".join(yolo_lines))
            n_ok += 1
        except Exception as e:
            print(f"Error processing {xml_path}: {e}")
            n_skip += 1

    return n_ok, n_skip


YOLO_ROOT = DATASET_DIR / "yolo_dataset"

for split in ("train", "val", "test"):
    print(f"Converting {split}...")
    ok, skip = convert_to_yolo(split, dataset_root, YOLO_ROOT)
    print(f"  OK: {ok}  Skipped: {skip}")

# Write data.yaml
DATA_YAML = YOLO_ROOT / "data.yaml"
DATA_YAML.write_text(
    f"path: {YOLO_ROOT}\\n"
    "train: images/train\\n"
    "val: images/val\\n"
    "test: images/test\\n"
    f"nc: {len(class_list)}\\n"
    f"names: {class_list}\\n"
)
print(f"\\ndata.yaml written: {DATA_YAML}")"""))

cells.append(new_markdown_cell("## 9. Dataset Verification"))

cells.append(new_code_cell("""\
def count_split(split):
    imgs = list((YOLO_ROOT / "images" / split).glob("*.*")) if (YOLO_ROOT / "images" / split).exists() else []
    lbls = list((YOLO_ROOT / "labels" / split).glob("*.txt")) if (YOLO_ROOT / "labels" / split).exists() else []
    return len(imgs), len(lbls)

ti, tl = count_split("train")
vi, vl = count_split("val")
sti, stl = count_split("test")

print(f"Train: {ti:4d} images, {tl:4d} labels")
print(f"Val  : {vi:4d} images, {vl:4d} labels")
print(f"Test : {sti:4d} images, {stl:4d} labels")

assert ti >= 50, f"Too few training images: {ti}"
assert vi >= 10, f"Too few val images: {vi}"
assert DATA_YAML.exists(), "data.yaml missing"
print("Dataset verification passed.")"""))

cells.append(new_markdown_cell("## 10. Data Integrity Audit"))

cells.append(new_code_cell("""\
train_img_dir = YOLO_ROOT / "images" / "train"
train_lbl_dir = YOLO_ROOT / "labels" / "train"
all_train = list(train_img_dir.glob("*.*"))

sample = random.sample(all_train, min(300, len(all_train)))
corrupt, widths, heights = 0, [], []

for img_path in sample:
    img = cv2.imread(str(img_path))
    if img is None:
        corrupt += 1
        continue
    h, w = img.shape[:2]
    widths.append(w)
    heights.append(h)

print(f"Sampled      : {len(sample)}")
print(f"Corrupt      : {corrupt}")
if widths:
    print(f"Width range  : {min(widths)}–{max(widths)} px")
    print(f"Height range : {min(heights)}–{max(heights)} px")

# Instance counts per class
from collections import Counter
inst_counts = Counter()
for lbl_file in list(train_lbl_dir.glob("*.txt"))[:500]:
    for line in lbl_file.read_text().strip().splitlines():
        parts = line.split()
        if parts:
            inst_counts[int(parts[0])] += 1

print("\\nClass instance counts (train sample):")
for cls_id in sorted(inst_counts.keys()):
    cls_name = class_list[cls_id] if cls_id < len(class_list) else str(cls_id)
    cnt = inst_counts[cls_id]
    print(f"  [{cls_id}] {cls_name:20s}: {cnt:5d}")"""))

cells.append(new_markdown_cell("## 11. Sample Visualization — Faces with GT Bboxes"))

cells.append(new_code_cell("""\
sample_6 = random.sample(all_train, min(6, len(all_train)))
COLORS = {i: tuple(np.random.randint(0, 255, 3).tolist()) for i in range(len(class_list))}

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()

for ax, img_path in zip(axes, sample_6):
    img_bgr = cv2.imread(str(img_path))
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    H, W = img_rgb.shape[:2]

    lbl_path = train_lbl_dir / img_path.with_suffix(".txt").name
    n_faces = 0
    if lbl_path.exists():
        for line in lbl_path.read_text().strip().splitlines():
            parts = line.split()
            if len(parts) != 5:
                continue
            cls_id = int(parts[0])
            xc, yc, bw, bh = map(float, parts[1:])
            x1 = int((xc - bw/2) * W)
            y1 = int((yc - bh/2) * H)
            x2 = int((xc + bw/2) * W)
            y2 = int((yc + bh/2) * H)
            color = COLORS[cls_id]
            cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, 2)
            cls_name = class_list[cls_id] if cls_id < len(class_list) else str(cls_id)
            cv2.putText(img_bgr, cls_name, (x1, max(y1-4, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            n_faces += 1

    ax.imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    ax.set_title(f"{n_faces} face(s)", fontsize=8)
    ax.axis("off")

plt.suptitle("Sample training images with GT face mask bboxes", fontsize=12)
plt.tight_layout()
fig.savefig(str(ARTIFACTS_DIR / "sample_images.png"), dpi=100)
plt.close(fig)
print("Saved sample_images.png")"""))

cells.append(new_markdown_cell("## 12. Preprocessing / Augmentation Strategy"))

cells.append(new_code_cell("""\
# YOLO26m handles augmentation internally (mosaic, flip, HSV, scale).
# Below is an Albumentations preview.
import albumentations as A

aug = A.Compose([
    A.RandomBrightnessContrast(p=0.5),
    A.HueSaturationValue(p=0.4),
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=10, p=0.3),
], bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"]))

sample_img_path = random.choice(all_train)
sample_lbl_path = train_lbl_dir / sample_img_path.with_suffix(".txt").name
img_rgb = cv2.cvtColor(cv2.imread(str(sample_img_path)), cv2.COLOR_BGR2RGB)

bboxes, class_labels = [], []
if sample_lbl_path.exists():
    for line in sample_lbl_path.read_text().strip().splitlines():
        p = line.split()
        if len(p) == 5:
            class_labels.append(int(p[0]))
            bboxes.append(list(map(float, p[1:])))

try:
    aug_result = aug(image=img_rgb, bboxes=bboxes, class_labels=class_labels)
    aug_img = aug_result["image"]
except Exception:
    aug_img = img_rgb

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].imshow(cv2.resize(img_rgb, (320, 240))); axes[0].set_title("Original"); axes[0].axis("off")
axes[1].imshow(cv2.resize(aug_img, (320, 240))); axes[1].set_title("Augmented (demo)"); axes[1].axis("off")
plt.tight_layout()
fig.savefig(str(ARTIFACTS_DIR / "augmentation_preview.png"), dpi=100)
plt.close(fig)
print("Saved augmentation_preview.png")"""))

cells.append(new_markdown_cell("## 13. Train/Validation/Test Split Verification"))

cells.append(new_code_cell("""\
split_df = pd.DataFrame({
    "split":  ["train", "val", "test"],
    "images": [ti, vi, sti],
    "labels": [tl, vl, stl],
})
print(split_df.to_string(index=False))
assert ti >= 50, "Insufficient training data"
print("\\nSplit verification passed.")"""))

cells.append(new_markdown_cell("## 14. Baseline — Pretrained YOLO26m (COCO)"))

cells.append(new_code_cell("""\
from ultralytics import YOLO

baseline = YOLO("yolo26m.pt")
val_imgs = list((YOLO_ROOT / "images" / "val").glob("*.*"))
sanity_6 = random.sample(val_imgs, min(6, len(val_imgs)))

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()
for ax, ip in zip(axes, sanity_6):
    res = baseline.predict(str(ip), verbose=False)[0]
    n = len(res.boxes) if res.boxes is not None else 0
    ax.imshow(res.plot()[:, :, ::-1])
    ax.set_title(f"Pretrained: {n} face(s)", fontsize=8)
    ax.axis("off")
plt.suptitle("Baseline — pretrained YOLO26m (COCO weights)", fontsize=11)
plt.tight_layout()
fig.savefig(str(ARTIFACTS_DIR / "sanity_inference.png"), dpi=100)
plt.close(fig)
print("Saved sanity_inference.png")"""))

cells.append(new_markdown_cell("## 15. Main Model Setup"))

cells.append(new_code_cell("""\
PREFERRED_MODEL = "yolo26m.pt"
FALLBACK_MODEL  = "yolo26s.pt"

IMG_SIZE = 640
EPOCHS   = 30
BATCH    = 16
WORKERS  = 2

TRAIN_PROJECT = RUNS_DIR / "face_mask_det"
TRAIN_NAME    = "yolo26m_mask"

print(f"Model   : {PREFERRED_MODEL}")
print(f"Epochs  : {EPOCHS}")
print(f"Batch   : {BATCH}")
print(f"Classes : {class_list}")
print(f"Data    : {DATA_YAML}")"""))

cells.append(new_markdown_cell("## 16. Training"))

cells.append(new_code_cell("""\
from ultralytics import YOLO


def run_training(model_name: str, batch: int):
    model = YOLO(model_name)
    return model.train(
        data=str(DATA_YAML),
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=batch,
        workers=WORKERS,
        project=str(TRAIN_PROJECT),
        name=TRAIN_NAME,
        exist_ok=True,
        device=DEVICE,
        seed=SEED,
        verbose=True,
    )


train_model_name = PREFERRED_MODEL
current_batch    = BATCH
try:
    train_results = run_training(train_model_name, current_batch)
    print(f"Training complete with {train_model_name}.")
except RuntimeError as exc:
    if "out of memory" in str(exc).lower():
        print(f"OOM with {train_model_name}, retrying with {FALLBACK_MODEL}...")
        torch.cuda.empty_cache()
        train_model_name = FALLBACK_MODEL
        current_batch    = max(4, current_batch // 2)
        train_results    = run_training(train_model_name, current_batch)
        print(f"Training complete with {train_model_name} (OOM fallback).")
    else:
        raise"""))

cells.append(new_markdown_cell("## 17. Validation and Core Metrics"))

cells.append(new_code_cell("""\
from ultralytics import YOLO
import json

best_path = TRAIN_PROJECT / TRAIN_NAME / "weights" / "best.pt"
if not best_path.exists():
    raise FileNotFoundError(f"best.pt not found at {best_path}")

best_model  = YOLO(str(best_path))
val_metrics = best_model.val(
    data=str(DATA_YAML),
    split="val",
    imgsz=IMG_SIZE,
    device=DEVICE,
    verbose=True,
)

map50     = float(val_metrics.box.map50)
map50_95  = float(val_metrics.box.map)
precision = float(val_metrics.box.mp)
recall    = float(val_metrics.box.mr)

print(f"mAP50     : {map50:.4f}")
print(f"mAP50-95  : {map50_95:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")

names     = val_metrics.names
cls_list  = list(names.values()) if isinstance(names, dict) else names
class_aps = {}
ap50_vals = val_metrics.box.ap50
for cls_name, ap in zip(cls_list, ap50_vals):
    class_aps[cls_name] = float(ap)
    print(f"  AP50 [{cls_name}]: {ap:.4f}")

metrics_out = {
    "map50": map50, "map50_95": map50_95,
    "precision": precision, "recall": recall,
    "per_class_ap50": class_aps,
}
with open(ARTIFACTS_DIR / "metrics.json", "w") as f:
    json.dump(metrics_out, f, indent=2)
print("Saved metrics.json")"""))

cells.append(new_markdown_cell("## 18. Error Analysis — Training Curves + Per-Class AP"))

cells.append(new_code_cell("""\
run_dir = TRAIN_PROJECT / TRAIN_NAME

for fname in ["results.png", "PR_curve.png", "F1_curve.png", "confusion_matrix.png"]:
    fpath = run_dir / fname
    if fpath.exists():
        img = cv2.cvtColor(cv2.imread(str(fpath)), cv2.COLOR_BGR2RGB)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(img); ax.axis("off"); ax.set_title(fname)
        plt.tight_layout()
        fig.savefig(str(ARTIFACTS_DIR / fname), dpi=80)
        plt.close(fig)
        print(f"Saved {fname}")

# Per-class AP50 bar chart
if class_aps:
    names_sorted, aps_sorted = zip(*sorted(class_aps.items(), key=lambda x: -x[1]))
    fig, ax = plt.subplots(figsize=(max(6, len(names_sorted) * 0.8), 5))
    ax.bar(range(len(names_sorted)), aps_sorted, color="steelblue")
    ax.set_xticks(range(len(names_sorted)))
    ax.set_xticklabels(names_sorted, rotation=45, ha="right")
    ax.set_ylabel("AP50")
    ax.set_title("Per-class Detection AP50 (val set)")
    ax.set_ylim(0, 1)
    plt.tight_layout()
    fig.savefig(str(ARTIFACTS_DIR / "per_class_ap50.png"), dpi=100)
    plt.close(fig)
    print("Saved per_class_ap50.png")"""))

cells.append(new_markdown_cell("## 19. Inference on Holdout Examples"))

cells.append(new_code_cell("""\
test_img_dir = YOLO_ROOT / "images" / "test"
test_all_imgs = list(test_img_dir.glob("*.*")) if test_img_dir.exists() else list((YOLO_ROOT / "images" / "val").glob("*.*"))
holdout_sample = random.sample(test_all_imgs, min(6, len(test_all_imgs)))

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()
for ax, img_path in zip(axes, holdout_sample):
    result = best_model.predict(str(img_path), conf=0.25, verbose=False)[0]
    n_det = len(result.boxes) if result.boxes is not None else 0
    ax.imshow(result.plot()[:, :, ::-1])
    ax.set_title(f"{img_path.name[:20]}\\n{n_det} face(s)", fontsize=7)
    ax.axis("off")
plt.suptitle("Holdout inference — fine-tuned YOLO26m", fontsize=11)
plt.tight_layout()
fig.savefig(str(ARTIFACTS_DIR / "holdout_inference.png"), dpi=100)
plt.close(fig)
print("Saved holdout_inference.png")"""))

cells.append(new_markdown_cell("## 20. Save Model / Artifacts"))

cells.append(new_code_cell("""\
import shutil

saved_pt = ARTIFACTS_DIR / "best.pt"
shutil.copy2(best_path, saved_pt)
print(f"Saved best.pt → {saved_pt}")

export_model = YOLO(str(saved_pt))
onnx_path = export_model.export(format="onnx", imgsz=IMG_SIZE, opset=12)
saved_onnx = ARTIFACTS_DIR / "best.onnx"
shutil.copy2(onnx_path, saved_onnx)
print(f"Exported ONNX → {saved_onnx}")

with open(ARTIFACTS_DIR / "metrics.json") as f:
    m = json.load(f)

manifest = {
    "project":    "Face Mask Detection (YOLO26m)",
    "model_file": "best.pt",
    "onnx_file":  "best.onnx",
    "dataset":    "andrewmvd/face-mask-detection",
    "classes":    class_list,
    "metrics":    m,
}
with open(ARTIFACTS_DIR / "artifact_manifest.json", "w") as f:
    json.dump(manifest, f, indent=2)
print("Saved artifact_manifest.json")"""))

cells.append(new_markdown_cell("""\
## 21. Reproducibility Notes

- Random seed fixed to `42` throughout
- YOLO26m training uses `seed=42`
- Dataset split: 70% train, 15% val, 15% test (deterministic by XML file order)
- Kaggle API caches downloads; re-running reuses cache"""))

cells.append(new_markdown_cell("""\
## 22. Conclusion and Limitations

This notebook implements a real end-to-end face mask detection pipeline:
- Real Kaggle download with fail-loud credential checks
- Pascal VOC XML → YOLO bbox format conversion
- YOLO26m fine-tuned on 2–3 mask-wearing classes with OOM fallback to YOLO26s
- mAP50, mAP50-95, precision, recall, per-class AP50 + metrics.json export
- Per-class AP50 bar chart + training curves + holdout inference grid

**Limitations:**
- Dataset is relatively small (~850 images); larger datasets improve generalization
- Masks are hand-drawn annotations (not pixel-perfect); VOC format has some boundary noise
- Classes may have imbalance (e.g. more 'with_mask' than 'without_mask' samples)
- Real-world performance depends on lighting, angle, occlusion level beyond train distribution"""))

cells.append(new_markdown_cell("""\
## 23. How to Improve This Project

1. Train on larger face mask datasets (Kaggle competitions, synthetic augmentation)
2. Use **data augmentation** strategies: Mosaic, CutMix, AutoAugment
3. Apply **Focal Loss** if class imbalance is severe
4. Add **ensemble** predictions (multiple model snapshots)
5. Fine-tune on **domain-specific data** (e.g. real surveillance/shop camera frames)
6. Export to **TensorRT** for real-time edge deployment"""))

nb = new_notebook(cells=cells)
nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
nb.metadata["language_info"] = {"name": "python", "version": "3.10.0"}

NB_PATH = Path(
    r"e:\Github\Machine-Learning-Projects\Computer Vision"
    r"\Face Mask Detection\Source Code\face_mask_detection_pipeline.ipynb"
)
NB_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(NB_PATH, "w", encoding="utf-8") as f:
    nbformat.write(nb, f)

print(f"Notebook written : {NB_PATH}")
print(f"Total cells      : {len(nb.cells)}")
