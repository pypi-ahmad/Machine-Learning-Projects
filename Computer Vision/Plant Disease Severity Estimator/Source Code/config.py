"""Plant Disease Severity Estimator — configuration and label mappings.

Severity buckets are derived from agronomic knowledge of each disease's
typical impact on plant health.  These are **heuristic estimates**, not
ground-truth severity annotations — the original PlantVillage dataset
labels disease *type*, not degree.

Severity levels:
    0  none     — healthy leaf, no disease
    1  mild     — cosmetic / surface-level damage, low yield impact
    2  moderate — significant leaf damage, moderate yield impact
    3  severe   — systemic infection or devastating disease
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# ── Severity definitions (explicit) ──────────────────────

SEVERITY_LEVELS = {
    0: "none",
    1: "mild",
    2: "moderate",
    3: "severe",
}

SEVERITY_NAMES = ("none", "mild", "moderate", "severe")


# ── Disease → severity mapping ───────────────────────────
# Key  = ImageFolder class name (as produced by PlantVillage)
# Value = (plant, disease, severity_index)
#
# Rationale for each severity assignment is documented inline.

DISEASE_SEVERITY_MAP: dict[str, tuple[str, str, int]] = {
    # ── Apple ─────────────────────────────────────────────
    "Apple___Apple_scab":           ("Apple", "Apple scab", 2),          # defoliating fungus
    "Apple___Black_rot":            ("Apple", "Black rot", 3),           # kills branches, fruit rot
    "Apple___Cedar_apple_rust":     ("Apple", "Cedar apple rust", 1),    # cosmetic, rarely lethal
    "Apple___healthy":              ("Apple", "Healthy", 0),

    # ── Blueberry ─────────────────────────────────────────
    "Blueberry___healthy":          ("Blueberry", "Healthy", 0),

    # ── Cherry ────────────────────────────────────────────
    "Cherry_(including_sour)___Powdery_mildew":
                                    ("Cherry", "Powdery mildew", 1),     # surface fungus
    "Cherry_(including_sour)___healthy":
                                    ("Cherry", "Healthy", 0),

    # ── Corn ──────────────────────────────────────────────
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot":
                                    ("Corn", "Gray leaf spot", 2),       # moderate yield loss
    "Corn_(maize)___Common_rust_":  ("Corn", "Common rust", 1),          # usually mild
    "Corn_(maize)___Northern_Leaf_Blight":
                                    ("Corn", "Northern leaf blight", 2), # moderate
    "Corn_(maize)___healthy":       ("Corn", "Healthy", 0),

    # ── Grape ─────────────────────────────────────────────
    "Grape___Black_rot":            ("Grape", "Black rot", 3),           # destroys fruit
    "Grape___Esca_(Black_Measles)": ("Grape", "Esca (Black Measles)", 3),# systemic, vine death
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)":
                                    ("Grape", "Leaf blight", 2),         # moderate
    "Grape___healthy":              ("Grape", "Healthy", 0),

    # ── Orange ────────────────────────────────────────────
    "Orange___Haunglongbing_(Citrus_greening)":
                                    ("Orange", "Citrus greening", 3),    # devastating, incurable

    # ── Peach ─────────────────────────────────────────────
    "Peach___Bacterial_spot":       ("Peach", "Bacterial spot", 1),      # cosmetic spots
    "Peach___healthy":              ("Peach", "Healthy", 0),

    # ── Pepper ────────────────────────────────────────────
    "Pepper,_bell___Bacterial_spot":("Pepper", "Bacterial spot", 1),     # leaf spots, mild
    "Pepper,_bell___healthy":       ("Pepper", "Healthy", 0),

    # ── Potato ────────────────────────────────────────────
    "Potato___Early_blight":        ("Potato", "Early blight", 2),       # progressive
    "Potato___Late_blight":         ("Potato", "Late blight", 3),        # historically devastating
    "Potato___healthy":             ("Potato", "Healthy", 0),

    # ── Raspberry ─────────────────────────────────────────
    "Raspberry___healthy":          ("Raspberry", "Healthy", 0),

    # ── Soybean ───────────────────────────────────────────
    "Soybean___healthy":            ("Soybean", "Healthy", 0),

    # ── Squash ────────────────────────────────────────────
    "Squash___Powdery_mildew":      ("Squash", "Powdery mildew", 1),     # surface fungus

    # ── Strawberry ────────────────────────────────────────
    "Strawberry___Leaf_scorch":     ("Strawberry", "Leaf scorch", 2),    # moderate
    "Strawberry___healthy":         ("Strawberry", "Healthy", 0),

    # ── Tomato ────────────────────────────────────────────
    "Tomato___Bacterial_spot":      ("Tomato", "Bacterial spot", 2),     # moderate spots
    "Tomato___Early_blight":        ("Tomato", "Early blight", 2),       # progressive
    "Tomato___Late_blight":         ("Tomato", "Late blight", 3),        # devastating
    "Tomato___Leaf_Mold":           ("Tomato", "Leaf mold", 1),          # usually mild
    "Tomato___Septoria_leaf_spot":  ("Tomato", "Septoria leaf spot", 2), # moderate defoliation
    "Tomato___Spider_mites Two-spotted_spider_mite":
                                    ("Tomato", "Spider mites", 1),       # pest, manageable
    "Tomato___Target_Spot":         ("Tomato", "Target spot", 2),        # moderate
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus":
                                    ("Tomato", "Yellow Leaf Curl Virus", 3),  # systemic viral
    "Tomato___Tomato_mosaic_virus": ("Tomato", "Mosaic virus", 3),       # systemic viral
    "Tomato___healthy":             ("Tomato", "Healthy", 0),
}


def parse_class(class_name: str) -> tuple[str, str, int, str]:
    """Parse an ImageFolder class name into structured fields.

    Returns (plant, disease, severity_index, severity_name).
    """
    if class_name in DISEASE_SEVERITY_MAP:
        plant, disease, sev = DISEASE_SEVERITY_MAP[class_name]
        return plant, disease, sev, SEVERITY_NAMES[sev]
    # Fallback for unseen classes
    parts = class_name.split("___")
    plant = parts[0].replace("_", " ") if parts else class_name
    disease = parts[1].replace("_", " ") if len(parts) > 1 else "Unknown"
    sev = 0 if "healthy" in class_name.lower() else 2
    return plant, disease, sev, SEVERITY_NAMES[sev]


# ── Lesion-area proxy hook (placeholder for future work) ─

def estimate_lesion_ratio(
    image_bgr,
    *,
    hue_range: tuple[int, int] = (15, 45),
    sat_min: int = 50,
    val_min: int = 50,
) -> float:
    """Estimate fraction of leaf area showing discolouration.

    This is a **proxy** based on HSV thresholding — not a true
    lesion segmentation.  Useful as an auxiliary severity signal
    but should not be treated as ground truth.

    Returns a float in [0, 1].
    """
    import cv2
    import numpy as np

    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # Green-ish pixels → healthy tissue
    green_mask = (
        (h >= 35) & (h <= 85) &
        (s >= sat_min) & (v >= val_min)
    )
    # Brown/yellow/necrotic pixels → lesion proxy
    lesion_mask = (
        (h >= hue_range[0]) & (h <= hue_range[1]) &
        (s >= sat_min) & (v >= val_min)
    )

    total = max(int(green_mask.sum()) + int(lesion_mask.sum()), 1)
    return float(lesion_mask.sum()) / total


# ── Config dataclass ──────────────────────────────────────

@dataclass
class SeverityConfig:
    """All tuneable knobs for the plant disease severity pipeline."""

    # ── Model ─────────────────────────────────────────────
    model_name: str = "resnet18"
    num_classes: int = 38               # PlantVillage classes
    imgsz: int = 224
    device: str | None = None

    # ── Training ──────────────────────────────────────────
    epochs: int = 25
    batch_size: int = 32
    lr: float = 1e-3
    val_split: float = 0.2
    num_workers: int = 4

    # ── Inference ─────────────────────────────────────────
    weights_path: str = "runs/plant_disease_cls/best_model.pt"
    confidence_threshold: float = 0.3

    # ── Lesion proxy ──────────────────────────────────────
    enable_lesion_proxy: bool = False
    lesion_hue_range: tuple[int, int] = (15, 45)

    # ── Visualisation ─────────────────────────────────────
    font_scale: float = 0.6
    color_none: tuple[int, int, int] = (80, 200, 80)       # green
    color_mild: tuple[int, int, int] = (0, 220, 255)        # yellow
    color_moderate: tuple[int, int, int] = (0, 140, 255)    # orange
    color_severe: tuple[int, int, int] = (0, 0, 220)        # red
    text_color: tuple[int, int, int] = (255, 255, 255)
    grid_thumb_size: int = 180
    grid_cols: int = 4

    # ── Output ────────────────────────────────────────────
    output_dir: str = "output"

    # ── Helpers ────────────────────────────────────────────
    def severity_color(self, sev: int) -> tuple[int, int, int]:
        return [self.color_none, self.color_mild,
                self.color_moderate, self.color_severe][min(sev, 3)]

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> SeverityConfig:
        valid = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {}
        for k, v in d.items():
            if k not in valid:
                continue
            if k.endswith("_color") or k.endswith("_range") and isinstance(v, list):
                v = tuple(v)
            filtered[k] = v
        return cls(**filtered)


def load_config(path: str | Path | None) -> SeverityConfig:
    """Load config from JSON or YAML, falling back to defaults."""
    if path is None:
        return SeverityConfig()
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    if p.suffix in {".yaml", ".yml"}:
        try:
            import yaml
            data = yaml.safe_load(text) or {}
        except ImportError:
            data = json.loads(text)
    else:
        data = json.loads(text)
    return SeverityConfig.from_dict(data)
