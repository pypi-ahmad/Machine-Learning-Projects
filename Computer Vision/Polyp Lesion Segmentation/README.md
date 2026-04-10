# Polyp Lesion Segmentation

Segment polyp regions in colonoscopy images using **YOLO26m-seg** as the baseline, with an optional **MedSAM comparison path** for future integration.

> **DISCLAIMER — NOT A MEDICAL DEVICE**
>
> This software is provided for **research and educational purposes only**.
> It is **NOT** intended to be used as a medical device, diagnostic tool,
> or substitute for professional medical judgement.  **Do not** use outputs
> from this tool to make clinical decisions.  Always consult a qualified
> gastroenterologist for polyp assessment and treatment.

---

## Features

| Capability | Details |
|---|---|
| **Polyp Segmentation** | Pixel-level polyp masks from images, video, or webcam |
| **Per-Image Metrics** | Area (px), coverage ratio, instance count, mean confidence |
| **Dice & IoU Scoring** | When ground-truth masks are provided |
| **Comparison Hooks** | Pluggable backend registry — YOLO baseline + optional MedSAM |
| **Visual Reports** | Polyp mask overlay with contours, bboxes, stats panel |
| **GT Comparison View** | Side-by-side prediction vs ground truth |
| **Structured Export** | JSON per image, CSV batch, mask PNGs |
| **Public Dataset** | Auto-download Kvasir-SEG from Kaggle (`dankok/kvasir-seg`) |

---

## Architecture

The project cleanly separates the **baseline** and **comparison** paths:

```
┌─────────────┐     ┌──────────────────────┐
│  Controller  │────▶│  comparison.py       │
│              │     │  ┌────────────────┐  │
│  process()   │     │  │ YOLO backend   │  │  ← always available
│              │     │  └────────────────┘  │
│              │     │  ┌────────────────┐  │
│              │     │  │ MedSAM backend │  │  ← optional, auto-detected
│              │     │  └────────────────┘  │
└─────────────┘     └──────────────────────┘
```

- `--backend yolo` (default): Uses YOLO26m-seg baseline
- `--backend medsam`: Uses MedSAM if `segment_anything` is installed

---

## Metrics

### Per-Image

$$\text{polyp\_coverage} = \frac{\text{polyp\_area\_px}}{\text{total\_image\_px}}$$

### With Ground Truth

$$\text{Dice} = \frac{2 \cdot |P \cap G|}{|P| + |G|}$$

$$\text{IoU} = \frac{|P \cap G|}{|P \cup G|}$$

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Single image
python "Source Code/infer.py" --source polyp.jpg

# Webcam
python "Source Code/infer.py" --source 0

# Directory with ground-truth evaluation
python "Source Code/infer.py" --source images/ --gt-dir masks/ \
    --save-annotated --export-csv results.csv

# Full export
python "Source Code/infer.py" --source polyp.jpg \
    --export-json report.json --save-annotated --save-masks
```

---

## CLI Reference

| Argument | Default | Description |
|---|---|---|
| `--source` | `0` | `0` for webcam, or path to image/video/directory |
| `--backend` | `yolo` | Segmentation backend: `yolo` or `medsam` |
| `--gt-dir` | — | Directory with GT masks for Dice/IoU evaluation |
| `--config` | — | YAML/JSON config override |
| `--no-display` | off | Headless mode |
| `--output-dir` | `output` | Output directory |
| `--export-json` | — | JSON report path |
| `--export-csv` | — | CSV export path |
| `--save-annotated` | off | Save annotated images/video |
| `--save-masks` | off | Save binary mask PNGs |
| `--force-download` | off | Re-download dataset |

---

## Training

```bash
# Fine-tune YOLO26m-seg on polyp data
python "Source Code/train.py" --data path/to/data.yaml --epochs 80

# Evaluate on dataset images (auto-downloads Kvasir-SEG)
python "Source Code/train.py" --eval

# Evaluate with a specific backend
python "Source Code/train.py" --eval --backend medsam

# Force re-download
python "Source Code/train.py" --eval --force-download
```

---

## Adding a New Backend

1. Create a class inheriting from `SegmentationBackend` in [comparison.py](Source%20Code/comparison.py)
2. Decorate with `@register_backend("your_name")`
3. Implement `load()`, `segment()`, and optionally `close()`
4. The backend auto-registers and becomes selectable via `--backend your_name`

```python
from comparison import SegmentationBackend, register_backend

@register_backend("my_model")
class MyBackend(SegmentationBackend):
    name = "my_model"

    def load(self): ...
    def segment(self, image): ...
```

---

## Project Structure

```
Polyp Lesion Segmentation/
├── Source Code/
│   ├── config.py           # PolypConfig dataclass + loader
│   ├── segmentation.py     # PolypSegmenter — YOLO26m-seg wrapper
│   ├── comparison.py       # Backend registry + YOLO/MedSAM hooks
│   ├── metrics.py          # PolypMetrics + Dice/IoU computation
│   ├── visualize.py        # Overlay rendering + GT comparison view
│   ├── export.py           # JSON / CSV / mask export
│   ├── validator.py        # Input validation
│   ├── controller.py       # PolypController (backend → metrics)
│   ├── infer.py            # CLI entry point
│   ├── train.py            # Training + evaluation
│   ├── modern.py           # CVProject registry entry
│   └── data_bootstrap.py   # Idempotent dataset download
├── requirements.txt
└── README.md
```

---

## Dataset

**Kvasir-SEG** — [Kaggle](https://www.kaggle.com/datasets/dankok/kvasir-seg)

- 1,000 gastrointestinal polyp images with pixel-level segmentation masks
- Annotated and verified by experienced gastroenterologists
- Resolution: 332×487 to 1920×1072 pixels
- Format: JPEG images + PNG binary masks
- Bounding boxes in JSON
- CC BY-NC 4.0 licence
- ~67 MB

```bash
# Auto-download via bootstrap
python "Source Code/data_bootstrap.py"

# Force re-download
python "Source Code/data_bootstrap.py" --force-download
```

---

## Configuration

```yaml
model_name: yolo26m-seg.pt
confidence_threshold: 0.25
iou_threshold: 0.45
imgsz: 640
backend: yolo              # or "medsam"
mask_alpha: 0.40
```

---

## Requirements

- Python 3.10+
- ultralytics >= 8.3.0
- opencv-python >= 4.10.0
- numpy >= 1.26.0
- torch >= 2.0.0

**Optional** (for MedSAM comparison):
- segment-anything

---

## Scope and Limitations

- **Baseline only**: The YOLO26m-seg model ships as a general-purpose pretrained model; fine-tuning on Kvasir-SEG or clinical data is required for production-grade accuracy.
- **No absolute measurements**: All area values are pixel counts. Physical size (mm²) requires external calibration.
- **MedSAM path is a stub**: The MedSAM backend is structured and auto-detected but returns empty results until a checkpoint is integrated.
- **Not validated clinically**: This tool has not undergone clinical validation and must not be used for patient care.
- **Dataset scope**: Kvasir-SEG contains 1,000 images — a useful starting point but small compared to clinical datasets.
- **Single-class**: The current pipeline treats all detections as "polyp". Multi-class lesion typing (adenoma, hyperplastic, etc.) would require label refinement.

---

## Disclaimer

This project is for **educational and research purposes only**.
It does not provide medical advice.  All polyp area values are
**relative pixel counts** that depend on image resolution, endoscope
optics, and capture conditions.  **Do not rely on this tool for
clinical decisions.**
