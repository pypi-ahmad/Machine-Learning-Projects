# Industrial Scratch / Crack Segmentation

Detect and segment surface defects — scratches, cracks, and fractures — in
industrial and infrastructure imagery using **YOLO26m-seg** instance
segmentation with transparent, coverage-based severity estimation.

| Feature | Detail |
|---|---|
| **Model** | YOLO26m-seg (instance segmentation) |
| **Defect types** | Scratches, cracks, surface fractures |
| **Severity estimation** | Coverage-based thresholds (transparent heuristics) |
| **Shape metrics** | Major-axis length, aspect ratio per instance |
| **Outputs** | Annotated frames, binary masks, JSON, CSV |
| **Sources** | Image · directory · video · webcam |
| **Dataset** | `yidazhang07/bridge-cracks-image` (Kaggle, ODbL) |

---

## Severity Heuristics

Severity is determined by **defect coverage** — the fraction of image area
covered by all detected defect masks:

| Severity | Condition |
|---|---|
| **None** | 0 defects detected |
| **Low** | coverage ≤ 0.5 % |
| **Medium** | 0.5 % < coverage ≤ 2.0 % |
| **High** | coverage > 2.0 % |

Thresholds are configurable via `DefectConfig.severity_low` / `severity_medium`.

### Per-Instance Shape Metrics

Each detected defect also reports:

- **Length (px)** — major axis of a fitted ellipse (`cv2.fitEllipse` when
  ≥ 5 contour points, `cv2.minAreaRect` fallback)
- **Aspect ratio** — length / width of the fitted shape

---

## Pipeline

```
Image  ──►  YOLO26m-seg  ──►  Instance masks
                                     │
                          ┌──────────┴──────────┐
                          │  Per-instance        │
                          │  • area, length      │
                          │  • aspect ratio      │
                          │  • confidence        │
                          └──────────┬──────────┘
                                     │
                           Coverage-based severity
                                     │
                          ┌──────────┴──────────┐
                          │  Annotated overlay   │
                          │  Binary mask         │
                          │  JSON / CSV export   │
                          └─────────────────────┘
```

---

## Quick Start

```bash
# Single image
python "Industrial Scratch Crack Segmentation/Source Code/infer.py" \
    --source path/to/image.jpg

# Directory with JSON export
python "Industrial Scratch Crack Segmentation/Source Code/infer.py" \
    --source path/to/images/ --export-json --save-annotated

# Video
python "Industrial Scratch Crack Segmentation/Source Code/infer.py" \
    --source path/to/video.mp4 --save-annotated --save-masks

# Webcam
python "Industrial Scratch Crack Segmentation/Source Code/infer.py" \
    --source 0

# Evaluate on dataset
python "Industrial Scratch Crack Segmentation/Source Code/train.py" --eval

# Fine-tune
python "Industrial Scratch Crack Segmentation/Source Code/train.py" \
    --data path/to/data.yaml --epochs 80
```

---

## CLI Reference

### `infer.py`

| Argument | Default | Description |
|---|---|---|
| `--source` | `0` | Image, directory, video path, or webcam index |
| `--config` | — | JSON/YAML config file |
| `--output-dir` | `output` | Output directory |
| `--no-display` | off | Suppress window display |
| `--export-json` | off | Save per-frame JSON results |
| `--export-csv` | off | Save summary CSV |
| `--save-annotated` | off | Save annotated images |
| `--save-masks` | off | Save binary masks |
| `--force-download` | off | Force re-download dataset |

### `train.py`

| Argument | Default | Description |
|---|---|---|
| `--eval` | off | Run evaluation instead of training |
| `--data` | auto | Path to YOLO data.yaml |
| `--model` | `yolo26m-seg.pt` | Base model weights |
| `--epochs` | `80` | Training epochs |
| `--batch` | `16` | Batch size |
| `--imgsz` | `640` | Image size |
| `--device` | auto | CUDA device |
| `--max-images` | `50` | Max images for evaluation |
| `--force-download` | off | Force re-download dataset |

---

## Configuration

All parameters are tuneable via `DefectConfig` (in `config.py`) or a
JSON/YAML config file passed with `--config`:

```json
{
    "confidence_threshold": 0.25,
    "iou_threshold": 0.45,
    "min_area_px": 32,
    "severity_low": 0.005,
    "severity_medium": 0.02,
    "mask_alpha": 0.40
}
```

---

## Project Structure

```
Industrial Scratch Crack Segmentation/
├── Source Code/
│   ├── config.py           # DefectConfig dataclass
│   ├── segmentation.py     # YOLO26m-seg wrapper + DefectInstance
│   ├── metrics.py          # Coverage-based severity + shape stats
│   ├── visualize.py        # Overlay rendering with severity panel
│   ├── export.py           # JSON / CSV / mask export
│   ├── validator.py        # Input validation
│   ├── controller.py       # High-level orchestrator
│   ├── infer.py            # CLI entry point
│   ├── train.py            # Training & evaluation
│   ├── modern.py           # CVProject registry entry
│   └── data_bootstrap.py   # Idempotent dataset download
├── requirements.txt
└── README.md
```

---

## Dataset

**Bridge Cracks & Surface Defects** (`yidazhang07/bridge-cracks-image`)

- **Source:** Kaggle (ODbL license)
- **Size:** ~195 MB, 61.5k files
- **Sub-datasets:** CrackForest (pixel-level GT), Magnetic Tile (pixel-level
  GT), Bridge Crack, DeepPCB

Auto-downloaded on first run via `data_bootstrap.py` or the `--force-download`
flag.

---

## Requirements

```
ultralytics>=8.3.0
opencv-python>=4.10.0,<5
numpy>=1.26.0
torch>=2.0.0
```
