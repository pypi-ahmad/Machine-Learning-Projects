# Conveyor Part Defect Detector

Industrial **visual inspection** system using YOLO object detection.
Detects part defects (scratches, dents, cracks, missing components, etc.)
on a conveyor-style setup, produces a per-frame **pass / fail** verdict,
saves **cropped defect thumbnails**, and exports structured defect logs
to CSV / JSON.

---

## Architecture

```
Source Code/
├── config.py          # InspectionConfig dataclass, YAML/JSON loader
├── data_bootstrap.py  # Dataset download & preparation (Roboflow)
├── inspector.py       # Detection dataclass, Inspector (pass/fail evaluator)
├── visualize.py       # Overlay renderer (defect boxes, banner, dashboard)
├── export.py          # CSV / JSON logging + defect crop saving
├── infer.py           # Full inference pipeline (image / video / webcam)
├── train.py           # YOLO training entry-point
├── evaluate.py        # YOLO val() + per-class metrics report
├── modern.py          # CVProject adapter for repo registry
├── inspection.yaml    # Sample inspection configuration
└── requirements.txt   # Python dependencies
```

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Train (optional — download + fine-tune)

```bash
cd "Conveyor Part Defect Detector/Source Code"
python train.py --epochs 30 --batch 16
```

### 3. Run inference

```bash
# Single image
python infer.py --source part.jpg --config inspection.yaml

# Video
python infer.py --source conveyor_feed.mp4 --config inspection.yaml

# Webcam
python infer.py --source 0 --config inspection.yaml

# Headless with video output
python infer.py --source conveyor_feed.mp4 --config inspection.yaml --save-video --no-display
```

### 4. Evaluate

```bash
python evaluate.py --model runs/conveyor_part_defect_detector/train/weights/best.pt
```

## Configuration

Edit `inspection.yaml`:

| Key | Description |
|-----|-------------|
| `model` | YOLO weights path |
| `conf_threshold` | Detection confidence threshold |
| `defect_classes` | List of defect class names |
| `all_classes_are_defects` | If `true`, every detection counts as a defect |
| `fail_threshold` | Min defect count to mark a frame as FAIL |
| `save_crops` | Save cropped defect thumbnails |
| `crop_padding` | Pixels of padding around each crop |
| `export_dir` | Output directory for logs and crops |

## Pipeline Flow

```
Camera / Video / Image
     │
     ▼
  YOLO Detection
     │
     ▼
  ┌───────────────────┐
  │  Inspector         │  ← classify detections as defects
  │                     │  ← produce PASS / FAIL verdict
  └────────┬────────────┘
           │
     ┌─────┴─────┐
     ▼           ▼
  Visualize   Export
  (overlays)  (CSV / JSON / crops)
```

## Outputs

| File / Directory | Content |
|------------------|---------|
| `outputs/inspection_log.csv` | Per-frame verdict + defect summary |
| `outputs/inspection_log.json` | Same data in JSON format |
| `outputs/defect_crops/` | Cropped defect thumbnails (JPEG) |
| `outputs/result.jpg` | Annotated image (single-image mode) |
| `outputs/output.mp4` | Annotated video (with `--save-video`) |
| `outputs/eval_report.json` | Model evaluation metrics |

## Registration

Registered as `conveyor_part_defect_detector` in the repo's project registry:

```python
from core.registry import list_projects
print(list_projects())  # includes "conveyor_part_defect_detector"
```

## Dataset

Uses the **PCB Defect Detection** dataset from Roboflow Universe,
auto-downloaded via `configs/datasets/conveyor_part_defect_detector.yaml`.

If the real dataset cannot be downloaded (no Roboflow SDK, network issues),
`data_bootstrap.py` automatically generates a **synthetic YOLO-format
dataset** with PCB-like board images and random defect marks so that
training, evaluation, and inference can still run out of the box.

Contains annotated defect classes on PCB / industrial part images in
YOLO format. Classes include scratch, dent, crack, missing_part,
missing_hole, mouse_bite, open_circuit, short, spur, spurious_copper.

### Dataset bootstrap behaviour

- **Idempotent** — skips download if `.ready` marker exists
- **`--force-download`** — forces re-download and re-preparation
- **Synthetic fallback** — generates 90 images (60 train / 15 valid / 15 test) if download fails
- Layout: `data/conveyor_part_defect_detector/`
