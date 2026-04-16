# Waste Sorting Detector

**Waste detection and classification** system using YOLO26m object detection.
Detects recyclable materials (plastic, paper, cardboard, metal, glass, trash)
in images or video, provides **per-class counts**, optional **bin-zone
validation** for misplaced items, and exports structured summaries to
CSV / JSON.

---

## Architecture

```
Source Code/
├── config.py          # WasteConfig + BinZoneConfig dataclasses, YAML/JSON loader
├── data_bootstrap.py  # Dataset download & synthetic fallback
├── sorter.py          # WasteSorter: detection + per-class counting + zone validation
├── visualize.py       # Overlay renderer (boxes, zones, dashboard, misplacement alerts)
├── export.py          # CSV / JSON per-frame export
├── infer.py           # Full inference pipeline (image / video / webcam)
├── train.py           # YOLO training entry-point
├── evaluate.py        # YOLO val() + per-class metrics
├── modern.py          # CVProject adapter for repo registry
├── waste.yaml         # Sample configuration with bin zones
└── requirements.txt   # Python dependencies
```

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Train (optional — download + fine-tune)

```bash
cd "Waste Sorting Detector/Source Code"
python train.py --epochs 30 --batch 8
```

### 3. Run inference

```bash
# Single image
python infer.py --source waste_photo.jpg --config waste.yaml

# Video
python infer.py --source sorting_line.mp4 --config waste.yaml

# Webcam
python infer.py --source 0 --config waste.yaml

# Headless with exports
python infer.py --source video.mp4 --no-display --export-csv output/results.csv --export-json output/results.json
```

### 4. Evaluate

```bash
python evaluate.py --model runs/train/weights/best.pt
```

## Configuration

Edit `waste.yaml`:

| Key | Description |
|-----|-------------|
| `model` | YOLO weights path |
| `conf_threshold` | Detection confidence threshold |
| `waste_classes` | List of waste class names to detect |
| `bin_zones` | Optional list of bin-zone polygons with `accepted_classes` |
| `show_counts` | Show per-class count dashboard overlay |
| `show_zones` | Show bin-zone polygon overlays |
| `export_csv` | Path for CSV export |
| `export_json` | Path for JSON export |

## Pipeline Flow

```
Camera / Video / Image
     │
     ▼
  YOLO Detection
     │
     ▼
  ┌───────────────────┐
  │  WasteSorter       │  ← per-class counting
  │                     │  ← bin-zone validation (optional)
  └────────┬────────────┘
           │
     ┌─────┴─────┐
     ▼           ▼
  Visualize   Export
  (overlays)  (CSV / JSON)
```

## Outputs

| File | Content |
|------|---------|
| `output/waste_results.csv` | Per-frame class counts + misplaced items |
| `output/waste_results.json` | Same data in JSON format |
| Annotated image/video | Bounding boxes, zone overlays, count dashboard |

## Dataset

Uses a public waste detection dataset, auto-downloaded via
`configs/datasets/waste_sorting_detector.yaml`.

If the real dataset cannot be downloaded (no Roboflow SDK, network issues),
`data_bootstrap.py` automatically generates a **synthetic YOLO-format
dataset** with colored waste-item shapes (60 train / 15 valid / 15 test)
so that training, evaluation, and inference can run out of the box.

### Dataset bootstrap behaviour

- **Idempotent** — skips download if `.ready` marker exists
- **`--force-download`** — forces re-download and re-preparation
- **Synthetic fallback** — generates 90 images if download fails
- Classes: plastic, paper, cardboard, metal, glass, trash

## Bin-Zone Validation

Define physical bin zones as polygons in `waste.yaml`. Each zone lists
its `accepted_classes`. Detections whose centres fall inside a zone but
whose class is not accepted are flagged as **misplaced** — shown in red
on the overlay and logged in the export.
