# Waste Sorting Detector

Detect and classify waste items (plastic, paper, cardboard, metal, glass, trash) in images, video, or live webcam feeds.  Optionally validate that items are placed in the correct **bin zone** and flag misplaced waste.

---

## Features

| Feature | Description |
|---|---|
| **Per-class counting** | Real-time counts for each waste category |
| **Bin-zone validation** | Define polygon zones for physical bins; misplaced items are flagged |
| **CSV / JSON export** | Per-frame summaries with class counts and misplacement details |
| **Dashboard overlay** | On-screen waste counts, zone outlines, and alerts |
| **Image / video / webcam** | Unified inference pipeline for all source types |
| **Auto dataset download** | One-command training data bootstrap via Roboflow |

---

## Architecture

```
Waste Sorting Detector/
└── Source Code/
    ├── config.py          # WasteConfig / BinZoneConfig dataclasses + YAML loader
    ├── data_bootstrap.py  # Dataset download & preparation
    ├── sorter.py          # Detection + per-class counting + bin-zone validation
    ├── visualize.py       # Overlay renderer (boxes, zones, dashboard)
    ├── export.py          # CSV / JSON per-frame export
    ├── infer.py           # CLI inference pipeline
    ├── train.py           # Training script (delegates to shared infra)
    ├── evaluate.py        # Evaluation with per-class mAP
    ├── modern.py          # Registry entry (@register)
    ├── waste.yaml         # Sample configuration
    └── requirements.txt   # Python dependencies
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r "Waste Sorting Detector/Source Code/requirements.txt"
```

### 2. Run inference

```bash
# Webcam (default)
python "Waste Sorting Detector/Source Code/infer.py"

# Video file with config
python "Waste Sorting Detector/Source Code/infer.py" \
    --source video.mp4 \
    --config "Waste Sorting Detector/Source Code/waste.yaml"

# Image
python "Waste Sorting Detector/Source Code/infer.py" --source waste_photo.jpg

# Headless with export
python "Waste Sorting Detector/Source Code/infer.py" \
    --source video.mp4 \
    --no-display \
    --save-video output.mp4 \
    --export-csv results.csv \
    --export-json results.json
```

### 3. Train a custom model

```bash
python "Waste Sorting Detector/Source Code/train.py" --epochs 100 --batch 16
```

### 4. Evaluate

```bash
python "Waste Sorting Detector/Source Code/evaluate.py" \
    --model runs/waste_sorting/train/weights/best.pt
```

---

## Configuration

Edit `waste.yaml` (or create your own) to customise:

| Key | Type | Description |
|---|---|---|
| `model` | `str` | YOLO weights path |
| `conf_threshold` | `float` | Minimum detection confidence |
| `waste_classes` | `list[str]` | Classes to report (empty = all) |
| `bin_zones` | `list` | Polygon zones with accepted classes |
| `export_csv` | `str` | CSV output path |
| `export_json` | `str` | JSON output path |
| `show_counts` | `bool` | Display count dashboard |
| `show_zones` | `bool` | Render zone polygons |

### Bin-Zone Example

```yaml
bin_zones:
  - name: "Recyclables"
    polygon: [[50,50],[350,50],[350,400],[50,400]]
    accepted_classes: ["plastic","paper","cardboard","metal","glass"]

  - name: "General-Waste"
    polygon: [[400,50],[700,50],[700,400],[400,400]]
    accepted_classes: ["trash"]
```

Detections inside a zone whose class is **not** in `accepted_classes` are flagged as **misplaced**.

---

## Keyboard Controls

| Key | Action |
|---|---|
| `q` | Quit |

---

## Dataset

The training dataset is auto-downloaded from Roboflow via the shared dataset infrastructure.  See `configs/datasets/waste_sorting_detector.yaml` for source details.

If the real dataset cannot be downloaded (no Roboflow SDK, network issues),
`data_bootstrap.py` automatically generates a **synthetic YOLO-format
dataset** with waste item images (6 classes: plastic, paper, cardboard,
metal, glass, trash) so that training, evaluation, and inference can
still run out of the box.

To force a re-download:

```bash
python "Waste Sorting Detector/Source Code/train.py" --force-download
```
