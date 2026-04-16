# Drone Ship OBB Detector

**Oriented bounding-box (OBB) detection** for aerial and satellite imagery
using YOLO26m-OBB. Detects ships, vehicles, aircraft, and infrastructure
with **rotated boxes** that tightly fit elongated or angled objects.

## Why OBB?

Standard axis-aligned bounding boxes waste area on rotated objects —
a long ship at 45 degrees gets a huge square box that overlaps with
neighbouring objects. OBB uses 4 corner points to represent the true
orientation, which:

- **Improves IoU scores** for thin/elongated objects
- **Reduces false overlaps** in dense scenes (harbors, parking lots)
- **Enables angle estimation** for downstream analysis

---

## Architecture

```
Source Code/
├── config.py          # OBBConfig dataclass, YAML/JSON loader
├── data_bootstrap.py  # Dataset download & synthetic OBB fallback
├── detector.py        # OBBDetector: YOLO-OBB inference + corner parsing
├── visualize.py       # Overlay renderer (rotated boxes, angle labels, dashboard)
├── export.py          # JSON + YOLO-OBB TXT export
├── infer.py           # Full inference pipeline (image / video / webcam)
├── train.py           # YOLO-OBB training entry-point
├── evaluate.py        # OBB val() + per-class rotated-box metrics
├── modern.py          # CVProject adapter for repo registry
├── obb_config.yaml    # Sample configuration
└── requirements.txt   # Python dependencies
```

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Train (optional)

```bash
cd "Drone Ship OBB Detector/Source Code"
python train.py --epochs 100 --batch 8 --imgsz 1024
```

### 3. Run inference

```bash
# Single aerial image
python infer.py --source aerial.jpg --config obb_config.yaml

# Video
python infer.py --source drone_feed.mp4 --config obb_config.yaml

# Headless with exports
python infer.py --source aerial.jpg --no-display --export-json output/results.json --export-txt output/labels
```

### 4. Evaluate

```bash
python evaluate.py --model runs/train/weights/best.pt
```

## Configuration

Edit `obb_config.yaml`:

| Key | Description |
|-----|-------------|
| `model` | YOLO-OBB weights path (`yolo26m-obb.pt`) |
| `conf_threshold` | Detection confidence threshold |
| `imgsz` | Input image size (1024 recommended for aerial) |
| `target_classes` | Filter to specific classes (empty = all) |
| `export_json` | JSON export path |
| `export_txt` | Directory for YOLO-OBB TXT label files |
| `show_angle` | Show rotation angle on labels |
| `line_width` | OBB polygon line thickness |

## Label Format

YOLO-OBB uses 4 normalized corner points per object:

```
class_id x1 y1 x2 y2 x3 y3 x4 y4
```

Each (x, y) pair is a corner of the oriented bounding box, normalized
to [0, 1] by image dimensions.

## Dataset

Uses a public aerial OBB dataset (DOTA-ship subset) from Roboflow Universe,
auto-downloaded via `configs/datasets/drone_ship_obb_detector.yaml`.

If the real dataset cannot be downloaded, `data_bootstrap.py` generates a
**synthetic YOLO-OBB dataset** with rotated aerial objects (ships, vehicles,
aircraft) on water/land backgrounds (60 train / 15 valid / 15 test).

### Dataset bootstrap behaviour

- **Idempotent** — skips download if `.ready` marker exists
- **`--force-download`** — forces re-download
- **Synthetic fallback** — generates 90 OBB-labeled images if download fails
- 8 classes: ship, large-vehicle, small-vehicle, plane, helicopter, harbor, storage-tank, container-crane

### Source and license

The DOTA dataset family is publicly available for research use. See
[DOTA](https://captain-whu.github.io/DOTA/) for license details.
