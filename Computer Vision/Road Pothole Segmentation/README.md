# Road Pothole Segmentation

Detect, segment, and assess the severity of road potholes from images and
video using **YOLO26m-seg** instance segmentation.  Each detected pothole
receives a binary mask, a bounding box, and a transparent **severity
classification** (minor / moderate / severe) based on pixel area thresholds.

> **Severity estimation** uses simple, auditable pixel-area thresholds — not
> a black-box model.  Thresholds are configurable and the methodology is
> documented in [How It Works](#how-it-works).

---

## Features

| Feature | Description |
|---|---|
| **YOLO26m-seg segmentation** | Per-instance pothole masks with confidence |
| **Severity estimation** | minor / moderate / severe based on configurable area thresholds |
| **Road condition score** | Overall assessment: good / acceptable / fair / poor |
| **Area estimation** | Pixel area; optional m² with calibration factor |
| **Image + video + webcam** | Full source flexibility |
| **Directory batch mode** | Process folder of images in one command |
| **Colour-coded overlays** | Severity-coloured masks, bboxes, legend, summary panel |
| **JSON export** | Per-image/frame structured output with per-pothole details |
| **CSV export** | Frame-level tabular output |
| **Video annotation** | Save annotated `.mp4` with `--save-annotated` |
| **CVProject registry** | `@register("road_pothole_segmentation")` |
| **Idempotent bootstrap** | Auto-download dataset, `--force-download` |

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Single image
cd "Road Pothole Segmentation/Source Code"
python infer.py --source pothole.jpg

# 3. Video
python infer.py --source road_clip.mp4 --save-annotated

# 4. Webcam
python infer.py --source 0

# 5. Batch directory
python infer.py --source images/ --save-annotated --export-json results/

# 6. Full export
python infer.py --source road.mp4 \
    --export-json output/report.json \
    --export-csv output/metrics.csv \
    --save-annotated
```

---

## CLI Reference

### `infer.py` — Inference

```
python infer.py [OPTIONS]

  --source PATH|0         Image, video, directory, or webcam (default: 0)
  --config PATH           YAML/JSON config override
  --no-display            Headless mode (no GUI windows)
  --output-dir DIR        Output directory (default: output/)
  --export-json PATH      JSON export (file or directory for batch)
  --export-csv PATH       CSV export (appended per frame)
  --save-annotated        Save annotated images/video
  --force-download        Re-download dataset
```

### `train.py` — Training & Evaluation

```
# Fine-tune on pothole segmentation data
python train.py --data path/to/data.yaml --epochs 50

# Evaluate on dataset images
python train.py --eval --max-images 100

Options:
  --model MODEL           Base model (default: yolo26m-seg.pt)
  --epochs N              Training epochs (default: 50)
  --batch N               Batch size (default: 16)
  --imgsz N               Image size (default: 640)
  --device DEVICE         cuda:0 | cpu
  --max-images N          Max evaluation images (default: 50)
  --force-download        Re-download dataset
```

---

## Keyboard Controls (Display Mode)

| Key | Action |
|---|---|
| `q` | Quit |

---

## Configuration

All settings live in `config.py` (`PotholeConfig` dataclass).  Override via
`--config path/to/config.yaml`:

```yaml
# Model
model_name: yolo26m-seg.pt
confidence_threshold: 0.35
iou_threshold: 0.45
imgsz: 640

# Severity thresholds (pixels)
severity_thresholds: [1500, 6000]

# Calibration (0 = uncalibrated)
pixel_area_per_m2: 0.0

# Visualisation
mask_alpha: 0.45
```

---

## How It Works

### 1. Segmentation

YOLO26m-seg detects pothole instances and produces per-instance binary masks:

$$M_i \in \{0, 1\}^{H \times W}, \quad \text{conf}(M_i) \geq \tau$$

Each mask is extracted at full resolution via `retina_masks=True`.

### 2. Severity Estimation

Severity is a **transparent, threshold-based** classification on mask pixel
area $A_i = \sum M_i$:

| Severity | Condition |
|---|---|
| **Minor** | $A_i < t_1$ (default $t_1 = 1500$ px) |
| **Moderate** | $t_1 \leq A_i < t_2$ (default $t_2 = 6000$ px) |
| **Severe** | $A_i \geq t_2$ |

Thresholds are fully configurable.  No hidden logic.

### 3. Area Calibration (Optional)

If the camera's ground-sample distance is known, set `pixel_area_per_m2` to
convert pixel area to real-world square metres:

$$A_{m^2} = \frac{A_{px}}{\text{pixel\_area\_per\_m2}}$$

### 4. Road Condition

Overall condition is derived from the severity distribution:

| Condition | Rule |
|---|---|
| **Good** | No potholes detected |
| **Acceptable** | Only minor or single moderate pothole |
| **Fair** | Multiple moderate potholes |
| **Poor** | Any severe pothole |

---

## Dataset

**keremberke/pothole-segmentation** — A public pothole instance segmentation
dataset from Hugging Face with YOLO-format polygon annotations.

The dataset auto-downloads on first run:

```bash
python data_bootstrap.py               # download + organise
python data_bootstrap.py --force-download  # re-download
```

---

## Training

1. **Prepare data** — Use YOLO-format segmentation labels:
   ```
   data.yaml
   images/train/  images/val/
   labels/train/  labels/val/    # polygon .txt files
   ```

2. **Fine-tune**:
   ```bash
   python train.py --data path/to/pothole_data/data.yaml --epochs 80
   ```

3. Trained weights are auto-registered in the model registry.

4. **Evaluate**:
   ```bash
   python train.py --eval --max-images 100
   ```

---

## Project Structure

```
Road Pothole Segmentation/
├── README.md
├── requirements.txt
└── Source Code/
    ├── config.py           ← PotholeConfig dataclass
    ├── segmentation.py     ← YOLO26m-seg wrapper, instance extraction
    ├── severity.py         ← threshold-based severity + area estimation
    ├── visualize.py        ← severity-coloured overlays + legend
    ├── export.py           ← JSON / CSV structured output
    ├── controller.py       ← pipeline orchestration
    ├── validator.py        ← input validation
    ├── infer.py            ← CLI entry point (image/video/webcam/batch)
    ├── train.py            ← training & evaluation
    ├── modern.py           ← CVProject registry (@register)
    └── data_bootstrap.py   ← idempotent dataset download
```

---

## Module Separation

| Layer | Module | Responsibility |
|---|---|---|
| **Segmentation** | `segmentation.py` | YOLO inference → instance masks |
| **Severity** | `severity.py` | Area → severity bucket, road condition |
| **Visualisation** | `visualize.py` | Render overlays, legend, summary |
| **Export** | `export.py` | Serialise to JSON / CSV |
| **Orchestration** | `controller.py` | Wire everything together |
| **Validation** | `validator.py` | Input source checks |

---

## License

This project is part of the **Computer-Vision-Projects** repository.
See the top-level LICENSE file for details.
