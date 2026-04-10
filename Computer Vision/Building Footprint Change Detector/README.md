# Building Footprint Change Detector

Detect and quantify building footprint changes between **before** and **after**
aerial / satellite images.  The pipeline segments building footprints with
**YOLO26m-seg**, diffs the resulting masks, highlights new constructions and
demolitions, computes change metrics, and exports visual comparisons plus
structured JSON / CSV output.

> **Note** — The pretrained COCO model does **not** contain a "building" class.
> Fine-tune on a building segmentation dataset (see [Training](#training))
> before running change detection for production use.

---

## Features

| Feature | Description |
|---|---|
| **YOLO26m-seg segmentation** | Instance-level building mask extraction |
| **Before / after diff** | Pixel-level comparison with morphological cleanup |
| **Change classification** | New construction, demolished, unchanged |
| **Connected-component analysis** | Individual change region extraction with area and bbox |
| **Quantitative metrics** | IoU, coverage, change ratio, growth ratio |
| **Histogram matching** | Optional illumination normalisation for different capture dates |
| **Visual reports** | Side-by-side, change overlay with legend, metrics panel |
| **Batch mode** | Process entire directories of matched image pairs |
| **JSON / CSV export** | Structured per-pair output |
| **CVProject registry** | `@register("building_footprint_change_detector")` |
| **Idempotent bootstrap** | Auto-download LEVIR-CD dataset, `--force-download` |

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Single pair
cd "Building Footprint Change Detector/Source Code"
python infer.py --before path/to/before.png --after path/to/after.png

# 3. Batch mode (matched filenames in two directories)
python infer.py --before-dir images/A --after-dir images/B --save-visuals

# 4. Export results
python infer.py --before a.png --after b.png \
    --export-json results.json --export-csv metrics.csv
```

---

## CLI Reference

### `infer.py` — Inference

```
python infer.py [OPTIONS]

Input (single pair):
  --before PATH           Path to before image
  --after PATH            Path to after image

Input (batch):
  --before-dir DIR        Directory of before images
  --after-dir DIR         Directory of after images

Options:
  --config PATH           YAML/JSON config override
  --match-histograms      Histogram match after→before (reduces false positives)
  --no-display            Headless (no GUI windows)
  --output-dir DIR        Output directory (default: output/)

Export:
  --export-json PATH      Save JSON per pair
  --export-csv PATH       Append row per pair to CSV
  --save-visuals          Save report, side-by-side, and overlay images

Data:
  --force-download        Re-download dataset
```

### `train.py` — Training & Evaluation

```
# Fine-tune YOLO26m-seg on building segmentation data
python train.py --data path/to/data.yaml --epochs 50

# Evaluate change detection on LEVIR-CD pairs
python train.py --eval

# Options
  --model MODEL           Base model (default: yolo26m-seg.pt)
  --epochs N              Training epochs (default: 50)
  --batch N               Batch size (default: 8)
  --imgsz N               Image size (default: 1024)
  --device DEVICE         cuda:0 | cpu
  --max-pairs N           Max evaluation pairs (default: 50)
  --force-download        Re-download dataset
```

---

## Keyboard Controls (Display Mode)

| Key | Action |
|---|---|
| `q` | Quit |
| Any | Next pair (batch mode) |

---

## Configuration

All settings live in `config.py` (`ChangeConfig` dataclass).  Override via
`--config path/to/config.yaml`:

```yaml
# Model
model_name: yolo26m-seg.pt
confidence_threshold: 0.30
iou_threshold: 0.45
imgsz: 1024

# Diff
morph_kernel_size: 5
min_change_area: 100

# Visualisation (BGR)
mask_alpha: 0.40
```

---

## How It Works

### 1. Preprocessing

Both images are loaded, resized to a common resolution (default 1024 × 1024),
and optionally histogram-matched to normalise illumination differences between
capture dates.

### 2. Building Segmentation

YOLO26m-seg runs on each image independently.  All detected instance masks are
combined into a single binary footprint mask per image:

$$M = \bigcup_{i=1}^{N} m_i \quad \text{where } \text{conf}(m_i) \geq \tau$$

### 3. Change Detection

The diff engine classifies every pixel by comparing the before mask $M_b$ and
after mask $M_a$:

| Pixel class | Condition |
|---|---|
| **New** | $M_a = 1 \wedge M_b = 0$ |
| **Demolished** | $M_b = 1 \wedge M_a = 0$ |
| **Unchanged** | $M_a = 1 \wedge M_b = 1$ |
| **Background** | $M_a = 0 \wedge M_b = 0$ |

Morphological opening + closing removes noise, and connected-component analysis
extracts individual change regions (filtered by `min_change_area`).

### 4. Metrics

| Metric | Formula |
|---|---|
| IoU | $\frac{M_b \cap M_a}{M_b \cup M_a}$ |
| Change ratio | $\frac{\text{new} + \text{demolished}}{\text{total pixels}}$ |
| Growth ratio | $\frac{|M_a|}{|M_b|}$ |
| Coverage | $\frac{|M|}{W \times H}$ |

---

## Dataset

**LEVIR-CD** — 637 pairs of 1024 × 1024 VHR (0.5 m/pixel) aerial images from
Google Earth spanning 5–14 years, with binary change masks.

The dataset auto-downloads on first run:

```bash
python data_bootstrap.py               # download + organise
python data_bootstrap.py --force-download  # re-download
```

Structure after bootstrap:

```
data/building_footprint_change_detector/
  raw/          ← original download
  processed/
    A/          ← before images
    B/          ← after images
    label/      ← ground-truth change masks
    dataset_info.json
```

---

## Training

1. **Prepare data** — Provide building segmentation labels in YOLO format
   (`data.yaml` + images + label `.txt` files with polygon annotations).

2. **Fine-tune**:
   ```bash
   python train.py --data path/to/buildings/data.yaml --epochs 80 --imgsz 1024
   ```

3. Trained weights are auto-registered in the model registry and used by
   `infer.py` on subsequent runs.

4. **Evaluate** on LEVIR-CD:
   ```bash
   python train.py --eval --max-pairs 100
   ```

---

## Project Structure

```
Building Footprint Change Detector/
├── README.md
├── requirements.txt
└── Source Code/
    ├── config.py           ← ChangeConfig dataclass
    ├── preprocess.py       ← image loading, resizing, histogram matching
    ├── segmentation.py     ← YOLO26m-seg wrapper, mask extraction
    ├── diff_engine.py      ← mask differencing, morphological cleanup
    ├── metrics.py          ← IoU, coverage, change/growth ratios
    ├── visualize.py        ← side-by-side, change overlay, metrics panel
    ├── export.py           ← JSON / CSV structured output
    ├── controller.py       ← pipeline orchestration
    ├── validator.py        ← input validation
    ├── infer.py            ← CLI entry point
    ├── train.py            ← training & evaluation
    ├── modern.py           ← CVProject registry (@register)
    └── data_bootstrap.py   ← idempotent dataset download
```

---

## Module Separation

| Layer | Module | Responsibility |
|---|---|---|
| **Preprocessing** | `preprocess.py` | Load, resize, histogram match |
| **Segmentation** | `segmentation.py` | YOLO inference → binary mask |
| **Diff** | `diff_engine.py` | Mask comparison, morphology, regions |
| **Metrics** | `metrics.py` | Quantitative change scores |
| **Visualisation** | `visualize.py` | Render overlays and reports |
| **Export** | `export.py` | Serialise results to JSON / CSV |
| **Orchestration** | `controller.py` | Wire everything together |

---

## License

This project is part of the **Computer-Vision-Projects** repository.
See the top-level LICENSE file for details.
