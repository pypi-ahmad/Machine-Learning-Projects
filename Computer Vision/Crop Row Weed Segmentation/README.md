# Crop Row & Weed Segmentation

Multi-class agricultural segmentation that separates **crop rows**, **weeds**,
and **soil/background** from aerial or ground-level field imagery using
**YOLO26m-seg** instance segmentation.  Computes per-class area statistics,
saves binary masks, and renders colour-coded overlay previews.

> **Note** — The pretrained COCO model does not contain agricultural classes.
> Fine-tune on a crop/weed dataset (see [Training](#training)) for domain-
> specific accuracy.

---

## Features

| Feature | Description |
|---|---|
| **YOLO26m-seg multi-class** | Per-instance masks with class labels and confidence |
| **Per-class area statistics** | Instance count, total area, coverage ratio, mean confidence |
| **Background ratio** | Fraction of image not covered by any class (soil/bare ground) |
| **Class-coloured overlays** | Crop → green, weed → red, soil → brown (configurable) |
| **Binary mask export** | Per-class PNG mask files |
| **Image + video + webcam** | Full source flexibility including batch directories |
| **JSON export** | Per-image structured output with per-instance details |
| **CSV export** | Frame-level tabular output with per-class columns |
| **Video annotation** | Save annotated `.mp4` with `--save-annotated` |
| **Class map render** | Pure segmentation map (no original image) |
| **CVProject registry** | `@register("crop_row_weed_segmentation")` |
| **Idempotent bootstrap** | Auto-download dataset, `--force-download` |

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Single image
cd "Crop Row Weed Segmentation/Source Code"
python infer.py --source field_photo.jpg

# 3. Batch directory
python infer.py --source images/ --save-annotated --save-masks

# 4. Video (drone footage)
python infer.py --source drone_clip.mp4 --save-annotated

# 5. Full export
python infer.py --source field.jpg \
    --export-json report.json \
    --export-csv stats.csv \
    --save-annotated --save-masks
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
  --save-masks            Save per-class binary mask PNGs
  --force-download        Re-download dataset
```

### `train.py` — Training & Evaluation

```
# Fine-tune on crop/weed segmentation data
python train.py --data path/to/data.yaml --epochs 80

# Evaluate on dataset images
python train.py --eval --max-images 100

Options:
  --model MODEL           Base model (default: yolo26m-seg.pt)
  --epochs N              Training epochs (default: 80)
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

All settings in `config.py` (`CropWeedConfig` dataclass).  Override via
`--config path/to/config.yaml`:

```yaml
# Model
model_name: yolo26m-seg.pt
confidence_threshold: 0.30
iou_threshold: 0.45
imgsz: 640

# Classes (update after fine-tuning)
class_names: [crop, weed, soil]
class_colours:
  crop: [0, 200, 0]       # green (BGR)
  weed: [0, 0, 220]       # red
  soil: [140, 120, 100]   # brown

# Visualisation
mask_alpha: 0.45
```

---

## How It Works

### 1. Segmentation

YOLO26m-seg produces per-instance masks with class labels:

$$M_i^{(c)} \in \{0,1\}^{H \times W}, \quad c \in \{\text{crop}, \text{weed}, \text{soil}\}$$

Instance masks are extracted at full resolution via `retina_masks=True` and
grouped into per-class combined masks:

$$M^{(c)} = \bigcup_{i : c_i = c} M_i^{(c)}$$

### 2. Per-Class Area Statistics

For each class $c$, the pipeline computes:

| Metric | Formula |
|---|---|
| Instance count | $N_c = \lvert\{i : c_i = c\}\rvert$ |
| Total area | $A_c = \sum_{i : c_i = c} \lvert M_i \rvert$ |
| Coverage ratio | $R_c = A_c / (W \times H)$ |
| Mean confidence | $\bar{p}_c = \frac{1}{N_c} \sum_{i : c_i = c} p_i$ |

### 3. Background Estimation

Background (soil / bare ground) is estimated as the complement of all
segmented pixels:

$$R_\text{bg} = 1 - \frac{\lvert M^{(\text{crop})} \cup M^{(\text{weed})} \cup \cdots \rvert}{W \times H}$$

This avoids double-counting in overlapping instances.

---

## Dataset

**Mobiusi/Field-Crop-and-Weed-Segmentation-Dataset** — A public agricultural
segmentation dataset from Hugging Face with multi-class annotations for field
crops and weeds.

The dataset auto-downloads on first run:

```bash
python data_bootstrap.py                  # download + organise
python data_bootstrap.py --force-download # re-download
```

---

## Training

1. **Prepare data** — YOLO-format segmentation labels:
   ```
   data.yaml          # nc, names, train/val paths
   images/train/      images/val/
   labels/train/      labels/val/   # polygon .txt files
   ```

2. **Fine-tune**:
   ```bash
   python train.py --data path/to/cropweed/data.yaml --epochs 80 --imgsz 640
   ```

3. Trained weights are auto-registered in the model registry and used by
   `infer.py` on subsequent runs.

4. **Evaluate**:
   ```bash
   python train.py --eval --max-images 100
   ```

---

## Project Structure

```
Crop Row Weed Segmentation/
├── README.md
├── requirements.txt
└── Source Code/
    ├── config.py           ← CropWeedConfig dataclass
    ├── segmentation.py     ← YOLO26m-seg wrapper, multi-class extraction
    ├── class_stats.py      ← per-class area statistics
    ├── visualize.py        ← class-coloured overlays, legend, stats panel
    ├── export.py           ← JSON / CSV / mask PNG export
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
| **Segmentation** | `segmentation.py` | YOLO inference → per-instance masks with class labels |
| **Statistics** | `class_stats.py` | Per-class area, coverage, confidence aggregation |
| **Visualisation** | `visualize.py` | Render overlays, class maps, legends |
| **Export** | `export.py` | Serialise to JSON / CSV / mask PNGs |
| **Orchestration** | `controller.py` | Wire everything together |
| **Validation** | `validator.py` | Input source checks |

---

## License

This project is part of the **Computer-Vision-Projects** repository.
See the top-level LICENSE file for details.
