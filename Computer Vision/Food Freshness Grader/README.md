# Food Freshness Grader

Classify produce images as **fresh** or **stale** with confidence scores.
Uses transfer learning on a pretrained ResNet-18 to grade 6 produce types
into 12 freshness classes, with annotated output images and batch inference.

---

## Pipeline

```
image.jpg ──► ResNet-18 (transfer learned) ──► 12-class softmax
                                                 │
                                                 ├─ freshness: fresh / stale
                                                 ├─ produce:   apple / banana / …
                                                 └─ confidence: 0–100%
```

## Freshness Classes

| Fresh              | Stale              |
| ------------------ | ------------------ |
| `fresh_apple`      | `stale_apple`      |
| `fresh_banana`     | `stale_banana`     |
| `fresh_bitter_gourd` | `stale_bitter_gourd` |
| `fresh_capsicum`   | `stale_capsicum`   |
| `fresh_orange`     | `stale_orange`     |
| `fresh_tomato`     | `stale_tomato`     |

---

## Quick Start

### 1. Download Dataset

```bash
python data_bootstrap.py                   # idempotent
python data_bootstrap.py --force-download  # re-download
```

Uses the [Fresh and Stale Images](https://www.kaggle.com/datasets/raghavrpotdar/fresh-and-stale-images-of-fruits-and-vegetables)
dataset — 14,700 images, 12 classes, CC0 Public Domain.

### 2. Train

```bash
python train.py
python train.py --model resnet18 --epochs 25 --batch 32
python train.py --model efficientnet_b0 --epochs 30
python train.py --force-download   # auto-download + train
```

Training uses the shared `train/train_classification.py` pipeline
with transfer learning from ImageNet.

### 3. Inference — Single Image

```bash
python infer.py --source apple.jpg
python infer.py --source apple.jpg --save output/graded.jpg
python infer.py --source apple.jpg --show
```

### 4. Inference — Batch

```bash
python infer.py --source photos/ --batch
python infer.py --source photos/ --batch --save-grid output/grid.jpg
python infer.py --source photos/ --batch --export-json output/results.json
python infer.py --source photos/ --batch --export-csv output/results.csv
```

---

## CLI Reference

### `train.py` — Train Model

| Flag               | Description                              |
| ------------------ | ---------------------------------------- |
| `--data`           | Path to dataset directory                |
| `--model`          | Architecture: resnet18, resnet50, efficientnet_b0, mobilenet_v2 |
| `--epochs`         | Training epochs (default: 25)            |
| `--batch`          | Batch size (default: 32)                 |
| `--imgsz`          | Image size (default: 224)                |
| `--lr`             | Learning rate (default: 1e-3)            |
| `--device`         | cpu / cuda                               |
| `--force-download` | Force re-download dataset                |

### `infer.py` — Grade Images

| Flag             | Description                                |
| ---------------- | ------------------------------------------ |
| `--source`       | Image path or directory (required)         |
| `--batch`        | Enable batch mode for directories          |
| `--weights`      | Override model weights path                |
| `--model`        | Override model architecture                |
| `--save`         | Save annotated image (single mode)         |
| `--save-grid`    | Save batch result grid                     |
| `--show`         | Display annotated result                   |
| `--export-json`  | Export results to JSON                     |
| `--export-csv`   | Export results to CSV                      |
| `--batch-size`   | Batch inference size (default: 16)         |
| `--device`       | Compute device                             |

### `evaluate.py` — Evaluate Model

| Flag         | Description                            |
| ------------ | -------------------------------------- |
| `--eval`     | Run evaluation (required flag)         |
| `--data`     | Dataset directory (required)           |
| `--weights`  | Model weights path                     |
| `--model`    | Model architecture                     |

---

## Model Options

| Model             | Parameters | Notes                       |
| ----------------- | ---------- | --------------------------- |
| `resnet18`        | 11.7M      | Default — fast, good balance|
| `resnet50`        | 25.6M      | Higher capacity             |
| `efficientnet_b0` | 5.3M       | Efficient, modern           |
| `mobilenet_v2`    | 3.4M       | Lightweight, mobile-ready   |

---

## Output

### Annotated Images

Each image gets:
- Freshness grade label (`FRESH` / `STALE` / `UNCERTAIN`)
- Produce type (e.g. "Apple", "Banana")
- Confidence percentage
- Colour-coded confidence bar (green → fresh, red → stale)

### Batch Grid

Thumbnail grid with all graded images, colour-coded borders,
and per-image labels.

### Export Formats

| Format | Contents                                          |
| ------ | ------------------------------------------------- |
| JSON   | Total counts, fresh/stale breakdown, per-image    |
| CSV    | path, freshness, produce, class_name, confidence  |

---

## Evaluation Metrics

The evaluator reports:
- **Overall accuracy** — exact class match (12 classes)
- **Freshness accuracy** — fresh vs stale only (binary)
- **Per-class accuracy** — breakdown by each of the 12 classes

---

## Project Structure

```
Source Code/
├── config.py          # FreshnessConfig dataclass + grading labels
├── grader.py          # FreshnessGrader — model loading + inference
├── train.py           # Training script (shared pipeline)
├── infer.py           # CLI — single + batch inference
├── evaluate.py        # CLI — model evaluation
├── visualize.py       # Image annotation + batch grids
├── export.py          # JSON / CSV export
├── validator.py       # Input validation + image collection
├── controller.py      # High-level pipeline facade
├── modern.py          # CVProject registry integration
└── data_bootstrap.py  # Kaggle dataset download + train/val split
```

---

## Requirements

- Python ≥ 3.10
- PyTorch ≥ 2.0
- torchvision ≥ 0.15
- OpenCV ≥ 4.10
- NumPy ≥ 1.26

```bash
pip install -r requirements.txt
```

---

## Dataset

**Fresh and Stale Images of Fruits and Vegetables** by Raghav R Potdar  
14,700 images · 12 classes · 6 produce types · ~1.5 GB · CC0 Public Domain  
[kaggle.com/datasets/raghavrpotdar/fresh-and-stale-images-of-fruits-and-vegetables](https://www.kaggle.com/datasets/raghavrpotdar/fresh-and-stale-images-of-fruits-and-vegetables)
