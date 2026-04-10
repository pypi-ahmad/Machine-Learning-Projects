# Product Counterfeit Visual Checker

Visual screening tool that compares product images against approved
references to flag mismatch risk.  Uses embedding similarity,
region-aware patch comparison, and colour-histogram checks to produce
a composite screening score.

> **Disclaimer:** This is a *screening* tool.  It highlights visual
> differences between a suspect image and approved references.  It does
> **not** prove or disprove counterfeit status.  Any flagged item
> requires further investigation by qualified personnel.

---

## How It Works

```
suspect.jpg ──► embedder ──► global embedding ─────────┐
                  │                                     │ weighted
                  ├──► region patches (3×3 grid) ──────┤ composite
                  │                                     │ score
                  └──► colour histogram (HSV) ─────────┘
                                                        │
references.npz ──► top-k nearest ──────────────────────►│
                                                        ▼
                                              risk_level: low / medium / high
```

Three signals are combined:

| Signal               | Weight | What it measures                        |
| -------------------- | ------ | --------------------------------------- |
| Global embedding     | 0.60   | Overall visual similarity               |
| Region patches       | 0.25   | Local detail match (3×3 grid)           |
| Colour histogram     | 0.15   | Colour distribution consistency (HSV)   |

The composite score is mapped to a risk level:

| Composite Score | Risk Level | Meaning                                   |
| --------------- | ---------- | ----------------------------------------- |
| ≥ 0.75          | **Low**    | Visually consistent with references       |
| 0.55 – 0.74     | **Medium** | Some visual differences — review advised  |
| < 0.55          | **High**   | Significant visual mismatch — investigate |

---

## Quick Start

### 1. Download Dataset

```bash
python data_bootstrap.py                   # idempotent
python data_bootstrap.py --force-download  # re-download
```

Uses the [Grocery Store Dataset](https://www.kaggle.com/datasets/validmodel/grocery-store-dataset)
— 5,125 images, 81 product classes, MIT licence.

### 2. Build Reference Store

```bash
python reference_builder.py --image-dir data/grocery/processed/products
python reference_builder.py --image-dir data/grocery/processed/products --force
```

### 3. Screen a Suspect Image

```bash
python infer.py --source suspect.jpg
python infer.py --source suspect.jpg --save-grid output/grid.jpg
python infer.py --source suspect.jpg --save-heatmap output/heatmap.jpg
python infer.py --source suspect.jpg --export-json output/result.json
python infer.py --source suspect.jpg --product "Granny-Smith"
```

---

## CLI Reference

### `infer.py` — Screen Images

| Flag              | Description                                  |
| ----------------- | -------------------------------------------- |
| `--source`        | Path to suspect image (required)             |
| `--product`       | Filter references to this product label      |
| `--save-grid`     | Save comparison grid image                   |
| `--save-heatmap`  | Save region similarity heatmap               |
| `--show`          | Display comparison grid in a window          |
| `--export-json`   | Export results to JSON                        |
| `--export-csv`    | Export results to CSV                         |
| `--top-k`         | Number of reference matches (default: 3)     |
| `--backbone`      | Override backbone model                       |
| `--device`        | Compute device (cpu/cuda)                     |

### `reference_builder.py` — Build References

| Flag           | Description                                     |
| -------------- | ----------------------------------------------- |
| `--image-dir`  | Directory of approved product images (required) |
| `--force`      | Rebuild even if reference store exists           |
| `--batch-size` | Embedding batch size (default: 32)               |
| `--backbone`   | Override backbone model                          |

---

## Backbone Options

| Backbone             | Dim  | Notes                   |
| -------------------- | ---- | ----------------------- |
| `efficientnet_b0`    | 1280 | Default — good balance  |
| `efficientnet_b2`    | 1408 | Higher capacity         |
| `resnet50`           | 2048 | Classic, widely used    |
| `resnet18`           | 512  | Lightweight             |
| `mobilenet_v3_small` | 576  | Very fast, mobile-ready |
| `mobilenet_v3_large` | 960  | Fast, good accuracy     |

---

## Output Formats

### JSON Export

Includes a disclaimer, risk level, composite scores, and per-reference
breakdown with global/region/histogram scores.

### CSV Export

One row per reference comparison with all score components.

### Comparison Grid

Visual grid showing the suspect image alongside matched references
with colour-coded borders (green/orange/red) and score breakdowns.

### Region Heatmap

Overlay showing per-patch similarity scores on a 3×3 grid, highlighting
which regions of the image differ most from the best reference.

---

## Evaluation

Leave-one-out product matching accuracy:

```bash
python evaluate.py --eval
python evaluate.py --eval --max-queries 200
```

---

## Project Structure

```
Source Code/
├── config.py            # CounterfeitConfig dataclass
├── embedder.py          # CNN feature extraction (6 backbones)
├── reference_store.py   # Approved reference embedding store (.npz)
├── comparator.py        # Multi-signal comparison engine
├── visualize.py         # Comparison grids + region heatmaps
├── export.py            # JSON / CSV export with disclaimers
├── validator.py         # Input validation + image collection
├── controller.py        # High-level pipeline facade
├── infer.py             # CLI — screen suspect images
├── reference_builder.py # CLI — build reference store
├── evaluate.py          # CLI — screening quality evaluation
├── modern.py            # CVProject registry integration
└── data_bootstrap.py    # Kaggle dataset download + organise
```

---

## Important Limitations

- **Screening only** — flags visual mismatch risk, does not determine
  counterfeit status.
- **Embedding-based** — captures high-level visual features; may miss
  subtle text or fine-print differences.
- **Reference-dependent** — quality depends on the breadth and quality
  of approved reference images.
- **Not forensic** — this is not a replacement for expert product
  authentication or laboratory analysis.
- **Thresholds are tuneable** — the default risk thresholds should be
  calibrated for each product domain.

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

**Grocery Store Dataset** by Marcus Klasson et al.  
5,125 images · 81 fine-grained classes · 42 coarse categories · MIT licence  
[kaggle.com/datasets/validmodel/grocery-store-dataset](https://www.kaggle.com/datasets/validmodel/grocery-store-dataset)

Paper: *A Hierarchical Grocery Store Image Dataset with Visual and
Semantic Labels* (WACV 2019)
