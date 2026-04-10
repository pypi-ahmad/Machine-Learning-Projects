# Logo Retrieval Brand Match

Detect logos in images and retrieve the most similar brand matches from
a prebuilt embedding index using **EfficientNet-B0** feature extraction
and **cosine similarity** search.

| Feature | Detail |
|---|---|
| **Embedding model** | EfficientNet-B0 (torchvision pretrained) |
| **Index** | Numpy-based cosine similarity — no FAISS required |
| **Detection** | Optional YOLO-based logo cropper |
| **Retrieval** | Top-k matches with similarity scores |
| **Outputs** | Preview grids, JSON, CSV |
| **Dataset** | Popular Brand Logos (Kaggle, CC0, ~25 MB) |

---

## Pipeline

```
Query image  ──►  [Detect/Crop]  ──►  EfficientNet-B0  ──►  Embedding
    (optional)                          backbone              vector
                                                                │
                                                       ┌───────┴────────┐
                                                       │  Cosine search │
                                                       │  against index │
                                                       └───────┬────────┘
                                                                │
                                                      Top-k brand matches
                                                       ├── similarity scores
                                                       ├── brand vote aggregation
                                                       └── preview grid
```

Detection and retrieval are **fully separated** — detection is an
optional preprocessing step that can be toggled via `--use-detector`.

---

## Quick Start

### 1. Build the Index

```bash
# Auto-download dataset & build embedding index
python "Logo Retrieval Brand Match/Source Code/index_builder.py"

# From custom directory (brand sub-folders)
python "Logo Retrieval Brand Match/Source Code/index_builder.py" \
    --data path/to/logos/

# Update existing index with new images
python "Logo Retrieval Brand Match/Source Code/index_builder.py" \
    --data new_logos/ --update
```

### 2. Query

```bash
# Single image
python "Logo Retrieval Brand Match/Source Code/infer.py" \
    --source query_logo.png --save-grid

# Directory of queries with CSV export
python "Logo Retrieval Brand Match/Source Code/infer.py" \
    --source test_logos/ --export-csv results.csv --save-grid

# Custom top-k
python "Logo Retrieval Brand Match/Source Code/infer.py" \
    --source logo.jpg --top-k 10

# With logo detection (for scene images)
python "Logo Retrieval Brand Match/Source Code/infer.py" \
    --source scene_photo.jpg --use-detector
```

### 3. Evaluate

```bash
python "Logo Retrieval Brand Match/Source Code/evaluate.py" --eval
```

---

## CLI Reference

### `index_builder.py` — Build Index

| Argument | Default | Description |
|---|---|---|
| `--data` | auto | Directory with brand/image sub-folders |
| `--config` | — | JSON/YAML config file |
| `--index` | from config | Output index path |
| `--update` | off | Add to existing index |
| `--force-download` | off | Force re-download dataset |

### `infer.py` — Query

| Argument | Default | Description |
|---|---|---|
| `--source` | *(required)* | Image or directory of queries |
| `--config` | — | Config file |
| `--index` | from config | Path to index `.npz` |
| `--top-k` | `5` | Number of matches to return |
| `--use-detector` | off | Enable YOLO logo cropping |
| `--no-display` | off | Suppress GUI |
| `--save-grid` | off | Save preview grids |
| `--export-json` | — | Save JSON results |
| `--export-csv` | — | Save CSV results |

### `evaluate.py` — Evaluation

| Argument | Default | Description |
|---|---|---|
| `--eval` | off | Run evaluation |
| `--max-queries` | `200` | Max queries for leave-one-out |
| `--index` | from config | Path to index |

---

## Configuration

All parameters tuneable via `LogoConfig` or a JSON/YAML file:

```json
{
    "backbone": "efficientnet_b0",
    "embedding_dim": 1280,
    "imgsz": 224,
    "index_path": "index/logo_index.npz",
    "top_k": 5,
    "min_similarity": 0.0,
    "use_detector": false,
    "grid_cols": 5,
    "grid_thumb_size": 128
}
```

### Backbone Options

| Backbone | Dim | Notes |
|---|---|---|
| `efficientnet_b0` | 1280 | **Default** — fast, good accuracy |
| `efficientnet_b2` | 1408 | Slightly better accuracy |
| `resnet50` | 2048 | Classic, larger embeddings |
| `resnet18` | 512 | Fastest, smallest index |
| `mobilenet_v3_small` | 576 | Mobile-friendly |

---

## How It Works

### Index Building

1. Download the logo dataset (auto, idempotent).
2. Organise images into `brand_name/` sub-folders.
3. Extract L2-normalised embedding vectors via EfficientNet-B0.
4. Save the index as a compressed `.npz` file.

### Retrieval

1. Load query image (optionally detect/crop logo region).
2. Extract embedding with the same backbone.
3. Compute cosine similarity against all index entries.
4. Return top-k matches ranked by similarity score.
5. Aggregate scores by brand (**brand voting**) for robust matching.

### Brand Voting

When multiple examples of the same brand appear in the top-k, their
similarity scores are summed.  This makes the matching robust to
intra-class variation.

---

## Project Structure

```
Logo Retrieval Brand Match/
├── Source Code/
│   ├── config.py           # LogoConfig dataclass
│   ├── detector.py         # Optional YOLO logo cropper
│   ├── embedder.py         # EfficientNet feature extractor
│   ├── index.py            # LogoIndex — build, save, load, search
│   ├── retriever.py        # LogoRetriever — top-k query
│   ├── visualize.py        # Preview grids & overlays
│   ├── export.py           # JSON / CSV export
│   ├── validator.py        # Input validation
│   ├── controller.py       # LogoController orchestrator
│   ├── infer.py            # Query CLI entry point
│   ├── index_builder.py    # Index build / update script
│   ├── evaluate.py         # Retrieval accuracy evaluation
│   ├── modern.py           # CVProject registry entry
│   └── data_bootstrap.py   # Dataset download & organisation
├── requirements.txt
└── README.md
```

---

## Dataset

**Popular Brand Logos** (`kkhandekar/popular-brand-logos-image-dataset`)

- **Source:** Kaggle (CC0 Public Domain)
- **Size:** ~25 MB, 1,471 logo images
- **Includes:** CSV metadata with brand labels
- **Auto-organised** into brand sub-folders on first run

The project also works on arbitrary local images — point
`index_builder.py --data` at any directory of `brand_name/image.ext`
sub-folders.

---

## Requirements

```
torch>=2.0.0
torchvision>=0.15.0
opencv-python>=4.10.0,<5
numpy>=1.26.0
```

Optional for logo detection: `ultralytics>=8.3.0`
