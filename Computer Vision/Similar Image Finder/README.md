# Similar Image Finder

Visual similarity search over a local or downloaded image corpus.  
Embeds images with a pretrained CNN backbone, indexes embeddings for
fast nearest-neighbour lookup, and returns top-k matches with similarity
scores and preview grids.

---

## Pipeline

```
images/          embedder.py        index.py          retriever.py
  ├─ cat/    ──►  EfficientNet-B0  ──►  .npz index  ──►  top-k results
  ├─ dog/         (1280-d vectors)      (cosine sim)     + category votes
  └─ …
```

## Quick Start

### 1. Download Dataset

```bash
python data_bootstrap.py                   # idempotent
python data_bootstrap.py --force-download  # re-download
```

Uses the [Natural Images](https://www.kaggle.com/datasets/prasunroy/natural-images)
dataset — 6,899 images across 8 categories (airplane, car, cat, dog, flower, fruit,
motorbike, person).

### 2. Build Index

```bash
python index_builder.py --image-dir data/natural_images/processed/images
python index_builder.py --image-dir data/natural_images/processed/images --force  # rebuild
```

### 3. Query

```bash
python infer.py --source photo.jpg --top-k 8
python infer.py --source photo.jpg --save-grid results/grid.jpg
python infer.py --source photo.jpg --export-json results/out.json
python infer.py --source photo.jpg --show
```

---

## Backbone Options

| Backbone             | Embedding Dim | Notes                   |
| -------------------- | ------------- | ----------------------- |
| `efficientnet_b0`    | 1280          | Default — good balance  |
| `efficientnet_b2`    | 1408          | Higher capacity         |
| `resnet50`           | 2048          | Classic, widely used    |
| `resnet18`           | 512           | Lightweight             |
| `mobilenet_v3_small` | 576           | Very fast, mobile-ready |
| `mobilenet_v3_large` | 960           | Fast, good accuracy     |

Override with `--backbone`:

```bash
python index_builder.py --image-dir images/ --backbone resnet50
python infer.py --source photo.jpg --backbone resnet50
```

---

## Index Caching

- The index is saved as a `.npz` file (default: `index/image_index.npz`)
- Subsequent runs load the cached index — no re-embedding needed
- Pass `--force` to `index_builder.py` to rebuild from scratch
- Override path with `--index-path`

---

## Export Formats

| Flag             | Format | Contents                                |
| ---------------- | ------ | --------------------------------------- |
| `--export-json`  | JSON   | Matches + category votes                |
| `--export-csv`   | CSV    | query, rank, match_path, category, score|

---

## Evaluation

Leave-one-out retrieval accuracy on the indexed dataset:

```bash
python evaluate.py --eval
python evaluate.py --eval --max-queries 200 --top-k 5
```

---

## Project Structure

```
Source Code/
├── config.py          # SimilarityConfig dataclass
├── embedder.py        # CNN feature extraction (6 backbones)
├── index.py           # Numpy embedding index + cosine search
├── retriever.py       # Query orchestration + category voting
├── visualize.py       # Result grid + overlay rendering
├── export.py          # JSON / CSV export
├── validator.py       # Input validation + image collection
├── controller.py      # High-level pipeline facade
├── infer.py           # CLI — query similar images
├── index_builder.py   # CLI — build / rebuild index
├── evaluate.py        # CLI — retrieval quality evaluation
├── modern.py          # CVProject registry integration
└── data_bootstrap.py  # Kaggle dataset download + organise
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

**Natural Images** by Prasun Roy  
6,899 images · 8 categories · ~180 MB · CC BY-NC-SA 4.0  
[kaggle.com/datasets/prasunroy/natural-images](https://www.kaggle.com/datasets/prasunroy/natural-images)
