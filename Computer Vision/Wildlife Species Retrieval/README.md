# Wildlife Species Retrieval

Retrieve **visually similar wildlife images** from a 90-species embedding
index, with optional **classifier-based reranking** to boost same-species
matches.

---

## Pipeline Overview

```
query image
    │
    ▼
┌──────────────────────┐
│  EfficientNet-B0     │   Pretrained backbone (head removed)
│  Feature Extractor   │   → 1280-D L2-normalised embedding
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│  Cosine Similarity   │   Compare query embedding against index
│  Index Search        │   → top-k nearest neighbours
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│  (Optional)          │   Species classifier predicts query species;
│  Classifier Rerank   │   boosts hits with matching species label
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│  Results             │   Preview grid, species votes, JSON/CSV
└──────────────────────┘
```

---

## Key Design: Retrieval + Classification Separation

The **retrieval** and **classification** components are fully independent:

| Component | File | Purpose |
|-----------|------|---------|
| Embedder | `embedder.py` | Backbone feature extraction (6 backbones) |
| Index | `index.py` | Numpy `.npz` storage + cosine search |
| Retriever | `retriever.py` | Query orchestration (embedder + index) |
| Classifier | `classifier.py` | Species classification (separate model) |
| Controller | `controller.py` | Wires together retrieval + optional reranking |

The classifier is **never required** — retrieval works independently.
When `enable_rerank=True`, the controller blends similarity scores with a
species-match bonus:

$$\text{score}_{\text{final}} = (1 - w) \times \text{sim} + w \times \mathbb{1}[\text{species match}]$$

where $w$ is configurable via `rerank_weight` (default 0.4).

---

## Dataset

**Animal Image Dataset (90 Different Animals)**
— [iamsouravbanerjee/animal-image-dataset-90-different-animals](https://www.kaggle.com/datasets/iamsouravbanerjee/animal-image-dataset-90-different-animals)

- **5,400 images** across **90 species**
- ImageFolder format: `animals/<species>/*.jpg`
- ~699 MB

### 90 Species

antelope, badger, bat, bear, bee, beetle, bison, boar, butterfly, cat,
caterpillar, chimpanzee, cockroach, cow, coyote, crab, crow, deer, dog,
dolphin, donkey, dragonfly, duck, eagle, elephant, flamingo, fly, fox,
goat, goldfish, goose, gorilla, grasshopper, hamster, hare, hedgehog,
hippopotamus, hornbill, horse, hummingbird, hyena, jellyfish, kangaroo,
koala, ladybugs, leopard, lion, lizard, lobster, mosquito, moth, mouse,
octopus, okapi, orangutan, otter, owl, ox, oyster, panda, parrot,
pelecaniformes, penguin, pig, pigeon, porcupine, possum, raccoon, rat,
reindeer, rhinoceros, sandpiper, seahorse, seal, shark, sheep, snake,
sparrow, squid, squirrel, starfish, swan, tiger, turkey, turtle, whale,
wolf, wombat, woodpecker, zebra

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Download dataset

```bash
python "Source Code/data_bootstrap.py"

# Force re-download
python "Source Code/data_bootstrap.py" --force-download
```

### 3. Build embedding index

```bash
python "Source Code/index_builder.py" --image-dir data/animals
python "Source Code/index_builder.py" --image-dir data/animals --force --batch-size 64
```

### 4. Query for similar images

```bash
# Basic query
python "Source Code/infer.py" --source query.jpg

# With grid output and exports
python "Source Code/infer.py" --source query.jpg --top-k 8 \
    --save-grid output/grid.jpg \
    --export-json results.json --export-csv results.csv

# With classifier reranking
python "Source Code/infer.py" --source query.jpg --rerank
```

### 5. Train classifier (for reranking)

```bash
python "Source Code/train.py"
python "Source Code/train.py" --model efficientnet_b0 --epochs 30
```

### 6. Evaluate retrieval

```bash
python "Source Code/evaluate.py" --val-dir data/val --top-k 5
```

---

## CLI Reference

### `index_builder.py`

| Argument | Default | Description |
|----------|---------|-------------|
| `--image-dir` | required | Directory with species sub-folders |
| `--index-path` | `index/wildlife_index.npz` | Index output path |
| `--backbone` | `efficientnet_b0` | Feature extractor |
| `--batch-size` | 32 | Embedding batch size |
| `--force` | — | Rebuild even if index exists |
| `--device` | auto | `cpu` or `cuda` |

### `infer.py`

| Argument | Default | Description |
|----------|---------|-------------|
| `--source` | required | Query image path |
| `--index-path` | auto | Index path override |
| `--top-k` | 8 | Number of results |
| `--backbone` | `efficientnet_b0` | Feature extractor |
| `--rerank` | — | Enable classifier reranking |
| `--save-grid` | — | Save result grid |
| `--show` | — | Display result grid |
| `--export-json` | — | Export to JSON |
| `--export-csv` | — | Export to CSV |

### `evaluate.py`

| Argument | Default | Description |
|----------|---------|-------------|
| `--val-dir` | required | Validation set (ImageFolder) |
| `--index-path` | auto | Index path |
| `--top-k` | 5 | Retrieval depth |

### `train.py`

| Argument | Default | Description |
|----------|---------|-------------|
| `--data` | auto | Dataset path |
| `--model` | `resnet18` | Classifier architecture |
| `--epochs` | 25 | Training epochs |
| `--batch` | 32 | Batch size |
| `--force-download` | — | Force re-download |

---

## Project Structure

```
Wildlife Species Retrieval/
├── README.md
├── requirements.txt
└── Source Code/
    ├── config.py           # WildlifeConfig dataclass
    ├── embedder.py         # CNN feature extraction (6 backbones)
    ├── index.py            # Numpy embedding index + cosine search
    ├── retriever.py        # Query orchestration
    ├── classifier.py       # Species classifier (optional reranking)
    ├── controller.py       # High-level facade
    ├── index_builder.py    # CLI: build embedding index
    ├── infer.py            # CLI: query for similar images
    ├── evaluate.py         # CLI: retrieval accuracy evaluation
    ├── train.py            # CLI: train classifier for reranking
    ├── data_bootstrap.py   # Auto-download dataset
    ├── visualize.py        # Result grids with species + scores
    ├── export.py           # JSON / CSV export
    ├── validator.py        # Input validation
    └── modern.py           # CVProject registry entry
```

---

## Output Formats

### Result Grid

Query image on the left, top-k matches on the right with species labels
and similarity scores overlaid.

### JSON Export

```json
{
  "query": "query.jpg",
  "matches": [
    {"rank": 1, "path": "data/tiger/001.jpg", "species": "tiger", "score": 0.9234},
    {"rank": 2, "path": "data/leopard/042.jpg", "species": "leopard", "score": 0.8712}
  ],
  "species_votes": {
    "tiger": 3.456,
    "leopard": 1.234
  }
}
```

### CSV Export

| query | rank | match_path | species | score |
|-------|------|-----------|---------|-------|
| query.jpg | 1 | data/tiger/001.jpg | tiger | 0.9234 |
| query.jpg | 2 | data/leopard/042.jpg | leopard | 0.8712 |

---

## Supported Backbones

| Backbone | Embedding Dim | Speed | Quality |
|----------|--------------|-------|---------|
| `efficientnet_b0` | 1280 | Fast | Best (default) |
| `efficientnet_b2` | 1408 | Medium | Better |
| `resnet18` | 512 | Fastest | Good |
| `resnet50` | 2048 | Slow | Better |
| `mobilenet_v3_small` | 576 | Fastest | Fair |
| `mobilenet_v3_large` | 960 | Fast | Good |
