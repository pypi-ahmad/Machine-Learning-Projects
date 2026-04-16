# Face Clustering Photo Organizer

> **Task:** Face Clustering &nbsp;|&nbsp; **Key:** `face_clustering_photo_organizer` &nbsp;|&nbsp; **Framework:** InsightFace + scikit-learn

---

## Overview

Unsupervised face clustering pipeline that scans a collection of photos, detects faces, extracts ArcFace embeddings, and groups them by likely identity. Outputs organized per-identity folders, preview collages, and cluster manifests.

## Technology

| Aspect | Details |
|--------|---------|
| **Task Type** | Face Clustering / Photo Organization |
| **Detection** | YOLO face detector (primary) / InsightFace RetinaFace (fallback) |
| **Embeddings** | InsightFace ArcFace (buffalo_l, 512-d) |
| **Clustering** | Agglomerative (default) or DBSCAN on cosine distance |
| **Dataset** | LFW multi-identity face images (automatic local bootstrap) |
| **Key Metrics** | Cluster purity, cluster count |

## Dataset

- **Source:** `sklearn.datasets.fetch_lfw_people` (LFW). If that path is unavailable, the bootstrap falls back to the repo's shared dataset helper.
- **Layout:** ImageFolder — `identity_name/img1.jpg, img2.jpg, ...`
- **Prepared Path:** `data/face_clustering_photo_organizer/processed/identities/`
- **Download:** Automatic on first `python train.py` or `python data_bootstrap.py` run
- **Force re-download:** `python train.py --force-download` or `python data_bootstrap.py --force`

## Project Structure

```
Face Clustering Photo Organizer/
└── Source Code/
    ├── config.py          # FaceClusterConfig dataclass
    ├── face_detector.py   # YOLO / InsightFace face detection
    ├── embedder.py        # InsightFace ArcFace embedding extraction
    ├── clusterer.py       # Agglomerative / DBSCAN clustering
    ├── parser.py          # Full pipeline: detect → embed → cluster
    ├── organizer.py       # File organizer — cluster dirs + manifest
    ├── collage.py         # Preview collage builder per cluster
    ├── validator.py       # Quality checks and validation
    ├── export.py          # JSON / CSV export of cluster assignments
    ├── infer.py           # CLI — scan, cluster, organize
    ├── modern.py          # CVProject subclass — @register
    ├── train.py           # Dataset download + clustering evaluation
    ├── data_bootstrap.py  # Idempotent dataset bootstrap
    ├── requirements.txt   # Project dependencies
    └── README.md          # This file
```

## Quick Start

### Cluster Photos

```bash
# Scan a directory and cluster faces
python infer.py --source photos/

# Use DBSCAN instead of agglomerative
python infer.py --source photos/ --algorithm dbscan --threshold 0.55

# Export results
python infer.py --source photos/ --export-json clusters.json --export-csv faces.csv

# Save collages without GUI
python infer.py --source photos/ --no-display --save-collages

# Don't copy photos (manifest only)
python infer.py --source photos/ --no-copy
```

### Programmatic API

```python
from core import discover_projects
from core.registry import PROJECT_REGISTRY

discover_projects()
proj = PROJECT_REGISTRY["face_clustering_photo_organizer"]()
proj.load()

# Cluster a directory
result = proj.predict("path/to/photos/")
print(f"Found {result['num_clusters']} identity clusters")
print(f"Total faces: {result['total_faces']}")

# Visualize
collage = proj.visualize("path/to/photos/", result)
```

### Evaluation

```bash
cd "Face Clustering Photo Organizer/Source Code"
python data_bootstrap.py                     # prepare LFW locally
python train.py                              # download + evaluate
python train.py --force-download             # re-download dataset
python train.py --max-identities 100         # more identities
python train.py --algorithm dbscan           # try DBSCAN
python train.py --threshold 0.70             # adjust threshold
```

## CLI Reference

```
python infer.py [OPTIONS]

Required:
  --source PATH              Directory of photos or single image

Clustering:
  --algorithm {agglomerative,dbscan}   Clustering algorithm (default: agglomerative)
  --threshold FLOAT          Distance threshold (default: 0.85 agglom / 0.55 dbscan)
  --min-cluster-size INT     Minimum faces per cluster (default: 2)

Output:
  --output-dir DIR           Output directory (default: output/)
  --no-copy                  Don't copy photos into cluster folders
  --save-collages            Save collage preview images

Export:
  --export-json PATH         JSON export path
  --export-csv PATH          CSV export path

Other:
  --config PATH              YAML/JSON config file
  --no-display               Disable GUI windows
  --force-download           Force dataset re-download
```

## Output Structure

```
output/
├── clusters/
│   ├── person_000/          # Largest cluster
│   │   ├── photo1.jpg
│   │   └── photo3.jpg
│   ├── person_001/
│   │   └── photo2.jpg
│   └── ...
├── collages/
│   ├── cluster_000.jpg      # Face grid preview
│   ├── cluster_001.jpg
│   └── ...
└── cluster_manifest.json    # Full cluster metadata
```

## Features

- **InsightFace ArcFace embeddings** — 512-d normalized vectors (buffalo_l)
- **Dual clustering** — Agglomerative (default) or DBSCAN with cosine distance
- **Automatic cluster merging** — Close centroids merged automatically
- **Preview collages** — Grid of face thumbnails per cluster
- **Photo organization** — Source photos copied/symlinked into cluster folders
- **Cluster manifest** — JSON metadata with all cluster assignments
- **CSV/JSON export** — Per-face cluster assignment export
- **Recursive scan** — Finds images in subdirectories

## Configuration

```yaml
# face_cluster_config.yaml
algorithm: agglomerative
distance_threshold: 0.85
min_cluster_size: 2
collage_cols: 5
collage_thumb_size: 112
copy_photos: true
save_collages: true
```

```bash
python infer.py --source photos/ --config face_cluster_config.yaml
```

## Dependencies

```bash
pip install -r requirements.txt
```

Optional GPU runtime:

```bash
pip install onnxruntime-gpu
```
