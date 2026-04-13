# Logo Detection and Brand Recognition

> **Task:** Detection / matching &nbsp;|&nbsp; **Key:** `logo_detection` &nbsp;|&nbsp; **Framework:** YOLO26m detect + SIFT fallback

---

## Overview

Detects logos inside scenes and supports optional brand recognition. The modern pipeline is detection-first: use custom YOLO logo weights when available, or fall back to SIFT template matching for known logos.

## Technology

| Aspect | Details |
|--------|---------|
| **Task Type** | Detection (`detect`) with optional brand matching |
| **Legacy Stack** | MobileNetV2 (notebook) |
| **Modern Stack** | YOLO26m detect (custom logo weights) or SIFT template matching |
| **Dataset** | Flickr Logos / OpenLogo / LogoDet-style data prepared for YOLO if training detection |
| **Key Metrics** | mAP, precision, recall; brand-match quality if a recognition stage is added |
| **Download** | URL dataset resolver for raw assets; YOLO-format annotations required for detector training |

## Project Structure

```
Logo Detection and Brand Recognition/
└── Source Code/
    ├── modern.py        # CVProject subclass — @register("logo_detection")
    ├── train.py         # CLI training entry point
    └── ...              # Legacy code, notebooks, original assets
```

## Quick Start

### Inference

```python
from core import discover_projects, run

discover_projects()
result = run("logo_detection", "path/to/image.jpg")
```

**How it works:** `modern.py` calls `resolve("logo_detection", "detect")` to look for custom logo-detector weights. If they are available, it runs YOLO detection and returns logo bounding boxes. If not, the project falls back to SIFT template matching against a local template directory.

### Training

Preferred training is YOLO detection training with a prepared `data.yaml`:

```bash
cd "Logo Detection and Brand Recognition/Source Code"
python train.py --data path/to/data.yaml --epochs 50 --model yolo26m.pt
```

`train.py` validates that you provide a YOLO-format `data.yaml` (or a directory containing one) and then delegates to the shared detection training helper.

If you only want closed-set brand classification on cropped logos, keep that as a separate baseline rather than the default scene-logo method.

### Dataset

The repository dataset resolver can download the raw Flickr Logos archive into `data/logo_detection/`.

If you want to train a detector, you still need YOLO-format annotations and a `data.yaml`. Prepare those before launching `train.py`.

```bash
python -m utils.data_downloader logo_detection       # download (if enabled)
python scripts/validate_datasets.py          # status report
```

### Evaluation Notes

For scene logo detection, prefer localization metrics such as precision, recall, and mAP. If you add a second-stage brand recognizer, evaluate that stage separately from detection quality.

## Links

- [Root README](../../README.md)
- [Project Inventory](../../reports/PROJECT_INVENTORY.md)
- [Dataset Resolver Entry](../../utils/datasets.py)
