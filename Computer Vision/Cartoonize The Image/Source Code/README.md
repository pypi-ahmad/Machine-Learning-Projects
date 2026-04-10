# Cartoonize The Image

> **Task:** Utility &nbsp;|&nbsp; **Key:** `cartoonize_image` &nbsp;|&nbsp; **Framework:** Same OpenCV pipeline, wrapped in CVProject interface

---

## Overview

Applies cartoon-style effects to images using a pure OpenCV pipeline: grayscale → medianBlur → adaptiveThreshold → bilateralFilter → bitwise_and. No neural network or model loading.

## Technology

| Aspect | Details |
|--------|---------|
| **Task Type** | Utility (`utility`) |
| **Legacy Stack** | OpenCV bilateral filter + adaptive threshold |
| **Modern Stack** | Same (OpenCV) — wrapped in unified interface |
| **Dataset** | N/A — provide any image |
| **Key Metrics** | N/A (visual quality) |
| **Download** | N/A (enabled: no) |

## Project Structure

```
Cartoonize The Image/
└── Source Code/
    ├── modern.py        # CVProject subclass — @register("cartoonize_image")
    └── ...              # Legacy code, notebooks, original assets
```

## Quick Start

### Inference

```python
from core import discover_projects, run

discover_projects()
result = run("cartoonize_image", "path/to/image.jpg")
```

**How it works:** `modern.py` performs **no model loading** — `load()` is a no-op.  Inference: `predict(input_data)` runs a pure OpenCV pipeline and returns an annotated BGR image (numpy array).  No `resolve()`, no YOLO, no neural network.

### Dataset

Config: `configs/datasets/cartoonize_image.yaml`

> Automatic download is **disabled** for this project.

```bash
python -m utils.data_downloader cartoonize_image       # download (if enabled)
python scripts/validate_datasets.py          # status report
```

## Links

- [Root README](../../README.md)
- [Project Inventory](../../reports/PROJECT_INVENTORY.md)
- [Dataset Config](../../configs/datasets/cartoonize_image.yaml)
