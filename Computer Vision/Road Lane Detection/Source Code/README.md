# Road Lane Detection

> **Task:** Utility &nbsp;|&nbsp; **Key:** `road_lane_detection` &nbsp;|&nbsp; **Framework:** Same OpenCV pipeline, wrapped in CVProject interface

---

## Overview

Detects road lane markings using classical CV: grayscale → dilate(3×3) → Canny(130, 220) → ROI triangle mask → HoughLinesP(1, π/180, 10, minLen=15, maxGap=2) → draws green lines. No neural network or model loading.

## Technology

| Aspect | Details |
|--------|---------|
| **Task Type** | Utility (`utility`) |
| **Legacy Stack** | OpenCV Canny + HoughLinesP |
| **Modern Stack** | Same (OpenCV) — wrapped in unified interface |
| **Dataset** | N/A — provide dashcam image/video |
| **Key Metrics** | N/A (visual accuracy) |
| **Download** | N/A (enabled: no) |

## Project Structure

```
Road Lane Detection/
└── Source Code/
    ├── modern.py        # CVProject subclass — @register("road_lane_detection")
    └── ...              # Legacy code, notebooks, original assets
```

## Quick Start

### Inference

```python
from core import discover_projects, run

discover_projects()
result = run("road_lane_detection", "path/to/image.jpg")
```

**How it works:** `modern.py` performs **no model loading** — `load()` is a no-op.  Inference: `predict(input_data)` runs a pure OpenCV pipeline and returns an annotated BGR image (numpy array).  No `resolve()`, no YOLO, no neural network.

### Dataset

Config: `configs/datasets/road_lane_detection.yaml`

> Automatic download is **disabled** for this project.

```bash
python -m utils.data_downloader road_lane_detection       # download (if enabled)
python scripts/validate_datasets.py          # status report
```

## Links

- [Root README](../../README.md)
- [Project Inventory](../../reports/PROJECT_INVENTORY.md)
- [Dataset Config](../../configs/datasets/road_lane_detection.yaml)
