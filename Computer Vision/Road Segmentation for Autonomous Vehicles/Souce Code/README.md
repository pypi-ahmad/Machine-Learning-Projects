# Road Segmentation for Autonomous Vehicles

> **Task:** Segmentation &nbsp;|&nbsp; **Key:** `road_segmentation` &nbsp;|&nbsp; **Framework:** Ultralytics YOLO26-seg

---

## Overview

Semantic segmentation of road surfaces and driveable areas for autonomous driving.

## Technology

| Aspect | Details |
|--------|---------|
| **Task Type** | Segmentation (`seg`) |
| **Legacy Stack** | Custom segmentation (notebook) |
| **Modern Stack** | Ultralytics YOLO26-seg |
| **Dataset** | KITTI Road / CamVid (manual download) |
| **Key Metrics** | mIoU, road F1 |
| **Download** | manual_page (enabled: no) |

## Project Structure

```
Road segmentation for autonomous vechicles/
└── Souce Code/
    ├── modern.py        # CVProject subclass — @register("road_segmentation")
    ├── train.py         # CLI training entry point
    └── ...              # Legacy code, notebooks, original assets
```

## Quick Start

### Inference

```python
from core import discover_projects, run

discover_projects()
result = run("road_segmentation", "path/to/image.jpg")
```

**How it works:** `modern.py` calls `resolve("road_segmentation", "seg")` → `load_yolo(weights)`.  Default pretrained: `yolo26n-seg.pt`.  Inference: `self.model(input_data, verbose=False)`.  Visualize with `output[0].plot()`.

### Training

```bash
cd "Road segmentation for autonomous vechicles/Souce Code"
python train.py --epochs 50 --batch 16
```

Delegates to `train.train_segmentation.train_segmentation()`.
Trained weights are auto-registered in `ModelRegistry`.

```python
from train.train_segmentation import train_segmentation
train_segmentation(data_yaml="data/road_segmentation/data.yaml", model="yolo26n-seg.pt", epochs=50)
```

### Dataset

Config: `configs/datasets/road_segmentation.yaml`

> **Manual download required.** Visit [https://www.cityscapes-dataset.com/](https://www.cityscapes-dataset.com/), then place files into `data/road_segmentation/`.

```bash
python -m utils.data_downloader road_segmentation       # download (if enabled)
python scripts/validate_datasets.py          # status report
```

### Evaluate Accuracy

```bash
python -m benchmarks.evaluate_accuracy --project road_segmentation
```

## Links

- [Root README](../../README.md)
- [Project Inventory](../../reports/PROJECT_INVENTORY.md)
- [Dataset Config](../../configs/datasets/road_segmentation.yaml)
