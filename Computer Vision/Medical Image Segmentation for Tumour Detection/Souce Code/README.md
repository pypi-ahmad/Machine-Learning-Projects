# Medical Image Segmentation for Tumour Detection

> **Task:** Segmentation &nbsp;|&nbsp; **Key:** `medical_image_segmentation` &nbsp;|&nbsp; **Framework:** Ultralytics YOLO26-seg

---

## Overview

Brain tumour segmentation in medical images (MRI/CT). Legacy code used a custom segmentation notebook.

## Technology

| Aspect | Details |
|--------|---------|
| **Task Type** | Segmentation (`seg`) |
| **Legacy Stack** | Custom segmentation (notebook) |
| **Modern Stack** | Ultralytics YOLO26-seg |
| **Dataset** | BraTS / LGG Segmentation (manual download) |
| **Key Metrics** | mIoU, Dice |
| **Download** | manual_page (enabled: no) |

## Project Structure

```
Medical Image Segmentaion for Tumour Detection/
└── Souce Code/
    ├── modern.py        # CVProject subclass — @register("medical_image_segmentation")
    ├── train.py         # CLI training entry point
    └── ...              # Legacy code, notebooks, original assets
```

## Quick Start

### Inference

```python
from core import discover_projects, run

discover_projects()
result = run("medical_image_segmentation", "path/to/image.jpg")
```

**How it works:** `modern.py` calls `resolve("medical_image_segmentation", "seg")` → `load_yolo(weights)`.  Default pretrained: `yolo26n-seg.pt`.  Inference: `self.model(input_data, verbose=False)`.  Visualize with `output[0].plot()`.

### Training

```bash
cd "Medical Image Segmentaion for Tumour Detection/Souce Code"
python train.py --epochs 50 --batch 16
```

Delegates to `train.train_segmentation.train_segmentation()`.
Trained weights are auto-registered in `ModelRegistry`.

```python
from train.train_segmentation import train_segmentation
train_segmentation(data_yaml="data/medical_image_segmentation/data.yaml", model="yolo26n-seg.pt", epochs=50)
```

### Dataset

Config: `configs/datasets/medical_image_segmentation.yaml`

> **Manual download required.** Visit [https://www.cancerimagingarchive.net/collection/rsna-asnr-miccai-brats-2021/](https://www.cancerimagingarchive.net/collection/rsna-asnr-miccai-brats-2021/), then place files into `data/medical_image_segmentation/`.

```bash
python -m utils.data_downloader medical_image_segmentation       # download (if enabled)
python scripts/validate_datasets.py          # status report
```

### Evaluate Accuracy

```bash
python -m benchmarks.evaluate_accuracy --project medical_image_segmentation
```

## Links

- [Root README](../../README.md)
- [Project Inventory](../../reports/PROJECT_INVENTORY.md)
- [Dataset Config](../../configs/datasets/medical_image_segmentation.yaml)
