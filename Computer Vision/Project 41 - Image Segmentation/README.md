# P41: Image Segmentation (YOLO-Seg)

![Ultralytics YOLO-Seg](https://img.shields.io/badge/Framework-Ultralytics_YOLO-Seg-blue) ![Segmentation](https://img.shields.io/badge/Task-Segmentation-green) ![Trainable](https://img.shields.io/badge/Trainable-yes-orange)

## Overview

Performs instance segmentation on live camera frames using YOLO26-Seg. Training supports both YOLO-Seg and DeepLabV3 backends (see `train.py deeplab`).

## Entry Points

### Run / Inference

**Via the unified runner:**

```bash
python -m core.runner image_segmentation_v2 --source 0 --import-all
```

**Programmatic usage:**

```python
from core.runner import run_camera
run_camera("image_segmentation_v2", source=0)
```

### Train

```bash
cd "Project 41 - Image Segmentation"
python train.py --data path/to/data.yaml
```

Training registers the resulting model version in the model registry
(`models/metadata.json`) and auto-promotes based on the primary metric.

## Model Resolution

Resolves weights via `models.registry.resolve("image_segmentation", "seg")`, falls back to `yolo26n-seg.pt`.

The model registry (`models/registry.py`) resolves weights in this order:

1. **Trained model** — `models/<project>/<version>/best.pt` if registered and file exists
2. **YOLO26 pretrained fallback** — auto-downloaded by Ultralytics on first use

## Dataset

Configuration: `configs/datasets/image_segmentation.yaml`

Download method: **manual** (manual download required — see URL in config)

Expected layout after download:

```
data/image_segmentation/
  data.yaml
  train/images/
  valid/images/
```

## Processing Pipeline

- **Load**: See [Model Resolution](#model-resolution) above.
- **Predict**: Runs YOLO-Seg inference on frame with confidence 0.4.
- **Visualize**: Calls `output[0].plot()` for segmentation masks, shows segment count text.

## Outputs

- OpenCV display window showing annotated frames in real-time
- Training: `runs/detect/train/weights/best.pt` (registered in model registry)
- Press `q` to quit the camera loop

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `Cannot open video source` | Check webcam index — try `--source 1` |
| Model downloads on first run | Normal — Ultralytics auto-downloads YOLO26 pretrained weights |
| Dataset not found | Run `python -m utils.data_downloader --project image_segmentation` |
| Training OOM | Reduce `--batch` or use a smaller model (`yolo26n.pt`) |

## Testing

No project-level test suite. Use workspace-level smoke tests:

```bash
python scripts/smoke_3b3.py
```
