# P49: Text Detection (YOLO)

![Ultralytics YOLO](https://img.shields.io/badge/Framework-Ultralytics_YOLO-blue) ![Detection](https://img.shields.io/badge/Task-Detection-green) ![Trainable](https://img.shields.io/badge/Trainable-yes-orange)

## Overview

Detects text regions in images using YOLO26, with a training pipeline for custom text-region datasets.

## Entry Points

### Run / Inference

**Via the unified runner:**

```bash
python -m core.runner text_detection_v2 --source 0 --import-all
```

**Programmatic usage:**

```python
from core.runner import run_camera
run_camera("text_detection_v2", source=0)
```

### Train

```bash
cd "Project 49 - Real Time Text Detection"
python train.py --data path/to/data.yaml
```

Training registers the resulting model version in the model registry
(`models/metadata.json`) and auto-promotes based on the primary metric.

## Model Resolution

Resolves weights via `models.registry.resolve("text_detection", "detect")`, falls back to `yolo26n.pt`.

The model registry (`models/registry.py`) resolves weights in this order:

1. **Trained model** — `models/<project>/<version>/best.pt` if registered and file exists
2. **YOLO26 pretrained fallback** — auto-downloaded by Ultralytics on first use

## Dataset

Configuration: `configs/datasets/text_detection.yaml`

Download method: **http** (auto-download enabled)

```bash
python -m utils.data_downloader --project text_detection
```

Expected layout after download:

```
data/text_detection/
  data.yaml
  train/images/
  valid/images/
```

## Processing Pipeline

- **Load**: See [Model Resolution](#model-resolution) above.
- **Predict**: Runs YOLO detection on frame with confidence 0.3.
- **Visualize**: Calls `output[0].plot()`, adds text noting PaddleOCR or fine-tuned YOLO recommended for actual text detection.

## Outputs

- OpenCV display window showing annotated frames in real-time
- Training: `runs/detect/train/weights/best.pt` (registered in model registry)
- Press `q` to quit the camera loop

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `Cannot open video source` | Check webcam index — try `--source 1` |
| Model downloads on first run | Normal — Ultralytics auto-downloads YOLO26 pretrained weights |
| Dataset not found | Run `python -m utils.data_downloader --project text_detection` |
| Training OOM | Reduce `--batch` or use a smaller model (`yolo26n.pt`) |

## Testing

No project-level test suite. Use workspace-level smoke tests:

```bash
python scripts/smoke_3b3.py
```
