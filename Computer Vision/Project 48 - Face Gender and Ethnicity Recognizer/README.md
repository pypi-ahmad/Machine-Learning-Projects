# P48: Face Attributes (YOLO + Torch)

![Ultralytics YOLO](https://img.shields.io/badge/Framework-Ultralytics_YOLO-blue) ![Detection](https://img.shields.io/badge/Task-Detection-green) ![Trainable](https://img.shields.io/badge/Trainable-yes-orange)

## Overview

Detects faces and predicts age, gender, and ethnicity. Training pipeline uses ResNet-18 multi-output classification on UTKFace via torchvision.

## Entry Points

### Run / Inference

**Via the unified runner:**

```bash
python -m core.runner face_attributes_v2 --source 0 --import-all
```

**Programmatic usage:**

```python
from core.runner import run_camera
run_camera("face_attributes_v2", source=0)
```

### Train

```bash
cd "Project 48 - Face Gender and Ethnicity Recognizer"
python train.py --data path/to/data.yaml
```

Training registers the resulting model version in the model registry
(`models/metadata.json`) and auto-promotes based on the primary metric.

## Model Resolution

Resolves weights via `models.registry.resolve("face_attributes", "detect")`, falls back to `yolo26n.pt`.

The model registry (`models/registry.py`) resolves weights in this order:

1. **Trained model** — `models/<project>/<version>/best.pt` if registered and file exists
2. **YOLO26 pretrained fallback** — auto-downloaded by Ultralytics on first use

## Dataset

Configuration: `configs/datasets/face_attributes.yaml`

Download method: **manual** (manual download required — see URL in config)

Expected layout after download:

```
data/face_attributes/
  data.yaml
  train/images/
  valid/images/
```

## Processing Pipeline

- **Load**: See [Model Resolution](#model-resolution) above.
- **Predict**: Runs YOLO detection filtered to class 0 (person) with confidence 0.5.
- **Visualize**: Calls `output[0].plot()`, adds text noting fine-tuned age/gender model needed.

## Outputs

- OpenCV display window showing annotated frames in real-time
- Training: `runs/detect/train/weights/best.pt` (registered in model registry)
- Press `q` to quit the camera loop

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `Cannot open video source` | Check webcam index — try `--source 1` |
| Model downloads on first run | Normal — Ultralytics auto-downloads YOLO26 pretrained weights |
| Dataset not found | Run `python -m utils.data_downloader --project face_attributes` |
| Training OOM | Reduce `--batch` or use a smaller model (`yolo26n.pt`) |

## Testing

No project-level test suite. Use workspace-level smoke tests:

```bash
python scripts/smoke_3b3.py
```
