# P10: Pose Detector (YOLO-Pose)

![Ultralytics YOLO-Pose](https://img.shields.io/badge/Framework-Ultralytics_YOLO-Pose-blue) ![Pose](https://img.shields.io/badge/Task-Pose-green) ![Trainable](https://img.shields.io/badge/Trainable-yes-orange)

## Overview

Detects human body poses in real-time using YOLO-Pose keypoint estimation with person counting.

## Entry Points

### Run / Inference

**Via the unified runner:**

```bash
python -m core.runner pose_detector_v2 --source 0 --import-all
```

**Programmatic usage:**

```python
from core.runner import run_camera
run_camera("pose_detector_v2", source=0)
```

### Train

```bash
cd "Project 10 - Live Pose Detector"
python train.py --data path/to/data.yaml
```

Training registers the resulting model version in the model registry
(`models/metadata.json`) and auto-promotes based on the primary metric.

## Model Resolution

Resolves weights via `models.registry.resolve("pose_detection", "pose")`, falls back to `yolo26n-pose.pt`.

The model registry (`models/registry.py`) resolves weights in this order:

1. **Trained model** — `models/<project>/<version>/best.pt` if registered and file exists
2. **YOLO26 pretrained fallback** — auto-downloaded by Ultralytics on first use

## Dataset

Configuration: `configs/datasets/pose_detector.yaml`

Download method: **http** (auto-download enabled)

```bash
python -m utils.data_downloader --project pose_detector
```

Expected layout after download:

```
data/pose_detector/
  data.yaml
  train/images/
  valid/images/
```

## Processing Pipeline

- **Load**: See [Model Resolution](#model-resolution) above.
- **Predict**: Runs YOLO-Pose inference on frame with confidence 0.5.
- **Visualize**: Calls `output[0].plot()`, shows person count text at bottom of frame.

## Outputs

- OpenCV display window showing annotated frames in real-time
- Training: `runs/detect/train/weights/best.pt` (registered in model registry)
- Press `q` to quit the camera loop

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `Cannot open video source` | Check webcam index — try `--source 1` |
| Model downloads on first run | Normal — Ultralytics auto-downloads YOLO26 pretrained weights |
| Dataset not found | Run `python -m utils.data_downloader --project pose_detector` |
| Training OOM | Reduce `--batch` or use a smaller model (`yolo26n.pt`) |

## Testing

No project-level test suite. Use workspace-level smoke tests:

```bash
python scripts/smoke_3b3.py
```
