# P17: Blink Detection (YOLO-Pose)

![Ultralytics YOLO-Pose](https://img.shields.io/badge/Framework-Ultralytics_YOLO-Pose-blue) ![Pose](https://img.shields.io/badge/Task-Pose-green) ![Trainable](https://img.shields.io/badge/Trainable-yes-orange)

## Overview

Detects eye blinks using YOLO-Pose keypoints. Requires a fine-tuned face-keypoint model for accurate Eye Aspect Ratio computation.

## Entry Points

### Run / Inference

**Via the unified runner:**

```bash
python -m core.runner blink_detection_v2 --source 0 --import-all
```

**Programmatic usage:**

```python
from core.runner import run_camera
run_camera("blink_detection_v2", source=0)
```

### Train

```bash
cd "Project 17 - Blink Detection"
python train.py --data path/to/data.yaml
```

Training registers the resulting model version in the model registry
(`models/metadata.json`) and auto-promotes based on the primary metric.

## Model Resolution

Resolves weights via `models.registry.resolve("blink_detection", "pose")`, falls back to `yolo26n-pose.pt`.

The model registry (`models/registry.py`) resolves weights in this order:

1. **Trained model** — `models/<project>/<version>/best.pt` if registered and file exists
2. **YOLO26 pretrained fallback** — auto-downloaded by Ultralytics on first use

## Dataset

Configuration: `configs/datasets/blink_detection.yaml`

Download method: **manual** (manual download required — see URL in config)

Expected layout after download:

```
data/blink_detection/
  data.yaml
  train/images/
  valid/images/
```

## Processing Pipeline

- **Load**: See [Model Resolution](#model-resolution) above.
- **Predict**: Runs YOLO-Pose inference on frame with confidence 0.5.
- **Visualize**: Calls `output[0].plot()`, adds text noting face-keypoint model needed for blink EAR (Eye Aspect Ratio).

## Outputs

- OpenCV display window showing annotated frames in real-time
- Training: `runs/detect/train/weights/best.pt` (registered in model registry)
- Press `q` to quit the camera loop

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `Cannot open video source` | Check webcam index — try `--source 1` |
| Model downloads on first run | Normal — Ultralytics auto-downloads YOLO26 pretrained weights |
| Dataset not found | Run `python -m utils.data_downloader --project blink_detection` |
| Training OOM | Reduce `--batch` or use a smaller model (`yolo26n.pt`) |

## Testing

No project-level test suite. Use workspace-level smoke tests:

```bash
python scripts/smoke_3b3.py
```
