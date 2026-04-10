# P5: Finger Counter (YOLO-Pose)

![Ultralytics YOLO-Pose](https://img.shields.io/badge/Framework-Ultralytics_YOLO-Pose-blue) ![Pose](https://img.shields.io/badge/Task-Pose-green)

## Overview

Counts visible fingers using YOLO-Pose body keypoints for wrist detection, replacing legacy MediaPipe hand tracking. No training pipeline — uses pretrained pose model.

## Entry Points

### Run / Inference

**Via the unified runner:**

```bash
python -m core.runner finger_counter_v2 --source 0 --import-all
```

**Programmatic usage:**

```python
from core.runner import run_camera
run_camera("finger_counter_v2", source=0)
```

## Model Resolution

Resolves weights via `models.registry.resolve("finger_counter", "pose")`, falls back to `yolo26n-pose.pt`.

The model registry (`models/registry.py`) resolves weights in this order:

1. **Trained model** — `models/<project>/<version>/best.pt` if registered and file exists
2. **YOLO26 pretrained fallback** — auto-downloaded by Ultralytics on first use

## Processing Pipeline

- **Load**: See [Model Resolution](#model-resolution) above.
- **Predict**: Runs YOLO-Pose inference on frame with confidence 0.5.
- **Visualize**: Calls `output[0].plot()`, highlights wrist keypoints (COCO indices 9, 10) with red circles and labels.

## Outputs

- OpenCV display window showing annotated frames in real-time
- Press `q` to quit the camera loop

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `Cannot open video source` | Check webcam index — try `--source 1` |
| Model downloads on first run | Normal — Ultralytics auto-downloads YOLO26 pretrained weights |

## Testing

No project-level test suite. Use workspace-level smoke tests:

```bash
python scripts/smoke_3b3.py
```
