# P9: Real-Time Painter (v2)

![OpenCV](https://img.shields.io/badge/Framework-OpenCV-blue) ![Utility](https://img.shields.io/badge/Task-Utility-green)

## Overview

Lets users paint on the camera feed in real-time by tracking a blue-colored marker using HSV color detection.

## Entry Points

### Run / Inference

**Via the unified runner:**

```bash
python -m core.runner painter_v2 --source 0 --import-all
```

**Programmatic usage:**

```python
from core.runner import run_camera
run_camera("painter_v2", source=0)
```

## Processing Pipeline

- **Load**: Initializes a `deque` of canvas points (max 1024).
- **Predict**: HSV color tracking (blue marker range), finds largest contour, tracks center point via minEnclosingCircle.
- **Visualize**: Flips frame horizontally, draws connected lines between tracked points in red, shows green circle at current tracked center.

## Outputs

- OpenCV display window showing annotated frames in real-time
- Press `q` to quit the camera loop

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `Cannot open video source` | Check webcam index — try `--source 1` |

## Testing

No project-level test suite. Use workspace-level smoke tests:

```bash
python scripts/smoke_3b3.py
```
