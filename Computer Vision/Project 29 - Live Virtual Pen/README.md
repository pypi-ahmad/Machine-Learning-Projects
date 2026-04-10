# P29: Virtual Pen (v2)

![OpenCV](https://img.shields.io/badge/Framework-OpenCV-blue) ![Utility](https://img.shields.io/badge/Task-Utility-green)

## Overview

Draws on the camera feed in real-time by tracking a green-colored marker using HSV color detection.

## Entry Points

### Run / Inference

**Via the unified runner:**

```bash
python -m core.runner virtual_pen_v2 --source 0 --import-all
```

**Programmatic usage:**

```python
from core.runner import run_camera
run_camera("virtual_pen_v2", source=0)
```

## Processing Pipeline

- **Load**: Initializes a `deque` of points (max 2048).
- **Predict**: HSV color tracking (green marker range), erode/dilate morphology, finds largest contour, tracks center via minEnclosingCircle.
- **Visualize**: Draws connected red lines between tracked points, shows green circle at current center.

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
