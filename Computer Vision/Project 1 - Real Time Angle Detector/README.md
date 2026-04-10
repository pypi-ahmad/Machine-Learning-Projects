# P1: Angle Detector (v2)

![OpenCV](https://img.shields.io/badge/Framework-OpenCV-blue) ![Utility](https://img.shields.io/badge/Task-Utility-green)

## Overview

Detects edges and lines in a live camera feed and computes the angle between the first two detected lines using OpenCV Hough line transform.

## Entry Points

### Run / Inference

**Via the unified runner:**

```bash
python -m core.runner angle_detector_v2 --source 0 --import-all
```

**Programmatic usage:**

```python
from core.runner import run_camera
run_camera("angle_detector_v2", source=0)
```

## Processing Pipeline

- **Load**: Initializes empty points list (no model needed).
- **Predict**: Converts to grayscale, applies Gaussian blur + Canny edge detection, finds lines via HoughLinesP.
- **Visualize**: Draws detected lines on frame, computes angle between first two lines using atan2, displays angle text.

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
