# P25: Object Measurement (v2)

![OpenCV](https://img.shields.io/badge/Framework-OpenCV-blue) ![Utility](https://img.shields.io/badge/Task-Utility-green)

## Overview

Measures real-world object dimensions from a camera feed using contour detection and a pixel-per-metric calibration constant.

## Entry Points

### Run / Inference

**Via the unified runner:**

```bash
python -m core.runner object_measurement_v2 --source 0 --import-all
```

**Programmatic usage:**

```python
from core.runner import run_camera
run_camera("object_measurement_v2", source=0)
```

## Processing Pipeline

- **Load**: No-op (no model to load).
- **Predict**: Grayscale + blur + Canny + dilate/erode, finds external contours, computes minAreaRect, calculates real-world dimensions using PIXELS_PER_METRIC.
- **Visualize**: Draws rotated bounding boxes with center dots, labels width × height, shows object count.

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
