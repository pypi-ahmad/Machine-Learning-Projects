# P7: Object Size Detector (v2)

![OpenCV](https://img.shields.io/badge/Framework-OpenCV-blue) ![Utility](https://img.shields.io/badge/Task-Utility-green)

## Overview

Measures real-world object dimensions in centimeters from a camera feed using contour detection and a pixel-per-cm calibration constant.

## Entry Points

### Run / Inference

**Via the unified runner:**

```bash
python -m core.runner object_size_v2 --source 0 --import-all
```

**Programmatic usage:**

```python
from core.runner import run_camera
run_camera("object_size_v2", source=0)
```

## Processing Pipeline

- **Load**: No-op (no model to load).
- **Predict**: Grayscale + blur + Canny + dilate/erode, finds external contours, computes minAreaRect, calculates dimensions in cm using PIXELS_PER_CM calibration.
- **Visualize**: Draws rotated bounding boxes, labels each object with width x height in cm, shows object count.

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
