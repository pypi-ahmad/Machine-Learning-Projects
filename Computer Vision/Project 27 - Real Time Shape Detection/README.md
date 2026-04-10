# P27: Shape Detection (v2)

![OpenCV](https://img.shields.io/badge/Framework-OpenCV-blue) ![Utility](https://img.shields.io/badge/Task-Utility-green)

## Overview

Classifies geometric shapes (triangle, square, circle, etc.) in real-time using contour analysis and polygon approximation.

## Entry Points

### Run / Inference

**Via the unified runner:**

```bash
python -m core.runner shape_detection_v2 --source 0 --import-all
```

**Programmatic usage:**

```python
from core.runner import run_camera
run_camera("shape_detection_v2", source=0)
```

## Processing Pipeline

- **Load**: No-op (no model to load).
- **Predict**: Grayscale + blur + adaptive threshold, finds contours, classifies shapes by polygon vertex count (triangle, square, rectangle, pentagon, hexagon, circle).
- **Visualize**: Draws contours in green, labels each shape name at centroid, shows total shape count.

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
