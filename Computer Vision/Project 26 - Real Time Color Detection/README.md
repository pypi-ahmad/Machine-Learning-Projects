# P26: Color Detection (v2)

![OpenCV](https://img.shields.io/badge/Framework-OpenCV-blue) ![Utility](https://img.shields.io/badge/Task-Utility-green)

## Overview

Detects red-colored regions in a live camera feed using HSV color space and morphological filtering.

## Entry Points

### Run / Inference

**Via the unified runner:**

```bash
python -m core.runner color_detection_v2 --source 0 --import-all
```

**Programmatic usage:**

```python
from core.runner import run_camera
run_camera("color_detection_v2", source=0)
```

## Processing Pipeline

- **Load**: No-op (no model to load).
- **Predict**: HSV conversion, dual-range inRange mask for red detection, morphological open/close, finds contours with area > 500.
- **Visualize**: Draws red rectangles around detected red regions, labels them, shows detection count.

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
