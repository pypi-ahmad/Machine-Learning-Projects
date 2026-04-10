# P30: Contrast Enhancement Color (v2)

![OpenCV](https://img.shields.io/badge/Framework-OpenCV-blue) ![Utility](https://img.shields.io/badge/Task-Utility-green)

## Overview

Enhances color image contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization) in LAB color space.

## Entry Points

### Run / Inference

**Via the unified runner:**

```bash
python -m core.runner contrast_color_v2 --source 0 --import-all
```

**Programmatic usage:**

```python
from core.runner import run_camera
run_camera("contrast_color_v2", source=0)
```

## Processing Pipeline

- **Load**: Creates a CLAHE instance with clipLimit=3.0, tileGridSize=(8, 8).
- **Predict**: Converts to LAB color space, applies CLAHE to L channel, converts back to BGR.
- **Visualize**: Side-by-side split view — original on left, CLAHE enhanced on right, separated by a green line.

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
