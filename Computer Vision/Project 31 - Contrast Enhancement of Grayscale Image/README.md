# P31: Contrast Enhancement Gray (v2)

![OpenCV](https://img.shields.io/badge/Framework-OpenCV-blue) ![Utility](https://img.shields.io/badge/Task-Utility-green)

## Overview

Compares histogram equalization and CLAHE for grayscale contrast enhancement.

## Entry Points

### Run / Inference

**Via the unified runner:**

```bash
python -m core.runner contrast_gray_v2 --source 0 --import-all
```

**Programmatic usage:**

```python
from core.runner import run_camera
run_camera("contrast_gray_v2", source=0)
```

## Processing Pipeline

- **Load**: Creates a CLAHE instance with clipLimit=2.0, tileGridSize=(8, 8).
- **Predict**: Converts to grayscale, applies both histogram equalization and CLAHE.
- **Visualize**: 3-image horizontal strip: Original, Hist EQ, CLAHE — each labeled.

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
