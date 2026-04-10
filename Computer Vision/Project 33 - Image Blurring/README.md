# P33: Image Blurring (v2)

![OpenCV](https://img.shields.io/badge/Framework-OpenCV-blue) ![Utility](https://img.shields.io/badge/Task-Utility-green)

## Overview

Demonstrates and compares four OpenCV blurring methods: Gaussian, Median, Bilateral, and Box blur.

## Entry Points

### Run / Inference

**Via the unified runner:**

```bash
python -m core.runner image_blurring_v2 --source 0 --import-all
```

**Programmatic usage:**

```python
from core.runner import run_camera
run_camera("image_blurring_v2", source=0)
```

## Processing Pipeline

- **Load**: No-op (no model to load).
- **Predict**: Applies 4 blur methods: GaussianBlur, medianBlur, bilateralFilter, box blur (`cv2.blur`) with kernel size 15.
- **Visualize**: 2×2 grid showing Gaussian, Median, Bilateral, and Box blur results — each labeled.

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
