# P42: Image Resizing (v2)

![OpenCV](https://img.shields.io/badge/Framework-OpenCV-blue) ![Utility](https://img.shields.io/badge/Task-Utility-green)

## Overview

Compares four OpenCV interpolation methods for image resizing: Nearest, Bilinear, Bicubic, and Lanczos4.

## Entry Points

### Run / Inference

**Via the unified runner:**

```bash
python -m core.runner image_resizing_v2 --source 0 --import-all
```

**Programmatic usage:**

```python
from core.runner import run_camera
run_camera("image_resizing_v2", source=0)
```

## Processing Pipeline

- **Load**: No-op (no model to load).
- **Predict**: Downscales to 25%, then upscales back using 4 interpolation methods: NEAREST, LINEAR, CUBIC, LANCZOS4.
- **Visualize**: 2×2 grid: Nearest, Linear, Cubic, Lanczos4 — each labeled.

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
