# P44: Image Joining (v2)

![OpenCV](https://img.shields.io/badge/Framework-OpenCV-blue) ![Utility](https://img.shields.io/badge/Task-Utility-green)

## Overview

Demonstrates how to join multiple processed images into a single display grid using OpenCV.

## Entry Points

### Run / Inference

**Via the unified runner:**

```bash
python -m core.runner image_joining_v2 --source 0 --import-all
```

**Programmatic usage:**

```python
from core.runner import run_camera
run_camera("image_joining_v2", source=0)
```

## Processing Pipeline

- **Load**: No-op (no model to load).
- **Predict**: Creates 4 variants of frame: original, grayscale, Canny edges, Gaussian blur.
- **Visualize**: 2×2 grid: Original, Gray, Edges, Blurred — each labeled.

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
