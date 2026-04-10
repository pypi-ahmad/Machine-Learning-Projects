# P34: Motion Blur (v2)

![OpenCV](https://img.shields.io/badge/Framework-OpenCV-blue) ![Utility](https://img.shields.io/badge/Task-Utility-green)

## Overview

Applies directional (horizontal and vertical) motion blur effects using custom convolution kernels.

## Entry Points

### Run / Inference

**Via the unified runner:**

```bash
python -m core.runner motion_blur_v2 --source 0 --import-all
```

**Programmatic usage:**

```python
from core.runner import run_camera
run_camera("motion_blur_v2", source=0)
```

## Processing Pipeline

- **Load**: Creates horizontal and vertical motion blur kernels (size 30).
- **Predict**: Applies horizontal and vertical motion blur via `cv2.filter2D` with custom kernels.
- **Visualize**: 3-image horizontal strip: Original, H-Motion, V-Motion — each labeled.

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
