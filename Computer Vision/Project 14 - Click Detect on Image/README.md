# P14: Warp Perspective (v2)

![OpenCV](https://img.shields.io/badge/Framework-OpenCV-blue) ![Utility](https://img.shields.io/badge/Task-Utility-green)

## Overview

Detects and warps quadrilateral regions in images using OpenCV contour detection and perspective transforms.

## Entry Points

### Run / Inference

**Via the unified runner:**

```bash
python -m core.runner warp_perspective_v2 --source 0 --import-all
```

**Programmatic usage:**

```python
from core.runner import run_camera
run_camera("warp_perspective_v2", source=0)
```

## Processing Pipeline

- **Load**: No-op (no model to load).
- **Predict**: Grayscale + blur + Canny edges, finds largest 4-sided contour (quadrilateral), applies 4-point perspective transform.
- **Visualize**: Draws detected quadrilateral contour in green, shows status text.

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
