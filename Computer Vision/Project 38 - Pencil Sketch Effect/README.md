# P38: Pencil Sketch (v2)

![OpenCV](https://img.shields.io/badge/Framework-OpenCV-blue) ![Utility](https://img.shields.io/badge/Task-Utility-green)

## Overview

Applies pencil sketch effects (grayscale and color) using OpenCV's built-in `pencilSketch()` function.

## Entry Points

### Run / Inference

**Via the unified runner:**

```bash
python -m core.runner pencil_sketch_v2 --source 0 --import-all
```

**Programmatic usage:**

```python
from core.runner import run_camera
run_camera("pencil_sketch_v2", source=0)
```

## Processing Pipeline

- **Load**: No-op (no model to load).
- **Predict**: Calls `cv2.pencilSketch()` to produce gray and color pencil sketch variants.
- **Visualize**: 3-image horizontal strip: Original, Pencil B&W, Pencil Color — each labeled.

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
