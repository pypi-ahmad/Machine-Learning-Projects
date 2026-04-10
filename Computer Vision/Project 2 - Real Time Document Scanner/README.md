# P2: Document Scanner (v2)

![OpenCV](https://img.shields.io/badge/Framework-OpenCV-blue) ![Utility](https://img.shields.io/badge/Task-Utility-green)

## Overview

Scans documents in real-time by detecting rectangular contours and applying a 4-point perspective warp to produce a top-down view.

## Entry Points

### Run / Inference

**Via the unified runner:**

```bash
python -m core.runner doc_scanner_v2 --source 0 --import-all
```

**Programmatic usage:**

```python
from core.runner import run_camera
run_camera("doc_scanner_v2", source=0)
```

## Processing Pipeline

- **Load**: No-op (no model to load).
- **Predict**: Grayscale + blur + Canny edges, finds contours, searches for 4-point quadrilateral (document contour).
- **Visualize**: Draws detected document contour, applies 4-point perspective warp, shows warped thumbnail inset on frame.

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
