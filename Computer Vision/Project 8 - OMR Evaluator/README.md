# P8: OMR Evaluator (v2)

![OpenCV](https://img.shields.io/badge/Framework-OpenCV-blue) ![Utility](https://img.shields.io/badge/Task-Utility-green)

## Overview

Evaluates Optical Mark Recognition (OMR) sheets by detecting filled bubbles via contour analysis and fill-ratio thresholding.

## Entry Points

### Run / Inference

**Via the unified runner:**

```bash
python -m core.runner omr_evaluator_v2 --source 0 --import-all
```

**Programmatic usage:**

```python
from core.runner import run_camera
run_camera("omr_evaluator_v2", source=0)
```

## Processing Pipeline

- **Load**: No-op (no model to load).
- **Predict**: Edge detection + contour-based document detection, adaptive thresholding for bubble detection, computes fill ratio for each circular bubble contour.
- **Visualize**: Draws document contour, highlights filled bubbles (fill_ratio > 0.5) with red rectangles, shows filled-bubble count.

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
