# P32: Coin Lines (v2)

![OpenCV](https://img.shields.io/badge/Framework-OpenCV-blue) ![Utility](https://img.shields.io/badge/Task-Utility-green)

## Overview

Detects coins in images using Hough Circle Transform and draws vertical reference lines through their centers.

## Entry Points

### Run / Inference

**Via the unified runner:**

```bash
python -m core.runner coin_lines_v2 --source 0 --import-all
```

**Programmatic usage:**

```python
from core.runner import run_camera
run_camera("coin_lines_v2", source=0)
```

## Processing Pipeline

- **Load**: No-op (no model to load).
- **Predict**: Grayscale + median blur, detects circles via HoughCircles (dp=1.2, minDist=50).
- **Visualize**: Draws detected circles in green, center dots in red, vertical blue lines through each coin center, shows coin count.

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
