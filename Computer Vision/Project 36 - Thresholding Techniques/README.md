# P36: Thresholding Techniques (v2)

![OpenCV](https://img.shields.io/badge/Framework-OpenCV-blue) ![Utility](https://img.shields.io/badge/Task-Utility-green)

## Overview

Demonstrates and compares four OpenCV thresholding techniques on live camera frames.

## Entry Points

### Run / Inference

**Via the unified runner:**

```bash
python -m core.runner thresholding_v2 --source 0 --import-all
```

**Programmatic usage:**

```python
from core.runner import run_camera
run_camera("thresholding_v2", source=0)
```

## Processing Pipeline

- **Load**: No-op (no model to load).
- **Predict**: Applies 4 thresholding methods: Binary, Binary Inverse, Otsu, Adaptive Gaussian.
- **Visualize**: 2×2 grid: Binary, Binary Inv, Otsu, Adaptive — each labeled.

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
