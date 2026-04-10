# P28: Watermarking (v2)

![OpenCV](https://img.shields.io/badge/Framework-OpenCV-blue) ![Utility](https://img.shields.io/badge/Task-Utility-green)

## Overview

Applies a semi-transparent text watermark to images using OpenCV alpha blending.

## Entry Points

### Run / Inference

**Via the unified runner:**

```bash
python -m core.runner watermarking_v2 --source 0 --import-all
```

**Programmatic usage:**

```python
from core.runner import run_camera
run_camera("watermarking_v2", source=0)
```

## Processing Pipeline

- **Load**: No-op (no model to load).
- **Predict**: Creates text watermark overlay ("CV Projects v2") at bottom-right, blends via `addWeighted` (alpha=0.7).
- **Visualize**: Returns the watermarked image directly.

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
