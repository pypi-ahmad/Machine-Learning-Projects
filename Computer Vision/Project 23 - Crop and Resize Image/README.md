# P23: Crop & Resize (v2)

![OpenCV](https://img.shields.io/badge/Framework-OpenCV-blue) ![Utility](https://img.shields.io/badge/Task-Utility-green)

## Overview

Demonstrates center-cropping and resizing of images using OpenCV with various interpolation modes.

## Entry Points

### Run / Inference

**Via the unified runner:**

```bash
python -m core.runner crop_resize_v2 --source 0 --import-all
```

**Programmatic usage:**

```python
from core.runner import run_camera
run_camera("crop_resize_v2", source=0)
```

## Processing Pipeline

- **Load**: No-op (no model to load).
- **Predict**: Center-crops image to 50%, then resizes by SCALE_FACTOR (0.5) using INTER_AREA.
- **Visualize**: Draws green rectangle showing crop region, shows original and crop dimensions text.

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
