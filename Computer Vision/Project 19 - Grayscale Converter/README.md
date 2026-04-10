# P19: Grayscale Converter (v2)

![OpenCV](https://img.shields.io/badge/Framework-OpenCV-blue) ![Utility](https://img.shields.io/badge/Task-Utility-green)

## Overview

Converts live camera frames to grayscale using OpenCV color space conversion.

## Entry Points

### Run / Inference

**Via the unified runner:**

```bash
python -m core.runner grayscale_converter_v2 --source 0 --import-all
```

**Programmatic usage:**

```python
from core.runner import run_camera
run_camera("grayscale_converter_v2", source=0)
```

## Processing Pipeline

- **Load**: No-op (no model to load).
- **Predict**: Converts frame to grayscale via `cv2.cvtColor(BGR2GRAY)`.
- **Visualize**: Converts grayscale back to BGR for display, adds text label.

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
