# P20: Image Finder (v2)

![OpenCV](https://img.shields.io/badge/Framework-OpenCV-blue) ![Utility](https://img.shields.io/badge/Task-Utility-green)

## Overview

Finds occurrences of a template image within a larger image or camera frame using OpenCV template matching.

## Entry Points

### Run / Inference

**Via the unified runner:**

```bash
python -m core.runner image_finder_v2 --source 0 --import-all
```

**Programmatic usage:**

```python
from core.runner import run_camera
run_camera("image_finder_v2", source=0)
```

## Processing Pipeline

- **Load**: Initializes `self.template = None` (must call `set_template()` before use).
- **Predict**: Runs `cv2.matchTemplate` (TM_CCOEFF_NORMED) between frame and template, returns match locations above threshold 0.8.
- **Visualize**: Draws green rectangles at each match location, shows match count.

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
