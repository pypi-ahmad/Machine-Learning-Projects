# P22: Color Picker (v2)

![OpenCV](https://img.shields.io/badge/Framework-OpenCV-blue) ![Utility](https://img.shields.io/badge/Task-Utility-green)

## Overview

Picks and displays the BGR and HSV color values from the center of a live camera feed.

## Entry Points

### Run / Inference

**Via the unified runner:**

```bash
python -m core.runner color_picker_v2 --source 0 --import-all
```

**Programmatic usage:**

```python
from core.runner import run_camera
run_camera("color_picker_v2", source=0)
```

## Processing Pipeline

- **Load**: Initializes `self.last_color = (0, 0, 0)`.
- **Predict**: Samples center 10×10 pixel ROI, computes average BGR and HSV values.
- **Visualize**: Draws crosshair at center, renders colored info panel with BGR/HSV values.

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
