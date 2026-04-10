# P40: Non-Photorealistic Rendering (v2)

![OpenCV](https://img.shields.io/badge/Framework-OpenCV-blue) ![Utility](https://img.shields.io/badge/Task-Utility-green)

## Overview

Demonstrates OpenCV non-photorealistic rendering effects: stylization, edge-preserving filter, and detail enhancement.

## Entry Points

### Run / Inference

**Via the unified runner:**

```bash
python -m core.runner npr_rendering_v2 --source 0 --import-all
```

**Programmatic usage:**

```python
from core.runner import run_camera
run_camera("npr_rendering_v2", source=0)
```

## Processing Pipeline

- **Load**: No-op (no model to load).
- **Predict**: Applies 3 NPR effects: `cv2.stylization`, `cv2.edgePreservingFilter`, `cv2.detailEnhance`.
- **Visualize**: 2×2 grid: Original, Stylized, Edge Preserved, Detail Enhanced — each labeled.

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
