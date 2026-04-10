# P35: Image Sharpening (v2)

![OpenCV](https://img.shields.io/badge/Framework-OpenCV-blue) ![Utility](https://img.shields.io/badge/Task-Utility-green)

## Overview

Compares three image sharpening techniques: Laplacian kernel, edge enhancement kernel, and unsharp masking.

## Entry Points

### Run / Inference

**Via the unified runner:**

```bash
python -m core.runner image_sharpening_v2 --source 0 --import-all
```

**Programmatic usage:**

```python
from core.runner import run_camera
run_camera("image_sharpening_v2", source=0)
```

## Processing Pipeline

- **Load**: Creates Laplacian sharpen kernel and edge enhance kernel.
- **Predict**: Applies 3 sharpening methods: Laplacian kernel, edge-enhance kernel, unsharp mask (`addWeighted` with Gaussian blur).
- **Visualize**: 2×2 grid: Original, Laplacian Sharp, Edge Enhance, Unsharp Mask — each labeled.

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
