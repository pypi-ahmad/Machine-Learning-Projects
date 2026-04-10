# P39: Noise Removal (v2)

![OpenCV](https://img.shields.io/badge/Framework-OpenCV-blue) ![Utility](https://img.shields.io/badge/Task-Utility-green)

## Overview

Compares three noise removal techniques: Non-Local Means, Gaussian, and Median denoising. (Note: folder is named 'Pencil drawing effect' but contains noise removal code.)

## Entry Points

### Run / Inference

**Via the unified runner:**

```bash
python -m core.runner noise_removal_v2 --source 0 --import-all
```

**Programmatic usage:**

```python
from core.runner import run_camera
run_camera("noise_removal_v2", source=0)
```

## Processing Pipeline

- **Load**: No-op (no model to load).
- **Predict**: Applies 3 denoising methods: `fastNlMeansDenoisingColored`, GaussianBlur, medianBlur.
- **Visualize**: 2×2 grid: Original, NLMeans, Gaussian, Median — each labeled.

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
