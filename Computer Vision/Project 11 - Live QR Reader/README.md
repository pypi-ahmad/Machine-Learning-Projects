# P11: QR Reader (v2)

![OpenCV](https://img.shields.io/badge/Framework-OpenCV-blue) ![Utility](https://img.shields.io/badge/Task-Utility-green)

## Overview

Reads QR codes in real-time from a camera feed using OpenCV's built-in QR code detector.

## Entry Points

### Run / Inference

**Via the unified runner:**

```bash
python -m core.runner qr_reader_v2 --source 0 --import-all
```

**Programmatic usage:**

```python
from core.runner import run_camera
run_camera("qr_reader_v2", source=0)
```

## Processing Pipeline

- **Load**: Creates a `cv2.QRCodeDetector()` instance.
- **Predict**: Calls `detectAndDecode()` on frame to find and decode QR codes.
- **Visualize**: Draws green lines around QR bounding box, shows decoded QR data text.

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
