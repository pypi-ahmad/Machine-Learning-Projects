# P50: Video Reverse (v2)

![OpenCV](https://img.shields.io/badge/Framework-OpenCV-blue) ![Utility](https://img.shields.io/badge/Task-Utility-green)

## Overview

Records short video segments and plays them back in reverse in real-time using a frame buffer.

## Entry Points

### Run / Inference

**Via the unified runner:**

```bash
python -m core.runner video_reverse_v2 --source 0 --import-all
```

**Programmatic usage:**

```python
from core.runner import run_camera
run_camera("video_reverse_v2", source=0)
```

## Processing Pipeline

- **Load**: Initializes frame buffer `deque` (max 60 frames), sets recording mode.
- **Predict**: Records frames into buffer until full (60), then plays back in reverse order, cycles back to recording.
- **Visualize**: Shows "Recording..." with red dot during recording, shows reverse playback frame during playback.

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
