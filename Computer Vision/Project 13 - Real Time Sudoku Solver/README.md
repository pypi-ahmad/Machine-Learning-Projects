# P13: Sudoku Solver (YOLO-enhanced)

![Ultralytics YOLO](https://img.shields.io/badge/Framework-Ultralytics_YOLO-blue) ![Detection](https://img.shields.io/badge/Task-Detection-green) ![Trainable](https://img.shields.io/badge/Trainable-yes-orange)

## Overview

Runs YOLO26 detection to locate potential sudoku grids (book/cell_phone objects). Does not solve puzzles — digit OCR requires a custom-trained model. Separate `train.py` trains a ResNet-18 digit classifier on MNIST via torchvision.

## Entry Points

### Run / Inference

**Via the unified runner:**

```bash
python -m core.runner sudoku_solver_v2 --source 0 --import-all
```

**Programmatic usage:**

```python
from core.runner import run_camera
run_camera("sudoku_solver_v2", source=0)
```

### Train

```bash
cd "Project 13 - Real Time Sudoku Solver"
python train.py --data path/to/data.yaml
```

Training registers the resulting model version in the model registry
(`models/metadata.json`) and auto-promotes based on the primary metric.

## Model Resolution

Resolves weights via `models.registry.resolve("sudoku_solver", "detect")`, falls back to `yolo26n.pt`.

The model registry (`models/registry.py`) resolves weights in this order:

1. **Trained model** — `models/<project>/<version>/best.pt` if registered and file exists
2. **YOLO26 pretrained fallback** — auto-downloaded by Ultralytics on first use

## Dataset

Configuration: `configs/datasets/sudoku_solver.yaml`

Download method: **http** (auto-download enabled)

```bash
python -m utils.data_downloader --project sudoku_solver
```

Expected layout after download:

```
data/sudoku_solver/
  data.yaml
  train/images/
  valid/images/
```

## Processing Pipeline

- **Load**: See [Model Resolution](#model-resolution) above.
- **Predict**: Runs YOLO detection on frame (conf=0.4). Highlights `book`/`cell_phone` detections as potential sudoku grids.
- **Visualize**: Draws bounding boxes; grids (book/cell_phone) in green, other objects in orange. Shows status text: digit OCR requires a custom model.

## Outputs

- OpenCV display window showing annotated frames in real-time
- Training: `runs/detect/train/weights/best.pt` (registered in model registry)
- Press `q` to quit the camera loop

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `Cannot open video source` | Check webcam index — try `--source 1` |
| Model downloads on first run | Normal — Ultralytics auto-downloads YOLO26 pretrained weights |
| Dataset not found | Run `python -m utils.data_downloader --project sudoku_solver` |
| Training OOM | Reduce `--batch` or use a smaller model (`yolo26n.pt`) |

## Testing

No project-level test suite. Use workspace-level smoke tests:

```bash
python scripts/smoke_3b3.py
```
