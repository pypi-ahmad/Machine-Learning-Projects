# Number Plate Reader Pro

> **Task:** Detection + OCR &nbsp;|&nbsp; **Key:** `number_plate_reader_pro` &nbsp;|&nbsp; **Framework:** YOLO26m + PaddleOCR-first OCR

---

## Overview

Full ALPR pipeline: detects license plates with YOLO26m, crops and rectifies plate regions, reads text with PaddleOCR first, applies regex cleanup, and deduplicates reads across video frames. On local runtimes where PaddleOCR fails during inference, the OCR layer falls back to EasyOCR so the pipeline stays usable. Supports image, video, and live webcam inference with structured JSON/CSV export.

## Technology

| Aspect | Details |
|--------|---------|
| **Task Type** | Detection + OCR |
| **Modern Stack** | YOLO26m detection + PaddleOCR-first recognition |
| **Dataset** | Public license plate detection dataset when available, otherwise synthetic YOLO-format fallback |
| **Key Metrics** | mAP50, plate OCR accuracy |

## Pipeline

```
Frame → YOLO26m detect → crop + rectify → PaddleOCR → regex cleanup → dedup → export
```

1. **Detection** — YOLO26m localises plates in the frame with configurable confidence/IoU thresholds.
2. **Rectification** — Small crops are upscaled; adaptive thresholding sharpens character edges.
3. **OCR** — PaddleOCR reads text from the rectified plate crop; EasyOCR is used only as a runtime fallback on unsupported local setups.
4. **Cleanup** — Regex strips invalid characters, collapses whitespace, uppercases, and applies conservative OCR character corrections.
5. **Dedup** — Cooldown-based tracker suppresses the same plate within a configurable window.
6. **Validation** — Quality checks: confidence floor, pattern match, minimum length.
7. **Export** — JSON (full event records) and/or CSV (one row per plate read).

## Dataset

- **Primary source:** Roboflow Universe — `license-plate-recognition-rxg4e` (v4, YOLOv8 format) when the downloader environment is available.
- **License:** See the upstream dataset page for current licence terms.
- **Fallback:** If the public downloader is unavailable, bootstrap generates a synthetic YOLO-format plate dataset with `ocr_labels.json` sidecar metadata.
- **Download:** Automatic on first `python train.py` run.
- **Force re-download:** `python train.py --force-download`
- **Bootstrap:** `python data_bootstrap.py` (idempotent, writes `.ready`, `data.yaml`, `dataset_info.json`, and OCR label metadata)

## Project Structure

```
Number Plate Reader Pro/
└── Source Code/
    ├── config.py            # PlateConfig dataclass — all tunables
    ├── plate_detector.py    # YOLO26m plate detection + rectification
    ├── ocr_engine.py        # PaddleOCR-first wrapper with lazy init
    ├── plate_cleaner.py     # Regex cleanup + OCR error corrections
    ├── tracker.py           # Cooldown-based duplicate suppression
    ├── parser.py            # PlateReaderPipeline — orchestrates full pipeline
    ├── validator.py         # Quality checks + ValidationReport
    ├── visualize.py         # Annotated overlay renderer
    ├── export.py            # JSON / CSV exporter (context manager)
    ├── infer.py             # CLI — image / video / webcam inference
    ├── modern.py            # CVProject subclass — @register("number_plate_reader_pro")
    ├── train.py             # CLI training entry point
    ├── data_bootstrap.py    # Idempotent dataset download + synthetic fallback
    ├── plate_config.yaml    # Sample YAML configuration
    ├── requirements.txt     # Project dependencies
    └── README.md            # This file
```

## Quick Start

### CLI Inference

```bash
# Single image
python infer.py --source car.jpg

# Directory of images
python infer.py --source images/ --export-json results.json

# Video file with CSV export
python infer.py --source traffic.mp4 --export-csv plates.csv --save-crops

# Live webcam
python infer.py --source 0

# Headless mode with all exports
python infer.py --source highway.mp4 --no-display --export-json out.json --export-csv out.csv --save-annotated

# Override detector weights explicitly
python infer.py --source car.jpg --model runs/plate_detect/weights/best.pt
```

### Registry API

```python
from core import discover_projects, run

discover_projects()
result = run("number_plate_reader_pro", "path/to/car.jpg")
for read in result["result"].reads:
    print(f"Plate: {read.plate_text} (OCR: {read.ocr_confidence:.2f})")
```

### Training

```bash
cd "Number Plate Reader Pro/Source Code"
python train.py --epochs 80 --batch 16
python train.py --force-download  # re-download dataset
```

## CLI Arguments

| Flag | Description |
|------|-------------|
| `--source` | Image path, directory, video file, or webcam index |
| `--config` | Path to YAML/JSON config file |
| `--model` | Optional detector weights override |
| `--gpu` | Enable GPU for PaddleOCR |
| `--confidence` | Override detection confidence threshold |
| `--export-json` | JSON export path |
| `--export-csv` | CSV export path |
| `--save-annotated` | Save annotated images/frames |
| `--save-crops` | Save detected plate crops |
| `--output-dir` | Output directory (default: `output`) |
| `--no-display` | Disable GUI window |
| `--force-download` | Force dataset re-download |

## Configuration

All tunables are defined in `PlateConfig` (see [config.py](config.py)). Override via:

1. **YAML config file:** `python infer.py --config plate_config.yaml --source img.jpg`
2. **CLI flags:** `--gpu`, `--confidence 0.5`, `--no-display`, etc.
3. **Python:** Instantiate `PlateConfig(det_confidence=0.5, rectify=False)`.

## Features

- YOLO26m plate detection with configurable confidence threshold
- Plate crop rectification (upscale + adaptive threshold)
- PaddleOCR-first text recognition with lazy initialisation and runtime fallback
- Regex text cleanup with OCR error correction table
- Cooldown-based duplicate suppression across frames
- Quality validation with configurable rules
- Annotated overlay with colour-coded boxes (green=new, grey=dup, red=invalid)
- JSON and CSV structured export
- Image, video, and live webcam support
- Sample YAML configuration file
- Idempotent dataset bootstrap with `.ready`, `data.yaml`, `dataset_info.json`, and `ocr_labels.json`

## Dependencies

```bash
pip install -r requirements.txt
```

## Runtime Notes

- On this Windows environment, PaddleOCR can initialise successfully but fail during inference with a oneDNN runtime error.
- The project therefore keeps PaddleOCR as the primary OCR engine and automatically falls back to EasyOCR only when that runtime failure occurs.
