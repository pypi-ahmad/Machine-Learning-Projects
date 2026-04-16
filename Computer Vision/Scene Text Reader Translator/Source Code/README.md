# Scene Text Reader Translator

> **Task:** OCR + Translation Hook &nbsp;|&nbsp; **Key:** `scene_text_reader_translator` &nbsp;|&nbsp; **Framework:** PaddleOCR-first OCR

---

## Overview

Scene-text OCR pipeline that detects and recognises text from natural scene images (signs, billboards, menus, etc.) using PaddleOCR first, renders readable overlays, and exports structured JSON/CSV results with coordinates. The translation layer is an optional hook/stub and remains cleanly separated from OCR. On local runtimes where PaddleOCR fails during inference, the OCR layer falls back to EasyOCR so the pipeline stays usable.

## Pipeline

```
Scene image → PaddleOCR detect + recognise → (optional) translate → validate → overlay → export
```

1. **Detection** — PaddleOCR text detection locates text regions in the scene.
2. **Recognition** — PaddleOCR reads text from each detected region with confidence scores.
3. **Translation** *(optional)* — Pluggable translation hook (`TranslationProvider` ABC). No provider is bundled by default.
4. **Validation** — Quality checks: confidence floor, minimum text length.
5. **Overlay** — Colour-coded bounding polygons, text labels, confidence, translated text, and summary panel.
6. **Export** — JSON (full block records with coordinates) and/or CSV (one row per text region).

## Technology

| Aspect | Details |
|--------|---------|
| **Task Type** | Scene Text OCR + Translation |
| **Modern Stack** | PaddleOCR-first OCR (detection + recognition) |
| **Translation** | Optional — pluggable provider hook |
| **Dataset** | Public scene-text images when available, otherwise synthetic scene-text fallback |
| **Key Metrics** | OCR accuracy, detection coverage |

## Dataset

- **Primary source:** Public scene-text dataset via the repo downloader when available.
- **License:** See the upstream dataset page for current licence terms.
- **Fallback:** If the public download path is unavailable, bootstrap generates synthetic scene-text images plus `ocr_labels.json` metadata.
- **Download:** Automatic on first `python train.py` run.
- **Force re-download:** `python train.py --force-download`
- **Bootstrap:** `python data_bootstrap.py` (idempotent, writes `.ready`, `dataset_info.json`, and `ocr_labels.json`)

## Project Structure

```
Scene Text Reader Translator/
└── Source Code/
    ├── config.py                # SceneTextConfig dataclass — all tunables
    ├── ocr_engine.py            # PaddleOCR-first wrapper → OCRBlock dataclass
    ├── translator.py            # Translation hook — TranslationProvider ABC
    ├── parser.py                # SceneTextPipeline — OCR → translate → result
    ├── validator.py             # Quality checks + ValidationReport
    ├── visualize.py             # Annotated overlay renderer
    ├── export.py                # JSON / CSV exporter (context manager)
    ├── infer.py                 # CLI — image / video / webcam inference
    ├── modern.py                # CVProject subclass — @register("scene_text_reader_translator")
    ├── train.py                 # Evaluation entry point (PaddleOCR is pre-trained)
    ├── data_bootstrap.py        # Idempotent dataset download + synthetic fallback
    ├── scene_text_config.yaml   # Sample YAML configuration
    ├── requirements.txt         # Project dependencies
    └── README.md                # This file
```

## Quick Start

### CLI Inference

```bash
# Single image
python infer.py --source street_sign.jpg

# Directory of images with JSON export
python infer.py --source images/ --export-json results.json

# Video file with CSV export
python infer.py --source city_drive.mp4 --export-csv texts.csv

# Live webcam
python infer.py --source 0

# With translation enabled
python infer.py --source sign.jpg --translate --target-lang es

# Headless mode with all exports
python infer.py --source video.mp4 --no-display --export-json out.json --export-csv out.csv --save-annotated
```

### Registry API

```python
from core import discover_projects, run

discover_projects()
result = run("scene_text_reader_translator", "path/to/scene.jpg")
for read in result["result"].reads:
    print(f"Text: {read.text} (conf: {read.confidence:.2f})")
```

### Evaluation

```bash
cd "Scene Text Reader Translator/Source Code"
python train.py                    # evaluate on dataset
python train.py --force-download   # re-download dataset
python train.py --max-samples 50   # evaluate more samples
```

## CLI Arguments

| Flag | Description |
|------|-------------|
| `--source` | Image path, directory, video file, or webcam index |
| `--config` | Path to YAML/JSON config file |
| `--lang` | OCR language (default: `en`) |
| `--gpu` | Enable GPU for PaddleOCR |
| `--translate` | Enable the translation hook |
| `--target-lang` | Translation target language (e.g. `es`, `fr`, `de`) |
| `--translate-provider` | Optional provider label for a custom hook |
| `--export-json` | JSON export path |
| `--export-csv` | CSV export path |
| `--save-annotated` | Save annotated images/frames |
| `--output-dir` | Output directory (default: `output`) |
| `--no-display` | Disable GUI window |
| `--force-download` | Force dataset re-download |

## Translation

Translation is **cleanly separated** from OCR. The `translator.py` module provides:

- `TranslationProvider` — Abstract base class for any translation backend
- `NoOpProvider` — Default passthrough when translation is disabled
- `Translator` — Facade that selects provider based on config

### Adding a custom provider

```python
from translator import TranslationProvider, Translator

class MyProvider(TranslationProvider):
    def translate(self, text: str, target_lang: str) -> str:
        # Your translation logic here
        return translated_text

    def name(self) -> str:
        return "my_provider"

# Use it
from config import SceneTextConfig
cfg = SceneTextConfig(translate_enabled=True)
translator = Translator(cfg, provider=MyProvider())
```

No provider ships enabled by default. The project exposes a hook/stub only; real provider integrations should be added explicitly by the user or by a future scoped extension.

## Configuration

All tunables are defined in `SceneTextConfig` (see [config.py](config.py)). Override via:

1. **YAML config file:** `python infer.py --config scene_text_config.yaml --source img.jpg`
2. **CLI flags:** `--gpu`, `--lang ch`, `--translate`, `--target-lang fr`, etc.
3. **Python:** `SceneTextConfig(ocr_lang="ch", translate_enabled=True)`.

## Features

- PaddleOCR-first scene text detection and recognition with runtime fallback
- Optional translation hook with pluggable provider architecture
- Colour-coded overlay with bounding polygons, text labels, and confidence
- Translated text rendered below original in overlay
- Summary panel with block count, confidence, and text preview
- JSON and CSV structured export with coordinates
- Image, video, and live webcam support
- Configurable confidence threshold and validation rules
- Sample YAML configuration file
- Idempotent dataset bootstrap with `.ready`, `dataset_info.json`, and `ocr_labels.json`

## Dependencies

```bash
pip install -r requirements.txt
```

## Runtime Notes

- On this Windows environment, PaddleOCR can initialise successfully but fail during inference with a oneDNN runtime error.
- The project keeps PaddleOCR as the primary OCR engine and automatically falls back to EasyOCR only when that runtime failure occurs.
