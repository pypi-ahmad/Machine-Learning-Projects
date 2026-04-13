# Handwriting Recognition

> **Task:** OCR &nbsp;|&nbsp; **Key:** `handwriting_recognition` &nbsp;|&nbsp; **Framework:** TrOCR / PaddleOCR

---

## Overview

Recognizes handwritten text from images. The modern pipeline is OCR-first: TrOCR is the preferred recognizer for handwritten lines or words, with PaddleOCR as a broader fallback stack when TrOCR is unavailable.

## Technology

| Aspect | Details |
|--------|---------|
| **Task Type** | OCR (`ocr`) |
| **Legacy Stack** | Custom TF/Keras HTR CNN+RNN |
| **Modern Stack** | TrOCR (`microsoft/trocr-base-handwritten`) with PaddleOCR fallback |
| **Dataset** | IAM Handwriting Database (manual registration/download) |
| **Key Metrics** | CER, WER, qualitative OCR review |
| **Download** | manual_page (enabled: no) |

## Project Structure

```
Handwriting Recognition/
└── Source Code/
    ├── modern.py        # CVProject subclass — @register("handwriting_recognition")
    ├── train.py         # CLI training entry point
    └── ...              # Legacy code, notebooks, original assets
```

## Quick Start

### Inference

```python
from core import discover_projects, run

discover_projects()
result = run("handwriting_recognition", "path/to/image.jpg")
```

**How it works:** `modern.py` loads TrOCR first and generates text autoregressively from the image. If the HuggingFace TrOCR stack is not installed, the project falls back to PaddleOCR and returns recognized lines plus bounding boxes for visualization.

### Training

The packaged `train.py` script is retained only as a **legacy closed-set character-classification baseline**. It is not the recommended path for line/word handwriting OCR.

If you want to study or compare against that baseline:

```bash
cd "Handwriting Recognition/Source Code"
python train.py --epochs 25 --model resnet18
```

For the modern OCR pipeline, use `modern.py` for inference and treat TrOCR / PaddleOCR as the primary method.

### Dataset

Dataset registration is required. Download the IAM Handwriting Database from:

- [IAM Handwriting Database](http://www.fki.inf.unibe.ch/databases/iam-handwriting-database)

Then extract it into `data/handwriting_recognition/`.

```bash
python -m utils.data_downloader handwriting_recognition       # download (if enabled)
python scripts/validate_datasets.py          # status report
```

### Evaluation Notes

For OCR work, prefer character error rate (CER), word error rate (WER), and qualitative review of failure cases instead of plain image-classification accuracy.

## Links

- [Root README](../../README.md)
- [Project Inventory](../../reports/PROJECT_INVENTORY.md)
- [Dataset Resolver Entry](../../utils/datasets.py)
