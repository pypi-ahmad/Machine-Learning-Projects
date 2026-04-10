# Document Layout Block Detector

Detect and classify structural blocks in scanned documents and PDF pages — titles, paragraphs, tables, figures, stamps, headers, footers, lists, and captions.

---

## Motivation

Document layout analysis is the first step in any document understanding pipeline.  Before OCR can extract text, before tables can be parsed, before figures can be captioned — the system must know *where each region is* and *what type it is*.

This project provides that foundational layer: a YOLO-based detector that segments a page into typed blocks with bounding-box coordinates, ready for downstream OCR, table extraction, or content reflow.

**OCR is intentionally out of scope for v1** — the JSON export format includes coordinates and class labels designed to plug directly into Tesseract, PaddleOCR, or any OCR engine as a future extension.

---

## Features

| Feature | Description |
|---|---|
| **Block detection** | Titles, text, tables, figures, lists, captions, headers, footers, stamps |
| **PDF support** | Automatic PDF-to-image conversion (PyMuPDF or pdf2image) |
| **JSON export** | Block coordinates + classes for downstream pipelines |
| **Region cropping** | Save each detected block as a separate image |
| **Reading-order sort** | Blocks sorted top-to-bottom, left-to-right |
| **Dashboard overlay** | On-screen block counts with class-coloured boxes |
| **Directory batch** | Process an entire folder of page images |
| **Auto dataset download** | One-command training data bootstrap |

---

## Architecture

```
Document Layout Block Detector/
└── Source Code/
    ├── config.py           # LayoutConfig dataclass + YAML loader
    ├── data_bootstrap.py   # Dataset download & preparation
    ├── detector.py         # YOLO detection + LayoutBlock/PageResult dataclasses
    ├── pdf_utils.py        # PDF-to-image conversion (PyMuPDF / pdf2image)
    ├── visualize.py        # Block renderer + dashboard
    ├── export.py           # JSON export + optional crop saving
    ├── infer.py            # CLI inference (image / PDF / directory)
    ├── train.py            # Training (delegates to train/train_detection.py)
    ├── evaluate.py         # Evaluation with per-class mAP
    ├── modern.py           # Registry entry (@register)
    ├── layout_config.yaml  # Sample configuration
    └── requirements.txt    # Python dependencies
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r "Document Layout Block Detector/Source Code/requirements.txt"
```

### 2. Run inference

```bash
# Single image
python "Document Layout Block Detector/Source Code/infer.py" \
    --source scanned_page.jpg

# PDF document (all pages)
python "Document Layout Block Detector/Source Code/infer.py" \
    --source report.pdf \
    --config "Document Layout Block Detector/Source Code/layout_config.yaml"

# Directory of page images
python "Document Layout Block Detector/Source Code/infer.py" \
    --source pages/ \
    --export-json results.json

# Headless with crop saving
python "Document Layout Block Detector/Source Code/infer.py" \
    --source document.pdf \
    --no-display \
    --export-json layout.json \
    --save-crops \
    --save-annotated
```

### 3. Train a custom model

```bash
python "Document Layout Block Detector/Source Code/train.py" \
    --epochs 100 --batch 8 --imgsz 1024
```

### 4. Evaluate

```bash
python "Document Layout Block Detector/Source Code/evaluate.py" \
    --model runs/doc_layout/train/weights/best.pt
```

---

## Configuration

Edit `layout_config.yaml` to customise:

| Key | Type | Default | Description |
|---|---|---|---|
| `model` | `str` | `yolo26m.pt` | YOLO model weights |
| `conf_threshold` | `float` | `0.30` | Minimum detection confidence |
| `imgsz` | `int` | `1024` | Input size (higher = better for dense documents) |
| `target_classes` | `list[str]` | `[]` | Class filter (empty = all) |
| `export_json` | `str` | `""` | JSON output path |
| `save_crops` | `bool` | `false` | Save cropped block images |
| `crops_dir` | `str` | `output/crops` | Crop output directory |
| `pdf_dpi` | `int` | `300` | PDF rasterisation resolution |
| `save_annotated` | `bool` | `false` | Save annotated page images |

---

## JSON Export Format

```json
{
  "source": "page_001.png",
  "page_idx": 0,
  "image_height": 3300,
  "image_width": 2550,
  "total_blocks": 12,
  "class_counts": {"title": 1, "text": 6, "table": 2, "figure": 1, "caption": 2},
  "blocks": [
    {
      "block_id": 0,
      "class": "title",
      "confidence": 0.95,
      "bbox": [120, 80, 2400, 180],
      "area": 228000
    }
  ]
}
```

The `blocks` list is sorted in reading order (top-to-bottom, left-to-right).  Each block's `bbox` is `[x1, y1, x2, y2]` in pixel coordinates — ready for cropping or passing to an OCR engine.

---

## PDF Support

PDFs are automatically converted to images before detection.  Two backends are supported:

| Backend | Install | Notes |
|---|---|---|
| **PyMuPDF** | `pip install PyMuPDF` | Fast, no system deps (recommended) |
| **pdf2image** | `pip install pdf2image` + [Poppler](https://poppler.freedesktop.org/) | Requires system Poppler install |

The converter is auto-detected at runtime.  Set `pdf_dpi` in the config to control resolution (300 DPI is standard for OCR-quality output).

---

## Future Extensions

This project is designed as the **layout analysis** layer in a document understanding stack.  Planned extensions:

- **OCR integration**: pipe cropped text blocks to Tesseract / PaddleOCR
- **Table structure recognition**: parse detected table regions into rows/columns
- **Reading-order refinement**: multi-column layout ordering
- **Document classification**: classify page type (invoice, letter, report)

---

## Dataset

The training dataset is auto-downloaded from Roboflow via the shared dataset infrastructure (`configs/datasets/document_layout_block_detector.yaml`).

**Source**: DocLayNet — a diverse document layout dataset with annotations for 11 block types across financial, legal, scientific, and government documents.

**License**: Refer to the dataset page on Roboflow Universe for terms of use.

To force a re-download:

```bash
python "Document Layout Block Detector/Source Code/train.py" --force-download
```

---

## Keyboard Controls

| Key | Action |
|---|---|
| Any key | Next page (image/PDF mode) |
| `q` | Quit |
