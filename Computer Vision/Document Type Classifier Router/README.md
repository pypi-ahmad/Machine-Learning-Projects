# Document Type Classifier Router

Classify scanned/photographed documents into **16 types** and
**route each to the correct downstream pipeline** — invoice extraction,
OCR, archival, HR processing, and more.

---

## Pipeline Overview

```
document image
    │
    ▼
┌──────────────────────┐
│  Image Preprocessing │   Resize → 224×224, ImageNet normalisation
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│  ResNet-18 Backbone  │   Pretrained on ImageNet, fine-tuned on
│  (16-class head)     │   Real World Documents Collections
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│  Classification      │   → document_type, confidence
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│  Router              │   Look up routing table → pipeline name
│                      │   If conf < threshold → manual_review_pipeline
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│  Pipeline Dispatch   │   Stub hooks (replace with real handlers)
└──────────┬───────────┘
           ▼
  JSON / CSV / annotated image
```

---

## Key Design: Classifier + Router Separation

The **classifier** and **router** are fully independent modules:

| Component | File | Responsibility |
|-----------|------|----------------|
| **Classifier** | `classifier.py` | Predict document type from image |
| **Router** | `router.py` | Map predicted type → downstream pipeline |
| **Controller** | `controller.py` | Wire classifier + router together |

The router never touches images — it only consumes classification outputs.
This makes it easy to swap classifiers or routing logic independently.

---

## Routing Table

| Document Type | Pipeline | Use Case |
|--------------|----------|----------|
| Invoice | `invoice_extraction_pipeline` | Extract line items, totals, vendor |
| Form | `form_extraction_pipeline` | Extract field values |
| Questionnaire | `form_extraction_pipeline` | Extract responses |
| Handwritten | `ocr_handwriting_pipeline` | Handwriting recognition |
| Email | `correspondence_pipeline` | Thread, sender, body extraction |
| Letter | `correspondence_pipeline` | Sender/recipient, date, body |
| Memo | `correspondence_pipeline` | Internal communication parsing |
| Resume / CV | `hr_pipeline` | Skills, experience extraction |
| Budget | `finance_pipeline` | Financial data extraction |
| Advertisement | `marketing_pipeline` | Content analysis |
| News Article | `content_ingestion_pipeline` | Article indexing |
| Presentation | `content_ingestion_pipeline` | Slide content extraction |
| Scientific Publication | `research_pipeline` | Citation, abstract extraction |
| Scientific Report | `research_pipeline` | Findings, data extraction |
| Specification | `engineering_pipeline` | Requirements parsing |
| File Folder | `archive_pipeline` | Document archival |

> Documents below the confidence threshold are routed to
> `manual_review_pipeline` for human triage.

### Registering Custom Handlers

```python
from router import register_handler

def my_invoice_handler(doc_type: str, pipeline: str) -> str:
    # Submit to invoice processing queue
    return f"Invoice submitted to extraction service"

register_handler("invoice_extraction_pipeline", my_invoice_handler)
```

---

## Dataset

**Real World Documents Collections**
— [shaz13/real-world-documents-collections](https://www.kaggle.com/datasets/shaz13/real-world-documents-collections)

- **~5,000 images** across **16 document types**
- Derived from the RVL-CDIP dataset (converted from TIFF to JPG)
- ~300 images per class
- ~474 MB

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Download dataset

```bash
python "Source Code/data_bootstrap.py"

# Force re-download
python "Source Code/data_bootstrap.py" --force-download
```

### 3. Train

```bash
python "Source Code/train.py"
python "Source Code/train.py" --model efficientnet_b0 --epochs 30 --batch 16
```

### 4. Inference

```bash
# Single document
python "Source Code/infer.py" --source document.jpg

# Batch + exports
python "Source Code/infer.py" --source docs/ --batch \
    --save --save-grid \
    --export-json results.json --export-csv results.csv

# Custom confidence threshold
python "Source Code/infer.py" --source docs/ --batch --threshold 0.5
```

### 5. Evaluate

```bash
python "Source Code/evaluate.py" --data path/to/val
```

---

## CLI Reference

### `train.py`

| Argument | Default | Description |
|----------|---------|-------------|
| `--data` | auto | Path to image-folder dataset |
| `--model` | `resnet18` | Architecture: resnet18/34/50, efficientnet_b0, mobilenet_v2 |
| `--epochs` | 25 | Training epochs |
| `--batch` | 32 | Batch size |
| `--imgsz` | 224 | Input image size |
| `--lr` | 0.001 | Learning rate |
| `--device` | auto | `cpu` or `cuda` |
| `--force-download` | — | Force re-download dataset |

### `infer.py`

| Argument | Default | Description |
|----------|---------|-------------|
| `--source` | required | Image path or directory |
| `--batch` | — | Process all images in directory |
| `--weights` | auto | Path to model weights |
| `--config` | — | Config file (JSON/YAML) |
| `--model` | `resnet18` | Architecture override |
| `--threshold` | 0.3 | Confidence threshold for routing |
| `--save` | — | Save annotated images |
| `--save-grid` | — | Save thumbnail grid |
| `--export-json` | — | Export results to JSON |
| `--export-csv` | — | Export results to CSV |
| `--output-dir` | `output` | Output directory |

### `evaluate.py`

| Argument | Default | Description |
|----------|---------|-------------|
| `--data` | required | Validation set (ImageFolder) |
| `--weights` | auto | Model weights |
| `--model` | `resnet18` | Architecture |
| `--threshold` | 0.3 | Routing confidence threshold |

---

## Project Structure

```
Document Type Classifier Router/
├── README.md
├── requirements.txt
└── Source Code/
    ├── config.py           # 16-class config, routing table, display labels
    ├── classifier.py       # Document type classifier (separate)
    ├── router.py           # Routing logic + pipeline stubs (separate)
    ├── train.py            # Training (delegates to shared pipeline)
    ├── data_bootstrap.py   # Auto-download dataset, train/val split
    ├── visualize.py        # Annotated images with type + route badges
    ├── export.py           # JSON / CSV structured export
    ├── validator.py        # Input validation, image collection
    ├── controller.py       # High-level facade (classifier + router)
    ├── infer.py            # CLI inference (single + batch)
    ├── evaluate.py         # Per-class accuracy + routing stats
    └── modern.py           # CVProject registry entry
```

---

## Output Format

Each processed document produces:

```json
{
  "source": "scan_001.jpg",
  "document_type": "invoice",
  "display_label": "Invoice",
  "confidence": 0.9423,
  "pipeline": "invoice_extraction_pipeline",
  "routed": true,
  "reason": "Document type 'Invoice' (conf 94.23%) → invoice_extraction_pipeline"
}
```

---

## 16 Document Types

| # | Class | Display Label |
|---|-------|--------------|
| 1 | advertisement | Advertisement |
| 2 | budget | Budget |
| 3 | email | Email |
| 4 | file_folder | File Folder |
| 5 | form | Form |
| 6 | handwritten | Handwritten |
| 7 | invoice | Invoice |
| 8 | letter | Letter |
| 9 | memo | Memo |
| 10 | news_article | News Article |
| 11 | presentation | Presentation |
| 12 | questionnaire | Questionnaire |
| 13 | resume | Resume / CV |
| 14 | scientific_publication | Scientific Publication |
| 15 | scientific_report | Scientific Report |
| 16 | specification | Specification |
