# Document Layout Block Detector

YOLO26m-based detection of structural blocks in document images and PDFs.

## Classes (10)

title, text, table, figure, list, caption, header, footer, page-number, stamp

## Quick Start

```bash
# Train (synthetic data auto-generated if real dataset unavailable)
python train.py --epochs 50 --batch 8 --imgsz 1024

# Evaluate
python evaluate.py --model runs/train/weights/best.pt

# Infer on image or PDF
python infer.py --source document.pdf --save-annotated --export-json results.json
python infer.py --source page.png --save-crops --no-display
```

## Files

| File | Purpose |
|------|---------|
| config.py | LayoutConfig dataclass and YAML loader |
| data_bootstrap.py | Dataset download with synthetic fallback |
| detector.py | LayoutDetector, LayoutBlock, PageResult |
| train.py | Training via shared train_detection |
| evaluate.py | Validation metrics (mAP, per-class AP) |
| infer.py | CLI inference (images, PDFs, directories) |
| export.py | JSON export and crop saving |
| visualize.py | Overlay rendering with class colours |
| pdf_utils.py | PDF-to-image conversion (PyMuPDF / pdf2image) |
| modern.py | CVProject adapter for registry |
| layout_config.yaml | Sample YAML configuration |

## Configuration

Edit `layout_config.yaml` or pass CLI flags. Key options:

- `--model` -- YOLO weights (default: yolo26m.pt)
- `--imgsz` -- input resolution (default: 1024)
- `--conf` -- confidence threshold
- `--save-crops` -- save individual block crops
- `--export-json` -- export detection results as JSON
- `--save-annotated` -- save annotated images
