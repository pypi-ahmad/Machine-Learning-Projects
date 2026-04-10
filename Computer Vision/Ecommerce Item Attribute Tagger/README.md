# Ecommerce Item Attribute Tagger

Predict structured product attributes from images — category, colour, article type, season, usage, and gender — using a multi-head ResNet classifier with optional YOLO-based item isolation.

## Architecture

```
Product Image
      │
      ▼
┌─────────────┐     ┌──────────────────────┐
│ ItemDetector │────▶│ Crop / Isolate item  │
│   (YOLO)     │     │ (optional)           │
└─────────────┘     └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │  Shared ResNet-18    │
                    │  Backbone            │
                    └──────────┬───────────┘
                               │
              ┌────────┬───────┼───────┬────────┐
              ▼        ▼       ▼       ▼        ▼
          ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐  ...
          │Color │ │Categ.│ │Type  │ │Season│
          │ Head │ │ Head │ │ Head │ │ Head │
          └──────┘ └──────┘ └──────┘ └──────┘
              │        │       │       │
              ▼        ▼       ▼       ▼
          Structured JSON / CSV export
```

## Predicted Attributes

| Attribute | Examples | Max Classes |
|-----------|----------|-------------|
| masterCategory | Apparel, Accessories, Footwear | 7 |
| subCategory | Topwear, Bottomwear, Watches | 20 |
| articleType | Tshirts, Jeans, Shirts | 50 |
| baseColour | Black, White, Blue, Red | 20 |
| season | Summer, Fall, Winter, Spring | 5 |
| usage | Casual, Ethnic, Formal, Sports | 10 |
| gender | Men, Women, Boys, Girls, Unisex | 6 |

## Dataset

**Fashion Product Images (Small)** — [Kaggle](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small)
- 44,441 product images (80×60 px)
- `styles.csv` with 10 attribute columns
- MIT License

Auto-downloaded on first run:
```bash
python train.py --force-download
```

## Quick Start

### Train

```bash
cd "Ecommerce Item Attribute Tagger/Source Code"

# Train with defaults (ResNet-18, 15 epochs)
python train.py

# Custom backbone and epochs
python train.py --backbone resnet50 --epochs 25 --batch 32

# With explicit dataset path
python train.py --data path/to/dataset

# Force re-download
python train.py --force-download
```

### Inference

```bash
# Single image
python infer.py --source product.jpg

# Folder batch
python infer.py --source products_folder/

# Export catalog JSON
python infer.py --source folder/ --export catalog.json --no-display

# Export CSV
python infer.py --source folder/ --export catalog.csv --no-display

# Compact overlay mode
python infer.py --source image.jpg --overlay

# Save annotated images
python infer.py --source folder/ --save-dir output/
```

### Python API

```python
from modern import EcommerceItemAttributeTagger

tagger = EcommerceItemAttributeTagger()
tagger.load()

# Predict
result = tagger.predict("product.jpg")
print(result)
# {
#   "attributes": {
#     "masterCategory": {"value": "Apparel", "confidence": 0.95},
#     "baseColour": {"value": "Black", "confidence": 0.88},
#     "articleType": {"value": "Tshirts", "confidence": 0.92},
#     ...
#   },
#   "box": null
# }

# Visualize
annotated = tagger.visualize("product.jpg")
```

### Catalog Export

```python
from attribute_predictor import AttributePredictor
from export import export_catalog, export_csv

predictor = AttributePredictor(cfg)
predictor.load("best_model.pt")

results = []
for img_path in image_paths:
    pred = predictor.predict_proba(cv2.imread(str(img_path)))
    results.append({"source": str(img_path), "prediction": pred})

export_catalog(results, "catalog.json")
export_csv(results, "catalog.csv")
```

## Project Structure

```
Source Code/
├── config.py              # Configuration + attribute schema
├── detector.py            # YOLO-based item isolation (optional)
├── attribute_predictor.py # Multi-head ResNet classifier
├── train.py               # Training pipeline
├── infer.py               # Inference CLI
├── visualize.py           # Annotation overlays + grid view
├── export.py              # JSON/CSV catalog export
├── data_bootstrap.py      # Dataset download + label preparation
└── modern.py              # CVProject registry adapter
```

## Output Formats

**Structured JSON** (`catalog.json`):
```json
[
  {
    "source": "product_001.jpg",
    "attributes": {
      "masterCategory": {"value": "Apparel", "confidence": 0.95},
      "baseColour": {"value": "Black", "confidence": 0.88}
    }
  }
]
```

**Flat CSV** (`catalog.csv`):
```
source,masterCategory,baseColour,...,masterCategory_conf,baseColour_conf,...
product_001.jpg,Apparel,Black,...,0.95,0.88,...
```
