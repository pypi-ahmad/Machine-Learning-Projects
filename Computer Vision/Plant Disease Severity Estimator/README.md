# Plant Disease Severity Estimator

Classify leaf images into **38 PlantVillage disease classes** and assign a
**severity bucket** (none / mild / moderate / severe) using transfer-learned
image classification.

---

## Pipeline Overview

```
leaf image
    │
    ▼
┌──────────────────────┐
│  Image Preprocessing │   Resize → 224×224, ImageNet normalisation
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│  ResNet-18 Backbone  │   Pretrained on ImageNet, fine-tuned on PlantVillage
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│  38-class Softmax    │   → class name, confidence
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│  Severity Mapping    │   Agronomic lookup → none / mild / moderate / severe
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│  Optional: Lesion    │   HSV-based lesion-area proxy (auxiliary signal)
│  Area Proxy          │
└──────────┬───────────┘
           ▼
  JSON / CSV / annotated image
```

---

## Severity Bucket Definitions

The PlantVillage dataset labels disease **type**, not degree.  Severity is
derived from agronomic knowledge of each disease's typical impact:

| Level | Name       | Criteria                                              | Example diseases                                      |
|-------|------------|-------------------------------------------------------|-------------------------------------------------------|
| 0     | **None**   | Healthy leaf, no disease                              | All `*___healthy` classes (12 classes)                 |
| 1     | **Mild**   | Cosmetic / surface damage, low yield impact           | Cedar apple rust, Powdery mildew, Common rust, Spider mites, Leaf mold, Bacterial spot (Peach, Pepper) |
| 2     | **Moderate** | Significant leaf damage, moderate yield impact      | Apple scab, Gray leaf spot, Northern leaf blight, Early blight, Septoria leaf spot, Leaf scorch, Target spot |
| 3     | **Severe** | Systemic infection or devastating disease             | Black rot, Esca, Citrus greening, Late blight, Yellow Leaf Curl Virus, Mosaic virus |

> **Note:** These are heuristic severity assignments — not ground-truth
> annotations.  For true severity quantification, pixel-level lesion
> segmentation or field-calibrated ratings would be needed.

---

## Dataset

**PlantVillage** — [abdallahalidev/plantvillage-dataset](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset)

- **54,303 images** across **38 classes**
- **14 plant species**, **26 disease types** + 12 healthy classes
- ImageFolder format (`color/Plant___Disease/`)
- Licence: CC BY-NC-SA 4.0
- ~2.2 GB

### 38 Classes

| # | Class name | Plant | Disease | Severity |
|---|-----------|-------|---------|----------|
| 1 | Apple___Apple_scab | Apple | Apple scab | Moderate |
| 2 | Apple___Black_rot | Apple | Black rot | Severe |
| 3 | Apple___Cedar_apple_rust | Apple | Cedar apple rust | Mild |
| 4 | Apple___healthy | Apple | Healthy | None |
| 5 | Blueberry___healthy | Blueberry | Healthy | None |
| 6 | Cherry_(including_sour)___Powdery_mildew | Cherry | Powdery mildew | Mild |
| 7 | Cherry_(including_sour)___healthy | Cherry | Healthy | None |
| 8 | Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot | Corn | Gray leaf spot | Moderate |
| 9 | Corn_(maize)___Common_rust_ | Corn | Common rust | Mild |
| 10 | Corn_(maize)___Northern_Leaf_Blight | Corn | Northern leaf blight | Moderate |
| 11 | Corn_(maize)___healthy | Corn | Healthy | None |
| 12 | Grape___Black_rot | Grape | Black rot | Severe |
| 13 | Grape___Esca_(Black_Measles) | Grape | Esca (Black Measles) | Severe |
| 14 | Grape___Leaf_blight_(Isariopsis_Leaf_Spot) | Grape | Leaf blight | Moderate |
| 15 | Grape___healthy | Grape | Healthy | None |
| 16 | Orange___Haunglongbing_(Citrus_greening) | Orange | Citrus greening | Severe |
| 17 | Peach___Bacterial_spot | Peach | Bacterial spot | Mild |
| 18 | Peach___healthy | Peach | Healthy | None |
| 19 | Pepper,_bell___Bacterial_spot | Pepper | Bacterial spot | Mild |
| 20 | Pepper,_bell___healthy | Pepper | Healthy | None |
| 21 | Potato___Early_blight | Potato | Early blight | Moderate |
| 22 | Potato___Late_blight | Potato | Late blight | Severe |
| 23 | Potato___healthy | Potato | Healthy | None |
| 24 | Raspberry___healthy | Raspberry | Healthy | None |
| 25 | Soybean___healthy | Soybean | Healthy | None |
| 26 | Squash___Powdery_mildew | Squash | Powdery mildew | Mild |
| 27 | Strawberry___Leaf_scorch | Strawberry | Leaf scorch | Moderate |
| 28 | Strawberry___healthy | Strawberry | Healthy | None |
| 29 | Tomato___Bacterial_spot | Tomato | Bacterial spot | Moderate |
| 30 | Tomato___Early_blight | Tomato | Early blight | Moderate |
| 31 | Tomato___Late_blight | Tomato | Late blight | Severe |
| 32 | Tomato___Leaf_Mold | Tomato | Leaf mold | Mild |
| 33 | Tomato___Septoria_leaf_spot | Tomato | Septoria leaf spot | Moderate |
| 34 | Tomato___Spider_mites Two-spotted_spider_mite | Tomato | Spider mites | Mild |
| 35 | Tomato___Target_Spot | Tomato | Target spot | Moderate |
| 36 | Tomato___Tomato_Yellow_Leaf_Curl_Virus | Tomato | Yellow Leaf Curl Virus | Severe |
| 37 | Tomato___Tomato_mosaic_virus | Tomato | Mosaic virus | Severe |
| 38 | Tomato___healthy | Tomato | Healthy | None |

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Download dataset

```bash
# Auto-downloads PlantVillage from Kaggle (requires kaggle API key)
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
# Single image
python "Source Code/infer.py" --source leaf.jpg

# Batch + save outputs
python "Source Code/infer.py" --source images/ --batch \
    --save --save-grid \
    --export-json results.json --export-csv results.csv
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

---

## Project Structure

```
Plant Disease Severity Estimator/
├── README.md
├── requirements.txt
└── Source Code/
    ├── config.py           # Configuration, 38-class severity mapping
    ├── classifier.py       # Model loading, inference, PredictionResult
    ├── train.py            # Training (delegates to shared pipeline)
    ├── data_bootstrap.py   # Auto-download PlantVillage, train/val split
    ├── visualize.py        # Annotated images, severity badges, grids
    ├── export.py           # JSON / CSV structured output
    ├── validator.py        # Input validation, image collection
    ├── controller.py       # High-level facade
    ├── infer.py            # CLI inference (single + batch)
    ├── evaluate.py         # Per-class + per-severity accuracy
    └── modern.py           # CVProject registry entry
```

---

## Output Format

Each prediction produces:

```json
{
  "class_name": "Tomato___Late_blight",
  "plant": "Tomato",
  "disease": "Late blight",
  "severity_index": 3,
  "severity_name": "severe",
  "confidence": 0.9642,
  "lesion_ratio": null
}
```

---

## Licence

Dataset: [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)
Code: Same as parent repository.
