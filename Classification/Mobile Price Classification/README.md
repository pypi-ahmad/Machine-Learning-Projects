# Mobile price classification

![Category](https://img.shields.io/badge/Category-Classification-blue)
![Task](https://img.shields.io/badge/Task-Classification-green)
![Dataset](https://img.shields.io/badge/Dataset-Tabular-orange)
![AutoML](https://img.shields.io/badge/AutoML-Enabled-purple)

## Project Overview

**Mobile price classification** is a **Classification** project in the **Classification** category.

> We know that Screen size refers to the physical dimensions of a device’s display, measured in inches or centimeters. Screen resolution, on the other hand, refers to the number of pixels displayed on a screen, measured in pixels per inch (PPI) or pixels per centimeter (PPCM).

**Target variable:** `price_range`
**Models:** DecisionTree, GradientBoosting, LazyClassifier, PyCaret, RandomForest

## Dataset

| Property | Value |
|----------|-------|
| Type | Tabular |
| Source | Local |
| Path | `data/mobile_price_classification/train.csv` |
| Target | `price_range` |

```python
from core.data_loader import load_dataset
df = load_dataset('mobile_price_classification')
```

## Pipeline Files

| File | Lines |
|------|-------|
| `pipeline.py` | 409 |
| `train.py` | 337 |
| `evaluate.py` | 337 |
| `Mobile_price_classification.ipynb` | 33 code / 41 markdown cells |
| `test_mobile_price_classification.py` | test suite |

## ML Workflow

```mermaid
graph TD
    A[Load Dataset] --> B[Exploratory Data Analysis]
    B --> C[Preprocessing]
    C --> D[Feature Engineering]
    D --> E[Train-Test Split]
    E --> F[Model Training]
    F --> G[AutoML Comparison]
    G --> H[Evaluation]
    H --> I[Visualization]
```

## Core Logic

### Preprocessing

- Missing value imputation
- StandardScaler normalization
- Outlier removal
- Train-test split

### Feature Engineering

Feature engineering steps detected in notebook code cells.

### Visualizations

- Correlation heatmap
- Histograms / distributions
- Box plots
- Scatter plots
- Confusion matrix

## Models

| Model | Type |
|-------|------|
| DecisionTree | Tree-Based |
| GradientBoosting | Ensemble / Boosting |
| LazyClassifier | AutoML Benchmark (30+ classifiers) |
| PyCaret | AutoML Framework |
| RandomForest | Tree-Based |

AutoML is toggled via the `USE_AUTOML` flag in pipeline scripts.
**LazyPredict** (`LazyClassifier`) benchmarks 30+ models automatically.
**PyCaret** `compare_models()` runs cross-validated comparison.

## Reproducibility

```python
random.seed(42); np.random.seed(42); os.environ['PYTHONHASHSEED'] = '42'
```

```bash
python pipeline.py --seed 123    # custom seed
python pipeline.py --reproduce   # locked seed=42
```

## Project Structure

```
Classification/Mobile price classification/
  Dataset Link.pdf
  Mobile Price Classification.pdf
  Mobile_price_classification.ipynb
  README.md
  evaluate.py
  pipeline.py
  test_mobile_price_classification.py
  train.py
```

## How to Run

```bash
cd "Classification/Mobile price classification"
python pipeline.py
python train.py       # training only
python evaluate.py    # evaluation only
```

## Testing

```bash
pytest "Classification/Mobile price classification/test_mobile_price_classification.py" -v
```

## Setup

```bash
pip install lazypredict matplotlib numpy pandas pycaret scikit-learn seaborn
```

---
*README auto-generated from `Mobile_price_classification.ipynb` analysis.*