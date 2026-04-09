# Heart disease prediction

![Category](https://img.shields.io/badge/Category-Regression-blue)
![Task](https://img.shields.io/badge/Task-Regression-green)
![Dataset](https://img.shields.io/badge/Dataset-Tabular-orange)
![AutoML](https://img.shields.io/badge/AutoML-Enabled-purple)

## Project Overview

**Heart disease prediction** is a **Regression** project in the **Regression** category.

> `Diagnosis of heart disease (angiographic disease status)

**Target variable:** `target`
**Models:** LazyClassifier, PyCaret

## Dataset

| Property | Value |
|----------|-------|
| Type | Tabular |
| Source | Local |
| Path | `data/heart_disease_prediction/heart.csv` |
| Target | `target` |

```python
from core.data_loader import load_dataset
df = load_dataset('heart_disease_prediction')
```

## Pipeline Files

| File | Lines |
|------|-------|
| `pipeline.py` | 399 |
| `train.py` | 349 |
| `evaluate.py` | 349 |
| `heart_disease_predictions.ipynb` | 24 code / 24 markdown cells |
| `test_heart_disease_prediction.py` | test suite |

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
- One-hot encoding
- Train-test split

### Feature Engineering

Feature engineering steps detected in notebook code cells.

### Visualizations

- Correlation heatmap
- Count plots
- Pair plots
- Confusion matrix

### Helper Functions

- `count_plot()`
- `cramers_v()`

## Models

| Model | Type |
|-------|------|
| LazyClassifier | AutoML Benchmark (30+ classifiers) |
| PyCaret | AutoML Framework |

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
Regression/Heart disease prediction/
  Dataset Link.pdf
  Heart Disease Prediction.pdf
  README.md
  evaluate.py
  heart_disease_predictions.ipynb
  pipeline.py
  test_heart_disease_prediction.py
  train.py
```

## How to Run

```bash
cd "Regression/Heart disease prediction"
python pipeline.py
python train.py       # training only
python evaluate.py    # evaluation only
```

## Testing

```bash
pytest "Regression/Heart disease prediction/test_heart_disease_prediction.py" -v
```

## Setup

```bash
pip install lazypredict matplotlib numpy pandas pycaret scikit-learn seaborn
```

---
*README auto-generated from `heart_disease_predictions.ipynb` analysis.*