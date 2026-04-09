# Mobile Price Prediction

![Category](https://img.shields.io/badge/Category-Data%20Analysis-blue)
![Task](https://img.shields.io/badge/Task-Classification-green)
![Dataset](https://img.shields.io/badge/Dataset-Tabular-orange)
![AutoML](https://img.shields.io/badge/AutoML-Enabled-purple)

## Project Overview

**Mobile Price Prediction** is a **Classification** project in the **Data Analysis** category.

> Print the overview information of data frame

**Target variable:** `price_range`
**Models:** LazyClassifier, PyCaret

## Dataset

| Property | Value |
|----------|-------|
| Type | Tabular |
| Source | Local |
| Path | `data/mobile_price_prediction_da/train.csv` |
| Target | `price_range` |

```python
from core.data_loader import load_dataset
df = load_dataset('mobile_price_prediction')
```

## Pipeline Files

| File | Lines |
|------|-------|
| `pipeline.py` | 169 |
| `train.py` | 137 |
| `evaluate.py` | 137 |
| `code.ipynb` | 23 code / 24 markdown cells |
| `test_mobile_price_prediction.py` | test suite |

## ML Workflow

```mermaid
graph TD
    A[Load Dataset] --> B[Exploratory Data Analysis]
    B --> C[Preprocessing]
    C --> E[Train-Test Split]
    E --> F[Model Training]
    F --> G[AutoML Comparison]
    G --> H[Evaluation]
    H --> I[Visualization]
```

## Core Logic

### Preprocessing

- Train-test split

### Visualizations

- Histograms / distributions
- Box plots
- Pair plots

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
Data Analysis/Mobile Price Prediction/
  Mobile price Prediction.pdf
  README.md
  code.ipynb
  data/
  evaluate.py
  guideline.txt
  pipeline.py
  test_mobile_price_prediction.py
  train.py
```

## How to Run

```bash
cd "Data Analysis/Mobile Price Prediction"
python pipeline.py
python train.py       # training only
python evaluate.py    # evaluation only
```

## Testing

```bash
pytest "Data Analysis/Mobile Price Prediction/test_mobile_price_prediction.py" -v
```

## Setup

```bash
pip install lazypredict matplotlib numpy pandas pycaret scikit-learn seaborn
```

---
*README auto-generated from `code.ipynb` analysis.*