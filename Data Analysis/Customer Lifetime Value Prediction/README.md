# Customer Lifetime Value Prediction

![Category](https://img.shields.io/badge/Category-Data%20Analysis-blue)
![Task](https://img.shields.io/badge/Task-Classification-green)
![Dataset](https://img.shields.io/badge/Dataset-Tabular-orange)
![AutoML](https://img.shields.io/badge/AutoML-Enabled-purple)

## Project Overview

**Customer Lifetime Value Prediction** is a **Classification** project in the **Data Analysis** category.

> The code initializes the Plotly library for displaying interactive visualizations in a notebook. It then displays the first few rows of the "tx_data" DataFrame.

**Target variable:** `LTVCluster`
**Models:** KMeans, LazyClassifier, PyCaret, XGBoost

## Dataset

| Property | Value |
|----------|-------|
| Type | Tabular |
| Source | Local |
| Path | `data/customer_ltv_prediction_da/data.csv` |
| Target | `LTVCluster` |

```python
from core.data_loader import load_dataset
df = load_dataset('customer_lifetime_value_prediction')
```

## Pipeline Files

| File | Lines |
|------|-------|
| `pipeline.py` | 175 |
| `train.py` | 157 |
| `evaluate.py` | 157 |
| `code.ipynb` | 17 code / 20 markdown cells |
| `test_customer_lifetime_value_prediction.py` | test suite |

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

- Datetime feature extraction
- Train-test split

### Visualizations

- Confusion matrix

## Models

| Model | Type |
|-------|------|
| KMeans | Centroid Clustering |
| LazyClassifier | AutoML Benchmark (30+ classifiers) |
| PyCaret | AutoML Framework |
| XGBoost | Ensemble / Boosting |

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
Data Analysis/Customer Lifetime Value Prediction/
  Customer Lifetime value prediction.pdf
  README.md
  code.ipynb
  data.csv
  evaluate.py
  guideline.txt
  pipeline.py
  test_customer_lifetime_value_prediction.py
  train.py
```

## How to Run

```bash
cd "Data Analysis/Customer Lifetime Value Prediction"
python pipeline.py
python train.py       # training only
python evaluate.py    # evaluation only
```

## Testing

```bash
pytest "Data Analysis/Customer Lifetime Value Prediction/test_customer_lifetime_value_prediction.py" -v
```

## Setup

```bash
pip install lazypredict matplotlib numpy pandas pycaret scikit-learn seaborn xgboost
```

---
*README auto-generated from `code.ipynb` analysis.*