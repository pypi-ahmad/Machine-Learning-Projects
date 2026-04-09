# Bank Customer churn prediction

![Category](https://img.shields.io/badge/Category-Regression-blue)
![Task](https://img.shields.io/badge/Task-Regression-green)
![Dataset](https://img.shields.io/badge/Dataset-Tabular-orange)
![AutoML](https://img.shields.io/badge/AutoML-Enabled-purple)

## Project Overview

**Bank Customer churn prediction** is a **Regression** project in the **Regression** category.

> We aim to accomplist the following for this study:

**Target variable:** `Exited`
**Models:** LazyClassifier, PyCaret

## Dataset

| Property | Value |
|----------|-------|
| Type | Tabular |
| Source | Local |
| Path | `data/bank_customer_churn/Churn_Modelling.csv` |
| Target | `Exited` |

```python
from core.data_loader import load_dataset
df = load_dataset('bank_customer_churn_prediction')
```

## Pipeline Files

| File | Lines |
|------|-------|
| `pipeline.py` | 301 |
| `train.py` | 250 |
| `evaluate.py` | 250 |
| `bank_customer_churn_prediction.ipynb` | 25 code / 19 markdown cells |
| `test_bank_customer_churn_prediction.py` | test suite |

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

- Count plots
- Box plots

### Helper Functions

- `DfPrepPipeline()`

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
Regression/Bank Customer churn prediction/
  Bank Customer Churn Prediction.pdf
  Dataset Link.pdf
  README.md
  bank_customer_churn_prediction.ipynb
  evaluate.py
  pipeline.py
  test_bank_customer_churn_prediction.py
  train.py
```

## How to Run

```bash
cd "Regression/Bank Customer churn prediction"
python pipeline.py
python train.py       # training only
python evaluate.py    # evaluation only
```

## Testing

```bash
pytest "Regression/Bank Customer churn prediction/test_bank_customer_churn_prediction.py" -v
```

## Setup

```bash
pip install lazypredict matplotlib numpy pandas pycaret scikit-learn seaborn
```

---
*README auto-generated from `bank_customer_churn_prediction.ipynb` analysis.*