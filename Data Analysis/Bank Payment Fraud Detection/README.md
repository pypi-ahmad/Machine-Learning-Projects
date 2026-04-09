# Bank Payment Fraud Detection

![Category](https://img.shields.io/badge/Category-Data%20Analysis-blue)
![Task](https://img.shields.io/badge/Task-Classification-green)
![Dataset](https://img.shields.io/badge/Dataset-Tabular-orange)
![AutoML](https://img.shields.io/badge/AutoML-Enabled-purple)

## Project Overview

**Bank Payment Fraud Detection** is a **Classification** project in the **Data Analysis** category.

> The given dataset contains a table with 9 columns representing different features and one column representing the target variable. The features are as follows:

**Target variable:** `fraud`
**Models:** LazyClassifier, PyCaret, RandomForest, XGBoost

## Dataset

| Property | Value |
|----------|-------|
| Type | Tabular |
| Source | Local |
| Path | `data/bank_payment_fraud/data.csv` |
| Target | `fraud` |

```python
from core.data_loader import load_dataset
df = load_dataset('bank_payment_fraud_detection')
```

## Pipeline Files

| File | Lines |
|------|-------|
| `pipeline.py` | 228 |
| `train.py` | 181 |
| `evaluate.py` | 181 |
| `code.ipynb` | 20 code / 20 markdown cells |
| `test_bank_payment_fraud_detection.py` | test suite |

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

- SMOTE oversampling
- Train-test split

### Feature Engineering

Feature engineering steps detected in notebook code cells.

### Visualizations

- Histograms / distributions
- Count plots
- Box plots
- Confusion matrix
- ROC curve

## Models

| Model | Type |
|-------|------|
| LazyClassifier | AutoML Benchmark (30+ classifiers) |
| PyCaret | AutoML Framework |
| RandomForest | Tree-Based |
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
Data Analysis/Bank Payment Fraud Detection/
  Bank Payment Fraud Detection.pdf
  README.md
  code.ipynb
  data.csv
  evaluate.py
  guideline.txt
  pipeline.py
  test_bank_payment_fraud_detection.py
  train.py
```

## How to Run

```bash
cd "Data Analysis/Bank Payment Fraud Detection"
python pipeline.py
python train.py       # training only
python evaluate.py    # evaluation only
```

## Testing

```bash
pytest "Data Analysis/Bank Payment Fraud Detection/test_bank_payment_fraud_detection.py" -v
```

## Setup

```bash
pip install lazypredict matplotlib numpy pandas pycaret scikit-learn seaborn xgboost
```

---
*README auto-generated from `code.ipynb` analysis.*