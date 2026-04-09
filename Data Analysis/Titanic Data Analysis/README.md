# Titanic Data Analysis

![Category](https://img.shields.io/badge/Category-Data%20Analysis-blue)
![Task](https://img.shields.io/badge/Task-Classification-green)
![Dataset](https://img.shields.io/badge/Dataset-Tabular-orange)
![AutoML](https://img.shields.io/badge/AutoML-Enabled-purple)

## Project Overview

**Titanic Data Analysis** is a **Classification** project in the **Data Analysis** category.

> The code prints the column names (values) of the "train_df" DataFrame.

**Target variable:** `Survived`
**Models:** DecisionTree, LazyClassifier, LogisticRegression, PyCaret, RandomForest

## Dataset

| Property | Value |
|----------|-------|
| Type | Tabular |
| Source | Local |
| Path | `data/titanic_data_analysis/train.csv` |
| Target | `Survived` |

```python
from core.data_loader import load_dataset
df = load_dataset('titanic_data_analysis')
```

## Pipeline Files

| File | Lines |
|------|-------|
| `pipeline.py` | 412 |
| `train.py` | 381 |
| `evaluate.py` | 381 |
| `code.ipynb` | 45 code / 42 markdown cells |
| `test_titanic_data_analysis.py` | test suite |

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

- Bar charts

## Models

| Model | Type |
|-------|------|
| DecisionTree | Tree-Based |
| LazyClassifier | AutoML Benchmark (30+ classifiers) |
| LogisticRegression | Linear Classifier |
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
Data Analysis/Titanic Data Analysis/
  README.md
  Titanic Data anlysis.pdf
  code.ipynb
  data/
  evaluate.py
  guideline.txt
  pipeline.py
  test_titanic_data_analysis.py
  train.py
```

## How to Run

```bash
cd "Data Analysis/Titanic Data Analysis"
python pipeline.py
python train.py       # training only
python evaluate.py    # evaluation only
```

## Testing

```bash
pytest "Data Analysis/Titanic Data Analysis/test_titanic_data_analysis.py" -v
```

## Setup

```bash
pip install lazypredict matplotlib numpy pandas pycaret scikit-learn seaborn
```

---
*README auto-generated from `code.ipynb` analysis.*