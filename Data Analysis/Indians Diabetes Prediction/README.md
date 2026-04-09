# Indians Diabetes Prediction

![Category](https://img.shields.io/badge/Category-Data%20Analysis-blue)
![Task](https://img.shields.io/badge/Task-Classification-green)
![Dataset](https://img.shields.io/badge/Dataset-Tabular-orange)
![AutoML](https://img.shields.io/badge/AutoML-Enabled-purple)

## Project Overview

**Indians Diabetes Prediction** is a **Classification** project in the **Data Analysis** category.

> Printing the overview information of the data set

**Target variable:** `Outcome`
**Models:** LazyClassifier, PyCaret

## Dataset

| Property | Value |
|----------|-------|
| Type | Tabular |
| Source | Local |
| Path | `data/indians_diabetes_da/data.csv` |
| Target | `Outcome` |

```python
from core.data_loader import load_dataset
df = load_dataset('indians_diabetes_prediction')
```

## Pipeline Files

| File | Lines |
|------|-------|
| `pipeline.py` | 248 |
| `train.py` | 194 |
| `evaluate.py` | 194 |
| `code.ipynb` | 25 code / 25 markdown cells |
| `test_indians_diabetes_prediction.py` | test suite |

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

- Missing value imputation
- StandardScaler normalization
- Train-test split

### Visualizations

- Correlation heatmap
- Histograms / distributions
- Count plots
- Pair plots
- Bar charts
- Scatter plots

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
Data Analysis/Indians Diabetes Prediction/
  Indians Diabetes Prediction.pdf
  README.md
  code.ipynb
  data.csv
  evaluate.py
  guideline.txt
  pipeline.py
  test_indians_diabetes_prediction.py
  train.py
```

## How to Run

```bash
cd "Data Analysis/Indians Diabetes Prediction"
python pipeline.py
python train.py       # training only
python evaluate.py    # evaluation only
```

## Testing

```bash
pytest "Data Analysis/Indians Diabetes Prediction/test_indians_diabetes_prediction.py" -v
```

## Setup

```bash
pip install lazypredict matplotlib numpy pandas pycaret scikit-learn seaborn
```

---
*README auto-generated from `code.ipynb` analysis.*