# Heart Failure Prediction

![Category](https://img.shields.io/badge/Category-Data%20Analysis-blue)
![Task](https://img.shields.io/badge/Task-Classification-green)
![Dataset](https://img.shields.io/badge/Dataset-Tabular-orange)
![AutoML](https://img.shields.io/badge/AutoML-Enabled-purple)

## Project Overview

**Heart Failure Prediction** is a **Classification** project in the **Data Analysis** category.

> Cardiovascular diseases (CVDs) are responsible for the highest number of global deaths, with approximately 17.9 million lives lost each year, accounting for 31% of all worldwide deaths. This dataset focuses on predicting mortality caused by heart failure, a common occurrence related to CVDs, using 12 distinct features.

**Target variable:** `DEATH_EVENT`
**Models:** LSTM, LazyClassifier, NeuralNetwork, PyCaret

## Dataset

| Property | Value |
|----------|-------|
| Type | Tabular |
| Source | Local |
| Path | `data/heart_failure_prediction_da/data.csv` |
| Target | `DEATH_EVENT` |

```python
from core.data_loader import load_dataset
df = load_dataset('heart_failure_prediction')
```

## Pipeline Files

| File | Lines |
|------|-------|
| `pipeline.py` | 207 |
| `train.py` | 178 |
| `evaluate.py` | 178 |
| `code.ipynb` | 18 code / 23 markdown cells |
| `test_heart_failure_prediction.py` | test suite |

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

- StandardScaler normalization
- Outlier removal
- Train-test split

### Feature Engineering

Feature engineering steps detected in notebook code cells.

### Visualizations

- Correlation heatmap
- Count plots
- Confusion matrix

## Models

| Model | Type |
|-------|------|
| LSTM | Recurrent Neural Network |
| LazyClassifier | AutoML Benchmark (30+ classifiers) |
| NeuralNetwork | Neural Network |
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
Data Analysis/Heart Failure Prediction/
  Heart Failure Prediction.pdf
  README.md
  code.ipynb
  data.csv
  evaluate.py
  guideline.txt
  pipeline.py
  test_heart_failure_prediction.py
  train.py
```

## How to Run

```bash
cd "Data Analysis/Heart Failure Prediction"
python pipeline.py
python train.py       # training only
python evaluate.py    # evaluation only
```

## Testing

```bash
pytest "Data Analysis/Heart Failure Prediction/test_heart_failure_prediction.py" -v
```

## Setup

```bash
pip install lazypredict matplotlib numpy pandas pycaret scikit-learn seaborn tensorflow
```

---
*README auto-generated from `code.ipynb` analysis.*