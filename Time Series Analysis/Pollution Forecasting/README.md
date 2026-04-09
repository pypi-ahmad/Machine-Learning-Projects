# Pollution Forecasting

![Category](https://img.shields.io/badge/Category-Time%20Series%20Analysis-blue)
![Task](https://img.shields.io/badge/Task-Time%20Series%20Forecasting-green)
![Dataset](https://img.shields.io/badge/Dataset-Timeseries-orange)

## Project Overview

**Pollution Forecasting** is a **Time Series Forecasting** project in the **Time Series Analysis** category.

> The dataset we will be using comes from air quality sensors across South Korea. The sensors measure and record all types of air pollutants/particles in the air, but for this tutorial we will only look at PM2.5 (fine dust).

**Models:** PyCaret

## Dataset

| Property | Value |
|----------|-------|
| Type | Timeseries |
| Source | Local |
| Path | `data/pollution_forecasting/data.csv` |

```python
from core.data_loader import load_dataset
df = load_dataset('pollution_forecasting')
```

## Pipeline Files

| File | Lines |
|------|-------|
| `pipeline.py` | 178 |
| `train.py` | 148 |
| `evaluate.py` | 148 |
| `code.ipynb` | 10 code / 14 markdown cells |
| `test_pollution_forecasting.py` | test suite |

## ML Workflow

```mermaid
graph TD
    A[Load Dataset] --> B[Exploratory Data Analysis]
    B --> C[Preprocessing]
    C --> D[Feature Engineering]
    D --> E[Time Series Modeling]
    E --> G[Forecast & Evaluate]
    G --> H[Visualization]
```

## Core Logic

### Preprocessing

- Datetime feature extraction

### Feature Engineering

Feature engineering steps detected in notebook code cells.

## Models

| Model | Type |
|-------|------|
| PyCaret | AutoML Framework |

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
Time Series Analysis/Pollution Forecasting/
  Pollution Forecasting.pdf
  README.md
  code.ipynb
  data.csv
  evaluate.py
  guideline.txt
  pipeline.py
  test_pollution_forecasting.py
  train.py
```

## How to Run

```bash
cd "Time Series Analysis/Pollution Forecasting"
python pipeline.py
python train.py       # training only
python evaluate.py    # evaluation only
```

## Testing

```bash
pytest "Time Series Analysis/Pollution Forecasting/test_pollution_forecasting.py" -v
```

## Setup

```bash
pip install matplotlib numpy pandas pycaret scikit-learn seaborn statsmodels
```

## Limitations

- Forecast accuracy depends on the train/test split point chosen

---
*README auto-generated from `code.ipynb` analysis.*