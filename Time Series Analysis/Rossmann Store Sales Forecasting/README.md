# Rossmann Store Sales Forecasting

![Category](https://img.shields.io/badge/Category-Time%20Series%20Analysis-blue)
![Task](https://img.shields.io/badge/Task-Time%20Series%20Forecasting-green)
![Dataset](https://img.shields.io/badge/Dataset-Timeseries-orange)

## Project Overview

**Rossmann Store Sales Forecasting** is a **Time Series Forecasting** project in the **Time Series Analysis** category.

> Printing the unique value of `DayOfWeek` column

**Target variable:** `Sales`
**Models:** Prophet, PyCaret

## Dataset

| Property | Value |
|----------|-------|
| Type | Timeseries |
| Source | Local |
| Path | `data/rossmann_store_sales/data.csv` |
| Target | `Sales` |

```python
from core.data_loader import load_dataset
df = load_dataset('rossmann_store_sales_forecasting')
```

## Pipeline Files

| File | Lines |
|------|-------|
| `pipeline.py` | 318 |
| `train.py` | 169 |
| `evaluate.py` | 169 |
| `code.ipynb` | 66 code / 46 markdown cells |
| `test_rossmann_store_sales_forecasting.py` | test suite |

## ML Workflow

```mermaid
graph TD
    A[Load Dataset] --> B[Exploratory Data Analysis]
    B --> C[Preprocessing]
    C --> E[Time Series Modeling]
    E --> G[Forecast & Evaluate]
    G --> H[Visualization]
```

## Core Logic

### Preprocessing

- Missing value imputation

### Visualizations

- Correlation heatmap
- Histograms / distributions
- Bar charts

## Models

| Model | Type |
|-------|------|
| Prophet | Additive Time Series Model |
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
Time Series Analysis/Rossmann Store Sales Forecasting/
  README.md
  Rossmann Store Sales Forecasting.pdf
  code.ipynb
  data.csv
  evaluate.py
  guideline.txt
  pipeline.py
  store.csv
  test_rossmann_store_sales_forecasting.py
  train.py
```

## How to Run

```bash
cd "Time Series Analysis/Rossmann Store Sales Forecasting"
python pipeline.py
python train.py       # training only
python evaluate.py    # evaluation only
```

## Testing

```bash
pytest "Time Series Analysis/Rossmann Store Sales Forecasting/test_rossmann_store_sales_forecasting.py" -v
```

## Setup

```bash
pip install matplotlib numpy pandas prophet pycaret scikit-learn seaborn statsmodels
```

## Limitations

- Forecast accuracy depends on the train/test split point chosen

---
*README auto-generated from `code.ipynb` analysis.*