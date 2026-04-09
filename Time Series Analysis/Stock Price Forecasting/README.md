# Stock Price Forecasting

![Category](https://img.shields.io/badge/Category-Time%20Series%20Analysis-blue)
![Task](https://img.shields.io/badge/Task-Time%20Series%20Forecasting-green)
![Dataset](https://img.shields.io/badge/Dataset-Timeseries-orange)

## Project Overview

**Stock Price Forecasting** is a **Time Series Forecasting** project in the **Time Series Analysis** category.

> Time Series data is a series of data points indexed in time order. Time series data is everywhere, so manipulating them is important for any data analyst or data scientist.

**Models:** PyCaret

## Dataset

| Property | Value |
|----------|-------|
| Type | Timeseries |
| Source | Api |
| Path | `data/stock_price_forecasting_ts/stock_data_combined.csv` |

```python
from core.data_loader import load_dataset
df = load_dataset('stock_price_forecasting')
```

## Pipeline Files

| File | Lines |
|------|-------|
| `pipeline.py` | 324 |
| `train.py` | 254 |
| `evaluate.py` | 254 |
| `code.ipynb` | 25 code / 32 markdown cells |
| `test_stock_price_forecasting.py` | test suite |

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

- Missing value imputation
- MinMaxScaler normalization

### Feature Engineering

Feature engineering steps detected in notebook code cells.

### Visualizations

- Correlation heatmap
- Histograms / distributions
- Pair plots
- Scatter plots

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
Time Series Analysis/Stock Price Forecasting/
  README.md
  Stock price forecasting.pdf
  code.ipynb
  evaluate.py
  guideline.txt
  pipeline.py
  test_stock_price_forecasting.py
  train.py
```

## How to Run

```bash
cd "Time Series Analysis/Stock Price Forecasting"
python pipeline.py
python train.py       # training only
python evaluate.py    # evaluation only
```

## Testing

```bash
pytest "Time Series Analysis/Stock Price Forecasting/test_stock_price_forecasting.py" -v
```

## Setup

```bash
pip install matplotlib numpy pandas pycaret scikit-learn seaborn statsmodels yfinance
```

## Limitations

- Forecast accuracy depends on the train/test split point chosen
- Dataset originally fetched via API — local CSV snapshot is used in pipeline

---
*README auto-generated from `code.ipynb` analysis.*