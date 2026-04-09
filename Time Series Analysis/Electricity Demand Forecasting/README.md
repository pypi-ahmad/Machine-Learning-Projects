# Electricity Demand Forecasting

![Category](https://img.shields.io/badge/Category-Time%20Series%20Analysis-blue)
![Task](https://img.shields.io/badge/Task-Time%20Series%20Forecasting-green)
![Dataset](https://img.shields.io/badge/Dataset-Tabular-orange)

## Project Overview

**Electricity Demand Forecasting** is a **Time Series Forecasting** project in the **Time Series Analysis** category.

> The main goal of this lab is to test whether a general and simple approach based on Machine Learning models, can yield good enough results in a complex forecasting problem.

**Target variable:** `energy`

## Dataset

| Property | Value |
|----------|-------|
| Type | Tabular |
| Source | Local |
| Path | `data/electricity_demand_forecasting/data.csv` |
| Target | `energy` |

```python
from core.data_loader import load_dataset
df = load_dataset('electricity_demand_forecasting')
```

## Pipeline Files

| File | Lines |
|------|-------|
| `pipeline.py` | 301 |
| `code.ipynb` | 26 code / 36 markdown cells |
| `test_electricity_demand_forecasting.py` | test suite |

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
- One-hot encoding

### Feature Engineering

Feature engineering steps detected in notebook code cells.

### Visualizations

- Correlation heatmap
- Histograms / distributions
- Box plots
- Pair plots
- Scatter plots

### Helper Functions

- `shapiro_test()`

## Models

This project focuses on exploratory data analysis without explicit ML modeling.

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
Time Series Analysis/Electricity Demand Forecasting/
  Electricity demand forecasting.pdf
  README.md
  code.ipynb
  data.csv
  guideline.txt
  pipeline.py
  test_electricity_demand_forecasting.py
```

## How to Run

```bash
cd "Time Series Analysis/Electricity Demand Forecasting"
python pipeline.py
```

## Testing

```bash
pytest "Time Series Analysis/Electricity Demand Forecasting/test_electricity_demand_forecasting.py" -v
```

## Setup

```bash
pip install matplotlib numpy pandas scikit-learn seaborn statsmodels
```

---
*README auto-generated from `code.ipynb` analysis.*