# Video Game Sales Analysis

![Category](https://img.shields.io/badge/Category-Data%20Analysis-blue)
![Task](https://img.shields.io/badge/Task-Exploratory%20Data%20Analysis-green)
![Dataset](https://img.shields.io/badge/Dataset-Tabular-orange)
![AutoML](https://img.shields.io/badge/AutoML-Enabled-purple)

## Project Overview

**Video Game Sales Analysis** is a **Exploratory Data Analysis** project in the **Data Analysis** category.

> Check some random data using `.sample()` method. It will pick the random number of records.

**Target variable:** `Global_Sales`
**Models:** LazyRegressor, LinearRegression, PyCaret

## Dataset

| Property | Value |
|----------|-------|
| Type | Tabular |
| Source | Local |
| Path | `data/video_game_sales/data.csv` |
| Target | `Global_Sales` |

```python
from core.data_loader import load_dataset
df = load_dataset('video_game_sales_analysis')
```

## Pipeline Files

| File | Lines |
|------|-------|
| `pipeline.py` | 456 |
| `train.py` | 364 |
| `evaluate.py` | 364 |
| `code.ipynb` | 46 code / 53 markdown cells |
| `test_video_game_sales_analysis.py` | test suite |

## ML Workflow

```mermaid
graph TD
    A[Load Dataset] --> B[Exploratory Data Analysis]
    B --> C[Preprocessing]
    C --> D[Feature Engineering]
    D --> E[Model Training]
    E --> F[AutoML Comparison]
    F --> G[Evaluation]
    G --> H[Visualization]
```

## Core Logic

### Preprocessing

- Missing value imputation
- Label encoding
- Train-test split

### Feature Engineering

Feature engineering steps detected in notebook code cells.

### Visualizations

- Correlation heatmap
- Bar charts
- Scatter plots
- Word cloud

## Models

| Model | Type |
|-------|------|
| LazyRegressor | AutoML Benchmark (30+ regressors) |
| LinearRegression | Linear Regressor |
| PyCaret | AutoML Framework |

AutoML is toggled via the `USE_AUTOML` flag in pipeline scripts.
**LazyPredict** (`LazyRegressor`) benchmarks 30+ models automatically.
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
Data Analysis/Video Game Sales Analysis/
  README.md
  Video Game Sales Analysis.pdf
  code.ipynb
  data.csv
  evaluate.py
  guideline.txt
  pipeline.py
  test_video_game_sales_analysis.py
  train.py
```

## How to Run

```bash
cd "Data Analysis/Video Game Sales Analysis"
python pipeline.py
python train.py       # training only
python evaluate.py    # evaluation only
```

## Testing

```bash
pytest "Data Analysis/Video Game Sales Analysis/test_video_game_sales_analysis.py" -v
```

## Setup

```bash
pip install lazypredict matplotlib numpy pandas pycaret scikit-learn seaborn wordcloud
```

---
*README auto-generated from `code.ipynb` analysis.*