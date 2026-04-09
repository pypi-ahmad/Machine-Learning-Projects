# Top Billionaires List Analysis

![Category](https://img.shields.io/badge/Category-Data%20Analysis-blue)
![Task](https://img.shields.io/badge/Task-Exploratory%20Data%20Analysis-green)
![Dataset](https://img.shields.io/badge/Dataset-Tabular-orange)

## Project Overview

**Top Billionaires List Analysis** is a **Exploratory Data Analysis** project in the **Data Analysis** category.

> Calculating the total of missing value in each column


## Dataset

| Property | Value |
|----------|-------|
| Type | Tabular |
| Source | Local |
| Path | `data/top_billionaires/data.csv` |

```python
from core.data_loader import load_dataset
df = load_dataset('top_billionaires_list_analysis')
```

## Pipeline Files

| File | Lines |
|------|-------|
| `pipeline.py` | 185 |
| `code.ipynb` | 21 code / 23 markdown cells |
| `test_top_billionaires_list_analysis.py` | test suite |

## ML Workflow

```mermaid
graph TD
    A[Load Dataset] --> B[Exploratory Data Analysis]
    B --> C[Preprocessing]
    C --> E[Statistical Analysis]
    E --> F[Visualization]
```

## Core Logic

### Preprocessing

- Missing value imputation

### Visualizations

- Correlation heatmap
- Histograms / distributions
- Pair plots
- Scatter plots

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
Data Analysis/Top Billionaires List Analysis/
  README.md
  Top billionaires list analysis.pdf
  code.ipynb
  data.csv
  guideline.txt
  pipeline.py
  test_top_billionaires_list_analysis.py
```

## How to Run

```bash
cd "Data Analysis/Top Billionaires List Analysis"
python pipeline.py
```

## Testing

```bash
pytest "Data Analysis/Top Billionaires List Analysis/test_top_billionaires_list_analysis.py" -v
```

## Setup

```bash
pip install matplotlib numpy pandas scikit-learn seaborn
```

---
*README auto-generated from `code.ipynb` analysis.*