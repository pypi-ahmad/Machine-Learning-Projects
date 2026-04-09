# Price Elasticity of Demand Analysis

## 1. Project Overview

This project implements a **Regression** pipeline for **Price Elasticity of Demand Analysis**.

| Property | Value |
|----------|-------|
| **ML Task** | Regression |
| **Dataset Status** | OK LOCAL |

## 2. Dataset

**Data sources detected in code:**

- `beef.csv`

**Files in project directory:**

- `beef.csv`

**Standardized data path:** `data/price_elasticity_of_demand_analysis/`

## 3. Pipeline Overview

### Original Notebook Pipeline

**Preprocessing:**
- Date parsing
- Drop columns/rows
- Index setting

## 4. ML Workflow

```mermaid
flowchart TD
    A[Load beef.csv] --> B[Date parsing]
    B[Date parsing] --> C[Drop columns/rows]
    C[Drop columns/rows] --> D[Index setting]
```

## 5. Notebook Summary

| Metric | Value |
|--------|-------|
| Total cells | 30 |
| Code cells | 19 |
| Markdown cells | 11 |

## 6. Model Details

No model training in this project.

## 7. Project Structure

```
Price Elasticity of Demand Analysis/
├── Price Elasticity of Demand.ipynb
├── beef.csv
└── README.md
```

## 8. Setup & Installation

`pip install -r requirements.txt` from the workspace root.

**Key dependencies:**

- `matplotlib`
- `numpy`
- `pandas`
- `statsmodels`

## 9. How to Run

Open and run the notebook(s) sequentially:

```bash
jupyter notebook
```

- Open `Price Elasticity of Demand.ipynb` and run all cells

## 10. Testing

Automated tests are available in `tests/test_p114_*.py`:

```bash
python -m pytest tests/test_p114_*.py -v
```

Tests validate data loading and library imports.

## 11. Limitations

- No model training — this is an analysis/tutorial notebook only
