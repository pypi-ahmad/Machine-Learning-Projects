# Tesla Car Price Prediction

## 1. Project Overview

This project implements a **Regression** pipeline for **Tesla Car Price Prediction**.

| Property | Value |
|----------|-------|
| **ML Task** | Regression |
| **Dataset Status** | OK LOCAL |

## 2. Dataset

**Data sources detected in code:**

- `CarPrice_Assignment.csv`

**Files in project directory:**

- `Data Dictionary - carprices.xlsx`

**Standardized data path:** `data/tesla_car_price_prediction/`

## 3. Pipeline Overview

### Original Notebook Pipeline

**Preprocessing:**
- Drop columns/rows

## 4. ML Workflow

```mermaid
flowchart TD
    A[Load CarPrice_Assignment.csv] --> B[Drop columns/rows]
```

## 5. Notebook Summary

| Metric | Value |
|--------|-------|
| Total cells | 34 |
| Code cells | 34 |
| Markdown cells | 0 |

## 6. Model Details

No model training in this project.

## 7. Project Structure

```
Tesla Car Price Prediction/
├── eda-linear-regression(1).ipynb
├── Data Dictionary - carprices.xlsx
└── README.md
```

## 8. Setup & Installation

`pip install -r requirements.txt` from the workspace root.

**Key dependencies:**

- `matplotlib`
- `numpy`
- `pandas`
- `seaborn`

## 9. How to Run

Open and run the notebook(s) sequentially:

```bash
jupyter notebook
```

- Open `eda-linear-regression(1).ipynb` and run all cells

## 10. Testing

Automated tests are available in `tests/test_p158_*.py`:

```bash
python -m pytest tests/test_p158_*.py -v
```

Tests validate data loading and library imports.

## 11. Limitations

- No model training — this is an analysis/tutorial notebook only
