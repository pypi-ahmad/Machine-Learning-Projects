# China GDP Estimation

## 1. Project Overview

This project implements a **Exploratory Data Analysis** pipeline for **China GDP Estimation**.

| Property | Value |
|----------|-------|
| **ML Task** | Exploratory Data Analysis |
| **Dataset Status** | OK LOCAL |

## 2. Dataset

**Data sources detected in code:**

- `china_gdp.csv`

**Files in project directory:**

- `china_gdp.csv`

**Standardized data path:** `data/china_gdp_estimation/`

## 3. Pipeline Overview

### Original Notebook Pipeline

**Preprocessing:**
- Log transformation

**Evaluation metrics:**
- R² Score

## 4. ML Workflow

```mermaid
flowchart TD
    A[Load china_gdp.csv] --> B[Log transformation]
    B[Log transformation] --> C[Evaluate: R² Score]
```

## 5. Notebook Summary

| Metric | Value |
|--------|-------|
| Total cells | 30 |
| Code cells | 15 |
| Markdown cells | 15 |

## 6. Model Details

### Evaluation Metrics

- R² Score

No model training in this project.

## 7. Project Structure

```
Machine Learning Projects  88 -China GDP Estimation/
├── ChinaGDP.ipynb
├── china_gdp.csv
└── README.md
```

## 8. Setup & Installation

`pip install -r requirements.txt` from the workspace root.

**Key dependencies:**

- `matplotlib`
- `numpy`
- `pandas`
- `scikit-learn`
- `scipy`

## 9. How to Run

Open and run the notebook(s) sequentially:

```bash
jupyter notebook
```

- Open `ChinaGDP.ipynb` and run all cells

## 10. Testing

Automated tests are available in `tests/test_p088_*.py`:

```bash
python -m pytest tests/test_p088_*.py -v
```

Tests validate data loading and library imports.

## 11. Limitations

- No model training — this is an analysis/tutorial notebook only
