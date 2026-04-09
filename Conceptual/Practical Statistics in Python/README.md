# (Conceptual) Practical Statistics House Python_update

## 1. Project Overview

This project implements a **Exploratory Data Analysis** pipeline for **(Conceptual) Practical Statistics House Python_update**. The target variable is `SalePrice`.

| Property | Value |
|----------|-------|
| **ML Task** | Exploratory Data Analysis |
| **Target Variable** | `SalePrice` |
| **Dataset Status** | OK LOCAL |

## 2. Dataset

**Data sources detected in code:**

- `house_train.csv`

**Files in project directory:**

- `train (2).csv`

**Standardized data path:** `data/conceptual_practical_statistics_house_python_update/`

## 3. Pipeline Overview

### Original Notebook Pipeline

**Preprocessing:**
- Drop columns/rows
- Outlier removal (IQR/quantile)
- Handle missing values (fillna)

## 4. ML Workflow

```mermaid
flowchart TD
    A[Load house_train.csv] --> B[Drop columns/rows]
    B[Drop columns/rows] --> C[Outlier removal]
    C[Outlier removal] --> D[Handle missing values]
```

## 5. Notebook Summary

| Metric | Value |
|--------|-------|
| Total cells | 81 |
| Code cells | 42 |
| Markdown cells | 39 |

## 6. Model Details

No model training in this project.

## 7. Project Structure

```
(Conceptual) Practical Statistics House Python_update/
├── Practical Statistics House Python_update.ipynb
├── train (2).csv
└── README.md
```

## 8. Setup & Installation

`pip install -r requirements.txt` from the workspace root.

**Key dependencies:**

- `matplotlib`
- `numpy`
- `pandas`
- `plotly`
- `seaborn`

## 9. How to Run

Open and run the notebook(s) sequentially:

```bash
jupyter notebook
```

- Open `Practical Statistics House Python_update.ipynb` and run all cells

## 10. Testing

Automated tests are available in `tests/test_p113_*.py`:

```bash
python -m pytest tests/test_p113_*.py -v
```

Tests validate data loading and library imports.

## 11. Limitations

- No model training — this is an analysis/tutorial notebook only
