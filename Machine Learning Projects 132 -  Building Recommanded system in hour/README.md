# Building Recommanded system in hour

## 1. Project Overview

This project implements a **Exploratory Data Analysis** pipeline for **Building Recommanded system in hour**.

| Property | Value |
|----------|-------|
| **ML Task** | Exploratory Data Analysis |
| **Dataset Status** | OK LOCAL |

## 2. Dataset

**Data sources detected in code:**

- `BX-Books.csv`
- `BX-Users.csv`
- `BX-Book-Ratings.csv`

**Files in project directory:**

- `BX-Books.csv`

**Standardized data path:** `data/building_recommanded_system_in_hour/`

## 3. Pipeline Overview

### Original Notebook Pipeline

**Preprocessing:**
- Drop columns/rows
- Drop missing values (dropna)
- Index setting
- Outlier removal (IQR/quantile)
- Remove duplicate rows
- Handle missing values (fillna)

## 4. ML Workflow

```mermaid
flowchart TD
    A[Load BX-Books.csv] --> B[Drop columns/rows]
    B[Drop columns/rows] --> C[Drop missing values]
    C[Drop missing values] --> D[Index setting]
    D[Index setting] --> E[Outlier removal]
    E[Outlier removal] --> F[Remove duplicate rows]
    F[Remove duplicate rows] --> G[Handle missing values]
```

## 5. Notebook Summary

| Metric | Value |
|--------|-------|
| Total cells | 33 |
| Code cells | 26 |
| Markdown cells | 7 |

## 6. Model Details

No model training in this project.

## 7. Project Structure

```
 Building Recommanded system in hour/
├── Build Recommender System in an Hour - Part 2.ipynb
├── BX-Books.csv
└── README.md
```

## 8. Setup & Installation

`pip install -r requirements.txt` from the workspace root.

**Key dependencies:**

- `numpy`
- `pandas`
- `scikit-learn`
- `scipy`

## 9. How to Run

Open and run the notebook(s) sequentially:

```bash
jupyter notebook
```

- Open `Build Recommender System in an Hour - Part 2.ipynb` and run all cells

## 10. Testing

Automated tests are available in `tests/test_p132_*.py`:

```bash
python -m pytest tests/test_p132_*.py -v
```

Tests validate data loading and library imports.

## 11. Limitations

- No model training — this is an analysis/tutorial notebook only
