# articles_rec_system

## 1. Project Overview

This project implements a **Exploratory Data Analysis** pipeline for **articles_rec_system**.

| Property | Value |
|----------|-------|
| **ML Task** | Exploratory Data Analysis |
| **Dataset Status** | OK LOCAL |

## 2. Dataset

**Data sources detected in code:**

- `shared_articles.csv`
- `users_interactions.csv`

**Files in project directory:**

- `shared_articles.csv`
- `users_interactions.csv`

**Standardized data path:** `data/articles_rec_system/`

## 3. Pipeline Overview

### Original Notebook Pipeline

**Preprocessing:**
- Drop columns/rows
- Remove duplicate rows
- Index setting
- Data type conversion
- Feature scaling (MinMaxScaler)

## 4. ML Workflow

```mermaid
flowchart TD
    A[Load shared_articles.csv] --> B[Drop columns/rows]
    B[Drop columns/rows] --> C[Remove duplicate rows]
    C[Remove duplicate rows] --> D[Index setting]
    D[Index setting] --> E[Data type conversion]
    E[Data type conversion] --> F[Feature scaling]
```

## 5. Notebook Summary

| Metric | Value |
|--------|-------|
| Total cells | 37 |
| Code cells | 36 |
| Markdown cells | 1 |

## 6. Model Details

No model training in this project.

## 7. Project Structure

```
articles_rec_system/
├── Articles Rec System Implicit.ipynb
├── shared_articles.csv
├── users_interactions.csv
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

- Open `Articles Rec System Implicit.ipynb` and run all cells

## 10. Testing

Automated tests are available in `tests/test_p102_*.py`:

```bash
python -m pytest tests/test_p102_*.py -v
```

Tests validate data loading and library imports.

## 11. Limitations

- No model training — this is an analysis/tutorial notebook only
