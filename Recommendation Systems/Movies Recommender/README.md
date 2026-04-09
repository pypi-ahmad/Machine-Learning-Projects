# Movies Recommendation system

## 1. Project Overview

This project implements a **Exploratory Data Analysis** pipeline for **Movies Recommendation system**.

| Property | Value |
|----------|-------|
| **ML Task** | Exploratory Data Analysis |
| **Dataset Status** | OK LOCAL |

## 2. Dataset

**Data sources detected in code:**

- `movies.csv`
- `ratings.csv`

**Files in project directory:**

- `movies.csv`
- `ratings.csv`

**Standardized data path:** `data/movies_recommendation_system/`

## 3. Pipeline Overview

### Original Notebook Pipeline

**Preprocessing:**
- Train/test split

## 4. ML Workflow

```mermaid
flowchart TD
    A[Load movies.csv] --> B[Train/test split]
```

## 5. Notebook Summary

| Metric | Value |
|--------|-------|
| Total cells | 16 |
| Code cells | 13 |
| Markdown cells | 3 |

## 6. Model Details

No model training in this project.

## 7. Project Structure

```
Movies Recommendation system/
├── Movielens Recommender Metrics.ipynb
├── movies.csv
├── ratings.csv
└── README.md
```

## 8. Setup & Installation

`pip install -r requirements.txt` from the workspace root.

**Key dependencies:**

- `pandas`

## 9. How to Run

Open and run the notebook(s) sequentially:

```bash
jupyter notebook
```

- Open `Movielens Recommender Metrics.ipynb` and run all cells

## 10. Testing

Automated tests are available in `tests/test_p146_*.py`:

```bash
python -m pytest tests/test_p146_*.py -v
```

Tests validate data loading and library imports.

## 11. Limitations

- No model training — this is an analysis/tutorial notebook only
