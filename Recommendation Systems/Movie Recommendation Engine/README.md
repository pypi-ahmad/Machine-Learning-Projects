# Movie Recommendation Engine

## 1. Project Overview

This project implements a **Exploratory Data Analysis** pipeline for **Movie Recommendation Engine**.

| Property | Value |
|----------|-------|
| **ML Task** | Exploratory Data Analysis |
| **Dataset Status** | BLOCKED MISSING |

## 2. Dataset

> ⚠️ **Dataset not available locally.** movies.csv + ratings.csv (from data/movie_dataset/)

## 3. Pipeline Overview

### Original Notebook Pipeline

**Preprocessing:**
- Drop columns/rows

## 4. ML Workflow

```mermaid
flowchart TD
    A[Load Dataset] --> B[Drop columns/rows]
```

## 5. Notebook Summary

| Metric | Value |
|--------|-------|
| Total cells | 27 |
| Code cells | 15 |
| Markdown cells | 12 |

## 6. Model Details

No model training in this project.

## 7. Project Structure

```
Movie Recommendation Engine/
├── Movie_Recommendation_Engine.ipynb
└── README.md
```

## 8. Setup & Installation

`pip install -r requirements.txt` from the workspace root.

**Key dependencies:**

- `numpy`
- `pandas`
- `scipy`

## 9. How to Run

Open and run the notebook(s) sequentially:

```bash
jupyter notebook
```

- Open `Movie_Recommendation_Engine.ipynb` and run all cells

## 10. Testing

Automated tests are available in `tests/test_p092_*.py`:

```bash
python -m pytest tests/test_p092_*.py -v
```

Tests validate data loading and library imports.

## 11. Limitations

- Dataset is not available locally — notebook cannot run without manual data setup
- No model training — this is an analysis/tutorial notebook only
