# (Conceptual) Working with Databases

## 1. Project Overview

This project implements a **Exploratory Data Analysis** pipeline for **(Conceptual) Working with Databases**.

| Property | Value |
|----------|-------|
| **ML Task** | Exploratory Data Analysis |
| **Dataset Status** | OK LOCAL |

## 2. Dataset

**Files in project directory:**

- `iris_test.csv`
- `iris_train.csv`

**Standardized data path:** `data/conceptual_working_with_databases/`

## 3. Pipeline Overview

The original notebook primarily contains data loading and exploratory data analysis.

## 4. ML Workflow

```mermaid
flowchart TD
    A[Load Dataset] --> B[Explore / Visualize]
```

## 5. Notebook Summary

| Metric | Value |
|--------|-------|
| Total cells | 30 |
| Code cells | 18 |
| Markdown cells | 12 |

## 6. Model Details

No model training in this project.

## 7. Project Structure

```
(Conceptual) Working with Databases/
├── Working with Databases.ipynb
├── iris_test.csv
├── iris_train.csv
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

- Open `Working with Databases.ipynb` and run all cells

## 10. Testing

Automated tests are available in `tests/test_p130_*.py`:

```bash
python -m pytest tests/test_p130_*.py -v
```

Tests validate data loading and library imports.

## 11. Limitations

- No model training — this is an analysis/tutorial notebook only
