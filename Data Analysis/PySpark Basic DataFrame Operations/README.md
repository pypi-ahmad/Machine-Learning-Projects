# PySpark_Basic_DataFrame_Operations

## 1. Project Overview

This project implements a **Exploratory Data Analysis** pipeline for **PySpark_Basic_DataFrame_Operations**.

| Property | Value |
|----------|-------|
| **ML Task** | Exploratory Data Analysis |
| **Dataset Status** | OK LOCAL |

## 2. Dataset

**Files in project directory:**

- `apple.csv`

**Standardized data path:** `data/pyspark_basic_dataframe_operations/`

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
| Total cells | 67 |
| Code cells | 63 |
| Markdown cells | 4 |

## 6. Model Details

No model training in this project.

## 7. Project Structure

```
PySpark_Basic_DataFrame_Operations/
├── PySpark_Basic_DataFrame_Operations.ipynb
├── apple.csv
└── README.md
```

## 8. Setup & Installation

`pip install -r requirements.txt` from the workspace root.

## 9. How to Run

Open and run the notebook(s) sequentially:

```bash
jupyter notebook
```

- Open `PySpark_Basic_DataFrame_Operations.ipynb` and run all cells

## 10. Testing

Automated tests are available in `tests/test_p116_*.py`:

```bash
python -m pytest tests/test_p116_*.py -v
```

Tests validate data loading and library imports.

## 11. Limitations

- No model training — this is an analysis/tutorial notebook only
