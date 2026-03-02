# 2016 General Election Poll Analysis

## 1. Project Overview

This project implements a **Exploratory Data Analysis** pipeline for **2016 General Election Poll Analysis**.

| Property | Value |
|----------|-------|
| **ML Task** | Exploratory Data Analysis |
| **Dataset Status** | OK LOCAL |

## 2. Dataset

**Standardized data path:** `data/2016_general_election_poll_analysis/`

## 3. Pipeline Overview

### Original Notebook Pipeline

**Preprocessing:**
- Drop columns/rows
- DataFrame concatenation

## 4. ML Workflow

```mermaid
flowchart TD
    A[Load Dataset] --> B[Drop columns/rows]
    B[Drop columns/rows] --> C[DataFrame concatenation]
```

## 5. Notebook Summary

| Metric | Value |
|--------|-------|
| Total cells | 32 |
| Code cells | 19 |
| Markdown cells | 13 |

## 6. Model Details

No model training in this project.

## 7. Project Structure

```
2016 General Election Poll Analysis/
├── 2016 General Election Poll Analysis.ipynb
├── data
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

- Open `2016 General Election Poll Analysis.ipynb` and run all cells

## 10. Testing

Automated tests are available in `tests/test_p072_*.py`:

```bash
python -m pytest tests/test_p072_*.py -v
```

Tests validate data loading and library imports.

## 11. Limitations

- No model training — this is an analysis/tutorial notebook only
