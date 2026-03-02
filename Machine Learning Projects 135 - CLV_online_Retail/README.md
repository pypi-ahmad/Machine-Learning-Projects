# CLV_online_Retail

## 1. Project Overview

This project implements a **Exploratory Data Analysis** pipeline for **CLV_online_Retail**.

| Property | Value |
|----------|-------|
| **ML Task** | Exploratory Data Analysis |
| **Dataset Status** | OK LOCAL |

## 2. Dataset

**Data sources detected in code:**

- `Online_Retail.xlsx`

**Files in project directory:**

- `Online Retail.xlsx`

**Standardized data path:** `data/clv_online_retail/`

## 3. Pipeline Overview

### Original Notebook Pipeline

**Preprocessing:**
- Date parsing

## 4. ML Workflow

```mermaid
flowchart TD
    A[Load Online_Retail.xlsx] --> B[Date parsing]
```

## 5. Notebook Summary

| Metric | Value |
|--------|-------|
| Total cells | 47 |
| Code cells | 26 |
| Markdown cells | 21 |

## 6. Model Details

No model training in this project.

## 7. Project Structure

```
CLV_online_Retail/
├── CLV_Online_Retail.ipynb
├── Online Retail.xlsx
└── README.md
```

## 8. Setup & Installation

`pip install -r requirements.txt` from the workspace root.

**Key dependencies:**

- `pandas`
- `seaborn`

## 9. How to Run

Open and run the notebook(s) sequentially:

```bash
jupyter notebook
```

- Open `CLV_Online_Retail.ipynb` and run all cells

## 10. Testing

Automated tests are available in `tests/test_p135_*.py`:

```bash
python -m pytest tests/test_p135_*.py -v
```

Tests validate data loading and library imports.

## 11. Limitations

- No model training — this is an analysis/tutorial notebook only
