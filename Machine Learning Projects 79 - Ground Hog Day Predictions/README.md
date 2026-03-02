# Ground Hog Day Predictions

## 1. Project Overview

This project implements a **Regression** pipeline for **Ground Hog Day Predictions**.

| Property | Value |
|----------|-------|
| **ML Task** | Regression |
| **Dataset Status** | OK LOCAL |

## 2. Dataset

**Data sources detected in code:**

- `dataset.csv`

**Files in project directory:**

- `dataset.csv`
- `link_to_dataset.txt`

**Standardized data path:** `data/ground_hog_day_predictions/`

## 3. Pipeline Overview

### Original Notebook Pipeline

**Preprocessing:**
- Drop columns/rows

## 4. ML Workflow

```mermaid
flowchart TD
    A[Load dataset.csv] --> B[Drop columns/rows]
```

## 5. Notebook Summary

| Metric | Value |
|--------|-------|
| Total cells | 15 |
| Code cells | 15 |
| Markdown cells | 0 |

## 6. Model Details

No model training in this project.

## 7. Project Structure

```
Ground Hog Day Predictions/
├── ground-hog-day-predictions(1).ipynb
├── dataset.csv
├── link_to_dataset.txt
└── README.md
```

## 8. Setup & Installation

`pip install -r requirements.txt` from the workspace root.

**Key dependencies:**

- `matplotlib`
- `pandas`

## 9. How to Run

Open and run the notebook(s) sequentially:

```bash
jupyter notebook
```

- Open `ground-hog-day-predictions(1).ipynb` and run all cells

## 10. Testing

Automated tests are available in `tests/test_p079_*.py`:

```bash
python -m pytest tests/test_p079_*.py -v
```

Tests validate data loading and library imports.

## 11. Limitations

- No model training — this is an analysis/tutorial notebook only
