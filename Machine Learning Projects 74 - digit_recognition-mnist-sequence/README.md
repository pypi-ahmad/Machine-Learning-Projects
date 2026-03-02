# digit_recognition-mnist-sequence

## 1. Project Overview

This project implements a **Computer Vision** pipeline for **digit_recognition-mnist-sequence**.

| Property | Value |
|----------|-------|
| **ML Task** | Computer Vision |
| **Dataset Status** | OK LOCAL |

## 2. Dataset

**Standardized data path:** `data/digit_recognition-mnist-sequence/`

## 3. Pipeline Overview

### Original Notebook Pipeline

**Preprocessing:**
- Data type conversion

**Evaluation metrics:**
- Accuracy (Keras)

## 4. ML Workflow

```mermaid
flowchart TD
    A[Load Dataset] --> B[Data type conversion]
    B[Data type conversion] --> C[Evaluate: Accuracy (Keras)]
```

## 5. Notebook Summary

| Metric | Value |
|--------|-------|
| Total cells | 39 |
| Code cells | 26 |
| Markdown cells | 13 |

## 6. Model Details

### Evaluation Metrics

- Accuracy (Keras)

No model training in this project.

## 7. Project Structure

```
digit_recognition-mnist-sequence/
├── digit_recognition-mnist-sequence.ipynb
├── data
└── README.md
```

## 8. Setup & Installation

`pip install -r requirements.txt` from the workspace root.

**Key dependencies:**

- `keras`
- `matplotlib`
- `numpy`
- `scipy`
- `tensorflow`

## 9. How to Run

Open and run the notebook(s) sequentially:

```bash
jupyter notebook
```

- Open `digit_recognition-mnist-sequence.ipynb` and run all cells

## 10. Testing

Automated tests are available in `tests/test_p074_*.py`:

```bash
python -m pytest tests/test_p074_*.py -v
```

Tests validate data loading and library imports.

## 11. Limitations

- No model training — this is an analysis/tutorial notebook only
