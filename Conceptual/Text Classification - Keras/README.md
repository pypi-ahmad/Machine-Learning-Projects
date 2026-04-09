# (Conceptual) Text Classification Keras

## 1. Project Overview

This project implements a **Classification** pipeline for **(Conceptual) Text Classification Keras**.

| Property | Value |
|----------|-------|
| **ML Task** | Classification |
| **Dataset Status** | BLOCKED MISSING |

## 2. Dataset

**Data sources detected in code:**

- `Consumer_Complaints.csv`

> ⚠️ **Dataset not available locally.** Consumer_Complaints.csv

## 3. Pipeline Overview

### Original Notebook Pipeline

**Preprocessing:**
- Text tokenization (Keras)
- Label encoding (LabelEncoder)

**Models trained:**
- Sequential

**Evaluation metrics:**
- Accuracy (Keras)

## 4. ML Workflow

```mermaid
flowchart TD
    A[Load Consumer_Complaints.csv] --> B[Text tokenization]
    B[Text tokenization] --> C[Label encoding]
    C[Label encoding] --> D[Train: Sequential]
    D[Train: Sequential] --> E[Evaluate: Accuracy (Keras)]
```

## 5. Notebook Summary

| Metric | Value |
|--------|-------|
| Total cells | 18 |
| Code cells | 18 |
| Markdown cells | 0 |
| Original models | Sequential |

## 6. Model Details

### Original Models

- `Sequential`

**Neural network architecture:**

```
  Dense(512)
  Dropout(0.5)
```

### Evaluation Metrics

- Accuracy (Keras)

## 7. Project Structure

```
(Conceptual) Text Classification Keras/
├── Text Classification Keras.ipynb
└── README.md
```

## 8. Setup & Installation

`pip install -r requirements.txt` from the workspace root.

**Key dependencies:**

- `keras`
- `matplotlib`
- `numpy`
- `pandas`
- `scikit-learn`
- `tensorflow`

## 9. How to Run

Open and run the notebook(s) sequentially:

```bash
jupyter notebook
```

- Open `Text Classification Keras.ipynb` and run all cells

## 10. Testing

Automated tests are available in `tests/test_p124_*.py`:

```bash
python -m pytest tests/test_p124_*.py -v
```

Tests validate data loading and model instantiation.

## 11. Limitations

- Dataset is not available locally — notebook cannot run without manual data setup
