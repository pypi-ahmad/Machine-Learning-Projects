# Traffic Sign Recognizer

## 1. Project Overview

This project implements a **Image Classification** pipeline for **Traffic Sign Recognizer**.

| Property | Value |
|----------|-------|
| **ML Task** | Image Classification |
| **Dataset Status** | BLOCKED LINK ONLY |

## 2. Dataset

**Data sources detected in code:**

- `GT-final_test.csv`

> ⚠️ **Dataset not available locally.** Link-only but no downloadable URL identified

## 3. Pipeline Overview

### Original Notebook Pipeline

**Preprocessing:**
- Train/test split
- Image data augmentation

**Models trained:**
- Sequential

**Evaluation metrics:**
- Accuracy (Keras)
- Validation loss/accuracy
- Training loss tracking

## 4. ML Workflow

```mermaid
flowchart TD
    A[Load GT-final_test.csv] --> B[Train/test split]
    B[Train/test split] --> C[Image data augmentation]
    C[Image data augmentation] --> D[Train: Sequential]
    D[Train: Sequential] --> E[Evaluate: Accuracy (Keras), Validation loss/accuracy]
```

## 5. Notebook Summary

| Metric | Value |
|--------|-------|
| Total cells | 43 |
| Code cells | 23 |
| Markdown cells | 20 |
| Original models | Sequential |

## 6. Model Details

### Original Models

- `Sequential`

**Neural network architecture:**

```
  Dense(512)
  Conv2D(32)
  Conv2D(64)
  Conv2D(128)
  MaxPooling
  Dropout(0.2)
  Dropout(0.5)
  Flatten
```

### Evaluation Metrics

- Accuracy (Keras)
- Validation loss/accuracy
- Training loss tracking

## 7. Project Structure

```
Traffic Sign Recognizer/
├── Traffic-Sign-Recognition.ipynb
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

## 9. How to Run

Open and run the notebook(s) sequentially:

```bash
jupyter notebook
```

- Open `Traffic-Sign-Recognition.ipynb` and run all cells

## 10. Testing

Automated tests are available in `tests/test_p094_*.py`:

```bash
python -m pytest tests/test_p094_*.py -v
```

Tests validate data loading and model instantiation.

## 11. Limitations

- Dataset is not available locally — notebook cannot run without manual data setup
