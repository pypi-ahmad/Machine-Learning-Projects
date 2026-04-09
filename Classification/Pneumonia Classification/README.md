# Pneumonia Classification

## 1. Project Overview

This project implements a **Classification** pipeline for **Pneumonia Classification**.

| Property | Value |
|----------|-------|
| **ML Task** | Classification |
| **Dataset Status** | BLOCKED MISSING |

## 2. Dataset

> ⚠️ **Dataset not available locally.** chest_xray images (Kaggle: chest-xray-pneumonia)

## 3. Pipeline Overview

### Original Notebook Pipeline

**Preprocessing:**
- Image data augmentation
- Image loading from directory

**Models trained:**
- Sequential

**Evaluation metrics:**
- Accuracy (Keras)
- Custom accuracy function
- Validation loss/accuracy
- Training loss tracking

## 4. ML Workflow

```mermaid
flowchart TD
    A[Load Dataset] --> B[Image data augmentation]
    B[Image data augmentation] --> C[Image loading from directory]
    C[Image loading from directory] --> D[Train: Sequential]
    D[Train: Sequential] --> E[Evaluate: Accuracy (Keras), Custom accuracy function]
```

## 5. Notebook Summary

| Metric | Value |
|--------|-------|
| Total cells | 20 |
| Code cells | 20 |
| Markdown cells | 0 |
| Original models | Sequential |

## 6. Model Details

### Original Models

- `Sequential`

**Neural network architecture:**

```
  Dense(64)
  Dense(32)
  Dense(1)
  Conv2D(32)
  Conv2D(64)
  Flatten
```

### Evaluation Metrics

- Accuracy (Keras)
- Custom accuracy function
- Validation loss/accuracy
- Training loss tracking

## 7. Project Structure

```
Pneumonia Classification/
├── pneumonia.ipynb
└── README.md
```

## 8. Setup & Installation

`pip install -r requirements.txt` from the workspace root.

**Key dependencies:**

- `Pillow`
- `keras`
- `matplotlib`
- `numpy`
- `tensorflow`

## 9. How to Run

Open and run the notebook(s) sequentially:

```bash
jupyter notebook
```

- Open `pneumonia.ipynb` and run all cells

## 10. Testing

Automated tests are available in `tests/test_p030_*.py`:

```bash
python -m pytest tests/test_p030_*.py -v
```

Tests validate data loading and model instantiation.

## 11. Limitations

- Dataset is not available locally — notebook cannot run without manual data setup
- Hardcoded file paths detected — may need adjustment
