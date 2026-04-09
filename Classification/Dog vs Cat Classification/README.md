# Dog Vs Cat Classification

## 1. Project Overview

This project implements a **Classification** pipeline for **Dog Vs Cat Classification**.

| Property | Value |
|----------|-------|
| **ML Task** | Classification |
| **Dataset Status** | BLOCKED MISSING |

## 2. Dataset

> ⚠️ **Dataset not available locally.** Dog vs Cat image dataset (Kaggle: dogs-vs-cats)

## 3. Pipeline Overview

### Original Notebook Pipeline

**Preprocessing:**
- Image data augmentation
- Image loading from directory

**Models trained:**
- VGG16 (pretrained)

**Evaluation metrics:**
- Accuracy (Keras)

## 4. ML Workflow

```mermaid
flowchart TD
    A[Load Dataset] --> B[Image data augmentation]
    B[Image data augmentation] --> C[Image loading from directory]
    C[Image loading from directory] --> D[Train: VGG16]
    D[Train: VGG16] --> E[Evaluate: Accuracy (Keras)]
```

## 5. Notebook Summary

| Metric | Value |
|--------|-------|
| Total cells | 19 |
| Code cells | 17 |
| Markdown cells | 2 |
| Original models | VGG16 |

## 6. Model Details

### Original Models

- `VGG16 (pretrained)`

**Neural network architecture:**

```
  Dense(1)
  Flatten
```

### Evaluation Metrics

- Accuracy (Keras)

## 7. Project Structure

```
Dog Vs Cat Classification/
├── main.ipynb
└── README.md
```

## 8. Setup & Installation

`pip install -r requirements.txt` from the workspace root.

**Key dependencies:**

- `keras`
- `matplotlib`
- `numpy`
- `tensorflow`

## 9. How to Run

Open and run the notebook(s) sequentially:

```bash
jupyter notebook
```

- Open `main.ipynb` and run all cells

## 10. Testing

Automated tests are available in `tests/test_p024_*.py`:

```bash
python -m pytest tests/test_p024_*.py -v
```

Tests validate data loading and model instantiation.

## 11. Limitations

- Dataset is not available locally — notebook cannot run without manual data setup
- Hardcoded file paths detected — may need adjustment
- No train/test split detected in code
