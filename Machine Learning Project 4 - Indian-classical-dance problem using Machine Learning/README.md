# Indian-classical-dance problem using Machine Learning

## 1. Project Overview

This project implements a **Image Classification** pipeline for **Indian-classical-dance problem using Machine Learning**.

| Property | Value |
|----------|-------|
| **ML Task** | Image Classification |
| **Dataset Status** | BLOCKED MISSING |

## 2. Dataset

> ⚠️ **Dataset not available locally.** Indian classical dance image dataset

## 3. Pipeline Overview

### Original Notebook Pipeline

**Preprocessing:**
- Image data augmentation
- Image loading from directory
- Data type conversion
- Image augmentation

**Models trained:**
- Sequential

**Evaluation metrics:**
- Accuracy (Keras)
- Validation loss/accuracy
- Training loss tracking

## 4. ML Workflow

```mermaid
flowchart TD
    A[Load Dataset] --> B[Image data augmentation]
    B[Image data augmentation] --> C[Image loading from directory]
    C[Image loading from directory] --> D[Data type conversion]
    D[Data type conversion] --> E[Image augmentation]
    E[Image augmentation] --> F[Train: Sequential]
    F[Train: Sequential] --> G[Evaluate: Accuracy (Keras), Validation loss/accuracy]
```

## 5. Notebook Summary

| Metric | Value |
|--------|-------|
| Total cells | 22 |
| Code cells | 20 |
| Markdown cells | 2 |
| Original models | Sequential |

## 6. Model Details

### Original Models

- `Sequential`

**Neural network architecture:**

```
  Dense(256)
  Conv2D(32)
  Conv2D(64)
  Conv2D(128)
  Dropout(0.2)
  Flatten
```

### Evaluation Metrics

- Accuracy (Keras)
- Validation loss/accuracy
- Training loss tracking

## 7. Project Structure

```
Indian-classical-dance problem using Machine Learning/
├── main.ipynb
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

- Open `main.ipynb` and run all cells

## 10. Testing

Automated tests are available in `tests/test_p004_*.py`:

```bash
python -m pytest tests/test_p004_*.py -v
```

Tests validate data loading and model instantiation.

## 11. Limitations

- Dataset is not available locally — notebook cannot run without manual data setup
- Hardcoded file paths detected — may need adjustment
