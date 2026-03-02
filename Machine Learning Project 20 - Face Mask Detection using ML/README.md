# Face Mask Detection using ML

## 1. Project Overview

This project implements a **Classification** pipeline for **Face Mask Detection using ML**.

| Property | Value |
|----------|-------|
| **ML Task** | Classification |
| **Dataset Status** | BLOCKED KAGGLE |

## 2. Dataset

> ⚠️ **Dataset not available locally.** kaggle: ashishjangra27/face-mask-12k-images-dataset

## 3. Pipeline Overview

### Original Notebook Pipeline

**Preprocessing:**
- Image augmentation
- Normalization
- Image loading from directory (ImageFolder)
- Batch data loading (DataLoader)

**Models trained:**
- FaceMaskDetec (Custom PyTorch)

**Evaluation metrics:**
- Custom accuracy function
- Validation loss/accuracy

## 4. ML Workflow

```mermaid
flowchart TD
    A[Load Dataset] --> B[Image augmentation]
    B[Image augmentation] --> C[Normalization]
    C[Normalization] --> D[Image loading from directory]
    D[Image loading from directory] --> E[Batch data loading]
    E[Batch data loading] --> F[Train: FaceMaskDetec]
    F[Train: FaceMaskDetec] --> G[Evaluate: Custom accuracy function, Validation loss/accuracy]
```

## 5. Notebook Summary

| Metric | Value |
|--------|-------|
| Total cells | 25 |
| Code cells | 24 |
| Markdown cells | 1 |
| Original models | FaceMaskDetec |

## 6. Model Details

### Original Models

- `FaceMaskDetec (Custom PyTorch)`

**Neural network architecture:**

```
  Conv2d(3)
  Conv2d(16)
  Conv2d(32)
  MaxPooling
  BatchNorm
  Flatten
```

### Evaluation Metrics

- Custom accuracy function
- Validation loss/accuracy

## 7. Project Structure

```
Face Mask Detection using ML/
├── notebookf9ab511482.ipynb
└── README.md
```

## 8. Setup & Installation

`pip install -r requirements.txt` from the workspace root.

**Key dependencies:**

- `Pillow`
- `matplotlib`
- `torch`
- `torchvision`

## 9. How to Run

Open and run the notebook(s) sequentially:

```bash
jupyter notebook
```

- Open `notebookf9ab511482.ipynb` and run all cells

## 10. Testing

Automated tests are available in `tests/test_p020_*.py`:

```bash
python -m pytest tests/test_p020_*.py -v
```

Tests validate data loading and model instantiation.

## 11. Limitations

- Dataset is not available locally — notebook cannot run without manual data setup
- No train/test split detected in code
