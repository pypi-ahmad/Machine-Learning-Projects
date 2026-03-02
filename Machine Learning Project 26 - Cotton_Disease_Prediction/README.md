# Cotton_Disease_Prediction

## 1. Project Overview

This project implements a **Regression** pipeline for **Cotton_Disease_Prediction**.

| Property | Value |
|----------|-------|
| **ML Task** | Regression |
| **Dataset Status** | OK LOCAL |

## 2. Dataset

**Standardized data path:** `data/cotton_disease_prediction/`

## 3. Pipeline Overview

### Original Notebook Pipeline

**Preprocessing:**
- Image data augmentation
- Image loading from directory

**Models trained:**
- InceptionV3 (pretrained)
- ResNet50 (pretrained)

**Evaluation metrics:**
- Accuracy (Keras)
- Validation loss/accuracy
- Training loss tracking

## 4. ML Workflow

```mermaid
flowchart TD
    A[Load Dataset] --> B[Image data augmentation]
    B[Image data augmentation] --> C[Image loading from directory]
    C[Image loading from directory] --> D[Train: InceptionV3, ResNet50]
    D[Train: InceptionV3, ResNet50] --> E[Evaluate: Accuracy (Keras), Validation loss/accuracy]
```

## 5. Notebook Summary

| Metric | Value |
|--------|-------|
| Total cells | 42 |
| Code cells | 41 |
| Markdown cells | 1 |
| Original models | InceptionV3, ResNet50 |

## 6. Model Details

### Original Models

- `InceptionV3 (pretrained)`
- `ResNet50 (pretrained)`

**Neural network architecture:**

```
  Flatten
```

### Evaluation Metrics

- Accuracy (Keras)
- Validation loss/accuracy
- Training loss tracking

## 7. Project Structure

```
Cotton_Disease_Prediction/
├── Cotton_Disease_Resnet50.ipynb
├── Cotton_Disease_inceptionv3.ipynb
├── test
├── val
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

- Open `Cotton_Disease_inceptionv3.ipynb` and run all cells
- Open `Cotton_Disease_Resnet50.ipynb` and run all cells

## 10. Testing

Automated tests are available in `tests/test_p026_*.py`:

```bash
python -m pytest tests/test_p026_*.py -v
```

Tests validate data loading and model instantiation.

## 11. Limitations

- Hardcoded file paths detected — may need adjustment
- Contains Google Colab artifacts
