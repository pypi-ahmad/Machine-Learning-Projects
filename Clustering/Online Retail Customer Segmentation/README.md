# 3 Online retail customer segmentation

![Category](https://img.shields.io/badge/Category-Clustering-blue)
![Task](https://img.shields.io/badge/Task-Clustering-green)
![Dataset](https://img.shields.io/badge/Dataset-Tabular-orange)
![AutoML](https://img.shields.io/badge/AutoML-Enabled-purple)

## Project Overview

**3 Online retail customer segmentation** is a **Clustering** project in the **Clustering** category.

> As a first step, I load all the modules that will be used in this notebook:

**Models:** KMeans, PyCaret

## Dataset

| Property | Value |
|----------|-------|
| Type | Tabular |
| Source | Local |
| Path | `data/online_retail_segmentation/data.csv` |
| Fallback | `manual_required` |

```python
from core.data_loader import load_dataset
df = load_dataset('online_retail_customer_segmentation')
```

## Pipeline Files

| File | Lines |
|------|-------|
| `pipeline.py` | 490 |
| `train.py` | 435 |
| `evaluate.py` | 468 |
| `3 Online retail customer segmentation.ipynb` | 36 code / 38 markdown cells |
| `test_online_retail_customer_segmentation.py` | test suite |

## ML Workflow

```mermaid
graph TD
    A[Load Dataset] --> B[Exploratory Data Analysis]
    B --> C[Preprocessing]
    C --> D[Feature Engineering]
    D --> E[Clustering]
    E --> F[AutoML Comparison]
    F --> G[Cluster Evaluation]
    G --> H[Visualization]
```

## Core Logic

### Preprocessing

- Missing value imputation
- StandardScaler normalization
- Datetime feature extraction

### Feature Engineering

Feature engineering steps detected in notebook code cells.

### Visualizations

- Confusion matrix
- Word cloud
- Elbow method
- Silhouette analysis

## Models

| Model | Type |
|-------|------|
| KMeans | Centroid Clustering |
| PyCaret | AutoML Framework |

AutoML is toggled via the `USE_AUTOML` flag in pipeline scripts.
**PyCaret** `compare_models()` runs cross-validated comparison.

## Reproducibility

```python
random.seed(42); np.random.seed(42); os.environ['PYTHONHASHSEED'] = '42'
```

```bash
python pipeline.py --seed 123    # custom seed
python pipeline.py --reproduce   # locked seed=42
```

## Project Structure

```
Clustering/3 Online retail customer segmentation/
  3 Online retail customer segmentation.docx
  3 Online retail customer segmentation.ipynb
  Online Retail customer segmentation.pdf
  README.md
  evaluate.py
  pipeline.py
  test_online_retail_customer_segmentation.py
  train.py
```

## How to Run

```bash
cd "Clustering/3 Online retail customer segmentation"
python pipeline.py
python train.py       # training only
python evaluate.py    # evaluation only
```

## Testing

```bash
pytest "Clustering/3 Online retail customer segmentation/test_online_retail_customer_segmentation.py" -v
```

## Setup

```bash
pip install matplotlib nltk numpy pandas pycaret scikit-learn seaborn wordcloud
```

## Limitations

- Dataset requires manual download — not included in repository

---
*README auto-generated from `3 Online retail customer segmentation.ipynb` analysis.*