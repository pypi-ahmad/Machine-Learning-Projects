# Breast Cancer Detection using the Breast Cancer Wisconsin Dataset

## Documentation
- [Breast cancer detection using the Breast Cancer Wisconsin dataset.pdf](Breast%20cancer%20detection%20using%20the%20Breast%20Cancer%20Wisconsin%20dataset.pdf)

## Dataset
- **Source**: [Breast Cancer Wisconsin Data](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data)
- **Download**: Automatic via `opendatasets` (requires Kaggle API key)

## Run
```bash
python run.py                    # full training
python run.py --smoke-test       # quick validation (1 epoch / tiny subset)
python run.py --download-only    # download dataset only
python run.py --epochs 10        # custom epochs
python run.py --device cpu       # force CPU
python run.py --no-amp           # disable mixed precision
```

## Metrics Output
Results saved to `outputs/metrics.json`:
- accuracy
- macro_f1
- weighted_f1
- auc

## Approach
PyCaret AutoML pipeline for binary classification of malignant vs. benign breast tumors using cell nucleus features.
