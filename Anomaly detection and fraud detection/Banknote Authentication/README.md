# Banknote Authentication using the Banknote Authentication Dataset

## Documentation
- [Banknote authentication using the Banknote Authentication dataset.pdf](Banknote%20authentication%20using%20the%20Banknote%20Authentication%20dataset.pdf)

## Dataset
- **Source**: [Bank Note Authentication Dataset](https://www.kaggle.com/datasets/cdr0101/bank-note-authentication-dataset)
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
PyCaret AutoML pipeline for binary classification of authentic vs. forged banknotes based on image-derived features.
