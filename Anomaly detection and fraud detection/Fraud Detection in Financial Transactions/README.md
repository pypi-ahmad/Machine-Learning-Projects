# Fraud Detection in Financial Transactions

## Documentation
- [Fraud Detection in Financial Transactions.pdf](Fraud%20Detection%20in%20Financial%20Transactions.pdf)

## Dataset
- **Source**: [Bank Transaction Dataset for Fraud Detection](https://www.kaggle.com/datasets/valakhorasani/bank-transaction-dataset-for-fraud-detection)
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
PyCaret AutoML with fix_imbalance for binary classification of fraudulent vs. legitimate bank transactions.
