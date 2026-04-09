# Detecting Fraudulent Credit Card Transactions

## Documentation
- [Detecting fraudulent credit card transactions.pdf](Detecting%20fraudulent%20credit%20card%20transactions.pdf)

## Dataset
- **Source**: [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
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
PyCaret AutoML with fix_imbalance to handle the highly skewed class distribution in credit card fraud detection.
