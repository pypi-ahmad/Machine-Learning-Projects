# Fraud Detection in Insurance Claims using the Insurance Fraud Dataset

## Documentation
- [Fraud detection in insurance claims using the Insurance Fraud.pdf](Fraud%20detection%20in%20insurance%20claims%20using%20the%20Insurance%20Fraud.pdf)

## Dataset
- **Source**: [Insurance Fraud Detection](https://www.kaggle.com/datasets/arpan129/insurance-fraud-detection)
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
PyCaret AutoML pipeline for detecting fraudulent insurance claims based on policy and claim features.
