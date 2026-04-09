# Fraud Detection using the IEEE-CIS Fraud Detection Dataset

## Documentation
- [Fraud detection using the IEEE-CIS Fraud Detection.pdf](Fraud%20detection%20using%20the%20IEEE-CIS%20Fraud%20Detection.pdf)

## Dataset
- **Source**: [IEEE-CIS Fraud Detection](https://www.kaggle.com/datasets/lnasiri007/ieeecis-fraud-detection)
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
PyCaret AutoML with fix_imbalance on a 50k row sample of the IEEE-CIS dataset for scalable fraud classification.
