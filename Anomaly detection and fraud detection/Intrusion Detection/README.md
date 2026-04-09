# Intrusion Detection

## Documentation
- [Intrusion Detection.pdf](Intrusion%20Detection.pdf)

## Dataset
- **Source**: [NSL-KDD Dataset](https://www.kaggle.com/datasets/harivmv/nsl-kdd-dataset)
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
PyCaret AutoML classification pipeline to detect network intrusions using the NSL-KDD benchmark dataset.
