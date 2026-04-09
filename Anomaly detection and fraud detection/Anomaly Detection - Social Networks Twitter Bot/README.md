# Anomaly Detection in Social Networks Twitter Bot

## Documentation
- [Anomaly Detection in Social Networks Twitter Bot.pdf](Anomaly%20Detection%20in%20Social%20Networks%20Twitter%20Bot.pdf)

## Dataset
- **Source**: [Twitter Bot Detection Dataset](https://www.kaggle.com/datasets/goyaladi/twitter-bot-detection-dataset)
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
PyCaret AutoML classification pipeline to detect Twitter bot accounts based on social network features.
