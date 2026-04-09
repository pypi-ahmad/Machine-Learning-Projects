# Restaurant Recommendation System

## Documentation
- [Restaurant Recommendation System.pdf](Restaurant%20Recommendation%20System.pdf)
- [3 Restaurant recommendation system.docx](3%20Restaurant%20recommendation%20system.docx)

## Dataset
- **Source**: [Restaurant Data with Consumer Ratings](https://www.kaggle.com/datasets/uciml/restaurant-data-with-consumer-ratings)
- **Download**: Automatically downloaded via Kaggle API on first run

## Run
```bash
python run.py                    # full training
python run.py --smoke-test       # quick validation
python run.py --download-only    # download dataset only
python run.py --epochs 10        # custom epochs
python run.py --device cpu       # force CPU
python run.py --no-amp           # disable mixed precision
```

## Metrics
Results in `outputs/metrics.json`:
- mae, rmse, r2

## Approach
PyTorch NCF
