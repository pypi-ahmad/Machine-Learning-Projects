# Movie Recommendation System

## Documentation
- [Movie recommendation system.pdf](Movie%20recommendation%20system.pdf)
- [5 Movie Recommendation.docx](5%20Movie%20Recommendation.docx)

## Dataset
- **Source**: [MovieLens Latest Datasets](https://www.kaggle.com/datasets/ciroexe/movielens-latest-datasets)
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
PyTorch NCF (MovieLens)
