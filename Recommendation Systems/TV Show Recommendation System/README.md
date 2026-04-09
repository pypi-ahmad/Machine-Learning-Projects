# TV Show Recommendation System

## Documentation
- [TV Show Recommendation System.pdf](TV%20Show%20Recommendation%20System.pdf)
- [11 TV Shows recommendation system.docx](11%20TV%20Shows%20recommendation%20system.docx)

## Dataset
- **Source**: [10000 Popular TV Shows Dataset (TMDB)](https://www.kaggle.com/datasets/riteshswami08/10000-popular-tv-shows-dataset-tmdb)
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
PyCaret Regression
