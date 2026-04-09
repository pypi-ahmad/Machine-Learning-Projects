# Music Recommendation System

## Documentation
- [Music Recommendation System.pdf](Music%20Recommendation%20System.pdf)
- [6 Music Recommendation system.docx](6%20Music%20Recommendation%20system.docx)

## Dataset
- **Source**: [Spotify Tracks Dataset](https://www.kaggle.com/datasets/gauthamvijayaraj/spotify-tracks-dataset-updated-every-week)
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
NCF with PyCaret fallback
