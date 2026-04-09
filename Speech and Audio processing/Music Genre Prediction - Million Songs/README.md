# Predicting Music Genres from Audio Data

## Documentation
- [Predicting music genres from audio data using the Million Songs.pdf](Predicting%20music%20genres%20from%20audio%20data%20using%20the%20Million%20Songs.pdf)

## Dataset
- **Source**: [Spotify Million Song Dataset](https://www.kaggle.com/datasets/notshrirang/spotify-million-song-dataset)
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
- accuracy, macro_f1, weighted_f1, auc

## Approach
PyCaret on audio features
