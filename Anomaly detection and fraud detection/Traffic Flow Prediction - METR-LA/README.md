# Traffic Flow Prediction using the METR-LA Traffic Dataset

## Documentation
- [Traffic flow prediction using the METR-LA traffic.pdf](Traffic%20flow%20prediction%20using%20the%20METR-LA%20traffic.pdf)

## Dataset
- **Source**: [METR-LA Dataset](https://www.kaggle.com/datasets/annnnguyen/metr-la-dataset)
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
- mae
- rmse
- r2

## Approach
PyTorch LSTM model for time-series regression, predicting traffic flow speeds across Los Angeles highway sensors.
