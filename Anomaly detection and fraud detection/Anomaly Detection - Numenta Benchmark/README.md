# Anomaly Detection using the Numenta Anomaly Benchmark

## Documentation
- [Anomaly detection using the Numenta Anomaly Benchmark.pdf](Anomaly%20detection%20using%20the%20Numenta%20Anomaly%20Benchmark.pdf)

## Dataset
- **Source**: [Numenta Anomaly Benchmark (NAB)](https://www.kaggle.com/datasets/boltzmannbrain/nab)
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
- anomaly_ratio
- threshold
- num_anomalies

## Approach
PyTorch LSTM Autoencoder that learns normal time-series patterns and flags deviations as anomalies using reconstruction error.
