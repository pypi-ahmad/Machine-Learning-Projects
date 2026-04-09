# Anomaly Detection in Images CIFAR-10

## Documentation
- [Anomaly Detection in Images CIFAR-10.pdf](Anomaly%20Detection%20in%20Images%20CIFAR-10.pdf)

## Dataset
- **Source**: torchvision CIFAR-10 (auto-download, no credentials required)
- **Download**: Automatic via `torchvision.datasets`

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
timm EfficientNet-B0 binary anomaly classifier trained on CIFAR-10 images, distinguishing a target class from the rest as anomalies.
