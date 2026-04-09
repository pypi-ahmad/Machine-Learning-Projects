# Face Generation

## Documentation
- [Face generation.pdf](Face%20generation.pdf)

## Dataset
- **Source**: [CelebA Dataset](https://www.kaggle.com/datasets/jessicali9530/celeba-dataset)
- **Download**: Automatic via `opendatasets` (requires Kaggle API key)

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
- g_loss_final
- d_loss_final
- num_epochs_trained

## Approach
DCGAN with spectral normalization + AMP