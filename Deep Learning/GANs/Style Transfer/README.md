# Style Transfer

## Documentation
- [Style Transfer.pdf](Style%20Transfer.pdf)

## Dataset
- **Source**: [WikiArt](https://www.kaggle.com/datasets/steubk/wikiart)
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
- content_loss_final
- style_loss_final
- num_epochs_trained

## Approach
VGG19 neural style transfer