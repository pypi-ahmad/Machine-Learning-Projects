# Image-to-Image Translation

## Documentation
- [Image-to-Image Translation.pdf](Image-to-Image%20Translation.pdf)

## Dataset
- **Source**: [Pix2Pix Facades Dataset](https://www.kaggle.com/datasets/sabahesaraki/pix2pix-facades-dataset)
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
Pix2Pix (U-Net generator + PatchGAN discriminator)