# Audio Denoising

## Documentation
- [Audio Denoising.pdf](Audio%20Denoising.pdf)

## Dataset
- **Source**: [Denoising Audio Collection](https://www.kaggle.com/datasets/sayuksh/denoising-audio-collection)
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
- avg_snr_improvement, final_train_loss, final_val_loss

## Approach
PyTorch U-Net spectrogram denoiser
