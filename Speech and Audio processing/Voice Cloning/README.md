# Voice Cloning

## Documentation
- [Voice Cloning.pdf](Voice%20Cloning.pdf)

## Dataset
- **Source**: [English Multispeaker Corpus for Voice Cloning](https://www.kaggle.com/datasets/mfekadu/english-multispeaker-corpus-for-voice-cloning)
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
- val_loss

## Approach
SpeechT5 (HuggingFace) speaker embeddings
