# Image Captioning

## Documentation
- [Image Captioning.pdf](Image%20Captioning.pdf)

## Dataset
- **Source**: [COCO 2017 Dataset](https://www.kaggle.com/datasets/awsaf49/coco-2017-dataset)
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
- val_loss, bleu

## Approach
ViT + GPT-2 (HuggingFace)
