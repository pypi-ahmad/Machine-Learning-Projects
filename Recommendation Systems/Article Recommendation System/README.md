# Article Recommendation System

## Documentation
- [Article Recommendation System.pdf](Article%20Recommendation%20System.pdf)
- [10 Article Recommendation System.docx](10%20Article%20Recommendation%20System.docx)

## Dataset
- **Source**: [MIND News Dataset](https://www.kaggle.com/datasets/arashnic/mind-news-dataset)
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
- mae, rmse, r2

## Approach
PyTorch NCF (MIND news)
