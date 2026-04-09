# Book Recommendation System

## Documentation
- [Book Recommendation System.pdf](Book%20Recommendation%20System.pdf)
- [7 Book recommendation system.docx](7%20Book%20recommendation%20system.docx)

## Dataset
- **Source**: [goodbooks-10k](https://www.kaggle.com/datasets/zygmunt/goodbooks-10k)
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
PyTorch NCF
