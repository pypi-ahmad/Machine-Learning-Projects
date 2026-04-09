# Event Recommendation System

## Documentation
- [Event Recommendation System.pdf](Event%20Recommendation%20System.pdf)
- [1 Event Recommendation system.docx](1%20Event%20Recommendation%20system.docx)

## Dataset
- **Source**: [Event Dataset](https://www.kaggle.com/datasets/kilanisikiru/event-dataset)
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
PyTorch Neural Collaborative Filtering
