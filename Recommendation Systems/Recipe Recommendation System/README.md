# Recipe Recommendation System

## Documentation
- [Recipe recommendation system.pdf](Recipe%20recommendation%20system.pdf)
- [8 Recipe Recommendation System.docx](8%20Recipe%20Recommendation%20System.docx)

## Dataset
- **Source**: [Recipes Dataset 64K Dishes](https://www.kaggle.com/datasets/prashantsingh001/recipes-dataset-64k-dishes)
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
NCF with PyCaret fallback
