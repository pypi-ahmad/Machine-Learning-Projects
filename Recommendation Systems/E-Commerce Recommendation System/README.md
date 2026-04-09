# E-commerce Product Recommendation System

## Documentation
- [E-commerce Product Recommendation.pdf](E-commerce%20Product%20Recommendation.pdf)
- [2. E-commerce Product recommendation.docx](2.%20E-commerce%20Product%20recommendation.docx)

## Dataset
- **Source**: [eCommerce Behavior Data from Multi Category Store](https://www.kaggle.com/datasets/mkechinov/ecommerce-behavior-data-from-multi-category-store)
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
