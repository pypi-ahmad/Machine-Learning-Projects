# Online News Association Rule Learning

## Documentation
- [Online News.pdf](Online%20News.pdf)

## Dataset
- **Source**: [All the News](https://www.kaggle.com/datasets/snapcrack/all-the-news)
- **Download**: Automatic via `opendatasets` (requires Kaggle API key)

## Run
```bash
python run.py                    # full training
python run.py --smoke-test       # quick validation (1 epoch / tiny subset)
python run.py --download-only    # download dataset only
python run.py --epochs 10        # custom epochs
python run.py --device cpu       # force CPU
python run.py --no-amp           # disable mixed precision
```

## Metrics Output
Results saved to `outputs/metrics.json`:
- num_rules_apriori
- num_rules_fpgrowth
- avg_support
- avg_confidence
- avg_lift

## Approach
Apriori and FP-Growth algorithms (mlxtend) applied to TF-IDF transformed news articles to discover keyword association rules.
