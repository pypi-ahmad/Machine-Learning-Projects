# Grocery Store Association Rule Learning

## Documentation
- [Grocery Store.pdf](Grocery%20Store.pdf)

## Dataset
- **Source**: [Market Basket Analysis](https://www.kaggle.com/datasets/aslanahmedov/market-basket-analysis)
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
Apriori and FP-Growth algorithms (mlxtend) to discover frequent itemsets and association rules from grocery store transactions.
