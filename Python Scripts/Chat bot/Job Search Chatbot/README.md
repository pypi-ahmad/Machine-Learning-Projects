# Job Search Chatbot

## Documentation
- [Job Search Chatbot.pdf](Job%20Search%20Chatbot.pdf)

## Dataset
- **Source**: [LinkedIn Job Postings](https://www.kaggle.com/datasets/arshkon/linkedin-job-postings)
- **Download**: Automatic via `opendatasets` (requires Kaggle API key)

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
- accuracy
- macro_f1
- weighted_f1
- auc

## Approach
DistilBERT fine-tuning