# Personal Finance Chatbot

## Documentation
- [Personal Finance Chatbot.pdf](Personal%20Finance%20Chatbot.pdf)

## Dataset
- **Source**: [Bitext Training Dataset for Chatbots](https://www.kaggle.com/datasets/bitext/training-dataset-for-chatbotsvirtual-assistants)
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