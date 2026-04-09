# Language Learning Chatbot

## Documentation
- [Language Learning Chatbot.pdf](Language%20Learning%20Chatbot.pdf)

## Dataset
- **Source**: [English Sentences](https://www.kaggle.com/datasets/mayakaripel/eng-sentences)
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
PyCaret + TF-IDF on sentence features