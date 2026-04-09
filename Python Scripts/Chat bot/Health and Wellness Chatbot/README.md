# Health and Wellness Chatbot

## Documentation
- [Health and Wellness Chatbot.pdf](Health%20and%20Wellness%20Chatbot.pdf)

## Dataset
- **Source**: [MedQuAD Medical Question Answer for AI Research](https://www.kaggle.com/datasets/pythonafroz/medquad-medical-question-answer-for-ai-research)
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