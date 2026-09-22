# Health and Wellness Chatbot

## Documentation
- [Health and Wellness Chatbot.pdf](Health%20and%20Wellness%20Chatbot.pdf)

## Dataset used
- **Source**: [MedQuAD Medical Question Answer for AI Research](https://www.kaggle.com/datasets/pythonafroz/medquad-medical-question-answer-for-ai-research)
- **Download**: Automatic through the Kaggle CLI when available, with an `opendatasets` fallback.
- **Credentials**: Set `KAGGLE_API_TOKEN` before downloading. The token is never stored in this project.

## Run the project
This project uses the repository's shared uv environment because `run.py` imports `shared.utils`. From the repository root:

```powershell
uv sync
uv run python "Python Scripts/Chat bot/Health and Wellness Chatbot/run.py" --mode smoke
uv run python "Python Scripts/Chat bot/Health and Wellness Chatbot/run.py" --mode full
uv run python "Python Scripts/Chat bot/Health and Wellness Chatbot/run.py" --download-only
uv run python "Python Scripts/Chat bot/Health and Wellness Chatbot/run.py" --epochs 10
uv run python "Python Scripts/Chat bot/Health and Wellness Chatbot/run.py" --device cpu
uv run python "Python Scripts/Chat bot/Health and Wellness Chatbot/run.py" --no-amp
```

`--mode smoke` is the default and uses a small sample for one epoch. Use `--mode full` only when you intend to download and train on the complete dataset.

## Metrics
Results in `outputs/metrics.json`:
- accuracy
- macro_f1
- weighted_f1

The run also writes a confusion matrix and a text classification report.

## Approach
DistilBERT fine-tuning

## Safety

This learning project classifies question topics only. It does not provide medical advice, diagnosis, or treatment guidance and must not be used for clinical decisions.
