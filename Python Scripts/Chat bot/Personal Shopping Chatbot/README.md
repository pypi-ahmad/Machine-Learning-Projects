# Personal Shopping Chatbot

## Documentation
- [Personal Shopping Chatbot.pdf](Personal%20Shopping%20Chatbot.pdf)

## Dataset
- **Source**: [Amazon Fine Food Reviews](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews)
- **Download**: Automatic through the Kaggle CLI when available, with an `opendatasets` fallback.
- **Credentials**: Set `KAGGLE_API_TOKEN` before downloading. The token is never stored in this project.

## Run
This project uses the repository's shared uv environment because `run.py` imports `shared.utils`. From the repository root:

```powershell
uv sync
uv run python "Python Scripts/Chat bot/Personal Shopping Chatbot/run.py" --mode smoke
uv run python "Python Scripts/Chat bot/Personal Shopping Chatbot/run.py" --mode full
uv run python "Python Scripts/Chat bot/Personal Shopping Chatbot/run.py" --download-only
uv run python "Python Scripts/Chat bot/Personal Shopping Chatbot/run.py" --epochs 10
uv run python "Python Scripts/Chat bot/Personal Shopping Chatbot/run.py" --device cpu
uv run python "Python Scripts/Chat bot/Personal Shopping Chatbot/run.py" --no-amp
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
