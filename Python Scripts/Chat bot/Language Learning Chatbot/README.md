# Language Learning Chatbot

## Documentation
- [Language Learning Chatbot.pdf](Language%20Learning%20Chatbot.pdf)

## Dataset used
- **Source**: [English Sentences](https://www.kaggle.com/datasets/mayakaripel/eng-sentences)
- **Download**: Automatic through the Kaggle CLI when available, with an `opendatasets` fallback.
- **Credentials**: Set `KAGGLE_API_TOKEN` before downloading. The token is never stored in this project.

## Run the project
This project uses the repository's shared uv environment because `run.py` imports `shared.utils`. From the repository root:

```powershell
uv sync
uv run python "Python Scripts/Chat bot/Language Learning Chatbot/run.py" --mode smoke
uv run python "Python Scripts/Chat bot/Language Learning Chatbot/run.py" --mode full
uv run python "Python Scripts/Chat bot/Language Learning Chatbot/run.py" --download-only
uv run python "Python Scripts/Chat bot/Language Learning Chatbot/run.py" --seed 42
```

`--mode smoke` is the default and uses a small sample. Use `--mode full` only when you intend to download and process the complete dataset.

## Metrics
Results are written to `outputs/metrics.json`. Reported metrics depend on the selected AutoML classifier and the available target classes.

## Approach
PyCaret + TF-IDF on sentence features

## Labeling limitation

When the dataset has no categorical label column, the project creates labels from sentence length. Those labels are a reproducible demonstration target, not an assessment of language proficiency or learning progress.
