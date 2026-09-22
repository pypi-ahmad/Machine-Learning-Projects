# Weather Chatbot

## Documentation
- [Weather Chatbot.pdf](Weather%20Chatbot.pdf)

## Dataset used
- **Source**: [Historical Hourly Weather Data](https://www.kaggle.com/datasets/selfishgene/historical-hourly-weather-data)
- **Download**: Automatic through the Kaggle CLI when available, with an `opendatasets` fallback.
- **Credentials**: Set `KAGGLE_API_TOKEN` before downloading. The token is never stored in this project.

## Run the project
This project uses the repository's shared uv environment because `run.py` imports `shared.utils`. From the repository root:

```powershell
uv sync
uv run python "Python Scripts/Chat bot/Weather Chatbot/run.py" --mode smoke
uv run python "Python Scripts/Chat bot/Weather Chatbot/run.py" --mode full
uv run python "Python Scripts/Chat bot/Weather Chatbot/run.py" --download-only
uv run python "Python Scripts/Chat bot/Weather Chatbot/run.py" --seed 42
```

`--mode smoke` is the default and uses a small sample. Use `--mode full` only when you intend to download and process the complete dataset.

## Metrics
Results are written to `outputs/metrics.json`. Reported metrics depend on the selected AutoML classifier and the available target classes.

## Approach
PyCaret tabular classification

## Modeling scope

Temperature classes are derived from historical temperature readings. The model uses city, month, and hour features only; it does not use the source temperature or derived values as predictors. This is a historical classification exercise, not a weather forecasting service.
