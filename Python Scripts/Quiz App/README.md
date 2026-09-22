# Quiz App

A local Streamlit multiple-choice quiz with scoring, answer review, and optional JSON uploads.

## Setup

```powershell
uv sync --no-config
```

## Run

```powershell
uv run --no-config streamlit run main.py
```

Use the sample quiz or upload a JSON list whose questions include `question`, `options`, and `answer`; `explanation` is optional.

The app runs locally and does not persist quiz answers or make network requests.
