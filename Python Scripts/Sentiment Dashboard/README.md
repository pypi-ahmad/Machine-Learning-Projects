# Sentiment Dashboard

A Streamlit learning demo for inspecting the sentiment of individual text,
review batches, and a bundled sample set. It uses a small hand-maintained
lexicon rather than a trained machine-learning model.

## Run

```powershell
uv sync
uv run streamlit run main.py
```

## How it works

The app tokenizes text, looks for words from positive and negative lists, and
adjusts their contribution for nearby negations and intensifiers. It reports a
normalized score, label, matched words, and a heuristic confidence value.

The confidence value is not a calibrated probability. Results can be wrong for
sarcasm, context, domain-specific terms, mixed opinions, and words not in the
small built-in lexicon. Do not use this demo as the sole basis for a business,
employment, credit, health, or other consequential decision.

## Dependencies

uv manages pandas and Streamlit in `pyproject.toml`; exact resolved versions
are recorded in `uv.lock`.
