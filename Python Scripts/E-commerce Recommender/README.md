# E-commerce Product Recommender

A local Streamlit learning demo with two recommendation methods: user-based collaborative filtering over synthetic ratings and content-based similarity over product tags.

## Requirements

- Python 3.13+
- [uv](https://docs.astral.sh/uv/)

## Run

From this directory:

```powershell
uv sync
uv run streamlit run main.py
```

Use the tabs to view user recommendations, product-tag similarity, catalog filtering, and generated rating statistics.

## What the demo does

- Generates the same synthetic catalog and ratings on every run.
- Predicts unrated products from ratings by positively similar users.
- Ranks similar products using TF-IDF vectors built from tag lists.
- Keeps all data in memory for the current session.

## Important limits

- Products, users, prices, and ratings are illustrative only. Scores are not real customer preferences, probabilities, or business forecasts.
- Content similarity only reflects hand-written tags; collaborative scores only reflect the generated matrix.
- Do not use this project for production personalization, pricing, eligibility, or inventory decisions without real-data evaluation, governance, and monitoring.

## Verification

```powershell
uv run python -m py_compile main.py
uv lock --check
```
