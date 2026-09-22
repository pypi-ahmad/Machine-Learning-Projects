# E-Commerce Recommendation Demo

A local Streamlit demonstration of user-based collaborative filtering. It creates a deterministic synthetic rating matrix and ranks products a selected user has not rated.

## Requirements

- Python 3.13+
- [uv](https://docs.astral.sh/uv/)

## Run

From this directory:

```powershell
uv sync
uv run streamlit run main.py
```

Choose a user to view the generated ratings and product ranking.

## How it works

- Missing ratings are represented as empty cells.
- The app calculates cosine similarity between users from zero-filled rating vectors.
- It predicts an unrated product from positively similar users who rated it.

## Important limits

- All users, products, and ratings are synthetic. Rankings are illustrative and are not evidence of customer preferences or business value.
- The score is a simple weighted average, not a calibrated probability or expected revenue estimate.
- Do not use this demo for personalization, pricing, eligibility, or inventory decisions without appropriate real-data evaluation and safeguards.

## Verification

```powershell
uv run python -m py_compile main.py
uv lock --check
```
