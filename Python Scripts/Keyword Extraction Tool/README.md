# Keyword Extraction Tool

A standard-library CLI that ranks single-word keywords and short phrases with frequency, sentence-level TF-IDF, and RAKE-inspired scores.

```powershell
uv sync --no-config
uv run --no-config python main.py article.txt --top 10
```

Run without a file to use the interactive input and built-in demonstration text. The tool processes text locally and makes no network requests.
