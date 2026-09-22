# Text Summarizer

An offline extractive summarizer that scores sentences from word frequency and a small position bonus. It does not call external APIs or download models.

## Run it

```powershell
uv sync
uv run python main.py
```

To summarize a file directly:

```powershell
uv run python main.py article.txt --sentences 3
```

The input must be plain text. The `--sentences` value must be at least one.

## Limits

This is a simple extractive method: it selects sentences from the source rather than generating a new synopsis. Its output can miss context, nuance, or important low-frequency terms, so review it against the original text.

## Dependencies

- Python 3.14+
- No third-party packages
