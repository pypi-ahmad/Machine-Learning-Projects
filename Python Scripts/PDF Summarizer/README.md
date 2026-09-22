# PDF and Text Summarizer

A local Streamlit app that produces extractive summaries from pasted text, text files, and PDFs with embedded text.

## Run

```powershell
uv sync --no-config
uv run --no-config streamlit run main.py
```

Choose a sample, paste text, or upload a `.txt` or `.pdf` document. Set the summary ratio and maximum sentence count, then select **Summarize**.

## Behavior and limits

- The app scores sentences with local TF-IDF-style term weighting. It selects original sentences and does not generate or rewrite text.
- PDF support extracts embedded text with `pypdf`. Scanned or image-only PDFs require OCR before upload.
- Documents are processed by the running local app and are not sent to an external model or API.
- Summary quality depends on clear sentence boundaries and the source text. The keyword list reflects frequency, not semantic understanding.
