# Document Q&A Demo

A local Streamlit demonstration that ranks document sentences by keyword overlap with a question. It is a retrieval demo, not an LLM or factual question-answering system.

## Requirements

- Python 3.13+
- [uv](https://docs.astral.sh/uv/)

## Run

From this directory:

```powershell
uv sync
uv run streamlit run main.py
```

Replace the sample text with your own document, enter a question, and select **Find passages**.

## Limits and privacy

- Matching is based on shared keywords after punctuation is removed. It does not understand synonyms, context, or truth.
- The app neither saves nor sends the text you enter; it is used only in the active session.
- Do not treat retrieved passages as a complete answer without reading the source document.

## Verification

```powershell
uv run python -m py_compile main.py
uv lock --check
```
