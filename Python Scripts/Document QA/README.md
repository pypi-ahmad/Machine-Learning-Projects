# Document Q&A

A local Streamlit demo that retrieves relevant sentences from a pasted or uploaded UTF-8 text document using TF-IDF and cosine similarity. It does not generate answers or call an external model.

## Requirements

- Python 3.13+
- [uv](https://docs.astral.sh/uv/)

## Run

From this directory:

```powershell
uv sync
uv run streamlit run main.py
```

Load a sample, paste text, or upload a UTF-8 `.txt` file. Then ask a question to retrieve the most relevant source sentences.

## Privacy and limits

- Documents and Q&A history remain in the active Streamlit session; this project does not persist or send them to a remote service.
- Results are keyword-based retrieval, not factual reasoning. A high relevance score does not prove that a sentence answers the question.
- The app accepts plain UTF-8 text only. It does not extract text from PDF, Word, image, or scanned documents.

## Project files

```text
main.py         # Streamlit application and retrieval logic
pyproject.toml  # uv dependency definition
uv.lock         # Resolved dependency versions
```

## Verification

```powershell
uv run python -m py_compile main.py
uv lock --check
```
