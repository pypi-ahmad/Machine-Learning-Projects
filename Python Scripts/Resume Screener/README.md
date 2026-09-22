# Resume Screener

A small Streamlit demo that ranks uploaded resumes against a job description by
comparing normalized term-frequency vectors with cosine similarity. It is a
learning tool, not an applicant-tracking system or a hiring decision tool.

## Run

```powershell
uv sync
uv run streamlit run main.py
```

Open the local address Streamlit prints. The app starts with sample resumes, or
you can paste a job description and upload `.txt` or `.pdf` files.

## How scoring works

1. The app tokenizes the job description and each resume.
2. It removes a small set of common English words.
3. It creates normalized term-frequency vectors from the shared vocabulary.
4. It ranks candidates by cosine similarity and lists overlapping keywords.

The score measures keyword overlap only. It does not verify skills, infer
seniority, or account for qualifications that are expressed differently.

## PDF limitation

PDF uploads use a dependency-free byte-pattern extraction method. It may miss,
garble, or omit text from compressed, scanned, or image-based PDFs. Convert a
resume to text before uploading when reliable extraction matters.

## Dependencies

Dependencies are managed with uv in `pyproject.toml` and locked in `uv.lock`.
The project needs Python 3.14 or later, pandas, and Streamlit.
