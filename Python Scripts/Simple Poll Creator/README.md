# Simple Poll Creator

A local Streamlit app for creating polls, voting on their options, and viewing
live counts and percentages.

## Run

```powershell
uv sync
uv run streamlit run main.py
```

Create a question and enter one option per line. Each poll needs at least two
distinct options; duplicate questions are rejected to avoid resetting votes.

## Data storage

Polls and vote counts are stored in `polls.json` beside `main.py`. The file is
created after the first successful change and remains on the local machine.
Deleting a poll in the app permanently removes it from that file.

This is a local demo, not a multi-user polling system. It has no authentication,
voter identity, duplicate-vote prevention, concurrent-write protection, or
tamper resistance. Do not use it for binding, sensitive, or high-stakes votes.

## Dependencies

uv manages pandas and Streamlit in `pyproject.toml`; exact resolved versions
are recorded in `uv.lock`.
