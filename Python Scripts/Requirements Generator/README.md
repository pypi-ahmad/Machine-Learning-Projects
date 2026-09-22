# Requirements Generator

An interactive terminal tool that scans Python imports and can write a legacy `requirements.txt` for a selected project.

## Setup

```powershell
uv sync --no-config
```

## Run

```powershell
uv run --no-config python main.py .
```

Scan before generating. Options 2 and 3 write `requirements.txt` into the scanned project, so review the detected imports and target path first. This generator infers packages from imports and cannot guarantee an exact dependency graph; prefer each project’s `pyproject.toml` and `uv.lock` when available.
