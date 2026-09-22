# GitHub Repo Viewer

Read-only command-line viewer for GitHub repository metadata, languages, commits, issues, and contributors.

```powershell
uv sync --no-config
gh auth login
uv run --no-config python main.py --repo python/cpython --commits --issues
```

Requires Python 3.13 or later and the authenticated GitHub CLI (`gh`). Use `--n` to limit listed records. The tool does not create, edit, or store GitHub data.
