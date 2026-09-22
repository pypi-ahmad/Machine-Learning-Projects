# GitHub Profile Viewer

Command-line viewer for public GitHub profile and repository metadata through the authenticated GitHub CLI.

## Run

```powershell
uv sync --no-config
gh auth login
uv run --no-config python main.py --user octocat --repos
```

Use `--top` to limit displayed repositories. Omit `--user` for interactive mode.

## Requirements

- Python 3.13 or later.
- GitHub CLI (`gh`) installed and authenticated with `gh auth login`.

The app performs read-only GitHub API requests. It does not store profile data or credentials.
