# GitHub Repo Automation

Preview or explicitly create one private-by-default GitHub repository and a matching empty local folder.

## Setup

```powershell
uv sync --no-config
gh auth login
```

Requires Python 3.13 or later and the authenticated GitHub CLI. No Python packages are required.

## Preview

```powershell
uv run --no-config python GitHub.py my-new-project --directory "D:\Projects"
```

Preview validates the repository name and local destination without creating anything.

## Create

```powershell
uv run --no-config python GitHub.py my-new-project --directory "D:\Projects" --apply --confirm CREATE
```

The command creates a private remote repository and then an empty local directory at `D:\Projects\my-new-project`. Add `--visibility public` only when the repository is intended to be public.

## Limits

- The target parent directory must already exist and the destination folder must not exist.
- The script does not initialize Git locally, clone the repository, or push files.
- If remote creation succeeds but local directory creation fails, the remote remains; the script reports that partial result.
- No credentials are stored in the source code. GitHub CLI authentication manages credentials.
