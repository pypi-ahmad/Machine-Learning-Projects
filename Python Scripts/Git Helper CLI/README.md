# Git Helper CLI

Interactive shortcuts for common local Git inspection and user-confirmed actions.

## Run

```powershell
uv sync --no-config
uv run --no-config python main.py
```

Read-only shortcuts include `status`, `log`, `diff`, `branches`, `stash-list`, `info`, and `search`:

```powershell
uv run --no-config python main.py status
```

## Safety

- Run the helper from inside the repository you intend to inspect.
- Commits use only changes already staged by the user; the helper never runs `git add -A`.
- Commit, reset, branch deletion, pull, and push require typed confirmation.
- Branch deletion uses Git's safe `-d` mode and never force-deletes branches.
