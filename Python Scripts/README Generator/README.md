# README Generator

An interactive terminal wizard that builds a project README from supplied metadata and light project detection.

## Setup

```powershell
uv sync --no-config
```

## Run

```powershell
uv run --no-config python main.py --project . --template minimal --preview
```

Use `--preview` to print generated Markdown without writing a file. Without it, the tool prompts before overwriting the selected output path. Templates are `full` and `minimal`.

## Notes

Review generated commands, badges, links, claims, and license details before publishing. The tool runs locally and makes no network requests.
