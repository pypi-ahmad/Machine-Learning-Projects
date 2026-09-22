# Package Checker

A command-line utility for comparing installed package versions or requirement-file entries with current PyPI metadata.

## Run

```powershell
uv sync --no-config
uv run --no-config python main.py --help
```

Check one package or inspect a requirements file:

```powershell
uv run --no-config python main.py --check requests
uv run --no-config python main.py --requirements .\requirements.txt
```

Use `--outdated` to query every package installed in the interpreter that runs the command.

## Behavior and limits

- Each package check makes a read-only request to PyPI's JSON API, with a five-second timeout.
- `--outdated` can make many requests and may be slow or subject to PyPI rate limits.
- The tool compares version strings and reports package metadata; it does not install, upgrade, remove, or audit packages for security vulnerabilities.
- Run it with `uv run` to inspect the project's uv environment rather than an unrelated global Python installation.
