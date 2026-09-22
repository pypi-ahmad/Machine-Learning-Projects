# HTML Minifier

Command-line HTML minifier for local files or pasted markup.

```powershell
uv sync --no-config
uv run --no-config python main.py index.html --output dist\index.html
```

Use `--stats` to inspect savings without writing output. Minification can change markup semantics, especially with `--aggressive`; review generated files before publishing. Requires Python 3.13 or later and no third-party packages.
