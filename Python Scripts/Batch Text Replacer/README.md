# Batch Text Replacer

Preview text replacements across a directory, then apply them explicitly.

```powershell
cd "Python Scripts/Batch Text Replacer"
uv sync
uv run python main.py . --find old --replace new --ext .txt
uv run python main.py . --find old --replace new --ext .txt --apply
```

The first command is a dry run. Applied changes create `.bak` files unless
`--no-backup` is supplied.
