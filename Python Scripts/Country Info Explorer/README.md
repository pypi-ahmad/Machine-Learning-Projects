# Country Info Explorer

Country Info Explorer is a terminal client for the public REST Countries API.

```powershell
uv sync
uv run python main.py --country Germany
uv run python main.py --code US
uv run python main.py --region Europe
```

Run without arguments for interactive mode. Results depend on the live REST Countries service and its current data; no API key is required.

Verify syntax with `uv run python -m py_compile main.py`.
