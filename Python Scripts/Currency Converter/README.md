# Currency Converter

Currency Converter is a local CLI that converts currencies using the public ExchangeRate-API endpoint. If the live request fails, it uses the bundled fallback rate table.

```powershell
uv sync
uv run python main.py 100 USD EUR
uv run python main.py
```

The interactive menu supports individual conversions, currency listing, and a comparison table. Live and fallback values can be delayed or stale and are for informational use only, not financial advice.

`main.py` is the supported entrypoint. The older `cc.py` is retained as legacy code and is not used by this workflow.

Verify syntax with `uv run python -m py_compile main.py`.
