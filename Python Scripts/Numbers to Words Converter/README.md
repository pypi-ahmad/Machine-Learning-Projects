# Numbers to Words Converter

Convert signed integers into English words, from zero through nonillion.

## Run

```powershell
uv sync --no-config
uv run --no-config python converter.py 12345
```

The command prints:

```text
Twelve Thousand Three Hundred And Forty Five
```

Run without an argument for interactive mode:

```powershell
uv run --no-config python converter.py
```

Enter an integer, or type `exit` to stop.

## Limits

- Supported input range: -999,999,999,999,999,999,999,999,999,999,999 through 999,999,999,999,999,999,999,999,999,999,999.
- Decimal values, ordinals, currency formatting, and locale-specific wording are not supported.
- The tool uses only the Python standard library and does not write files or use a network connection.
