# Number Base Converter

An interactive terminal tool for converting integers between common and custom bases from 2 through 36. It also shows repeated-division steps, two's-complement representations, ASCII codes, and IEEE 754 float fields.

## Run

```powershell
uv sync --no-config
uv run --no-config python main.py
```

Choose an option from the menu and enter the requested value. Integer inputs may include `0x`, `0o`, or `0b` prefixes where Python accepts them.

## Notes

- All calculations are local and use the Python standard library.
- Custom base digits use `0-9` followed by `A-Z`.
- Two's-complement output does not enforce a fixed integer range; choose a bit width that fits the value you want to represent.
- IEEE 754 output describes a 32-bit single-precision representation, so very large or precise input values can be rounded.
