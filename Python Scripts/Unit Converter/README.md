# Unit Converter

A terminal converter for length, mass, temperature, volume, speed, time, area, digital storage, and pressure units.

## Run it

```powershell
uv sync
uv run python main.py
```

Enter conversions in this format:

```text
<value> <from_unit> <to_unit>
```

Examples:

```text
100 km mile
32 f c
1 gb mb
```

Type `list` to show supported units, or `quit` to exit. Units are case-insensitive.

## Dependencies

- Python 3.14+
- No third-party packages
