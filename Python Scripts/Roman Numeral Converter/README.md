# Roman Numeral Converter

An interactive command-line tool for converting and validating standard Roman
numerals from `I` through `MMMCMXCIX` (1 to 3999).

## Run

```powershell
uv run python main.py
```

Choose an option to convert an integer, convert a Roman numeral, explain a
valid numeral, print a range table, or convert several integers in one session.
Enter `0` to quit.

## Rules enforced

- Integers must be between 1 and 3999.
- Roman input is converted back to standard form to reject invalid sequences
  such as `IIV` or `MMMM`.
- Input is case-insensitive and surrounding whitespace is ignored.

## Dependencies

The project uses only Python's standard library. uv manages the Python version
and locks the reproducible project environment.
