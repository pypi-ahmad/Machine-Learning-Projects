# Slug Generator

A command-line tool that converts titles and other text into URL-style slugs.
It can create individual slugs, show common variants, and process multiple
titles in one session.

## Run

```powershell
uv run python main.py
```

Choose an option from the menu. For a custom slug, set the separator, optional
maximum length, and whether common English stop words should be removed.

## Behavior

- Creates kebab, snake, dot, no-stop-word, length-limited, and uppercase
  variants.
- Normalizes combining diacritics and maps a small set of special characters,
  such as `&` to `and` and `ß` to `ss`.
- Retains word characters recognized by Python; it is not a complete
  transliteration system for every writing system.
- Truncation cuts the generated slug at the requested character limit and
  removes a trailing separator.

## Dependencies

The project uses only Python's standard library. uv records the Python
requirement and provides the reproducible environment.
