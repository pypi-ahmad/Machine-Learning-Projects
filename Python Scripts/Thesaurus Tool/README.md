# Thesaurus Tool

A terminal client for the public Datamuse API. It can list synonyms, antonyms, related words, rhymes, and sound-alike words.

## Run it

```powershell
uv sync
uv run python main.py
```

For a one-off lookup:

```powershell
uv run python main.py --word happy
uv run python main.py --word fast --antonyms
uv run python main.py --word time --rhymes
```

The interactive prompt also accepts `<word>`, `ant <word>`, `rhyme <word>`, `sounds <word>`, and `quit`.

## Notes

Results come from Datamuse over the network. The API needs no key, but availability and returned terms are controlled by that service. This tool makes no request until you ask for a lookup.

## Dependencies

- Python 3.14+
- No third-party packages
