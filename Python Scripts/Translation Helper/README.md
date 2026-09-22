# Translation Helper

A terminal client for text translation through the public MyMemory API. It can translate supplied text and print the supported language-code list.

## Run it

```powershell
uv sync
uv run python main.py
```

For a one-off request:

```powershell
uv run python main.py --text "Hello, world!" --to es
uv run python main.py --text "Bonjour" --from fr --to en
uv run python main.py --list-languages
```

## Notes

Translation requests are sent to MyMemory over the network. The service requires no key for basic usage, but availability, quotas, privacy terms, and translation quality are controlled by the external provider. Avoid sending sensitive or confidential text, and review translations before using them.

## Dependencies

- Python 3.14+
- No third-party packages
