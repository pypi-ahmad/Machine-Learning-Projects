# Random Animal Image

Fetch random dog, cat, fox, or duck image URLs from public APIs.

## Setup

```powershell
uv sync --no-config
```

## Run

```powershell
uv run --no-config python main.py --animal dog
uv run --no-config python main.py --animal cat --fact
uv run --no-config python main.py --animal fox --count 3
uv run --no-config python main.py --breeds
```

Run without arguments for interactive prompts.

## Notes

The tool prints image URLs; it does not download images. It uses public live APIs, so it requires internet access and results depend on those services. No API key is required.
