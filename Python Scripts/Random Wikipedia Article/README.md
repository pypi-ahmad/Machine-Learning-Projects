# Random Wikipedia Article

Save the title and readable paragraph text from one random English Wikipedia article.

## Setup

Requirements: Python 3.13+.

```powershell
cd "Random Wikipedia Article"
uv sync
```

## Usage

```powershell
uv run python wiki_random.py --output random_wiki.txt
```

Options:

- `--output PATH`: Destination text file. Default: `random_wiki.txt`.
- `--overwrite`: Replace an existing output file. Without it, the script stops rather than overwriting content.

## Behavior

1. Requests Wikipedia's `Special:Random` page over HTTPS with a 20-second timeout.
2. Parses the redirected article's title and readable paragraphs.
3. Writes UTF-8 text to a new file.

Wikipedia can change its page markup or limit requests. This tool does not bypass access controls, rate limits, or CAPTCHA checks. It extracts text only; it does not save images, tables, or references.

## Project files

```text
Random Wikipedia Article/
├── wiki_random.py
├── pyproject.toml
└── uv.lock
```
