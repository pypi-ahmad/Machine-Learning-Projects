# Word Games: Letter Partners

Check whether a word follows the Letter Partners rules. Letters `a` through `m` are pre-partners; their matching post-partners are `n` through `z` at the same alphabet offset (`a`/`n`, `b`/`o`, and so on).

## Requirements

- Python 3.14 or newer
- [uv](https://docs.astral.sh/uv/)

The project uses only the Python standard library.

## Run

Evaluate a word directly:

```powershell
cd "Python Scripts\Word Games"
uv run python .\letter_partner.py abon
```

Or run interactively:

```powershell
uv run python .\letter_partner.py
```

## Rules

- Every pre-partner must appear before its matching post-partner.
- When pairs are not adjacent, closing post-partners must follow last-opened-first-closed nesting order.
- Extra post-partners without a matching pre-partner are ignored, matching the original game description.

Examples: `abon` and `aerfsbon` win; `abno` loses because `n` closes `a` before `o` closes `b`.
