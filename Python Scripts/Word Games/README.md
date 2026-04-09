# Word Games — Letter Partners

> A CLI word game that checks whether a given word satisfies the "Letter Partners" pairing rules between the first and second halves of the alphabet.

## Overview

This is a command-line game based on the concept of "letter partners." The English alphabet is split into two halves: pre-partners (`a`–`m`) and post-partners (`n`–`z`). Each pre-partner is paired with the post-partner at the same index (a↔n, b↔o, c↔p, … m↔z). The player enters a word, and the program determines whether the letter arrangement satisfies specific ordering and adjacency rules.

## Features

- Splits the alphabet into pre-partners (`a`–`m`) and post-partners (`n`–`z`)
- Validates that every pre-partner letter in the word has its corresponding post-partner present
- Checks that each pre-partner appears before its corresponding post-partner
- Enforces adjacency rules: post-partners must either immediately follow their pre-partner or satisfy a specific nesting order
- Outputs `GAME WON` or `YOU LOST` / `GAME LOST`

## Project Structure

```
WORDGAMES/
├── letter_partner.py    # Main game script
├── description          # Text file explaining the game rules
└── README.md
```

## Requirements

- Python 3.x
- No external dependencies (uses only `sys`)

## Installation

```bash
cd "WORDGAMES"
```

No packages to install.

## Usage

```bash
python letter_partner.py
```

The script prompts:
```
Enter a word
```

Enter a word and the program will evaluate it:
- `"abon"` → `GAME WON`
- `"abno"` → `YOU LOST`
- `"aerfsbon"` → `GAME WON`

## How It Works

1. **Alphabet split** — `list1 = ['a'...'m']`, `list2 = ['n'...'z']`. A letter in `list1` at index `i` is paired with the letter in `list2` at index `i`.
2. **Classification** — Each letter in the input word is classified as a pre-partner or post-partner.
3. **Partner existence check** — Verifies every pre-partner in the word has its corresponding post-partner present; exits with `YOU LOST` if not.
4. **Ordering check** — Ensures each pre-partner appears before its post-partner in the word. If the post-partner is not immediately adjacent (position `i+1`), it checks nesting order.
5. **Nesting validation** — For non-adjacent pairs, verifies that leftmost pre-partners have their post-partners appearing in reverse (outermost) order. Outputs `GAME WON` if all conditions pass, otherwise `GAME LOST`.

## Configuration

No configuration needed.

## Limitations

- Only handles lowercase letters; uppercase input is not converted or validated
- Uses mutable list aliasing (`prepartner1=prepartner`) which means both variables reference the same list — this is a bug-prone pattern
- `w.index()` only finds the first occurrence, so duplicate letters may cause incorrect results
- No input validation for empty strings or non-alphabetic characters
- Error messages don't explain which rule was violated
- The code uses `sys.exit()` for early termination on failure rather than structured control flow

## Security Notes

No security concerns identified.

## License

Not specified.
