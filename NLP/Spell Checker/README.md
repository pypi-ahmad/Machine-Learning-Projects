# Spell Checker

> An interactive spell checker using the TextBlob library.

## Overview

This script prompts the user to enter a word, uses TextBlob's `correct()` method to suggest the correct spelling, and allows repeated checks in a loop.

## Features

- Interactive spell checking via terminal input
- Automatic spelling correction using TextBlob's NLP-based `correct()` method
- Loop functionality — check multiple words without restarting the script

## Project Structure

```
Spell_checker/
└── spell_checker.py
```

## Requirements

- Python 3.x
- `textblob`

## Installation

```bash
cd "Spell_checker"
pip install textblob
```

> **Note:** TextBlob may require downloading its corpora on first use:
> ```bash
> python -m textblob.download_corpora
> ```

## Usage

```bash
python spell_checker.py
```

**Example session:**

```
Enter the word to be checked:- speling
original text: speling
corrected text: spelling
Try Again? 1 : 0 1
Enter the word to be checked:- wrld
original text: wrld
corrected text: world
Try Again? 1 : 0 0
```

- Enter `1` to check another word
- Enter `0` to exit

## How It Works

1. A `while` loop runs as long as the control variable `t` is truthy
2. The user enters a word via `input()`
3. The input is wrapped in a `TextBlob` object
4. `TextBlob.correct()` applies NLP-based spelling correction and returns the corrected text
5. Both the original and corrected text are displayed
6. The user is prompted to continue (`1`) or exit (`0`)

## Configuration

No configuration options. All input is provided interactively.

## Limitations

- `TextBlob.correct()` works on full sentences too, but the prompt says "word" — input is not validated for single words
- The loop control variable `t` is set via `int(input(...))` — entering a non-integer causes a `ValueError` crash
- No suggestion list — only the single most likely correction is shown
- Correction accuracy depends on TextBlob's underlying corpus and may not handle all words correctly
- The initial value of `t` is `1` (integer) but later set from `int(input())`, which is fine but inconsistent with boolean conventions

## Security Notes

No security concerns.

## License

Not specified.
