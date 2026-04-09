# Random Word from List

> A command-line Python script that picks a random word from a text file.

## Overview

This script reads a text file (specified via command-line argument or user input), counts the number of non-empty lines, generates a random line number, and prints the word on that line. It comes bundled with a large dictionary file (`file.txt`) containing approximately 176,839 words.

## Features

- Reads a word list from any text file
- Accepts filename as a command-line argument or interactive prompt
- Handles `FileNotFoundError` gracefully
- Strips trailing whitespace/newlines from output
- Counts only non-empty lines when determining the random range

## Project Structure

```
Random_word_from_list/
├── Random_word_from_list.py    # Main script
├── file.txt                     # Dictionary word list (~176,839 words)
└── README.md
```

## Requirements

- Python 3.x
- No external dependencies (uses only `sys`, `random`)

## Installation

```bash
cd "Random_word_from_list"
```

No additional packages need to be installed.

## Usage

### With command-line argument:

```bash
python Random_word_from_list.py file.txt
```

### Without argument (interactive prompt):

```bash
python Random_word_from_list.py
What is the name of the file? (extension included): file.txt
```

Output: a single random word from the file, e.g., `python`

## How It Works

1. Checks `sys.argv[1:]` for a filename argument; if not provided, prompts the user.
2. Opens the file and counts non-empty lines (`sum(1 for line in file if line.rstrip())`).
3. Generates a random line number with `random.randint(0, num_lines)`.
4. Seeks back to the beginning of the file with `file.seek(0)`.
5. Iterates through lines with `enumerate()` and prints the line at the random index (stripped of trailing whitespace).

## Configuration

- The included `file.txt` contains ~176,839 words (one per line)
- Any text file with one word/entry per line can be used

## Limitations

- `random.randint(0, num_lines)` can be off by one since `num_lines` is a count, not an index — the maximum index should be `num_lines - 1`
- The file is iterated twice (once to count lines, once to find the random line); this is inefficient for very large files
- The file handle is never explicitly closed (no `with` statement)
- Only prints one word per execution; no batch mode
- No encoding parameter specified when opening the file

## Security Notes

No security concerns identified.

## License

Not specified.
