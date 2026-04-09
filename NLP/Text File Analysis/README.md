# Text File Analysis

> CLI tool that analyzes a text file and reports line count, character count, word count, unique words, and special characters.

## Overview

This script reads a text file provided as a command-line argument and computes basic text statistics: total lines, total characters (excluding spaces and newlines), total words, unique words, and special character counts. Results are displayed as a dictionary.

## Features

- Counts total lines in the file
- Counts total characters (excluding spaces and line separators)
- Counts total words
- Counts unique words
- Counts special/punctuation characters (using Python's `string.punctuation`)
- UTF-8 file encoding support
- Error handling for missing arguments and unreadable files

## Project Structure

```
Textfile_analysis/
├── textfile_analysis.py
└── README.md
```

## Requirements

- Python 3.x
- No external dependencies (uses only `os`, `sys`, `collections`, and `string` from the standard library)

## Installation

```bash
cd Textfile_analysis
```

No additional installation needed — standard library only.

## Usage

```bash
python textfile_analysis.py <textfile>
```

**Example:**

```bash
python textfile_analysis.py sample.txt
```

**Sample output:**

```python
{
  'total_lines': 42,
  'total_characters': 1523,
  'total_words': 310,
  'unique_words': 185,
  'special_characters': 27
}
```

## How It Works

1. Reads the filename from `sys.argv[1]`
2. Opens the file with UTF-8 encoding and reads the entire content into memory
3. Computes statistics:
   - **Total lines:** Counts occurrences of `os.linesep` in the content
   - **Total characters:** Length of content with spaces removed, minus the line count
   - **Total words:** Uses `collections.Counter` on whitespace-split words, sums all counts
   - **Unique words:** Counts distinct keys in the word counter
   - **Special characters:** Sums counts of characters found in `string.punctuation`
4. Catches `IndexError` (no argument provided) and `IOError` (file not found/unreadable)
5. Prints the results dictionary

## Configuration

No configuration files. The text file path is provided as a command-line argument.

## Limitations

- Uses `os.linesep` for line counting, which may give incorrect counts on cross-platform files (e.g., `\r\n` files analyzed on Linux where `os.linesep` is `\n`)
- Character count subtracts line count, which is an approximation (doesn't account for multi-byte line separators)
- Word splitting is whitespace-based only — no tokenization or handling of contractions
- The results dictionary is initialized with empty strings, then overwritten; if an error occurs mid-computation, partial results may be printed
- Reads the entire file into memory, which may be problematic for very large files
- Output is a raw Python dict print, not formatted JSON

## Security Notes

No security concerns identified.

## License

Not specified.
