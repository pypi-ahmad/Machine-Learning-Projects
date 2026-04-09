# Unique Words in a File

> A script that extracts and displays alphabetically sorted unique words (appearing exactly once) from a text file.

## Overview

This script reads a text file, extracts all words using regex, counts occurrences of each word (case-insensitive), and prints a sorted list of words that appear exactly once in the file.

## Features

- Reads words from any text file using regex word extraction (`\w+`)
- Case-insensitive comparison (all words converted to lowercase)
- Counts word occurrences using a dictionary
- Filters to only words appearing exactly once (truly unique)
- Outputs results in alphabetical order
- Includes a sample text file (`text_file.txt`) with Lorem Ipsum content

## Project Structure

```
Unique_words_in_a_file/
├── unique.py
├── text_file.txt
└── README.md
```

## Requirements

- Python 3.x
- No external dependencies (uses only `re` from the standard library)

## Installation

```bash
cd "Unique_words_in_a_file"
```

No package installation required.

## Usage

```bash
python unique.py
```

The script reads from `text_file.txt` in the same directory and prints the sorted list of unique words.

To use a different file, edit the `filename` variable in the script or uncomment the `input()` line:

```python
# filename = input("Enter file name: ")
filename = "text_file.txt"
```

## How It Works

1. Opens the file specified by `filename` in read mode.
2. Iterates through each line, extracting words using `re.findall(r"[\w]+", line.lower())`.
3. Builds a dictionary mapping each word to its occurrence count.
4. Filters the dictionary for words with a count of exactly 1.
5. Sorts the filtered words alphabetically and prints the list.

## Configuration

- **Input file:** Hardcoded as `text_file.txt`. Can be changed by editing the `filename` variable or uncommenting the user input line.
- **Case sensitivity:** Currently case-insensitive (uses `line.lower()`). An alternative case-sensitive mode is commented out in the code.

## Limitations

- The filename is hardcoded; no command-line argument support.
- No error handling for missing files or read permissions.
- The regex `[\w]+` includes digits and underscores as part of words (e.g., `word123` would be treated as one word).
- For large files, the entire word list is held in memory.

## License

Not specified.
