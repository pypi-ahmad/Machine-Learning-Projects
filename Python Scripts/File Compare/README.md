# File Compare

> A Python script that finds and outputs the common lines between two text files.

## Overview

This script reads two text files, computes the set intersection of their lines, and writes the common lines to an output file. It uses Python's built-in set operations for efficient comparison.

## Features

- Compares two text files line by line
- Finds common lines using set intersection
- Outputs matching lines to a separate file (`soutput.txt`)
- Discards blank newline entries from the results

## Project Structure

```
File _Compare/
├── file.py       # Main comparison script
├── 1.txt         # Sample input file 1
├── 2.txt         # Sample input file 2
└── soutput.txt   # Output file (generated after running)
```

## Requirements

- Python 3.x
- No external dependencies

## Installation

```bash
cd "File _Compare"
```

No package installation needed.

## Usage

```bash
python file.py
```

The script reads `1.txt` and `2.txt` from the same directory and writes common lines to `soutput.txt`.

### Sample Input

**1.txt** (sample content — contains Python code as text):
```
with open('some_file_1.txt', 'r') as file1:
    with open('some_file_2.txt', 'r') as file2:
        same = set(file1).intersection(file2)
...
```

**2.txt** (similar content with slight differences):
```
with open('some_file_1.txt', 'r') as file1:
    with open('some_file_2.txt', 'r') as file2:
        same = set(file1).intersection(file2)
...
```

### Output

`soutput.txt` will contain only the lines that appear in both files.

## How It Works

1. Opens `1.txt` and `2.txt` for reading.
2. Converts each file's lines into a set.
3. Computes the intersection of the two sets (lines common to both files).
4. Removes standalone newline characters (`'\n'`) from the result.
5. Writes the remaining common lines to `soutput.txt`.

## Configuration

The filenames are hardcoded in the script:
- Input files: `1.txt`, `2.txt`
- Output file: `soutput.txt`

## Limitations

- Filenames are hardcoded — cannot be passed as arguments
- Uses set intersection, so **line order is not preserved** in the output
- **Duplicate lines** are collapsed (sets remove duplicates)
- Lines must match exactly (including whitespace and case) to be considered common
- No command-line interface or user prompts
- No error handling for missing input files

## Security Notes

No security concerns.

## License

Not specified.
