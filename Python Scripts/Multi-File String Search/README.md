# String Search from Multiple Files

> CLI tool that recursively searches for a text string across all files in a given directory.

## Overview

This script prompts the user for a search string and a directory path, then recursively traverses the directory tree to find files containing the specified text. When a match is found, it prints the absolute path of the matching file.

## Features

- Searches for a user-specified string across all files in a directory
- Recursively traverses subdirectories
- Prints the absolute path of matching files
- Interactive input for both search term and directory path

## Project Structure

```
String_search_from_multiple_files/
├── findstring.py
├── files/
│   ├── file1.txt
│   ├── file2.txt
│   ├── file3.txt
│   └── folder1/
│       ├── hello.txt
│       └── python.txt
└── README.md
```

## Requirements

- Python 3.x
- No external dependencies (uses only `os` from the standard library)

## Installation

```bash
cd String_search_from_multiple_files
```

No additional installation needed — standard library only.

## Usage

```bash
python findstring.py
```

The script will prompt for:

1. **Input text** — the string to search for
2. **Path** — the directory to search in

**Example:**

```
input text : hello
path : ./files
```

If `"hello"` is found in any file under `./files`, the script prints:

```
hello found in
C:\...\files\folder1\hello.txt
```

## How It Works

1. Prompts the user for a search string and directory path via `input()`
2. Changes the working directory to the specified path using `os.chdir()`
3. Lists all items in the directory with `os.listdir()`
4. For each item:
   - If it's a directory, recursively calls `getfiles()` on it
   - If it's a file, opens and reads its contents
5. Checks if the search string is present in the file content
6. If found, prints the string and the file's absolute path, then returns `True`

## Configuration

No configuration files. Input is gathered interactively at runtime.

## Limitations

- The function returns after finding the **first** matching file — it does not search all files for all matches
- The variable `f` is reused for both a flag (integer `0`) and a file handle, which is confusing and error-prone
- The "not found" message depends on `f == 1`, but `f` is overwritten by the file handle on each iteration, so the "not found" message is effectively unreachable
- Uses `os.chdir()` which changes the global working directory, affecting recursive calls and any subsequent code
- No error handling for binary files, permission errors, or encoding issues
- All files are read entirely into memory regardless of size
- The commented-out `os.chdir(path)` at the top is unused dead code

## Security Notes

No security concerns identified.

## License

Not specified.
