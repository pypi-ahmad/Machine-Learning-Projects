# Rename Files

> A collection of Python scripts for batch file renaming — removing prefixes and digits from filenames.

## Overview

This project contains several utility scripts for renaming files in a directory. The scripts perform two operations: stripping a `cheese_` prefix from filenames, and removing numeric digits from filenames. The `prank/` folder contains sample files with number prefixes, and `workedOn/` contains the cleaned results.

## Features

- **`rename.py`**: Strips the `cheese_` prefix from all files in a hardcoded directory
- **`project.py`**: Removes all numeric digits from filenames in a hardcoded directory
- **`list.py`**: Lists all files in the `workedOn/` directory
- **`remove.py`**: Removes digits from a test string (`"hello23you.jpg"` → `"helloyou.jpg"`)

## Project Structure

```
Rename-Files/
├── rename.py        # Removes "cheese_" prefix from filenames
├── project.py       # Removes digits from filenames
├── list.py          # Lists files in workedOn/ directory
├── remove.py        # Demo: digit removal from a string
├── structure.txt    # Brief task description
├── prank/           # Sample files with number-prefixed city names (e.g., "16los angeles.jpg")
├── workedOn/        # Cleaned files with digits removed (e.g., "los angeles.jpg")
└── README.md
```

## Requirements

- Python 3.x
- `os` (Python standard library)

## Installation

```bash
cd "Rename-Files"
```

No external dependencies required.

## Usage

> **Warning:** All scripts have hardcoded absolute paths (macOS-style). You must edit the paths before running.

**Remove `cheese_` prefix from filenames:**
```bash
python rename.py
```

**Remove digits from filenames:**
```bash
python project.py
```

**List files in workedOn/ directory:**
```bash
python list.py
```

**Demo digit removal on a test string:**
```bash
python remove.py
```

## How It Works

- **`rename.py`**: Uses `os.listdir()` to iterate files, checks if the filename starts with `"cheese_"`, and calls `os.rename()` to strip the first 7 characters (the prefix).
- **`project.py`**: Iterates files in the `workedOn/` directory, uses a list comprehension to filter out digit characters (`i.isdigit()`), joins the remaining characters, and renames the file. Switches working directory with `os.chdir()` and restores it afterward.
- **`remove.py`**: A standalone demo that filters digits from the string `"hello23you.jpg"` using the same list comprehension technique.

## Configuration

All directory paths are hardcoded to macOS-style absolute paths and **must be edited** before use:

- `rename.py`: `/Users/User/Desktop/uni/change file names`
- `project.py`: `/Users/User/Desktop/uni/change file names/workedOn`
- `list.py`: `/Users/User/Desktop/uni/change file names/workedOn`

## Limitations

- All paths are hardcoded absolute paths for a specific macOS system; scripts will fail on any other machine without modification.
- No command-line arguments or configuration file support.
- No error handling for missing directories, permission errors, or filename collisions.
- `rename.py` only handles the `cheese_` prefix — it is not a general-purpose renamer.
- `project.py` removes **all** digits, which may corrupt filenames that legitimately contain numbers.
- The `prank/` and `workedOn/` folders contain sample `.jpg` files (city names) used as test data.

## Security Notes

No security concerns.

## License

Not specified.
