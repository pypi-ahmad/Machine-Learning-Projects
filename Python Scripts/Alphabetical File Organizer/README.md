# Alphabetical File Organizer

> A Python script that sorts files in the current directory into folders named by their first letter (alphabetical order).

## Overview

This script reads all files in its current working directory, creates single-letter folders (or a `misc` folder for non-alphabetic names), and moves each file into the corresponding folder based on the first character of its filename.

## Features

- Scans all files in the current working directory
- Creates lowercase single-letter folders (`a`–`z`) as needed
- Files starting with non-alphabetic characters go into a `misc` folder
- Skips `main.py` itself (removes it from the file list before processing)
- Uses `shutil.move()` for reliable cross-filesystem file moves
- Prints progress messages during folder creation and file moves

## Project Structure

```
Write_script_to_move_files_into_alphabetically_ordered_folder/
├── main.py      # Main organizer script
└── README.md
```

## Requirements

- Python 3.x
- No external dependencies (uses `os` and `shutil` from the standard library)

## Installation

```bash
cd "Write_script_to_move_files_into_alphabetically_ordered_folder"
```

No packages to install.

## Usage

**Important**: Copy `main.py` into the directory you want to organize, then run:

```bash
python main.py
```

The script will:
1. List all files in the current directory
2. Create alphabetical folders as needed
3. Move each file into its corresponding folder

### Example

```
Before:                    After:
├── apple.txt              ├── a/
├── banana.txt             │   └── apple.txt
├── 123.dat                ├── b/
├── main.py                │   └── banana.txt
                           ├── misc/
                           │   └── 123.dat
                           └── main.py
```

## How It Works

1. **`getfoldername(filename)`** — Returns the lowercase first character if alphabetic; otherwise returns `'misc'`.
2. **`readdirectory()`** — Uses `os.listdir(os.getcwd())` to collect all files (not directories) into a global list, then removes `main.py` from the list.
3. **`createfolder()`** — Iterates through filenames, calls `os.mkdir()` for each needed folder if it doesn't already exist.
4. **`movetofolder()`** — Moves each file to its target folder using `shutil.move(source, destination)`.

## Configuration

No configuration needed. The script operates on the current working directory.

## Limitations

- Hardcodes the exclusion of `main.py` — will crash with `ValueError` if `main.py` is not in the directory or has been renamed
- Operates on `os.getcwd()`, not a configurable target directory
- Uses a global mutable list (`filenames`) shared across functions
- Case-insensitive only by virtue of `lower()` on the first character; files starting with uppercase letters get lowercase folder names
- No dry-run mode — files are moved immediately without confirmation
- Does not handle filename conflicts if a file with the same name already exists in the target folder
- Existing directories in the working directory are skipped but pre-existing target folders are reused

## Security Notes

No security concerns identified.

## License

Not specified.
