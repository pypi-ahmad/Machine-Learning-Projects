# Unzip File

> A script to extract all contents from a ZIP archive to a specified directory.

## Overview

A minimal Python script that takes a path to a ZIP file as user input and extracts all its contents to a `./Unzip file/Unzip files/` directory using the built-in `zipfile` module.

## Features

- Extracts all files from a ZIP archive
- Prompts user for the ZIP file path at runtime
- Uses Python's built-in `zipfile` module — no external dependencies

## Project Structure

```
Unzip file/
├── script.py
└── Readme.md
```

## Requirements

- Python 3.x
- No external dependencies (uses only `zipfile` from the standard library)

## Installation

```bash
cd "Unzip file"
```

No package installation required.

## Usage

```bash
python script.py
```

When prompted, enter the path to the ZIP file:

```
Enter file to be unzipped: path/to/archive.zip
```

The extracted files will be placed in `./Unzip file/Unzip files/`.

## How It Works

1. Prompts the user for a file path using `input()`.
2. Opens the ZIP file with `zipfile.ZipFile(target)`.
3. Calls `extractall("./Unzip file/Unzip files")` to extract all contents.
4. Closes the ZIP file handle.

## Configuration

- **Extraction directory:** Hardcoded as `./Unzip file/Unzip files`. Change the path in `handle.extractall()` to extract elsewhere.

## Limitations

- The extraction path `./Unzip file/Unzip files` is hardcoded and assumes the script is run from the parent directory.
- No error handling for invalid file paths, corrupted archives, or non-ZIP files.
- Does not create the output directory if it doesn't exist (relies on `extractall` behavior).
- No progress indication for large archives.
- Does not use a context manager (`with` statement) for the ZIP file — relies on manual `close()`.

## License

Not specified.
