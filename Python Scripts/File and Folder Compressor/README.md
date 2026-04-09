# File and Folder Compressor

> A CLI tool that compresses individual files or entire directories into ZIP archives using Python's `zipfile` module.

## Overview

This script accepts a file path or directory path as a command-line argument and creates a ZIP archive of the target. For directories, it recursively walks through all subdirectories and includes every file. For individual files, it compresses just that file. The output ZIP file is created alongside the original with a `.zip` extension appended.

## Features

- Compress individual files to ZIP format with `ZIP_DEFLATED` compression
- Compress entire directory trees recursively
- Automatically detects whether the target is a file or directory
- Prints the list of files to be zipped when compressing a directory
- Handles special files (sockets, FIFOs, device files) with an informative message

## Project Structure

```
Write_script_to_compress_folder_and_files/
├── zipfiles.py    # Main compression script
└── README.md
```

## Requirements

- Python 3.x
- No external dependencies (uses `zipfile`, `sys`, `os` from the standard library)

## Installation

```bash
cd "Write_script_to_compress_folder_and_files"
```

No packages to install.

## Usage

### Compress a file

```bash
python zipfiles.py test.txt
```

Creates `test.txt.zip` in the current directory.

### Compress a folder

```bash
python zipfiles.py ./test
```

Creates `./test.zip` containing all files from the `test` directory and its subdirectories.

## How It Works

1. **`zip_file(file_path)`** — Creates a `ZipFile` object in write mode with `ZIP_DEFLATED` compression and writes the single file into it.
2. **`retrieve_file_paths(dir_name)`** — Uses `os.walk()` to recursively traverse the directory tree, collecting full file paths into a list.
3. **`zip_dir(dir_path, file_paths)`** — Creates a `ZipFile` and writes each file from the collected paths list.
4. **Main block** — Reads `sys.argv[1]`, checks if it's a directory (`os.path.isdir`) or file (`os.path.isfile`), and calls the appropriate function.

## Configuration

No configuration needed. All behavior is controlled via the command-line argument.

## Limitations

- The `zip_file()` function references a global variable `path` instead of the `file_path` parameter — this is a bug
- No error handling for missing command-line arguments (will raise `IndexError` if no argument is provided)
- Output ZIP filename is always `<input>.zip` — no way to specify a custom output name
- Empty directories are not preserved in the archive
- No progress indicator for large directory trees
- The file paths stored in the ZIP may include relative path prefixes like `./`

## Security Notes

No security concerns identified.

## License

Not specified.
