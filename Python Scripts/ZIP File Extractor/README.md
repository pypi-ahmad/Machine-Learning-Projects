# Extract Zip Files

> A CLI tool that extracts the contents of a ZIP file into a new folder named after the archive.

## Overview

This script accepts a ZIP file path as a command-line argument, verifies the file exists, and extracts its contents into a new directory (named after the ZIP file) in the current working directory.

## Features

- **CLI interface** using `argparse` with a required `-l` / `--zippedfile` argument
- Validates that the specified ZIP file exists before extraction
- Creates a new folder named after the ZIP file (without the `.zip` extension)
- Extracts all contents using Python's built-in `zipfile` module
- Provides clear error messages for missing files or non-ZIP inputs

## Project Structure

```
Extract_zip_files/
├── extract_zip_files.py
└── README.md
```

## Requirements

- Python 3.x
- No external dependencies (uses only `os`, `zipfile`, `sys`, and `argparse` from the standard library)

## Installation

```bash
cd Extract_zip_files
```

No pip install required.

## Usage

```bash
python extract_zip_files.py -l <path_to_zip_file>
```

Example:

```bash
python extract_zip_files.py -l archive.zip
```

Output on success:

```
Extracted successfully!!!
```

The contents are extracted to `./archive/` in the current directory.

## How It Works

1. Parses command-line arguments using `argparse` to get the ZIP file path (`-l` / `--zippedfile`).
2. Checks if the file exists with `os.path.exists()` — exits with an error message if not found.
3. **`extract(zip_file)`**:
   - Strips the `.zip` extension to derive the output folder name.
   - Verifies the file ends with `.zip`.
   - Constructs the output path as `<current_working_directory>/<folder_name>`.
   - Opens the ZIP with `zipfile.ZipFile()` in read mode.
   - Extracts all contents with `extractall()` to the new directory.
4. Prints success or "Not a zip file" if the extension doesn't match.

## Configuration

No configuration files. The ZIP file path is provided via the `-l` CLI argument.

## Limitations

- Uses forward slash `/` for path joining instead of `os.path.join()`, which may cause issues on Windows.
- The `file_name` variable is assigned in both the outer scope and inside `extract()`, shadowing the outer one.
- Only supports `.zip` files — no support for `.tar.gz`, `.rar`, `.7z`, or other archive formats.
- No password-protected ZIP support.
- No option to specify the output directory — always extracts to the current working directory.
- No progress indication for large archives.
- The `sys` import is used only for `sys.exit()`.

## License

Not specified.
