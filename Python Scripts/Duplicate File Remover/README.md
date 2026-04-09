# Duplicate Files Remover

> A Python script that detects and removes duplicate files in the current directory by comparing MD5 hashes.

## Overview

This script scans all files in its current working directory, computes an MD5 hash for each file, and deletes any file whose hash matches a previously seen file. It reads files in blocks to handle large files efficiently.

## Features

- Scans all files in the current working directory
- Computes **MD5 hashes** using block-based reading (64 KB blocks) for memory efficiency
- Detects duplicates by comparing hash values
- Automatically **deletes** duplicate files (keeps the first occurrence)
- Reports which files were deleted, or confirms no duplicates found

## Project Structure

```
Duplicate files remover/
├── duplicatefileremover.py
├── Screenshot.png            # Sample screenshot
└── README.md
```

## Requirements

- Python 3.x
- No external dependencies (uses only `hashlib` and `os` from the standard library)

## Installation

```bash
cd "Duplicate files remover"
```

No pip install required.

## Usage

**Important:** Place the script in the directory containing the files you want to deduplicate, then run:

```bash
python duplicatefileremover.py
```

Sample output when duplicates are found:

```
Deleted Files
duplicate_copy.txt
another_copy.pdf
```

Sample output when no duplicates exist:

```
No duplicate files found
```

## How It Works

1. **`hashFile(filename)`** — Opens a file in binary mode, reads it in 65,536-byte blocks, and feeds each block to an MD5 hasher. Returns the hex digest string.
2. **Main logic** — Lists all files in the current directory (excluding subdirectories), computes each file's hash, and checks against a `hashMap` dictionary:
   - If the hash already exists → the file is a duplicate; it is deleted with `os.remove()` and added to the `deletedFiles` list.
   - If the hash is new → stored in `hashMap`.
3. Prints the list of deleted files, or "No duplicate files found".

## Configuration

No configuration files. The script operates on the current working directory.

## Limitations

- Only scans files in the **current directory** — does not recurse into subdirectories.
- Uses **MD5** which is not collision-resistant (though sufficient for duplicate detection in practice).
- The first file encountered with a given hash is kept; order depends on `os.listdir()` which is not guaranteed to be sorted.
- **Destructive operation** — deletes files immediately with no confirmation prompt, undo, or dry-run mode.
- Ignores the script file itself — could potentially delete its own copy if duplicated.
- No logging or output file — only prints to console.

## License

Not specified.
