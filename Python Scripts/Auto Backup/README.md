# Auto Backup

A multithreaded CLI backup utility that recursively copies files from one or more source directories to a target directory, with automatic gzip compression for files exceeding a configurable size threshold.

## Overview

- Mirrors source directory structures into a target backup folder, performing incremental sync (only newer files) with optional gzip compression and multithreaded transfers
- **Project type:** CLI / Utility

## Features

- Recursively walks one or more source directories and mirrors their structure in the target
- **Incremental sync** — only copies files whose modification time (`st_mtime`) is newer than the corresponding target file
- **Automatic gzip compression** for files exceeding a configurable byte threshold (default: 1,024,000 bytes ≈ 1 MB)
- **Multithreaded transfers** — each file copy/compress operation runs in its own `threading.Thread`
- Creates missing target directories automatically via `os.makedirs()`
- Supports multiple source directories in a single invocation
- Prints help message and exits if no arguments are provided

## Dependencies

| Package | Source |
|---------|--------|
| `argparse` | Python standard library |
| `gzip` | Python standard library |
| `os` | Python standard library |
| `shutil` | Python standard library |
| `sys` | Python standard library |
| `threading` | Python standard library |

No external dependencies — standard library only.

## How It Works

1. `parse_input()` uses `argparse` to parse CLI arguments: `--target` (required), `--source` (required, one or more paths), and `--compress` (optional threshold in bytes, default 1024000).
2. If no arguments are provided, the help message is printed and the script exits.
3. For each source root, `sync_root()` calls `os.walk()` to enumerate every file recursively.
4. For each file, `threaded_sync_file()` calls `size_if_newer()` to compare modification times:
   - Checks the target file's `st_mtime` (also checks `.gz` variant if the plain file is not found).
   - Returns the source file size if source is newer (by >1 second), otherwise `False`.
5. If the file is newer, a new `threading.Thread` is spawned running `transfer_file()`:
   - If the file size exceeds the compression threshold: opens the file with `gzip.open()` and writes a `.gz` compressed copy to the target.
   - Otherwise: copies the file with `shutil.copy2()` (preserves metadata).
   - If the target directory doesn't exist, creates it via `os.makedirs()` and retries.
6. All threads are joined before moving to the next source root.

## Project Structure

```
Auto_Backup/
├── Auto_Backup.py   # Main backup script
└── README.md
```

## Setup & Installation

No installation required — uses only the Python standard library.

## How to Run

```bash
cd Auto_Backup

# Basic usage: back up a single folder
python Auto_Backup.py --target ./Backup_Folder --source ./Source_Folder

# Multiple sources with a custom compression threshold (100 KB)
python Auto_Backup.py -t ./Backup -s ./Folder1 ./Folder2 -c 100000
```

### CLI Arguments

| Flag | Required | Description |
|------|----------|-------------|
| `-t` / `--target` | Yes | Target backup directory |
| `-s` / `--source` | Yes | One or more source directories |
| `-c` / `--compress` | No | Gzip compression threshold in bytes (default: `1024000`) |

## Configuration

All configuration is done via CLI arguments. No config files or environment variables.

## Testing

No formal test suite present.

## Limitations

- **No deletion tracking** — files deleted from the source are not removed from the target backup.
- Thread count is unbounded — a directory with many files spawns one thread per file, which may exhaust system resources.
- `size_if_newer()` compares only modification time, not content hashes; edits that don't change `st_mtime` are missed.
- The target path is constructed as `target + source` (string concatenation), which may produce unexpected paths on Windows or when source paths are absolute.
- No progress bar or summary statistics (file count, bytes transferred).
- No logging to file — output is printed to stdout only.
- The 1-second tolerance in `size_if_newer()` (`st_mtime - target_ts > 1`) can miss rapid successive modifications.

## Security Notes

No sensitive data is handled. Ensure the target directory has appropriate filesystem permissions if backing up sensitive files.

