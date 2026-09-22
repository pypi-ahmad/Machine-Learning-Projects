# Download Folder Organizer

A preview-first CLI that groups files from a selected source directory into extension-based category folders.

## Requirements

- Python 3.13+
- [uv](https://docs.astral.sh/uv/)

## Preview an organization plan

From this directory:

```powershell
uv sync
uv run python file-sortor.py "C:\path\to\downloads" "C:\path\to\organized-downloads"
```

The default command only prints the planned moves. It does not create folders or move files.

## Move files

After reviewing the preview, repeat the command with `--execute`:

```powershell
uv run python file-sortor.py "C:\path\to\downloads" "C:\path\to\organized-downloads" --execute
```

The tool creates category folders only during execution and skips a file when the destination path already exists.

## Categories

`images`, `videos`, `music`, `archives`, `documents`, `installers`, `programs`, and `design` are mapped from common file extensions. Files without a configured extension go to `others`. Matching is case-insensitive.

## Safety notes

- `--execute` performs filesystem moves, which change the selected source directory.
- The source and destination must be different directories.
- Review the preview before executing, especially when organizing a large or important folder.
- Hidden files and directories are ignored.

## Verification

```powershell
uv run python -m py_compile file-sortor.py
uv lock --check
```
