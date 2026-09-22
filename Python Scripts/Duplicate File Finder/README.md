# Duplicate File Finder

A CLI that locates files with identical content by grouping equal file sizes and then comparing hashes. It can report duplicates or permanently delete extra copies after an explicit confirmation.

## Requirements

- Python 3.13+
- [uv](https://docs.astral.sh/uv/)

## Run

From this directory:

```powershell
uv sync
uv run python main.py
```

Choose a directory, recursion mode, and either `md5` or `sha256`. MD5 is suitable for fast duplicate detection but is not a security or integrity guarantee.

## Menu actions

- **Find duplicates** and **dry-run** both report duplicate groups without modifying files.
- **Delete duplicates** keeps the lexicographically first path in each group and permanently deletes the other copies only after you type `DELETE` exactly.

## Safety notes

- Review every reported group before selecting deletion. Deletion bypasses the Recycle Bin.
- Do not use this tool on files that merely look similar; it only compares file content.
- The selected directory can be scanned recursively, which may take time on large or protected trees. Unreadable files are skipped.

## Project files

```text
main.py         # CLI entry point and hash-based scan logic
pyproject.toml  # uv project definition
uv.lock         # Resolved Python environment
```

## Verification

```powershell
uv run python -m py_compile main.py
uv lock --check
```
