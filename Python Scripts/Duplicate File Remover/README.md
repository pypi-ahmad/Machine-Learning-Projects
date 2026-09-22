# Duplicate File Remover

A non-recursive duplicate-file detector that previews matching-content groups by default. It can permanently remove extra copies only after explicit confirmation.

## Requirements

- Python 3.13+
- [uv](https://docs.astral.sh/uv/)

## Preview duplicates

From this directory:

```powershell
uv sync
uv run python duplicatefileremover.py "C:\path\to\folder"
```

The default uses SHA-256 and lists each duplicate group. The lexicographically first file is the planned retained copy; no file is changed.

## Permanently remove extra copies

```powershell
uv run python duplicatefileremover.py "C:\path\to\folder" --delete
```

The command asks you to type `DELETE` before it removes anything. Deletion is permanent and bypasses the Recycle Bin.

Use `--algorithm md5` only when compatibility or speed matters. MD5 is not a security or integrity guarantee.

## Limits

- Only the selected directory level is scanned; subdirectories are not traversed.
- Unreadable files are reported and skipped.
- Identical-content detection does not determine whether a duplicate is safe to remove. Review the preview first.

## Verification

```powershell
uv run python -m py_compile duplicatefileremover.py
uv lock --check
```
