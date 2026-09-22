# File Integrity Checker

A dependency-free command-line tool that generates and verifies SHA-256 manifests for one directory tree.

## Requirements

- Python 3.13 or later
- [uv](https://docs.astral.sh/uv/)

## Generate a manifest

```powershell
uv sync --no-config
uv run --no-config python main.py generate "C:\path\to\folder"
```

This creates `integrity_manifest.csv` inside the selected folder. Existing manifests are protected; add `--overwrite` only when replacement is intended.

```powershell
uv run --no-config python main.py generate "C:\path\to\folder" --overwrite
```

## Verify a manifest

```powershell
uv run --no-config python main.py verify "C:\path\to\folder"
```

Verification is read-only. It reports tracked files that are modified or missing, newly discovered files, and unsafe paths in a manifest. Unsafe entries that are absolute, include `..`, or resolve outside the selected directory are rejected.

Use a different manifest location when needed:

```powershell
uv run --no-config python main.py verify "C:\path\to\folder" --manifest "C:\manifests\folder.csv"
```

## Notes

- A manifest proves only that file contents still match the hashes captured when it was generated. Protect the manifest itself from unauthorized changes.
- Hashing reads every included file and can take time for large trees.
- The tool does not repair, delete, or alter tracked files during verification.
