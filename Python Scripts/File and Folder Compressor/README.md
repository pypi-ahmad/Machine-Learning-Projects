# File and Folder Compressor

A dependency-free command-line tool for creating ZIP archives from a single file or a directory tree.

## Requirements

- Python 3.13 or later
- [uv](https://docs.astral.sh/uv/)

## Setup

```powershell
uv sync --no-config
```

## Preview an archive

Preview the output path and file count without writing an archive:

```powershell
uv run --no-config python zipfiles.py "C:\path\to\folder" --dry-run
```

## Create an archive

```powershell
uv run --no-config python zipfiles.py "C:\path\to\folder"
```

By default, a file named `folder.zip` or `file.txt.zip` is created beside the source. Directory archives preserve paths relative to the selected source folder.

Choose a destination explicitly:

```powershell
uv run --no-config python zipfiles.py report.xlsx --output archives\report.zip
```

Existing archives are protected. Add `--overwrite` only after reviewing the destination:

```powershell
uv run --no-config python zipfiles.py report.xlsx --output archives\report.zip --overwrite
```

## Notes

- The output directory must already exist.
- ZIP archives are not encrypted by this tool. Do not treat them as protection for sensitive data.
