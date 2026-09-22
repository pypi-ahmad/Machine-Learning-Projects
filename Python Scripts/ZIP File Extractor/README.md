# ZIP File Extractor

A command-line tool that extracts a ZIP archive into a folder named after the archive. It uses only Python's standard library.

## Requirements

- Python 3.14 or newer
- [uv](https://docs.astral.sh/uv/)

## Install

```powershell
cd "Python Scripts\ZIP File Extractor"
uv sync
```

## Extract an archive

```powershell
uv run python .\extract_zip_files.py --zippedfile .\archive.zip
```

The command extracts `archive.zip` into `archive` in the current working directory. Use `--output` to choose another destination:

```powershell
uv run python .\extract_zip_files.py --zippedfile .\archive.zip --output .\extracted
```

## Notes

- The tool accepts `.zip` archives only.
- It creates the destination directory when needed and may add files to an existing destination.
- Extract archives from sources you trust.
