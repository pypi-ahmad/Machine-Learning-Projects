# File Metadata Viewer

A read-only command-line tool for inspecting filesystem metadata, image EXIF data, and PDF document metadata.

## Requirements

- Python 3.13 or later
- [uv](https://docs.astral.sh/uv/)

## Setup

```powershell
uv sync --no-config
```

## Inspect filesystem metadata

```powershell
uv run --no-config python main.py "C:\path\to\file.txt"
```

The output includes timestamps, size, MIME guess, permission bits, and basic platform-specific fields.

## Inspect image or PDF metadata

```powershell
uv run --no-config python main.py photo.jpg --exif
uv run --no-config python main.py report.pdf --pdf
```

The tool reads only the named path. It does not scan directories, modify files, remove metadata, or upload file contents. Metadata can contain sensitive information such as names, locations, software versions, or timestamps; avoid sharing output without reviewing it.
