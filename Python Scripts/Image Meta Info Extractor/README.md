# Get Meta Information of Images

> Extract EXIF metadata, file ownership, and GPS locations from image files.

## Overview

A command-line tool that reads local image metadata such as dimensions, EXIF data, file creation date, Windows file ownership, and GPS coordinates. Reverse geocoding is optional.

## Features

- Extracts image name, pixel dimensions, and file extension
- Reads EXIF data: `ExifImageWidth`, `ExifImageHeight`, `DateTimeOriginal`
- Retrieves file creation timestamp from the OS
- Determines file owner/author using Windows security APIs (`advapi32`, `kernel32`)
- Extracts GPS coordinates from EXIF when present
- Optionally reverse-geocodes coordinates through Nominatim (OpenStreetMap)

## Project Structure

```
Get_meta_information_of_images/
├── get_meta_from_pic.py
├── author_utils.py
├── gps_utils.py
├── pyproject.toml
├── uv.lock
└── README.md
```

## Requirements

- Python 3.13+
- Windows OS (author detection uses Win32 `ctypes` API)
- `Pillow` (PIL)
- `ExifRead`
- `geopy`
- `Pillow`, `ExifRead`, and `geopy`, managed by uv in `pyproject.toml`

## Installation

```bash
cd "Get_meta_information_of_images"
uv sync
```

## Usage

```bash
uv run python get_meta_from_pic.py <image_file>
```

Example:
```bash
uv run python get_meta_from_pic.py photo.jpg
```

Reverse-geocode GPS data only when needed, with a valid Nominatim user agent:

```bash
uv run python get_meta_from_pic.py photo.jpg --reverse-geocode --nominatim-user-agent your-app-name
```

Output:
```
ImageName: photo.jpg
size: 4032x3024
FileExtension: .jpg
ImageWidth: 4032
ImageHeight: 3024
DateTimeOriginal: 2020:06:15 14:30:00
CreateDate: 2020-06-15 14:30:00
Author: DOMAIN\Username
Coordinates: (12.34, 56.78)
```

## How it works

1. **`get_meta_from_pic.py`** — Main script. Opens the image through Pillow and safely reports available EXIF tags, timestamps, Windows owner, and coordinates without requiring EXIF or GPS data.

2. **`author_utils.py`** — Windows-only module that uses `ctypes` to call Win32 APIs (`advapi32.GetNamedSecurityInfoW`, `LookupAccountSidW`) to retrieve the file's NTFS owner. Returns the owner in `DOMAIN\Username` format.

3. **`gps_utils.py`** — Reads GPS EXIF tags, converts DMS (degrees/minutes/seconds) to signed decimal coordinates, and reverse-geocodes only when requested.

## Configuration

- **Nominatim user agent:** Pass `--nominatim-user-agent` only with `--reverse-geocode`.

## Limitations

- **Windows-only:** `author_utils.py` relies on Win32 APIs and will not work on Linux/macOS.
- **Windows ownership:** `author_utils.py` relies on Win32 APIs and may report `Unavailable` for unsupported filesystems or inaccessible files.
- **GPS availability:** Many images contain no GPS metadata; reverse geocoding requires both GPS data and network access.

## Security Notes

- Reverse geocoding sends image coordinates to Nominatim. Use it only when that disclosure is appropriate.

## License

Not specified.
