# Get Meta Information of Images

> Extract EXIF metadata, file owner, and GPS location from image files.

## Overview

A command-line tool that reads an image file and extracts comprehensive metadata including dimensions, EXIF data (width, height, original date), file creation date, file ownership (Windows-only via Win32 API), and GPS coordinates reverse-geocoded to a human-readable address using `geopy`.

## Features

- Extracts image name, pixel dimensions, and file extension
- Reads EXIF data: `ExifImageWidth`, `ExifImageHeight`, `DateTimeOriginal`
- Retrieves file creation timestamp from the OS
- Determines file owner/author using Windows security APIs (`advapi32`, `kernel32`)
- Extracts GPS coordinates from EXIF and reverse-geocodes to a street address via Nominatim (OpenStreetMap)

## Project Structure

```
Get_meta_information_of_images/
├── get_meta_from_pic.py
├── author_utils.py
├── gps_utils.py
├── requirements.txt
└── README.md
```

## Requirements

- Python 3.x
- Windows OS (author detection uses Win32 `ctypes` API)
- `Pillow` (PIL)
- `ExifRead`
- `geopy`
- `requests` (imported in `gps_utils.py` but not directly used)

From `requirements.txt`:
```
Pillow
ExifRead==2.3.1
geopy==2.0.0
```

## Installation

```bash
cd "Get_meta_information_of_images"
pip install Pillow ExifRead geopy requests
```

## Usage

```bash
python get_meta_from_pic.py <image_file>
```

Example:
```bash
python get_meta_from_pic.py photo.jpg
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
Location: 123 Main St, City, State, Country
```

## How It Works

1. **`get_meta_from_pic.py`** — Main script. Opens the image via `PIL.Image.open(sys.argv[1])`, verifies the image, and extracts EXIF tags using `PIL.ExifTags.TAGS`. Prints image name, size, extension, select EXIF fields, creation date, author, and location.

2. **`author_utils.py`** — Windows-only module that uses `ctypes` to call Win32 APIs (`advapi32.GetNamedSecurityInfoW`, `LookupAccountSidW`) to retrieve the file's NTFS owner. Returns the owner in `DOMAIN\Username` format.

3. **`gps_utils.py`** — Reads GPS EXIF tags (`GPS GPSLatitude`, `GPS GPSLongitude`) using `exifread`, converts DMS (degrees/minutes/seconds) to decimal, and reverse-geocodes using `geopy.Nominatim`.

## Configuration

- **Nominatim user agent:** In `gps_utils.py`, the `user_agent` is set to `"your email"` — replace with your actual email address per Nominatim's usage policy.

## Limitations

- **Windows-only:** `author_utils.py` relies on Win32 APIs and will not work on Linux/macOS.
- **GPS required:** `gps_utils.py` will fail with a `KeyError` if the image lacks GPS EXIF data (no error handling).
- **EXIF required:** `get_exif()` calls `image.verify()` then `image._getexif()` — images without EXIF data will cause an error.
- **`requests` imported but unused** in `gps_utils.py`.
- No command-line argument validation (crashes if no argument provided).

## Security Notes

- The Nominatim `user_agent` placeholder `"your email"` should be replaced before use.

## License

Not specified.
