# Organized Download Folder with Different Categories

> A Python script that automatically sorts files in a download directory into categorized subfolders based on file extensions.

## Overview

This script scans a specified download directory and moves each file into a category-specific subfolder (e.g., images, videos, documents) based on its file extension. Files that don't match any predefined category are moved to an "others" folder.

## Features

- Automatic file sorting by extension type
- Supports 8 file categories: images, videos, musics, zip, documents, setup, programs, and design
- Unrecognized extensions are moved to an "others" fallback folder
- Skips files that already exist in the destination (prints a message instead of overwriting)

## Project Structure

```
Organized_download_folder_with_different_categories/
└── file-sortor.py
```

## Requirements

- Python 3.x
- `os` (standard library)
- `shutil` (standard library)

No external dependencies required.

## Installation

```bash
cd "Organized_download_folder_with_different_categories"
```

No pip packages needed.

## Usage

Before running, edit the hardcoded paths in `file-sortor.py` to match your system:

```python
os.chdir("E:\\downloads")  # Change to your download directory
shutil.move(file, "../download-sorting/" + dist)  # Change destination path
```

Then run:

```bash
python file-sortor.py
```

## How It Works

1. Changes the working directory to the configured downloads folder (`E:\downloads`).
2. Lists all files in the directory.
3. For each file, checks its extension against a dictionary of categories.
4. Moves the file to the appropriate subfolder under `../download-sorting/`.
5. Files with no matching extension go to `../download-sorting/others`.
6. If a file already exists at the destination, it prints a message and skips it.

### Supported File Categories

| Category    | Extensions                                          |
|-------------|-----------------------------------------------------|
| images      | `.jpg`, `.png`, `.jpeg`, `.gif`                     |
| videos      | `.mp4`, `.mkv`                                      |
| musics      | `.mp3`, `.wav`                                      |
| zip         | `.zip`, `.tgz`, `.rar`, `.tar`                      |
| documents   | `.pdf`, `.docx`, `.csv`, `.xlsx`, `.pptx`, `.doc`, `.ppt`, `.xls` |
| setup       | `.msi`, `.exe`                                      |
| programs    | `.py`, `.c`, `.cpp`, `.php`, `.C`, `.CPP`           |
| design      | `.xd`, `.psd`                                       |

## Configuration

The following values are **hardcoded** and must be manually edited in `file-sortor.py`:

- **Source directory**: `E:\downloads` (line 3)
- **Destination base**: `../download-sorting/` (lines 39, 43)
- **Extension mappings**: The `extentions` dictionary (lines 8–19)

## Limitations

- Source and destination directories are hardcoded — no CLI arguments or config file support.
- Bare `except` clause catches all exceptions silently (only prints "already exist" message).
- Does not create destination folders automatically — they must exist beforehand.
- Case-sensitive extension matching (e.g., `.JPG` won't match `.jpg`, though `.C` and `.CPP` are explicitly included).
- No logging or dry-run mode.

## Security Notes

No security concerns identified.

## License

Not specified.
