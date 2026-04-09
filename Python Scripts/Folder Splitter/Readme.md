# Split Folder into Subfolders

> CLI tool that splits the files in a folder into numbered subfolders of a specified size.

## Overview

This script takes a source folder and a file count, then distributes the files from the source into newly created subfolders (`data_0`, `data_1`, `data_2`, etc.), each containing up to the specified number of files. Files are copied (not moved) using `shutil.copy2`, which preserves metadata.

## Features

- Splits files from a source folder into numbered subfolders
- Preserves file metadata using `shutil.copy2`
- Automatically creates destination subfolders as needed
- Configurable number of files per subfolder via command-line argument
- Recursively resolves absolute paths for both source files and destination folders

## Project Structure

```
Split_folder_into_subfolders/
├── split_and_copy.py
└── README.md
```

## Requirements

- Python 3.x
- No external dependencies (uses only `glob`, `os`, `shutil`, `sys` from the standard library)

## Installation

```bash
cd Split_folder_into_subfolders
```

No additional installation needed — standard library only.

## Usage

```bash
python split_and_copy.py <input_folder_path> <count>
```

**Arguments:**
| Argument | Description |
|---|---|
| `input_folder_path` | Path to the folder containing files to split |
| `count` | Maximum number of files per subfolder |

**Example:**

```bash
python split_and_copy.py ./my_images 20
```

This copies files from `./my_images` into `data_0/`, `data_1/`, etc., each holding up to 20 files.

## How It Works

1. Validates that exactly two command-line arguments are provided and the path is a valid directory
2. Uses `glob.glob()` to list all items in the source folder
3. A `split()` generator yields slices of the file list, each of length `count`
4. For each slice, creates a destination folder (`data_0`, `data_1`, ...) if it doesn't exist
5. Copies each file into the corresponding subfolder using `shutil.copy2`

## Configuration

No configuration files. All parameters are provided via command-line arguments.

## Limitations

- Output subfolders (`data_0`, `data_1`, ...) are created in the current working directory, not inside the source folder
- `glob.glob()` matches everything in the folder (including subdirectories), so directories themselves may be "copied" as well
- The `split()` generator has an off-by-one issue: it starts range at 1, so the first slice starts at index 0 but the logic uses `i-1` — this works but the first subfolder may get `count` files while the generator range starts at 1
- No progress output or summary of what was copied
- If destination folders already exist, files are silently overwritten

## Security Notes

No security concerns identified.

## License

Not specified.
