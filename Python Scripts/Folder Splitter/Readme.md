# Split Folder into Subfolders

> CLI tool that previews or copies direct files into numbered subfolders of a specified size.

## Overview

This script takes a source folder and a file count, then plans numbered subfolders (`data_0`, `data_1`, `data_2`, etc.), each containing up to the requested number of files. It previews by default. Copying requires `--apply --confirm COPY` and uses `shutil.copy2` to preserve metadata.

## Features

- Splits files from a source folder into numbered subfolders
- Preserves file metadata using `shutil.copy2`
- Automatically creates destination subfolders as needed
- Configurable number of files per subfolder via command-line argument
- Recursively resolves absolute paths for both source files and destination folders

## Project Structure

```
Folder Splitter/
├── split_and_copy.py
├── Readme.md
├── pyproject.toml
└── uv.lock
```

## Requirements

- Python 3.13 or later
- No external dependencies

## Installation

```bash
cd "Python Scripts/Folder Splitter"
uv sync --no-config
```

No additional installation is needed.

## Usage

```bash
uv run --no-config python split_and_copy.py <input_folder_path> <count>
```

**Arguments:**
| Argument | Description |
|---|---|
| `input_folder_path` | Path to the folder containing files to split |
| `count` | Maximum number of files per subfolder |

**Example:**

```bash
uv run --no-config python split_and_copy.py .\my_images 20
```

This only previews the planned groups. To copy into `my_images_split/`, run:

```bash
uv run --no-config python split_and_copy.py .\my_images 20 --apply --confirm COPY
```

## How It Works

1. Validates the source, group size, and new output path.
2. Lists direct regular files only; directories and symbolic links are skipped.
3. Shows the planned groups without changing files.
4. Copies only after `--apply --confirm COPY`.

## Configuration

No configuration files. All parameters are provided via command-line arguments.

## Limitations

- Only direct files are included; nested files are not split.
- The output folder must not already exist.
- Copying makes new copies; source files are never moved or deleted.

## Security Notes

The tool never changes files without `--apply --confirm COPY`.

## License

Not specified.
