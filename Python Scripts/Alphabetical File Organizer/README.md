# Alphabetical File Organizer

> A Python script that sorts files in the current directory into folders named for their first letter.

## Overview

This script reads the files in its current working directory, creates single-letter folders (or a `misc` folder for non-alphabetic names), and moves each file according to the first character of its filename.

## Features

- Organizes regular files in a directory you choose
- Uses lowercase initial folders (`a`–`z`) and `misc` for non-alphabetic names
- Shows a dry-run plan by default
- Moves files only when `--apply` is supplied
- Refuses to overwrite an existing destination file
- Leaves directories, symbolic links, and the running script in place

## Install and run

```powershell
cd "Python Scripts/Alphabetical File Organizer"
uv sync
uv run python main.py "C:\path\to\files"
```

The first command is a dry run. Review its plan, then apply the same operation:

```powershell
uv run python main.py "C:\path\to\files" --apply
```

### Example

```
Before:                    After:
├── apple.txt              ├── a/
├── banana.txt             │   └── apple.txt
├── 123.dat                ├── b/
├── main.py                │   └── banana.txt
                           ├── misc/
                           │   └── 123.dat
                           └── main.py
```

## Safety notes

The tool first builds the whole plan and checks every destination for an existing
file. It does not create folders or move files in dry-run mode. If a planned
destination already exists, it stops before moving any file.

The organizer does not recurse into subdirectories and skips symbolic links.
Use it on a dedicated directory rather than this project folder.
