# High Quality YouTube Video Downloader

> GUI-based YouTube video downloader with multiple versions, built with Tkinter and youtube_dl.

## Overview

A collection of three YouTube downloader scripts with Tkinter GUIs. The main version (`YTD.py`) provides a single-link downloader, while the V2.0 and V2.1 versions support downloading up to 4 videos at once with an optional auto-shutdown feature after downloads complete.

## Features

- **YTD.py**: Single-video download with a styled GUI (centered window, custom icon, placeholder text)
- **YouTube_downloader_V2.0**: Batch download of up to 4 videos with optional system shutdown after completion
- **YouTube_downloader_V2.1**: Enhanced V2.0 with progress bars per download, styled UI with colors, and "completed!" labels per entry

## Project Structure

```
High Qulity YouTube Video Downloader/
├── YTD.py                       # Single-link downloader GUI
├── YouTube_downloader_V2.0      # Batch downloader (plain Tkinter, no extension)
├── YouTube_downloader_V2.1.py   # Batch downloader with progress bars
├── yt.ico                       # Window icon for YTD.py
└── ReadMe.md
```

## Requirements

- Python 3.x
- `youtube_dl`
- `tkinter` (included with standard Python on most platforms)

## Installation

```bash
cd "High Qulity YouTube Video Downloader"
pip install youtube_dl
```

## Usage

```bash
# Single video downloader
python YTD.py

# Batch downloader with progress bars
python YouTube_downloader_V2.1.py
```

1. Paste a YouTube URL into the entry field(s).
2. Click **Download** (or **download all** for batch versions).
3. For V2.0/V2.1: Optionally click **yes** to enable auto-shutdown after downloads finish.

## How It Works

- **YTD.py**: Creates a 500×200 centered Tkinter window. On clicking Download, retrieves the URL from the entry field, strips whitespace, and calls `youtube_dl.YoutubeDL().download()`.
- **V2.0**: Provides 4 entry fields. Downloads each sequentially, skipping entries that still contain the placeholder text. If shutdown was enabled (`key=0`), runs `os.system("shutdown /s /t 1")`.
- **V2.1**: Same as V2.0 but adds `ttk.Progressbar` widgets (indeterminate mode) alongside each entry, and uses colored styling (`#dfe6e9` background, `#0984e3` text).

## Configuration

- **YTD.py**: Uses `yt.ico` as the window icon via `root.iconbitmap(r"yt.ico")`. The icon file must be in the working directory.
- **youtube_dl options**: `ydl_opts = {}` — default options are used (downloads best available quality).

## Limitations

- `youtube_dl` is largely unmaintained; `yt-dlp` is the recommended modern fork.
- The `YouTube_downloader_V2.0` file has no `.py` extension, which may cause issues on some systems.
- No download path selection — files are saved to the current working directory.
- No error handling if an invalid URL is provided.
- V2.1 progress bars are indeterminate spinners, not actual progress indicators.
- The auto-shutdown feature (`os.system("shutdown /s /t 1")`) is Windows-only and executes immediately with no confirmation beyond the initial button click.

## Security Notes

- The auto-shutdown feature (`os.system("shutdown /s /t 1")`) will forcefully shut down the computer after downloads. Use with caution.

## License

Not specified.
