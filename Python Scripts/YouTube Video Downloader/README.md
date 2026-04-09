# YouTube Video Downloader

> A Tkinter GUI application to download YouTube videos by pasting a URL.

## Overview

This application provides a minimal graphical interface where users paste a YouTube video URL and click a download button. It uses the `pytube` library to fetch the first available stream and downloads the video to the current working directory.

## Features

- Tkinter-based GUI with a fixed 700×300 window
- Text entry field for pasting YouTube video URLs
- One-click download via a "DOWNLOAD" button
- Displays "DOWNLOADED" label upon completion
- Uses the first available stream from `pytube` (typically the highest resolution progressive stream)

## Project Structure

```
YouTube-Video-Downloader/
├── youtube_vid_dl.py    # Main application script
└── README.md
```

## Requirements

- Python 3.x
- `pytube`
- `tkinter` (included with standard Python)

## Installation

```bash
cd "YouTube-Video-Downloader"
pip install pytube
```

## Usage

```bash
python youtube_vid_dl.py
```

1. A window titled "YouTube Video Downloader" opens
2. Paste a YouTube video URL into the "Paste Link Here" field
3. Click the **DOWNLOAD** button
4. The video downloads to the current working directory
5. A "DOWNLOADED" label appears when complete

## How It Works

1. **GUI Setup** — Creates a `Tk` root window (700×300, non-resizable) with a `Label` header, an `Entry` widget bound to a `StringVar`, and a `Button`.
2. **`Downloader()`** — Called on button click:
   - Creates a `YouTube` object from the entered URL
   - Calls `url.streams.first()` to get the first available stream
   - Calls `video.download()` to download to the current directory
   - Places a "DOWNLOADED" `Label` on the window

## Configuration

No configuration files. The download location is always the current working directory.

## Limitations

- No stream quality selection — always downloads the first available stream (`streams.first()`)
- No download progress indicator
- No error handling — invalid URLs or network failures will crash the application
- The "DOWNLOADED" label is placed at a fixed position and may overlap on subsequent downloads
- Window is non-resizable at 700×300
- Downloaded filename is determined by `pytube` (typically the video title)
- No option to choose download directory
- `pytube` frequently breaks due to YouTube API changes and may need updates

## Security Notes

No security concerns identified.

## License

Not specified.
