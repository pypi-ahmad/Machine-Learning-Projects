# Facebook Video Downloader

> A GUI application built with Tkinter that downloads Facebook videos by URL with a progress bar.

## Overview

This script provides a graphical interface for downloading Facebook videos. It takes a Facebook video post URL, extracts the direct video download link by scraping the mobile version of the page, and downloads the video with a real-time progress bar.

## Features

- Tkinter-based GUI with URL input field, download button, and progress bar
- Extracts direct video download links from Facebook mobile pages
- Threaded download to keep the GUI responsive during download
- Real-time progress bar showing download percentage
- Status bar with download state messages
- Saves downloaded video as `video.mp4` in the script's directory

## Project Structure

```
Facebook_Video_Downloader/
├── script.py          # Main GUI application with download logic
├── pyproject.toml     # Project metadata and dependencies
└── uv.lock            # Locked dependency versions
```

## Requirements

- Python 3.13+
- `requests`, managed by uv in `pyproject.toml`
- `tkinter` (included with standard Python)

## Installation

```bash
cd "Facebook_Video_Downloader"
uv sync
```

## Usage

```bash
uv run python script.py
```

1. A GUI window titled "Facebook Video Downloader" will appear.
2. Enter a Facebook video URL (must contain `www.facebook.com`).
3. Click the **Download** button.
4. Watch the progress bar fill as the video downloads.
5. The video is saved as `video.mp4` in the same directory.

## How It Works

1. **URL validation**: Accepts HTTP(S) URLs only from `facebook.com` or its subdomains.
2. **Link extraction** (`get_download_link`): Requests the mobile page and extracts its direct video URL.
3. **Threaded download** (`VideoDownload`): Downloads in 1KB chunks and sends progress updates through a queue.
4. **Progress monitoring** (`DownloaderApp`): Runs all Tkinter widget updates on the GUI thread.

## Configuration

No configuration files. The video is saved as `video.mp4` beside `script.py`.

## Limitations

- The mobile page scraping approach (`mbasic.facebook.com`) may break if Facebook changes its page structure
- The output filename is still `video.mp4` and overwrites any existing file at that path.
- The app cannot download private, restricted, or otherwise unavailable videos.

## Security Notes

No sensitive credentials in the code.

## License

Not specified.
