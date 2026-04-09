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
└── requirements.txt   # Python dependencies
```

## Requirements

- Python 3.x
- `requests==2.25.1`
- `tkinter` (included with standard Python)

## Installation

```bash
cd "Facebook_Video_Downloader"
pip install -r requirements.txt
```

## Usage

```bash
python script.py
```

1. A GUI window titled "Facebook Video Downloader" will appear.
2. Enter a Facebook video URL (must contain `www.facebook.com`).
3. Click the **Download** button.
4. Watch the progress bar fill as the video downloads.
5. The video is saved as `video.mp4` in the same directory.

## How It Works

1. **URL Validation**: Checks that the entered URL contains `www.facebook.com`.
2. **Link Extraction** (`get_downloadlink`): Replaces `www` with `mbasic` in the URL to access the mobile version, then uses regex to find `/video_redirect/` and extracts the direct video URL from the `?src=` parameter.
3. **Threaded Download** (`VideoDownload` class): Extends `Thread` to download the video in 1KB chunks, updating a shared `Queue` with the download percentage.
4. **Progress Monitoring** (`monitor`): Polls the queue every 10ms to update the progress bar widget.

## Configuration

No configuration files. The video is always saved as `video.mp4` in the current directory.

## Limitations

- Output filename is hardcoded as `video.mp4` — downloading multiple videos overwrites the previous one
- Only validates that the URL contains `www.facebook.com` — no deeper URL validation
- The mobile page scraping approach (`mbasic.facebook.com`) may break if Facebook changes its page structure
- The `queue` variable name shadows the `queue` module import (uses `queue = queue.Queue()`)
- `exit(0)` and `exit(1)` calls in `get_downloadlink` will terminate the GUI
- No error handling or user feedback for network failures during download
- `file.close()` is redundant inside a `with` block

## Security Notes

No sensitive credentials in the code.

## License

Not specified.
