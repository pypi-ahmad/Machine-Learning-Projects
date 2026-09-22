# YouTube Video Downloader

A Tkinter application for downloading a YouTube video from a pasted URL. It saves the highest-resolution progressive stream available to the current working directory.

## Requirements

- Python 3.14 or newer
- [uv](https://docs.astral.sh/uv/)
- Tkinter, included with standard Windows Python installations

## Install

```powershell
cd "Python Scripts\YouTube Video Downloader"
uv sync
```

## Run

```powershell
uv run python .\youtube_vid_dl.py
```

Paste a video URL and select **DOWNLOAD**. The application reports failures in a dialog and saves successful downloads to the directory from which you launched it.

## Notes

- The downloader requests the highest-resolution progressive stream. A matching stream may not be available for every video.
- Download availability can change when YouTube changes its service or a video has access restrictions.
- Download only videos you are authorized to save, and follow the applicable terms and copyright rules.
