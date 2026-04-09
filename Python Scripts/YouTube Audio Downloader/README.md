# YouTube Audio Downloader

> Download the highest-quality audio stream from a YouTube video and optionally convert `.webm` to `.mp3`.

## Overview

This project contains two scripts: one that downloads the best available audio stream from a YouTube video using the `pafy` library, and another that converts the downloaded `.webm` file to `.mp3` format using `moviepy`.

## Features

- Downloads the highest bitrate audio stream from a YouTube video
- Displays video metadata: title, rating, view count, author, and length
- Converts `.webm` audio files to `.mp3` format using `moviepy`
- Uses `pafy` and `youtube-dl` as the download backend

## Project Structure

```
Youtube-Audio-Downloader/
├── YouTubeAudioDownloader.py    # Downloads audio from YouTube
├── WebmToMp3.py                 # Converts .webm to .mp3
├── requirements.txt             # Pinned dependencies
└── README.md
```

## Requirements

- Python 3.x
- `pafy` (0.5.5)
- `youtube-dl` (2020.12.31)
- `moviepy` (for `.webm` to `.mp3` conversion — not in requirements.txt)

All download dependencies are pinned in `requirements.txt`.

## Installation

```bash
cd "Youtube-Audio-Downloader"
pip install -r requirements.txt
pip install moviepy   # For WebmToMp3.py
```

## Usage

### Step 1: Download audio

1. Edit `YouTubeAudioDownloader.py` and replace the `url` variable:
   ```python
   url = "https://www.youtube.com/watch?v=YOUR_VIDEO_ID"
   ```
2. Run the script:
   ```bash
   python YouTubeAudioDownloader.py
   ```
3. The script prints video details and downloads the best audio stream as a `.webm` file.

### Step 2: Convert to MP3 (optional)

1. Edit `WebmToMp3.py` and set the input/output paths:
   ```python
   clip = mp.AudioFileClip("path/to/file.webm").subclip()
   clip.write_audiofile("path/to/output.mp3")
   ```
2. Run:
   ```bash
   python WebmToMp3.py
   ```

## How It Works

### `YouTubeAudioDownloader.py`
1. Creates a `pafy` video object from the URL.
2. Prints video metadata (`title`, `rating`, `viewcount`, `author`, `length`).
3. Calls `video.getbestaudio()` to select the highest bitrate audio stream.
4. Calls `bestaudio.download()` to save the file (typically `.webm` format).

### `WebmToMp3.py`
1. Opens the `.webm` file as an `AudioFileClip` using `moviepy.editor`.
2. Calls `.subclip()` (no arguments = full clip).
3. Writes the audio to an `.mp3` file via `write_audiofile()`.
4. Closes the clip.

## Configuration

Both scripts require manual editing of hardcoded values:

| Variable | File | Description |
|----------|------|-------------|
| `url` | `YouTubeAudioDownloader.py` | YouTube video URL (placeholder by default) |
| `AudioFileClip(...)` path | `WebmToMp3.py` | Path to input `.webm` file |
| `write_audiofile(...)` path | `WebmToMp3.py` | Path for output `.mp3` file |

## Limitations

- YouTube URL must be manually pasted into the source code; no CLI argument or GUI
- `youtube-dl` (2020.12.31) is outdated and may fail with current YouTube; consider `yt-dlp` as a replacement
- `pafy` has known compatibility issues with newer YouTube changes
- `moviepy` is not listed in `requirements.txt`
- No error handling in either script
- The `.webm` output filename is determined by `pafy` (usually the video title) and cannot be customized
- `subclip()` with no arguments processes the entire file, which is the intended behavior but could be confusing

## Security Notes

No security concerns identified. No API keys are used.

## License

Not specified.
