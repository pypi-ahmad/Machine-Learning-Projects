# Video to Audio Converter

> A command-line tool that downloads a YouTube video's audio track and saves it as an MP3 file.

## Overview

This script takes a YouTube video URL as input, downloads only the audio stream using `pytube`, and renames the resulting `.mp4` file to `.mp3`. The output file is saved in the current working directory, named after the video ID.

## Features

- Downloads audio-only stream from YouTube videos
- Automatically selects the first available audio stream
- Converts output from `.mp4` to `.mp3` by renaming
- Cross-platform file renaming (uses `ren` on Windows, `mv` on Unix)
- Output filename is the YouTube video ID

## Project Structure

```
Video To Audio Converter in Python/
├── Video To Audio Converter in python.py
└── README.md
```

## Requirements

- Python 3.x
- pytube

## Installation

```bash
cd "Video To Audio Converter in Python"
pip install pytube
```

## Usage

```bash
python "Video To Audio Converter in python.py"
```

When prompted, enter a YouTube video URL:

```
Enter YouTube video URL: https://www.youtube.com/watch?v=dQw4w9WgXcQ
```

The audio file will be saved as `<video_id>.mp3` in the current directory.

## How It Works

1. Prompts the user for a YouTube video URL.
2. Extracts the video ID using `pytube.extract.video_id()`.
3. Creates a `YouTube` object and filters streams to `only_audio=True`.
4. Downloads the first audio stream with the video ID as the filename (saves as `.mp4`).
5. Renames the `.mp4` file to `.mp3` using an OS-appropriate command:
   - Windows: `os.system('ren ...')`
   - Unix/macOS: `os.system('mv ...')`

## Configuration

No configuration needed. Files are saved to the current working directory.

## Limitations

- The file is not actually transcoded — it's an MP4 audio container renamed to `.mp3`. Some players may not handle this correctly.
- Uses `os.system()` for file renaming instead of `os.rename()`, which is less portable and secure.
- No error handling for invalid URLs, unavailable videos, or network failures.
- `pytube` frequently breaks when YouTube updates its internal APIs and may require updating.
- File paths with spaces may cause issues with the `os.system()` rename commands since the paths are not quoted.

## License

Not specified.
