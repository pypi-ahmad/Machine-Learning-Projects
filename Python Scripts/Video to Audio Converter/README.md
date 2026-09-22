# Video to Audio Converter

Download the audio-only stream from a YouTube video. The tool preserves the file extension returned by YouTube instead of incorrectly renaming an MP4 or WebM container as an MP3.

## Requirements

- Python 3.14 or newer
- [uv](https://docs.astral.sh/uv/)

## Install

```powershell
cd "Python Scripts\Video to Audio Converter"
uv sync
```

`uv sync` installs the pinned `pytube` dependency from `uv.lock`.

## Usage

Pass a YouTube URL:

```powershell
uv run python ".\Video To Audio Converter in python.py" "https://www.youtube.com/watch?v=VIDEO_ID"
```

Or start it without an argument to enter the URL at the prompt:

```powershell
uv run python ".\Video To Audio Converter in python.py"
```

Choose a destination directory when needed:

```powershell
uv run python ".\Video To Audio Converter in python.py" "https://www.youtube.com/watch?v=VIDEO_ID" --output-dir ".\downloads"
```

Check URL parsing and the destination without contacting YouTube or writing a media file:

```powershell
uv run python ".\Video To Audio Converter in python.py" "https://www.youtube.com/watch?v=VIDEO_ID" --dry-run
```

## Behavior and limitations

- The output format is determined by YouTube's selected audio stream. The script does not transcode media to MP3.
- The first available audio-only stream is selected.
- Downloading requires network access and may fail when YouTube changes its delivery behavior or restricts a video.
- Only download material you have permission to save and use.
