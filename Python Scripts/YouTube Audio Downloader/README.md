# YouTube Audio Downloader

Download the best available audio-only stream from one YouTube URL, then optionally convert a local downloaded file to MP3.

## Requirements

- Python 3.13 or newer
- [uv](https://docs.astral.sh/uv/)
- FFmpeg available to MoviePy when converting to MP3

## Install

```powershell
cd "Python Scripts\YouTube Audio Downloader"
uv sync
```

## Download audio

Preview a download without contacting YouTube:

```powershell
uv run python .\YouTubeAudioDownloader.py "https://www.youtube.com/watch?v=VIDEO_ID" --dry-run
```

Download the best available audio stream:

```powershell
uv run python .\YouTubeAudioDownloader.py "https://www.youtube.com/watch?v=VIDEO_ID" --output-dir .\downloads
```

The downloaded stream retains its source container extension.

## Convert a local file to MP3

Preview conversion:

```powershell
uv run python .\WebmToMp3.py .\downloads\audio.webm --dry-run
```

Convert using the default adjacent `.mp3` output path:

```powershell
uv run python .\WebmToMp3.py .\downloads\audio.webm
```

Or choose the output path explicitly:

```powershell
uv run python .\WebmToMp3.py .\downloads\audio.webm --output .\downloads\audio.mp3
```

## Notes

- `yt-dlp` and YouTube availability can change; a given URL may be restricted or unavailable.
- Download and convert only content you have permission to save and use.
- No media is downloaded or converted by either `--dry-run` command.
