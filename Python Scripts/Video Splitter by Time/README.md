# Split a Video File by Given Time Period

> CLI tool that splits a media file into two chunks at a specified time boundary using FFmpeg.

## Overview

This script uses the `ffmpeg-python` wrapper to trim a video (or audio) file into two separate output files based on user-specified start and end times. The first output contains frames from the start time to the end time; the second output contains everything after the end time.

## Features

- Splits any FFmpeg-supported media file into two parts
- Accepts start time and end time in seconds via command-line arguments
- Uses FFmpeg's `trim` filter for frame-accurate splitting
- Supports arbitrary input/output filenames

## Project Structure

```
Split_a_video_file_by_given_time_period/
├── videosplitter.py
├── requirements.txt
└── README.md
```

## Requirements

- Python 3.x
- `ffmpeg-python` (`ffmpeg==1.4` specified in requirements.txt)
- FFmpeg binary installed and available on system PATH

## Installation

```bash
cd "Split_a_video_file_by_given_time_period"
pip install -r requirements.txt
```

> **Note:** You must also install the FFmpeg system binary separately (e.g., via `choco install ffmpeg` on Windows, `brew install ffmpeg` on macOS, or `apt install ffmpeg` on Linux).

## Usage

```bash
python videosplitter.py <inputfile> <starttime> <endtime> <outputfile1> <outputfile2>
```

**Arguments:**
| Argument | Description |
|---|---|
| `inputfile` | Path to the input media file |
| `starttime` | Start time in seconds (float) |
| `endtime` | End time in seconds (float) — split point |
| `outputfile1` | Output filename for the first chunk (start → end) |
| `outputfile2` | Output filename for the second chunk (end → remainder) |

**Example:**

```bash
python videosplitter.py movie.mp4 10.0 60.0 part1.mp4 part2.mp4
```

This trims `movie.mp4` from 10s–60s into `part1.mp4`, and everything after 60s into `part2.mp4`.

## How It Works

1. Parses five positional CLI arguments using `argparse`
2. Creates an FFmpeg input stream from the source file
3. Applies a `trim` filter twice:
   - First with `start=starttime, end=endtime` → output1
   - Second with `start=endtime` → output2
4. Runs both FFmpeg output pipelines sequentially

## Configuration

No configuration files. All parameters are provided via command-line arguments.

## Limitations

- The first output chunk uses `start` and `end` on the trim filter, so the output always begins from the specified start time — content before `starttime` is discarded entirely (there is no "third" file for the 0–start segment).
- No audio stream handling — the `trim` filter is applied only to video. Audio may be lost or out of sync depending on the container format.
- No error handling for invalid time ranges (e.g., endtime > file duration, starttime > endtime).
- Both FFmpeg commands run sequentially via `run()`, blocking until complete.

## Security Notes

No security concerns identified.

## License

Not specified.
