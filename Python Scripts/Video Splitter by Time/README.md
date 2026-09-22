# Video splitter by time

`videosplitter.py` writes two files from a local media input:

1. the interval from `start` to `end`
2. everything after `end`

Content before `start` is not written to either output.

## Requirements

- Python 3.13 or later
- FFmpeg and FFprobe available on `PATH`

The script uses the installed FFmpeg executables directly, so no Python media
wrapper is required.

## Usage

```powershell
cd "Python Scripts/Video Splitter by Time"
uv sync
uv run python videosplitter.py movie.mp4 10 60 selected.mp4 remainder.mp4
```

The example writes `selected.mp4` from 10 to 60 seconds and `remainder.mp4`
from 60 seconds to the end of the input.

## Safety checks

The command verifies that:

- the input exists
- `0 <= start < end <= duration`
- FFmpeg and FFprobe are available
- each output path is new and differs from the input and the other output

It never replaces an existing output file.

## Media behavior

FFmpeg copies streams instead of re-encoding them, which is fast and preserves
the source streams. On codecs with sparse keyframes, a cut can begin near the
preceding keyframe rather than at the exact requested frame. Use an editing tool
that re-encodes when frame-exact boundaries are required.

Run `uv run python videosplitter.py --help` for the full command reference.
