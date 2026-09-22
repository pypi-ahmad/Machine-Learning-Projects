# EasyVideoPlayer

> A terminal-based video player that searches for video files and plays them with audio using OpenCV and ffpyplayer.

## Overview

This script prompts for a video filename and search directory, locates the file recursively, and plays it with OpenCV frames and `ffpyplayer` audio.

## Features

- **Recursive file search**: Locates video files anywhere within a specified directory tree
- **Video playback**: Displays video frames using OpenCV's `cv2.VideoCapture` and `cv2.imshow()`
- **Audio playback**: Plays audio track via `ffpyplayer.player.MediaPlayer`
- **Quit support**: Press `q` to stop playback at any time

## Project Structure

```
EasyVideoPlayer/
├── EasyVideoPlayer.py
├── pyproject.toml
├── uv.lock
└── README.md
```

## Requirements

- Python 3.13
- `opencv-python` and `ffpyplayer`, managed by uv in `pyproject.toml`
- `pathlib` (Python standard library)

## Installation

```bash
cd EasyVideoPlayer
uv sync
```

## Usage

```bash
uv run python EasyVideoPlayer.py
```

Or provide both values directly:

```bash
uv run python EasyVideoPlayer.py video.mp4 C:\Videos
```

Interactive prompts:

```
Name of the video file that you want to play:    video.mp4
Directory that may contain the video:    /home/user/Videos
```

Press `q` to quit playback, or wait for the video to end.

## How it works

1. **`find_video(file_name, directory_name)`** — Walks the directory tree with `os.walk()` and returns the first matching path.
2. **`play_video(video_path)`**:
   - Opens the video with `cv2.VideoCapture(video_path)`
   - Opens the audio with `MediaPlayer(video_path)`
   - Reads frames in a loop, displaying each with `cv2.imshow()`
   - Reads audio frames in parallel with `player.get_frame()`
   - Exits on end of video or when `q` is pressed (checked every 28ms via `cv2.waitKey(28)`)
4. Releases the video capture and destroys all OpenCV windows on exit.

## Configuration

- **Frame delay**: Hardcoded to 28ms in `cv2.waitKey(28)`, targeting roughly 35 FPS playback regardless of the video's actual frame rate.

## Limitations

- Audio and video synchronization is approximate — no precise A/V sync mechanism.
- The 28ms frame delay is hardcoded and does not adapt to the video's actual FPS.
- Stops with a clear error if the video file is not found or cannot be opened.
- Requires a GUI environment for `cv2.imshow()`.
- No support for pause, seek, or volume control.

## License

Not specified.
