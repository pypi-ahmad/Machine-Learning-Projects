# EasyVideoPlayer

> A terminal-based video player that searches for video files on your system and plays them with audio using OpenCV and ffpyplayer.

## Overview

This script prompts the user for a video filename and a directory to search in, recursively locates the file, changes the working directory to the video's location, and plays the video with synchronized audio using OpenCV for video frames and `ffpyplayer` for audio playback.

## Features

- **Recursive file search**: Locates video files anywhere within a specified directory tree
- **Automatic directory change**: Sets the working directory to the video's parent folder
- **Video playback**: Displays video frames using OpenCV's `cv2.VideoCapture` and `cv2.imshow()`
- **Audio playback**: Plays audio track via `ffpyplayer.player.MediaPlayer`
- **Quit support**: Press `q` to stop playback at any time

## Project Structure

```
EasyVideoPlayer/
├── EasyVideoPlayer.py
├── requirements.txt
└── README.md
```

## Requirements

- Python 3.x
- `opencv-python==4.4.0.42`
- `ffpyplayer==4.3.1`
- `pathlib==1.0.1` (included in Python 3.4+ stdlib)

## Installation

```bash
cd EasyVideoPlayer
pip install -r requirements.txt
```

## Usage

```bash
python EasyVideoPlayer.py
```

Interactive prompts:

```
Name of the video file that you want to play:    video.mp4
Directory that may contain the video:    /home/user/Videos
```

Press `q` to quit playback, or wait for the video to end.

## How It Works

1. **`find_the_video(file_name, directory_name)`** — Walks the directory tree with `os.walk()`, finds all matching filenames, and returns the first match's full path.
2. Uses `pathlib.Path` to extract the video's parent directory and changes the working directory there.
3. **`PlayVideo(video_path)`**:
   - Opens the video with `cv2.VideoCapture(video_path)`
   - Opens the audio with `MediaPlayer(video_path)`
   - Reads frames in a loop, displaying each with `cv2.imshow()`
   - Reads audio frames in parallel with `player.get_frame()`
   - Exits on end of video or when `q` is pressed (checked every 28ms via `cv2.waitKey(28)`)
4. Releases the video capture and destroys all OpenCV windows on exit.

## Configuration

- **Frame delay**: Hardcoded to 28ms in `cv2.waitKey(28)`, targeting roughly 35 FPS playback regardless of the video's actual frame rate.

## Limitations

- `find_the_video()` is called **twice** (once to set the directory, once to get the path for playback), causing redundant directory traversals.
- Audio and video synchronization is approximate — no precise A/V sync mechanism.
- The 28ms frame delay is hardcoded and does not adapt to the video's actual FPS.
- Will crash with `IndexError` if the video file is not found.
- Requires a GUI environment for `cv2.imshow()`.
- `pathlib==1.0.1` in requirements is unnecessary for Python 3.4+.
- No support for pause, seek, or volume control.

## License

Not specified.
