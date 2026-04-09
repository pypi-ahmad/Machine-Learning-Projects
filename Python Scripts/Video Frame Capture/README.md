# Capture Video Frames

## Overview

A command-line utility that extracts every frame from a video file and saves them as individual JPEG images using OpenCV. Useful for video analysis, dataset creation, or frame-by-frame inspection.

**Type:** CLI Utility

## Features

- Extracts all frames from any OpenCV-supported video format
- Saves frames as sequentially numbered JPEG files (`frame0.jpg`, `frame1.jpg`, …)
- Automatically creates a `captured_frames/` output directory
- Cleans up (deletes and recreates) the output directory on each run to avoid mixing frames from different videos
- Object-oriented design with a `FrameCapture` class

## Dependencies

From `requirements.txt`:

| Package        | Version   |
|----------------|-----------|
| opencv-python  | 4.3.0.36  |

Additional standard library imports: `os`, `shutil`, `sys`

## How It Works

1. The video file path is read from `sys.argv[1]`.
2. A `FrameCapture` object is instantiated, which:
   - Defines the output directory as `captured_frames/`
   - If that directory already exists, it is deleted via `shutil.rmtree()` and recreated
3. The `capture_frames()` method:
   - Opens the video file with `cv2.VideoCapture()`
   - Iterates through every frame using `cv2_object.read()`
   - Writes each frame to `captured_frames/frameN.jpg` using `cv2.imwrite()`
   - Increments the frame counter until no more frames are found

## Project Structure

```
Capture_Video_Frames/
├── capture_video_frames.py   # Main script with FrameCapture class
├── requirements.txt          # Python dependencies
└── README.md
```

## Setup & Installation

```bash
cd Capture_Video_Frames
pip install -r requirements.txt
```

## How to Run

```bash
python capture_video_frames.py <path_to_video_file>
```

**Example:**

```bash
python capture_video_frames.py my_video.mp4
```

Extracted frames will be saved in the `captured_frames/` directory.

## Configuration

No environment variables or config files required. The video file path is the only input, passed as a command-line argument.

## Testing

No formal test suite present.

## Limitations

- No argument validation — running without a command-line argument raises an `IndexError`.
- The last iteration of the loop attempts to write a frame after `cv2_object.read()` returns `False`, which will write an invalid/empty image file.
- No option to specify the output directory — always writes to `captured_frames/` in the current working directory.
- No option to control frame extraction rate (e.g., every Nth frame or frames per second).
- No progress indicator for long videos.
- The output directory is always wiped on each run, with no confirmation prompt.
