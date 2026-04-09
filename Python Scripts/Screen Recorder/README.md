# Screen Recorder

> Records the computer screen to an AVI video file using PIL for screen capture and OpenCV for video encoding.

## Overview

A Python script that captures the screen in real time, displays a live preview window, and writes the frames to an AVI video file. The recording continues until the user presses the Escape key. The output file is named using the current timestamp in milliseconds.

## Features

- Captures the full screen in real time using PIL's `ImageGrab`
- Live preview window showing the recording in progress
- Saves recording as an AVI file with XVID codec
- Filename is auto-generated from the current timestamp (milliseconds)
- Press Escape to stop recording and save the file

## Project Structure

```
Screen-Recorder/
├── screen-recorder.py   # Main screen recording script
└── README.md
```

## Requirements

- Python 3.x
- `opencv-python`
- `numpy`
- `Pillow`

## Installation

```bash
cd "Screen-Recorder"
pip install opencv-python numpy Pillow
```

## Usage

```bash
python screen-recorder.py
```

- A "Screen Recorder" preview window will appear showing the live capture.
- Press **Escape** (key code 27) to stop recording.
- The video file (e.g., `1614567890123.avi`) is saved in the current directory.

## How It Works

1. Creates a `VideoWriter` object with XVID codec (`fourcc = cv2.VideoWriter_fourcc(*'XVID')`), 5.0 FPS, and 1920×1080 resolution.
2. In a loop, captures the full screen using `ImageGrab.grab()`.
3. Converts the PIL image to a NumPy array and then from BGR to RGB color space using `cv2.cvtColor()`.
4. Displays the frame in an OpenCV window and writes it to the video file.
5. Checks for the Escape key (`cv2.waitKey(1) == 27`) to break the loop.
6. Releases the video writer and destroys the preview window on exit.

## Configuration

- **Resolution**: Hardcoded to `1920 × 1080` pixels in the `VideoWriter` constructor.
- **Frame rate**: Hardcoded to `5.0` FPS.
- **Codec**: XVID (produces `.avi` files).
- **Output filename**: Auto-generated from `int(round(time.time() * 1000))`.

## Limitations

- Resolution is hardcoded to 1920×1080; will produce incorrectly sized output on displays with different resolutions.
- The 5 FPS frame rate produces choppy video; higher values will increase CPU load.
- `ImageGrab.grab()` is only available on **Windows** and **macOS** (not Linux without additional setup).
- No audio recording.
- No option to select a screen region — always captures the full screen.
- No CLI arguments for resolution, FPS, or output filename.
- The color conversion uses `COLOR_BGR2RGB`, but `ImageGrab` returns RGB and OpenCV writes BGR; this results in swapped red/blue channels in the output file.

## Security Notes

No security concerns.

## License

Not specified.
