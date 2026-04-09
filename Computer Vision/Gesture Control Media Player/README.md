# Gesture Control Media Player

> Control media playback (volume, play, pause, mute) using hand gestures detected via webcam.

## Overview

This project uses a webcam to detect hand gestures in real-time by analyzing finger count through convex hull and convexity defects on skin-colored regions. Based on the number of detected fingers, it triggers media control hotkeys (volume up/down, mute, play, pause) using `pyautogui`.

## Features

- Real-time hand detection via webcam within a defined green rectangle ROI (100,100 to 300,300)
- Skin color segmentation using HSV color space
- Finger counting using convex hull and convexity defects with cosine rule angle calculation
- Media control actions mapped to finger count:
  - **1 finger (0 defects):** Volume Up
  - **2 fingers (1 defect):** Volume Down
  - **3 fingers (2 defects):** Mute
  - **4 fingers (3 defects):** Play/Pause
  - **5 fingers (4 defects):** Stop
- On-screen text overlay showing current gesture action
- Press `q` to quit

## Project Structure

```
Gesture-Control-Media-Player/
├── vlc-Gesture-Control.py
├── LICENSE
└── README.md
```

## Requirements

- Python 3.x
- `numpy`
- `opencv-python` (cv2)
- `pyautogui`

## Installation

```bash
cd "Gesture-Control-Media-Player"
pip install numpy opencv-python pyautogui
```

## Usage

```bash
python vlc-Gesture-Control.py
```

Place your hand inside the green rectangle shown on the camera feed. The number of raised fingers determines the media action.

## How It Works

1. **Capture:** Opens the default webcam (`VideoCapture(0)`) and reads frames continuously.
2. **ROI Extraction:** Crops a 200×200 region (100,100 to 300,300) outlined by a green rectangle.
3. **Preprocessing:** Applies Gaussian blur, converts to HSV, and creates a binary mask for skin tones (HSV range `[2,0,0]` to `[20,255,255]`).
4. **Morphology:** Dilation followed by erosion to reduce noise, then another Gaussian blur and threshold.
5. **Contour Detection:** Finds the largest contour (assumed to be the hand).
6. **Convex Hull & Defects:** Computes the convex hull and convexity defects. For each defect, calculates the angle at the far point using the cosine rule.
7. **Finger Counting:** Defects with angles ≤ 90° are counted. The defect count maps to a media action triggered via `pyautogui.hotkey()`.

## Configuration

- **ROI Position:** Hardcoded rectangle at `(100, 100)` to `(300, 300)` — modify in the `cv2.rectangle()` and crop lines.
- **Skin Color Range:** HSV lower `[2, 0, 0]` and upper `[20, 255, 255]` — may need tuning for different skin tones or lighting.
- **Angle Threshold:** 90° — adjustable in the `if angle <= 90` condition.

## Limitations

- Skin color detection via fixed HSV range; may not work well for all skin tones or lighting conditions.
- Only one ROI region (top-left area); hand must be placed precisely inside the green rectangle.
- Bare `except: pass` blocks swallow all errors silently (e.g., when no contours are found).
- Despite the project name referencing "VLC," it uses system-wide media hotkeys via `pyautogui`, not VLC-specific controls.
- The `math` module is used for angle calculation with `3.14` approximation instead of `math.pi`.

## Security Notes

No security concerns identified.

## License

MIT License — Copyright (c) 2020 Arbaz Khan (see `LICENSE` file).

