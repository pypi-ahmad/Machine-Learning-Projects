# Lane Finder

> A computer vision application that detects road lane lines from video using OpenCV edge detection and Hough line transforms.

## Overview

This project implements a lane detection pipeline for road videos. It processes each video frame through grayscale conversion, Gaussian blur, Canny edge detection, region-of-interest masking, and Hough line detection, then overlays the detected lane lines onto the original frame.

## Features

- Real-time lane detection from video files
- Canny edge detection with configurable thresholds
- Triangular region-of-interest masking to focus on the road area
- Hough Line Transform for line segment detection
- Slope-based classification of left and right lane lines
- Averaging of multiple line segments into single smooth lane lines per side
- Overlays detected lanes onto the original video frame

## Project Structure

```
lane-finder/
├── finding_lanes.py
├── test2.mp4
└── test_image.jpg
```

## Requirements

- Python 3.x
- `opencv-python` (`cv2`)
- `numpy`

## Installation

```bash
cd "lane-finder"
pip install opencv-python numpy
```

## Usage

Ensure `test2.mp4` is in the project directory, then run:

```bash
python finding_lanes.py
```

- The application will open a window displaying the video with detected lane lines overlaid in blue.
- Press **q** to quit the application.

**For static image processing:** The script contains commented-out code (lines near the bottom) for processing `test_image.jpg` instead of video. Uncomment those lines and comment out the video loop to use it.

## How It Works

1. **`canny(img)`**: Converts the frame to grayscale, applies a 5×5 Gaussian blur, then runs Canny edge detection with thresholds 50 and 150.
2. **`region_of_interest(canny)`**: Creates a triangular mask with vertices at (200, height), (550, 250), and (1100, height) to isolate the road region.
3. **`cv2.HoughLinesP()`**: Detects line segments in the masked edge image with parameters: resolution of 2 pixels, angle resolution of π/180, threshold of 100, min line length of 40, max line gap of 5.
4. **`average_slope_intercept(image, lines)`**: Separates detected lines into left (negative slope) and right (positive slope) groups, fits a first-degree polynomial to each group, and averages them into a single line per side.
5. **`make_points(image, line)`**: Converts slope-intercept line parameters into pixel coordinates spanning from the bottom of the frame to 60% of the frame height.
6. **`display_lines(img, lines)`**: Draws the averaged lane lines (blue, 10px thick) on a blank image.
7. The lane overlay is blended with the original frame using `cv2.addWeighted()` (80% original, 100% overlay).

## Configuration

- **Canny thresholds:** 50 (low) and 150 (high) in the `canny()` function.
- **Region of interest vertices:** Hardcoded triangle at (200, height), (550, 250), (1100, height) — tuned for the provided test video resolution.
- **Hough transform parameters:** `minLineLength=40`, `maxLineGap=5`, `threshold=100`.
- **Video file:** Hardcoded as `test2.mp4`.

## Limitations

- Region-of-interest triangle vertices are hardcoded for a specific video resolution and camera position — will not work correctly for different camera angles or resolutions without adjustment.
- No error handling if `test2.mp4` is missing or unreadable.
- The `canny()` function applies Gaussian blur but then runs Canny on the un-blurred `gray` image instead of `blur` (line: `canny = cv2.Canny(gray, 50, 150)` should use `blur`).
- No handling for frames where no lines are detected — `average_slope_intercept` will fail on `np.average` of an empty list.
- Only supports straight lane lines; curved lanes are not handled.
- The video capture loop does not check if `frame` is valid before processing.

## Security Notes

No security concerns identified.

## License

Not specified.
