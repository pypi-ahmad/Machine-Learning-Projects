# Room Security Using Laptop Webcam

> A Flask-based web application that streams live webcam video with real-time face detection using OpenCV's Haar cascade classifier.

## Overview

This project creates a local web server that streams video from the laptop's webcam to a browser. The video feed includes real-time face detection — detected faces are highlighted with green rectangles. It uses Flask for the web server and OpenCV for video capture, processing, and face detection via a Haar cascade XML file.

## Features

- Live webcam video streaming to a web browser
- Real-time face detection using Haar cascade classifier (`faces.xml`)
- Detected faces highlighted with green bounding rectangles
- Flask web server with MJPEG streaming (`multipart/x-mixed-replace`)
- Frame downscaling by factor 0.6 for faster processing
- Accessible from any device on the local network

## Project Structure

```
Room Security Using Laptop Webcam/
├── main.py              # Flask web server with routes for index and video feed
├── camera.py            # VideoCamera class — captures frames, detects faces
├── faces.xml            # Haar cascade classifier for frontal face detection
├── templates/
│   └── index.html       # HTML page displaying the video stream
└── Readme.md
```

## Requirements

- Python 3.x
- `flask`
- `opencv-python`

## Installation

```bash
cd "Room Security Using Laptop Webcam"
pip install flask opencv-python
```

## Usage

```bash
python main.py
```

The server starts on `http://0.0.0.0:5000`. Open this URL in a browser (or use your machine's local IP address from another device on the same network) to view the live video feed with face detection.

## How It Works

1. **`main.py`** initializes a Flask app with two routes:
   - `/` — renders `index.html`, which contains an `<img>` tag pointing to the video feed.
   - `/video_feed` — returns a streaming `Response` using the `gen()` generator function that continuously yields JPEG frames.
2. **`camera.py`** defines the `VideoCamera` class:
   - `__init__()` opens the default webcam (`cv2.VideoCapture(0)`).
   - `get_frame()` reads a frame, downscales it by factor `ds_factor=0.6`, converts to grayscale, and runs `face_cascade.detectMultiScale()` to find faces. A green rectangle is drawn around the first detected face. The frame is encoded as JPEG and returned as bytes.
   - `__del__()` releases the webcam.
3. **`faces.xml`** is a pre-trained Haar cascade XML file loaded by `cv2.CascadeClassifier`.

## Configuration

- **Server address**: Hardcoded to `0.0.0.0:5000` in `main.py`. Debug mode is `False`.
- **Downscale factor**: `ds_factor = 0.6` in `camera.py` controls frame resizing before face detection.
- **Cascade parameters**: `detectMultiScale(gray, 1.3, 5)` — scale factor 1.3, minimum neighbors 5.

## Limitations

- Only draws a rectangle around the **first** detected face (the `break` statement exits the loop after the first detection).
- No recording, alerting, or notification functionality — purely a video streamer with face detection visualization.
- No authentication on the web server; anyone on the local network can access the feed.
- The webcam index is hardcoded to `0` (default camera).
- Flask debug mode is disabled (`debug=False`), but the built-in Flask server is not suitable for production.
- No error handling if the webcam is unavailable or already in use.

## Security Notes

- The web server has **no authentication** — any device on the local network can view the webcam feed by navigating to port 5000.
- The Flask development server is not designed for production use; consider using a WSGI server (e.g., Gunicorn) if deploying beyond local use.

## License

Not specified.





