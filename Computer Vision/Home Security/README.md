# Home Security

> A webcam-based face detection system that emails an image of detected intruders using OpenCV and SMTP.

## Overview

This project uses a Haar Cascade classifier with OpenCV to continuously monitor a webcam feed for faces. When a face is detected, it captures a frame and sends it as an email attachment via Gmail. An email rate limiter prevents flooding by enforcing a 1-minute cooldown between emails tracked via a log file.

## Features

- Real-time face detection using Haar Cascade (`haarcascade_frontalface_alt2.xml`)
- Automatic email notification with intruder image attachment
- Email rate limiting (1-minute minimum between sends) via `email.log`
- Live video feed display with face rectangles drawn
- Press 'q' to quit the monitoring

## Project Structure

```
Home Security/
├── detctor.py                        # Main detection loop using OpenCV
├── send_mail.py                      # Email sending with attachment
├── haarcascade_frontalface_alt2.xml  # Haar Cascade model for face detection
├── email.log                         # Tracks last email timestamp
├── about.txt                         # Project description
├── Further Improvements.txt          # Ideas for future enhancements
└── README.md
```

## Requirements

- Python 3.x
- `opencv-python` (`cv2`)
- A webcam connected to the system
- A Gmail account with "Less Secure Apps" enabled (or App Password)

## Installation

```bash
cd "Home Security"
pip install opencv-python
```

## Usage

```bash
python detctor.py
```

The webcam feed will open. When a face is detected, an email with the captured image is sent. Press **q** to stop.

## How It Works

1. **`detctor.py`**: Opens the default webcam (`VideoCapture(0)`), converts each frame to grayscale, and runs `detectMultiScale()` on the color frame (not the grayscale) using the Haar Cascade classifier. For each detected face, it draws a rectangle on the grayscale image and calls `send_mail(frame)`.
2. **`send_mail.py`**: Checks `email.log` for the last send time. If less than 1 minute has elapsed, it returns immediately. Otherwise, it saves the frame as `intrude.jpg`, constructs a MIME multipart email with the image attached, and sends it via Gmail's SMTP server (port 587 with TLS).

## Configuration

The following values in `send_mail.py` must be updated before use:

```python
gmail_user = 'Enter_gmail-ID_to_be_used_for_sending_the_email'
gmail_password = 'Enter_gmail-ID_password_here'
recipient = 'Enter_gmailID_of_the_recipient'
```

- The Haar Cascade XML file path is resolved relative to `sys.argv[0]`.
- Email cooldown is hardcoded to 1 minute (`timedelta(minutes=1)`).

## Limitations

- Gmail credentials are hardcoded as plaintext in the source code.
- "Less Secure Apps" access is deprecated by Google; App Passwords or OAuth2 should be used instead.
- No motion detection — emails fire on every frame where a face is present (rate-limited to once per minute).
- `recipient` variable is assigned twice, suggesting a copy-paste error.
- No video recording capability.
- The detection draws blue rectangles on the grayscale image but displays the color frame, so the rectangles are not visible in the display window.

## Security Notes

- **Hardcoded credentials**: Gmail username and password are stored in plaintext in `send_mail.py`. Use environment variables or a secrets manager instead.
- Gmail's "Less Secure Apps" setting is a security risk and has been deprecated.

## License

Not specified.
