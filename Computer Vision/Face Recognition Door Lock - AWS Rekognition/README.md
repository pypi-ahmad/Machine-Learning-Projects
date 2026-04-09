# Face Recognition Door Lock with AWS Rekognition & Raspberry Pi 3

> A Raspberry Pi–based door lock system that uses AWS Rekognition for face recognition to grant or deny physical access.

## Overview

This project implements a smart door lock using a Raspberry Pi, a camera module, a physical push button, and an electric door lock. When the button is pressed, the Pi captures an image via the camera, sends it to AWS Rekognition to match against a pre-indexed face collection, and unlocks the door (via GPIO) if a recognized face is detected.

## Features

- Captures images using the Raspberry Pi Camera (`picamera`)
- Indexes face images from an AWS S3 bucket into an AWS Rekognition collection (`train.py`)
- Matches captured faces against the indexed collection with configurable confidence threshold (85%) (`recognition.py`)
- Controls an electric door lock via GPIO pin 17 using `gpiozero` (`main.py`)
- Responds to a physical push button on GPIO pin 26 to trigger recognition
- Unlocks the door for 10 seconds upon successful face match
- Logs recognition output to `text.txt`

## Project Structure

```
Face-Recognition-Door-Lock-with-AWS-Rekognition-Raspberry-Pi3-master/
├── main.py           # Main entry point: button listener, triggers recognition, controls door lock
├── recognition.py    # Captures image via PiCamera, sends to AWS Rekognition for face matching
├── train.py          # Indexes faces from S3 bucket into a Rekognition collection
├── text.txt          # Log file for recognition output
├── img/              # Documentation images
└── LICENSE           # MIT License
```

## Requirements

- Python 2/3 (code references both `python3` and standard Python)
- `boto3` — AWS SDK for Python (S3 and Rekognition)
- `gpiozero` — GPIO control for Raspberry Pi
- `picamera` — Raspberry Pi Camera interface

### Hardware

- Raspberry Pi (any version)
- Raspberry Pi Camera or USB Webcam
- Push button (connected to GPIO 26)
- Electric door lock (connected to GPIO 17)

## Installation

```bash
cd "Face-Recognition-Door-Lock-with-AWS-Rekognition-Raspberry-Pi3-master"
pip install boto3 gpiozero picamera
```

## Usage

### Step 1: Train the Face Collection

1. Create an AWS S3 bucket and organize images into folders named after each person.
2. Configure AWS credentials and bucket/collection names in `train.py`.
3. Run:

```bash
python train.py
```

This deletes any existing Rekognition collection with the same name, creates a new one, and indexes all face images from the S3 bucket.

### Step 2: Configure Recognition

In `recognition.py`, set:
- `directory` — local folder path on the Raspberry Pi for captured images
- `collectionId` — the Rekognition collection name
- AWS credentials (`aws_access_key_id`, `aws_secret_access_key`, `region_name`)

### Step 3: Run the Door Lock System

```bash
python main.py
```

Press the physical button to trigger face capture and recognition. The door unlocks for 10 seconds if a match is found.

## How It Works

1. **`train.py`**: Connects to S3, lists all objects in a bucket, and indexes each face image into an AWS Rekognition collection. Folder names in S3 serve as labels (`ExternalImageId`).
2. **`recognition.py`**: Captures a photo using PiCamera, reads the image bytes, and calls `search_faces_by_image` on the Rekognition collection. Returns the matched person's name and confidence score.
3. **`main.py`**: Listens for a button press on GPIO 26. On press, calls `match.py` via `subprocess`. If the returned text contains `"---"` (expected to match a person's name), it activates the LED/lock on GPIO 17 for 10 seconds.

## Configuration

The following values **must** be configured before use:

| File             | Variable                  | Description                          |
|------------------|---------------------------|--------------------------------------|
| `train.py`       | `aws_access_key_id`       | AWS access key                       |
| `train.py`       | `aws_secret_access_key`   | AWS secret key                       |
| `train.py`       | `region_name`             | AWS region                           |
| `train.py`       | `bucket`                  | S3 bucket name                       |
| `train.py`       | `collectionId`            | Rekognition collection name          |
| `recognition.py` | `aws_access_key_id`       | AWS access key                       |
| `recognition.py` | `aws_secret_access_key`   | AWS secret key                       |
| `recognition.py` | `region_name`             | AWS region (default: `ap-south-1`)   |
| `recognition.py` | `directory`               | Local image save directory           |
| `recognition.py` | `collectionId`            | Rekognition collection name          |
| `main.py`        | `"---"` string check      | Name substring to match for unlock   |

## Limitations

- `main.py` calls `python3 match.py` via subprocess, but the actual recognition script is named `recognition.py` — this mismatch will cause a runtime error unless renamed
- The face match check in `main.py` uses `"---"` as the substring to look for, which needs to be replaced with the actual person's name
- Bare `except` clauses in `recognition.py` suppress all errors silently
- No multi-face support — `MaxFaces=1` in both training and recognition
- No graceful shutdown or error recovery in the main loop
- Indentation appears inconsistent in `main.py` (mixed tabs/spaces)

## Security Notes

- **AWS credentials are hardcoded as empty strings** — these must be filled in before use and should ideally use environment variables or AWS IAM roles instead
- No encryption or secure storage of credentials in the code
- The `text.txt` log file stores recognition results in plaintext

## License

MIT License — Copyright (c) 2020 Arbaz Khan
