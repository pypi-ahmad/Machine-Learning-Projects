# Video frame capture

`capture_video_frames.py` extracts JPEG frames from a local video with OpenCV.
It works with video formats that the installed OpenCV build can decode.

## Install

```powershell
cd "Python Scripts/Video Frame Capture"
uv sync
```

## Usage

Extract every frame into a new directory:

```powershell
uv run python capture_video_frames.py video.mp4 --output captured_frames
```

Save every tenth frame instead:

```powershell
uv run python capture_video_frames.py video.mp4 --output sampled_frames --every 10
```

Frames are named `frame_000000.jpg`, `frame_000001.jpg`, and so on.

## Safety and limits

The output directory must not already exist. This prevents a capture run from
deleting or mixing files from an earlier run. Choose a different `--output`
path when you need another capture.

The script checks that the input path is a file and reports a clear error when
OpenCV cannot open it. It releases the video handle when extraction finishes or
an error occurs.

Run `uv run python capture_video_frames.py --help` for the command reference.
