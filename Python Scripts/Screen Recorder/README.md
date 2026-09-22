# Screen Recorder

A Windows desktop utility that captures the primary screen to an AVI file and
shows a live OpenCV preview. Press Escape in the preview window to finish the
recording cleanly.

## Run

```powershell
uv sync
uv run python screen-recorder.py
```

The default output is a timestamped `.avi` file in the current directory.

```powershell
uv run python screen-recorder.py --output demo.avi --fps 10
```

## Behavior

- The recording dimensions are detected from the first captured frame, so the
  writer matches the active screen instead of assuming 1920x1080.
- Captured RGB frames are converted to OpenCV's BGR format before previewing
  and writing, avoiding red/blue channel swaps.
- The writer and preview window are closed even if the recording is interrupted.

## Limitations

- Captures only the primary full screen; it cannot select a region or monitor.
- Records video only, with no audio track.
- Screen-capture permission and an interactive desktop session are required.
- The XVID AVI codec must be available through the installed OpenCV build.

## Dependencies

uv manages NumPy, OpenCV, and Pillow in `pyproject.toml`; resolved versions are
recorded in `uv.lock`.
