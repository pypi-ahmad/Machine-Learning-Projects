# Video Launcher

A Tkinter desktop app for building a local video library, searching it, organizing playlists, and opening selected files in the system default video player.

## Run it

```powershell
uv sync
uv run python main.py
```

Use **Add Files** or **Add Folder** to add video paths. Double-click a row or choose **Play** to open the file with your operating system’s associated player.

## Data

The app stores library paths and playlist membership in `video_library.json` beside `main.py` after the first change. It does not copy or modify the video files themselves. Keep that JSON file private if your library paths are sensitive.

## Dependencies

- Python 3.14+
- Tkinter, included with standard Windows Python installations
