# Music Player

A local Tkinter desktop player for MP3, WAV, OGG, and FLAC files. Add individual tracks or a folder, then use the playlist controls to play, pause, stop, or move between tracks.

## Requirements

- Python 3.13
- A local audio device supported by pygame

Tkinter is included with the standard Windows Python distribution. The project pins Python below 3.14 because pygame does not currently provide a compatible Windows distribution for Python 3.14.

## Run

From this directory:

```powershell
uv sync --no-config
uv run --no-config python main.py
```

Use **Add Files** or **Add Folder** to build the playlist. Double-click a listed track, or select one and use the playback controls.

## Behavior and limits

- The player only reads files you select; it does not upload or modify them.
- It filters folder imports to MP3, WAV, OGG, and FLAC filenames. Actual playback support depends on pygame and the local audio backend.
- If pygame or the audio backend cannot initialize, the window still opens and playback controls are disabled with a visible reason.
- The progress slider is visual only; seeking is not implemented because pygame does not provide reliable duration and seek support for every supported format.
