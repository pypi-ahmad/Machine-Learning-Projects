# Music Player With Python

> A GUI music player built with Tkinter and Pygame that lets you browse, play, pause, and stop audio files from a selected directory.

## Overview

This application creates a Tkinter-based music player window. On launch, it prompts the user to select a directory via a file dialog. All files in that directory are listed in a scrollable listbox. The user can play, stop, pause, and unpause audio tracks using Pygame's mixer module.

## Features

- Directory selection dialog on startup to choose a music folder
- Displays all files from the selected directory in a listbox
- **Play** — loads and plays the currently selected track
- **Stop** — stops playback entirely
- **Pause** — pauses the currently playing track
- **Unpause** — resumes a paused track
- Displays the currently playing song title in a label

## Project Structure

```
Music Player With Python/
└── Music Player With Python.py    # Complete music player application
```

## Requirements

- Python 3.x
- pygame
- tkinter (included with standard Python installations)

## Installation

```bash
cd "Music Player With Python"
pip install pygame
```

## Usage

```bash
python "Music Player With Python.py"
```

1. A directory chooser dialog appears — select a folder containing audio files
2. All files in the folder are listed in the player window
3. Click a track in the list, then use the buttons:
   - **PLAY** — plays the selected track
   - **STOP** — stops playback
   - **PAUSE** — pauses playback
   - **UNPAUSE** — resumes playback

## How It Works

1. **Startup**: Initializes a Tkinter window (450×350) and opens `askdirectory()` to let the user pick a folder
2. **File listing**: Uses `os.listdir()` to populate a `Listbox` with all files in the chosen directory
3. **Pygame mixer**: Initializes `pygame` and `pygame.mixer` for audio playback
4. **Controls**:
   - `play()` — calls `pygame.mixer.music.load()` with the active listbox selection, updates the title label, and calls `play()`
   - `stop()` — calls `pygame.mixer.music.stop()`
   - `pause()` — calls `pygame.mixer.music.pause()`
   - `unpause()` — calls `pygame.mixer.music.unpause()`
5. Four colored buttons (PLAY=blue, STOP=red, PAUSE=purple, UNPAUSE=orange) are packed into the window along with the listbox

## Configuration

No configuration files. The music directory is selected interactively at runtime.

## Limitations

- Lists **all** files in the directory, not just audio files (no file type filtering)
- No volume control
- No seek/progress bar or track duration display
- No next/previous track functionality
- No playlist management or shuffle/repeat modes
- The `pos` variable in the listbox population loop is reset to 0 each iteration (effectively always inserts at position 0, resulting in reverse order)
- No error handling for non-audio files or unsupported formats
- The window size is fixed at 450×350 with no responsive layout
- Pygame supports limited audio formats (MP3 support varies by platform; OGG and WAV are reliable)

## License

Not specified.
