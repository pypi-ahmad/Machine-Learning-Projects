# AudioBook Creator

`Create Audio Book in Python.py` extracts selectable text from a PDF and sends
it to the local `pyttsx3` text-to-speech engine.

## Install and run

```powershell
cd "Python Scripts/AudioBook Creator"
uv sync
uv run python "Create Audio Book in Python.py" input.pdf --output audiobook.wav
```

Add `--speak` to also read the text through the system audio device. The default
output is `audiobook.wav`; support for other output formats depends on the
installed speech-engine backend.

The project uses `pypdf` for modern PDF extraction and `pyttsx3` for offline
speech. Image-only PDFs need OCR before they can be converted.

Automated verification checks extraction validation only. It does not invoke
audio playback or create an audio file.
