# Text to Speech

Convert a UTF-8 text file into an MP3 using Google Text-to-Speech.

## Setup

Requirements: Python 3.13+ and internet access for gTTS.

```powershell
cd "Text to Speech"
uv sync
```

## Usage

```powershell
uv run python txtToSpeech.py abc.txt --output voice.mp3
```

Options:

- `--output PATH`: New MP3 destination. Default: `voice.mp3`.
- `--language CODE`: gTTS language code. Default: `en`.
- `--slow`: Generate slower speech.
- `--play`: Open the generated MP3 with the Windows default player after saving.

The command refuses to replace an existing MP3. Choose another output path or remove the old file intentionally.

## Behavior

The script reads UTF-8 text, validates that it is not empty, sends it to the Google TTS service, and saves the returned audio as MP3. Playback is disabled unless `--play` is supplied.

Text is transmitted to Google for speech synthesis. Do not use sensitive content unless that data handling is acceptable to you.

## Project files

```text
Text to Speech/
├── txtToSpeech.py
├── abc.txt
├── pyproject.toml
└── uv.lock
```

`voice.mp3` is a legacy sample artifact and is not overwritten by default.
