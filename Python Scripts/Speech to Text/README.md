# Speech to Text

Record one microphone phrase and transcribe it with Google Speech Recognition.

## Setup

Requirements: Python 3.13+, a working microphone, and internet access for Google's recognition service.

```powershell
cd "Speech to Text"
uv sync
```

## Usage

```powershell
uv run python speech_to_text.py --output transcript.txt
```

Options:

- `--language CODE`: Google recognition language. Default: `en-US`.
- `--timeout SECONDS`: Time to wait for speech. Default: `10`; use `0` for no limit.
- `--phrase-time-limit SECONDS`: Maximum phrase length. Default: `30`; use `0` for no limit.
- `--device-index N`: Optional microphone device index.
- `--output PATH`: Transcript file. Default: `you_said_this.txt`.
- `--overwrite`: Replace an existing transcript.

## Behavior

The script calibrates ambient noise for one second, records one phrase, sends that audio to Google's recognition service, and saves only a successful transcript as UTF-8 text. It reports separate errors for no speech, unintelligible speech, microphone failures, and recognition-service failures.

Audio is transmitted to Google for transcription. Do not use this tool for sensitive speech unless that data handling is acceptable to you. The tool does not retain audio files locally.

## Project files

```text
Speech to Text/
├── speech_to_text.py
├── pyproject.toml
└── uv.lock
```
