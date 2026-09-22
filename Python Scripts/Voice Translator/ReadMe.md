# Voice Translator

Translate English speech or supplied text, then optionally speak the translation. The default direction is English to Catalan.

## Requirements

- Python 3.14 or newer
- [uv](https://docs.astral.sh/uv/)
- A working microphone for voice input
- Internet access for Google speech recognition and translation

## Install

```powershell
cd "Python Scripts\Voice Translator"
uv sync
```

The locked environment includes `SpeechRecognition`, `sounddevice`, `googletrans`, and `pyttsx3`. `sounddevice` is used for microphone capture; PyAudio is not required.

## Usage

Record five seconds of English speech and translate it to Catalan:

```powershell
uv run python .\trans.py
```

Translate text directly without using the microphone. `--mute` prevents text-to-speech output:

```powershell
uv run python .\trans.py --text "Good morning" --mute
```

Choose source and target language codes, or a recording duration:

```powershell
uv run python .\trans.py --source en --target fr --seconds 7
```

Inspect the selected configuration without recording, speaking, or making a network request:

```powershell
uv run python .\trans.py --target de --dry-run
```

## Notes

- Speech recognition is configured for `en-IN` by default. Change it with `--listen-language` when appropriate.
- Translation and speech recognition depend on external Google services. Availability and results can vary.
- The text-to-speech voice is the platform default; no machine-specific voice index is assumed.
- Use this tool only where recording and sending spoken content to third-party services is appropriate.
