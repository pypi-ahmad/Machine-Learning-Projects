# Voice Translator

> A voice-controlled English-to-Catalan translator using speech recognition and text-to-speech.

## Overview

This script listens for spoken English input via the microphone, translates it to Catalan using Google Translate, and then speaks the translated text aloud using a text-to-speech engine.

## Features

- Speech recognition via microphone input (Google Speech Recognition API)
- English to Catalan translation using Google Translate
- Text-to-speech output of the translated text using `pyttsx3` (SAPI5 engine on Windows)
- Error handling with voice prompt to repeat input on recognition failure

## Project Structure

```
Voice Translator/
├── License
├── ReadMe.md
└── trans.py
```

## Requirements

- Python 3.x
- `googletrans` — Google Translate API wrapper
- `pyttsx3` — Text-to-speech conversion
- `SpeechRecognition` — Speech recognition library
- `PyAudio` — Required by SpeechRecognition for microphone access

## Installation

```bash
cd "Voice Translator"
pip install googletrans pyttsx3 SpeechRecognition pyaudio
```

## Usage

```bash
python trans.py
```

The script will:
1. Speak "what I should Translate??"
2. Listen for your spoken English input
3. Translate the recognized text to Catalan
4. Print and speak the translated result

## How It Works

1. **`speak(audio)`** — Uses `pyttsx3` with the SAPI5 engine (Windows) to speak text aloud. Configured to use voice index `[1]` (typically a female voice).
2. **`takeCommand()`** — Opens the microphone via `speech_recognition`, listens for audio, and sends it to Google's speech recognition service with `language='en-in'` (English-India).
3. **`Translate()`** — Prompts the user vocally, captures speech, translates from English (`src='en'`) to Catalan (`dest='ca'`) using `googletrans.Translator`, then prints and speaks the result.

## Configuration

- **Translation language pair**: Hardcoded as English → Catalan (`src='en'`, `dest='ca'`) in the `Translate()` function. Change `dest` to target a different language.
- **Speech recognition language**: Hardcoded to `'en-in'` in `takeCommand()`.
- **TTS voice**: Uses `voices[1]` — this index may vary by system. Adjust the index in `engine.setProperty('voice', voices[1].id)` if needed.

## Limitations

- Translation is hardcoded to English → Catalan only; no runtime language selection.
- The `googletrans` library can be unreliable due to API changes; may require a specific version (e.g., `googletrans==4.0.0-rc1`).
- No graceful exit mechanism; the script runs once and terminates.
- Returns the string `"None"` (not `None`) on recognition failure, which would be passed to the translator if called in a loop.
- Requires a working microphone and internet connection.

## Security Notes

No sensitive data or credentials in the code.

## License

MIT License (see `License` file).
