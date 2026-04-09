# Speech-To-Text

> A Python script that captures speech from the microphone, converts it to text using Google Speech Recognition, and saves the result to a file.

## Overview

This script uses the `speech_recognition` library to listen to audio input from the microphone, converts it to text via Google's speech recognition API, and writes the transcribed text to `output.txt`. It also initializes a `pyttsx3` text-to-speech engine (configured for Windows SAPI5), though it is not actively used for output in the current code.

## Features

- Captures audio from the system microphone
- Converts speech to text using Google Speech Recognition (`recognize_google`)
- Saves transcribed text to `output.txt`
- Initializes a text-to-speech engine via `pyttsx3` with Windows SAPI5 voice

## Project Structure

```
Speech-To-Text/
├── LICENSE
├── output.txt
└── speech-to-text.py
```

## Requirements

- Python 3.x
- `pyttsx3`
- `SpeechRecognition`
- `PyAudio` (required by `speech_recognition` for microphone access)
- A working microphone

## Installation

```bash
cd "Speech-To-Text"
pip install pyttsx3 SpeechRecognition PyAudio
```

> **Note:** On some systems, `PyAudio` may require additional steps:
> - **Windows:** `pip install pyaudio`
> - **Linux:** `sudo apt-get install python3-pyaudio`
> - **macOS:** `brew install portaudio && pip install pyaudio`

## Usage

```bash
python speech-to-text.py
```

**Output:**

```
say something!
done
google think you said:
<your transcribed speech>
```

The transcribed text is also saved to `output.txt` in the same directory.

## How It Works

1. Initializes a `pyttsx3` engine with the Windows SAPI5 driver and sets the first available voice
2. Creates a `speech_recognition.Recognizer` instance
3. Opens the microphone as the audio source and listens for speech
4. Sends the audio to Google's speech recognition API via `r.recognize_google(audio)`
5. Prints the recognized text to the console
6. Writes the text to `output.txt` (overwriting any previous content)

## Configuration

- **Voice selection**: Change `voices[0].id` to `voices[1].id` (or another index) in the `engine.setProperty('voice', ...)` line to switch TTS voices
- **Output file**: Change `'output.txt'` in the `open()` call to customize the output filename
- **TTS engine**: Change `'sapi5'` to `'nsss'` (macOS) or `'espeak'` (Linux) for cross-platform TTS support

## Limitations

- The `pyttsx3` engine and `speak()` function are initialized but never called — the TTS functionality is unused
- No ambient noise adjustment before listening
- The `text` variable is referenced in the `except` block's `open()` call even if recognition fails, causing a `NameError`
- The file is opened with `'w'` mode, overwriting previous transcriptions
- Uses the generic `except Exception` which catches all errors but only prints them
- Windows-specific TTS driver (`sapi5`) is hardcoded

## Security Notes

- Audio is sent to Google's servers for recognition — not suitable for sensitive/private speech data

## License

MIT License (see LICENSE file). Copyright (c) 2020 Arbaz Khan.
