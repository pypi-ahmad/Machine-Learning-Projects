# Speech to Text

> A Python script that records speech from the microphone and saves the transcribed text to a file using Google Speech Recognition.

## Overview

This script listens to audio from the system microphone, adjusts for ambient noise, converts speech to text using Google's recognition API, and saves the result to `you_said_this.txt`.

## Features

- Captures audio input from the microphone
- Automatic ambient noise adjustment via `adjust_for_ambient_noise()`
- Speech-to-text conversion using Google Speech Recognition (English language)
- Saves the transcribed text to `you_said_this.txt`
- Error handling for unrecognized speech

## Project Structure

```
Speech_to_text/
├── requirements.txt
└── speech_to_text.py
```

## Requirements

- Python 3.x
- `SpeechRecognition==3.8.1`
- `PyAudio==0.2.11`
- A working microphone

## Installation

```bash
cd "Speech_to_text"
pip install -r requirements.txt
```

> **Note:** `PyAudio` may require additional system-level dependencies:
> - **Windows:** `pip install pyaudio`
> - **Linux:** `sudo apt-get install python3-pyaudio`
> - **macOS:** `brew install portaudio && pip install pyaudio`

## Usage

```bash
python speech_to_text.py
```

**Output:**

```
I'm trying to hear you:
the last sentence you spoke was saved in you_said_this.txt
```

The transcribed text is saved to `you_said_this.txt` in the same directory.

## How It Works

1. `record_voice()` creates a `speech_recognition.Recognizer` instance
2. Opens the microphone as a context manager via `speech_recognition.Microphone()`
3. Calls `adjust_for_ambient_noise()` to calibrate for background noise
4. Listens for audio input with `microphone.listen()`
5. Sends the audio to Google's API with `microphone.recognize_google(audio, language='en')`
6. Returns the recognized phrase (or an error message string)
7. The main block writes the returned phrase to `you_said_this.txt`

## Configuration

- **Language**: Change `language='en'` in the `recognize_google()` call to support other languages (e.g., `'es'` for Spanish, `'fr'` for French)
- **Output file**: Change `'you_said_this.txt'` in the `open()` call to customize the output filename

## Limitations

- The error handling references `speech_recognition.UnkownValueError` (typo — should be `UnknownValueError`), which will cause a `NameError` at runtime instead of catching the exception
- The file is opened with `'w'` mode, overwriting previous transcriptions each time
- No timeout for listening — the script will wait indefinitely for speech input
- Only processes a single speech input per run
- The returned error message string would be written to the file as if it were valid transcription

## Security Notes

- Audio is sent to Google's servers for recognition — not suitable for private or sensitive speech data

## License

Not specified.
