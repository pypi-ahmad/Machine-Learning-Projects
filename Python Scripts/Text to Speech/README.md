# Text to Speech

> Converts text from a file to speech using Google Text-to-Speech (gTTS), saves it as an MP3, and plays it.

## Overview

This script reads text from `abc.txt`, converts it to speech using the Google Text-to-Speech API via the `gTTS` library, saves the audio as `voice.mp3`, and then plays the audio file using the system's default media player.

## Features

- Reads text from a local file (`abc.txt`)
- Converts text to speech using Google's TTS engine
- Supports English language output
- Saves generated audio as an MP3 file
- Automatically plays the audio file after generation

## Project Structure

```
Text_to_speech/
├── txtToSpeech.py
├── abc.txt
├── voice.mp3
├── requirements.txt
└── README.md
```

## Requirements

- Python 3.x
- `gTTS==2.1.1` (specified in requirements.txt)
- Internet connection (gTTS uses Google's online TTS API)

## Installation

```bash
cd Text_to_speech
pip install -r requirements.txt
```

## Usage

1. **Edit the input file:** Place the text you want converted in `abc.txt`.

2. **Run the script:**

   ```bash
   python txtToSpeech.py
   ```

3. **Output:** The script generates `voice.mp3` and opens it with your system's default media player.

**Current `abc.txt` content:**

> Thanks to Gail Cleaver, Beth Barrack, Bingo Nightly, Emily Webber and Sharon Counts. Finally, special thanks to Casey Cromwell. Radio Lab is produced by WNYC New York public radio, and distributed by NPR, National Public Radio.

## How It Works

1. Opens and reads the entire content of `abc.txt`
2. Passes the text to `gTTS(text=file, lang='en', slow=False)` to create a speech object
3. Saves the speech as `voice.mp3` using `speech.save()`
4. Calls `os.system("voice.mp3")` to open the file with the system's default handler

## Configuration

| Item | Location | Description |
|---|---|---|
| Input file | `txtToSpeech.py` line 3 | Hardcoded to `abc.txt` |
| Output file | `txtToSpeech.py` line 6 | Hardcoded to `voice.mp3` |
| Language | `txtToSpeech.py` line 5 | Hardcoded to `'en'` (English) |
| Speed | `txtToSpeech.py` line 5 | `slow=False` (normal speed) |

## Limitations

- Input filename (`abc.txt`) and output filename (`voice.mp3`) are hardcoded — no CLI arguments
- Requires an active internet connection (gTTS sends text to Google's servers)
- `os.system("voice.mp3")` relies on OS file association — may not work on all systems (primarily works on Windows)
- No error handling for missing input file, network errors, or API failures
- The entire file is read and sent as a single request — very large files may exceed API limits
- The generated `voice.mp3` is committed to the repository (unnecessary binary file)

## Security Notes

- Text content is sent to Google's servers for TTS conversion — do not use with sensitive or private text.

## License

Not specified.
