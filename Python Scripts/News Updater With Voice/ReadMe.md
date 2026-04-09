# News Updater With Voice

> A Python script that fetches top news headlines using NewsAPI and reads them aloud using text-to-speech, repeating every 10 minutes.

## Overview

This script uses the NewsAPI client to fetch the top 5 coronavirus-related headlines from India, prints each headline description to the console, and reads them aloud using the `pyttsx3` text-to-speech engine. It runs in an infinite loop, refreshing the news every 10 minutes (600 seconds).

## Features

- Fetches top headlines from NewsAPI filtered by keyword (`corona`), country (`in`/India), and language (`en`)
- Reads news descriptions aloud using `pyttsx3` with the SAPI5 speech engine (Windows)
- Automatically refreshes and re-reads news every 10 minutes
- Prints numbered headlines to the console alongside voice output

## Project Structure

```
News Updater With Voice/
├── News.py      # Main script
└── License      # MIT License
```

## Requirements

- Python 3.x
- newsapi-python
- pyttsx3
- SpeechRecognition (imported but unused — required for the script to run without import errors)

## Installation

```bash
cd "News Updater With Voice"
pip install newsapi-python pyttsx3
```

## Usage

1. Obtain a free API key from [NewsAPI.org](https://newsapi.org/)
2. Edit `News.py` and insert your API key:
   ```python
   newsapi = NewsApiClient(api_key='YOUR_API_KEY_HERE')
   ```
3. Run the script:
   ```bash
   python News.py
   ```

The script will fetch and read the latest 5 corona-related headlines from India, then repeat every 10 minutes until stopped with `Ctrl+C`.

## How It Works

1. Initializes `pyttsx3` with the SAPI5 engine and selects voice index `[1]` (typically a female voice on Windows)
2. The `speak()` function queues text and calls `engine.runAndWait()` for synchronous speech
3. The `new()` function:
   - Creates a `NewsApiClient` with the configured API key
   - Calls `get_top_headlines()` with `q='corona'`, `country='in'`, `language='en'`, `page_size=5`
   - Iterates through the returned articles, printing and speaking each description
4. The `__main__` block runs `new()` in an infinite `while True` loop with a 600-second (10-minute) `sleep()` between iterations

## Configuration

| Setting | Location | Default | Description |
|---------|----------|---------|-------------|
| `api_key` | `News.py`, line 16 | `''` (empty) | NewsAPI API key — **must be set** |
| `q` | `News.py`, line 17 | `'corona'` | Search keyword for headlines |
| `country` | `News.py`, line 17 | `'in'` | Country code (India) |
| `page_size` | `News.py`, line 19 | `5` | Number of headlines to fetch |
| `sleep()` | `News.py`, line 34 | `600` | Refresh interval in seconds |
| Voice index | `News.py`, line 8 | `voices[1]` | SAPI5 voice selection |

## Limitations

- The API key is hardcoded as an empty string — script will fail without a valid key
- The search query (`corona`) is hardcoded — no way to change topics without editing code
- Country is hardcoded to India (`in`)
- Uses SAPI5 engine which is Windows-only
- `speech_recognition` is imported in the original README instructions but not actually used in the code
- No error handling for API failures, network issues, or missing article descriptions
- The voice index `[1]` assumes at least two SAPI5 voices are installed
- No graceful shutdown mechanism — requires `Ctrl+C` to stop

## Security Notes

- The API key placeholder is empty in the source code, but users should avoid committing their actual API key to version control

## License

MIT License — Copyright (c) 2020 Arbaz Khan. See the `License` file for full text.
