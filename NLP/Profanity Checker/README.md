# Profanity Checker

> Python scripts that read a text file and check its content for profanity using an external web API.

## Overview

This project contains two scripts: `openFile.py` reads and prints the contents of a text file, and `checkText.py` reads the same file and sends its content to a web-based profanity detection API (`wdylike.appspot.com`) to check for offensive language.

## Features

- Read text content from a local file
- Check text for profanity via the `wdylike.appspot.com` web API
- Print API response indicating profanity detection result

## Project Structure

```
Profanity-Checker/
├── checkText.py     # Reads file and checks for profanity via API
├── openFile.py      # Reads and prints file contents
├── important.txt    # Note about Python version compatibility
├── movie.txt        # Sample text file with movie quotes
└── structure.txt    # Project planning notes
```

## Requirements

- **Python 2.7** (as noted in `important.txt`)
- `urllib` (standard library — Python 2 version)

**Note**: This code uses `urllib.urlopen()` which is a Python 2 API. It will **not** work with Python 3 without modification (Python 3 uses `urllib.request.urlopen()`).

## Installation

```bash
cd "Profanity-Checker"
```

No external pip packages required.

## Usage

### Check for Profanity

```bash
python checkText.py
```

This reads the text file and sends its contents to the profanity API.

### Just Read the File

```bash
python openFile.py
```

This reads and prints the file contents to the console.

## How It Works

### `checkText.py`

1. **`readFile()`**: Opens a hardcoded file path (`/Users/User/Desktop/uni/profanity editor/movie.txt`), reads its contents, prints them, and passes the text to `checkProfanity()`.
2. **`checkProfanity(text_to_check)`**: Constructs a URL by appending the text to `http://www.wdylike.appspot.com/?q=`, opens the URL with `urllib.urlopen()`, reads the response, and prints the result.

### `openFile.py`

1. **`readFile()`**: Opens the same hardcoded file path, reads and prints the contents.

### Sample Data (`movie.txt`)

Contains famous movie quotes from Apollo 13, Forrest Gump, A Few Good Men, and A Shot in the Dark.

## Configuration

- **File path**: Hardcoded in both scripts as `/Users/User/Desktop/uni/profanity editor/movie.txt`. This must be changed to match your system.
- **API endpoint**: `http://www.wdylike.appspot.com/?q=` (hardcoded in `checkText.py`).

## Limitations

- **Python 2 only**: Uses `urllib.urlopen()` which does not exist in Python 3.
- File path is hardcoded to a macOS/Linux-style absolute path — will not work on Windows or other systems without modification.
- No command-line arguments for specifying input files.
- The API endpoint uses HTTP (not HTTPS).
- No error handling for file not found, network errors, or API failures.
- Files are opened without explicit encoding specification.
- The `wdylike.appspot.com` API may no longer be available or maintained.
- Text with special characters or spaces may not be properly URL-encoded for the API request.

## Security Notes

- **Unencrypted HTTP**: The profanity API is called over plain HTTP (`http://`), not HTTPS. Text content is transmitted in the clear.
- **External API dependency**: Text content is sent to a third-party web service.

## License

Not specified.
