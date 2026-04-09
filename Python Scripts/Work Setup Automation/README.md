# Work Setup Automation

> A Python script that automatically opens your code editor and a set of frequently used websites in Chrome on startup.

## Overview

This script launches a configured text editor/IDE and opens multiple websites in Google Chrome tabs. It's designed to be converted to an executable and placed in the Windows startup folder so your entire development workstation is ready when your computer boots.

## Features

- Opens a configurable text editor or IDE (default: Sublime Text 3)
- Launches multiple website tabs in Google Chrome using `webbrowser`
- Ships with a pre-built `workstation.exe` for direct use
- Designed to run automatically at Windows startup

## Project Structure

```
Work-Setup-Automation/
├── workstation.py       # Main Python script
├── workstation.exe      # Pre-built Windows executable
├── LICENSE              # MIT License
└── README.md
```

## Requirements

- Python 3.x
- `os` (standard library)
- `webbrowser` (standard library)
- Google Chrome installed
- **Optional**: `pyinstaller` to rebuild the `.exe`

No external packages required.

## Installation

```bash
cd "Work-Setup-Automation"
```

To rebuild the executable:

```bash
pip install pyinstaller
pyinstaller -F workstation.py
```

## Usage

### Run directly

```bash
python workstation.py
```

### Set up for Windows auto-start

1. Press `Win + R`, type `shell:startup`, and press Enter
2. Copy `workstation.exe` into the opened Startup folder
3. Restart your system — the script will run automatically on login

## How It Works

1. **`os.startfile(codePath)`** — Opens the application at the configured path (default: Sublime Text 3).
2. **`webbrowser.get(chrome_path).open(url)`** — Opens each URL in the configured Chrome installation. The default URLs are:
   - `stackoverflow.com`
   - `github.com/Arbazkhan4712`
   - `gmail.com`
   - `google.com`
   - `youtube.com`

## Configuration

Edit the following values in `workstation.py`:

| Variable | Default | Description |
|----------|---------|-------------|
| `codePath` | `C:\Program Files\Sublime Text 3\sublime_text.exe` | Path to your text editor / IDE |
| `chrome_path` | `C:/Program Files (x86)/Google/Chrome/Application/chrome.exe %s` | Path to Chrome executable |
| `URLS` | 5 default URLs | Tuple of websites to open at startup |

## Limitations

- All paths and URLs are hardcoded; must edit source code to customize
- Windows-only (`os.startfile` is a Windows-specific function)
- The Chrome path uses the 32-bit Program Files directory; may need updating for 64-bit Chrome
- No error handling if Chrome or the editor is not installed at the specified path
- The pre-built `.exe` will use the original hardcoded paths

## Security Notes

- Contains a hardcoded GitHub profile URL (`github.com/Arbazkhan4712`) which may need to be changed for your use case

## License

MIT License — Copyright (c) 2020 Arbaz Khan
