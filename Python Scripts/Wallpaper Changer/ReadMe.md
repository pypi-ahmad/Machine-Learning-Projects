# Wallpaper Changer using Python

> Automatically fetches random HD wallpapers from Unsplash and sets them as the Windows desktop background.

## Overview

This project contains two scripts: `wallpapers.py` fetches a random landscape wallpaper from the Unsplash API and sets it as the desktop wallpaper on Windows in a continuous loop, while `test.py` is a utility to set a specific local image as the wallpaper using the Windows API.

## Features

- Fetches random HD landscape wallpapers from the Unsplash API
- Downloads wallpaper images using `wget`
- Automatically sets the Windows desktop wallpaper via `ctypes` (`SystemParametersInfoW`)
- Continuous wallpaper rotation every 10 seconds (configurable)
- Graceful exit on `Ctrl+C` with a friendly message
- Utility script (`test.py`) to set a specific local image as wallpaper with 32/64-bit Windows detection

## Project Structure

```
Wallpaper Changer using Python/
├── License
├── ReadMe.md
├── test.py
└── wallpapers.py
```

## Requirements

- Python 3.x
- `requests` — HTTP requests to Unsplash API
- `wget` — Image downloading
- `ctypes` (stdlib) — Windows API calls for setting wallpaper
- An [Unsplash API key](https://unsplash.com/developers)
- **Windows OS** (uses `ctypes.windll.user32.SystemParametersInfoW`)

## Installation

```bash
cd "Wallpaper Changer using Python"
pip install requests wget
```

## Usage

### Main script (auto-rotate wallpapers)

1. Open `wallpapers.py` and set your Unsplash API key and download path:
   ```python
   access_key = 'YOUR_UNSPLASH_API_KEY'
   ```
2. Update the download path in both `get_wallpaper()` and `change_wallpaper()` to your desired location.
3. Run:
   ```bash
   python wallpapers.py
   ```
4. Press `Ctrl+C` to stop.

### Test script (set a specific wallpaper)

1. Update `WALLPAPER_PATH` in `test.py` to point to a local image file.
2. Run:
   ```bash
   python test.py
   ```

## How It Works

### wallpapers.py
1. Sends a GET request to `https://api.unsplash.com/photos/random` with the query `'HD wallpapers'` and orientation `'landscape'`.
2. Extracts the full-resolution image URL from the JSON response (`response['urls']['full']`).
3. Downloads the image to a local path using `wget.download()`.
4. Calls `ctypes.windll.user32.SystemParametersInfoW` with `SPI_SETDESKWALLPAPER` (20) to set the wallpaper.
5. Repeats every 10 seconds in an infinite loop.

### test.py
1. Detects whether the system is 32-bit or 64-bit using `struct.calcsize('P')`.
2. Selects the appropriate `SystemParametersInfoW` or `SystemParametersInfoA` function.
3. Sets the desktop wallpaper from a hardcoded local file path.

## Configuration

| Setting | File | Location |
|---|---|---|
| Unsplash API key | `wallpapers.py` | `access_key = ''` |
| Download path | `wallpapers.py` | `wget.download(...)` and `SystemParametersInfoW(...)` calls |
| Rotation interval | `wallpapers.py` | `time.sleep(10)` — change `10` to desired seconds |
| Wallpaper path | `test.py` | `WALLPAPER_PATH` constant |

## Limitations

- **Windows only** — uses `ctypes.windll` which is not available on Linux/macOS.
- Download path is hardcoded to `C:/Users/projects/wallpaper.jpg`; must be manually changed.
- Overwrites the same file on each download (no wallpaper history).
- All exceptions in the main loop are silently swallowed (`except Exception as e: pass`).
- No validation of the API response before accessing `response['urls']['full']`.
- The 10-second rotation interval is very aggressive for an API with rate limits.

## Security Notes

- The Unsplash API key is stored as a plaintext string in the source code. Consider using environment variables instead.

## License

MIT License (see `License` file).
